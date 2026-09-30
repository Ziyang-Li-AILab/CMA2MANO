# -*- coding: utf-8 -*-
"""每个被试一组 MANO beta（左右手共用、同一被试的所有录制共用）。

旧流程逐录制、逐手把 beta 自由拟合到 CMA 的逐段骨长（clamp 在 ±3）：40 只手全部有分量顶在边界上，
同一个人左右手 beta 的 L2 距离平均 2.36，同一个人不同录制的 beta 也各不相同。根源是 CMA 手套的骨架参数
本身不符合解剖，例如 lxy 的中节指骨只有 11-17 mm、近节 58-74 mm，拇指 T0->T1 在所有人身上都恰好是 30 mm；
而且 CMA 的 MCP 普遍比 MANO 的 MCP 靠近手腕 10-25 mm（手掌短、手指长，见诊断 split_residual_mm）。
逐段追这些数，beta 只能被顶到边界，得到畸形的手。

这里只拟合对“关节放在哪里”不敏感的整体尺寸：
  腕 -> MCP + 三节指骨之和     每根手指的整条链长（4 个）；MCP 放在哪都不影响
  拇指四段之和                 拇指整条链长（T0->T1 是常数 30 mm，所以容差放宽）
  食指 MCP <-> 小指 MCP        掌宽
同一被试所有录制、左右两手的目标先取平均，再解一个 beta 的 MAP：长度残差按 3 mm（拇指 6 mm）归一化，
先验 ||beta||^2（MANO 形状 PCA 系数本来就是标准正态），所以先验强度与该被试有几条录制无关。
MANO 左右模板在同一 beta 下骨长完全对称（左手 shapedirs 已按 ACE / smplx 的约定镜像），
所以左右共用 beta 是自洽的。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from robust_convert.cleaning import repair_hand
from robust_convert.cma_source import read_human
from robust_convert.mano_model import MANO_TO_OPENPOSE, TIP_VERTEX_IDS
from robust_convert.mcap_io import find_inputs

from .anatomy import SIDES, get_anatomy, rest_points

FINGER_MCP = (5, 9, 13, 17)
LENGTH_NAMES = ['index_chain', 'middle_chain', 'ring_chain', 'pinky_chain', 'palm_width', 'thumb_chain']
LENGTH_SIGMA_MM = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 6.0])   # 目标长度容差（残差按它归一化）
SPLIT_NAMES = ['wrist_I_MCP', 'wrist_M_MCP', 'wrist_R_MCP', 'wrist_P_MCP',
               'index_finger', 'middle_finger', 'ring_finger', 'pinky_finger']
BETA_PRIOR = 1.0           # ||beta||^2 的权重（相对于归一化残差平方）：N(0, I) 先验下的 MAP
BETA_BOUND = 3.0
METHOD = 'per_subject_shared_lr_beta_v2'


def cma_lengths(offsets):
    """CMA 骨架常量偏移 (20,3) mm -> (整体尺寸 (6,), 手掌/手指拆分 (8,)) mm。offsets[k] = 关节 k+1 相对父关节。"""
    off = np.asarray(offsets, np.float64)
    L = np.linalg.norm(off, axis=1)
    palm = [L[m - 1] for m in FINGER_MCP]
    finger = [L[m] + L[m + 1] + L[m + 2] for m in FINGER_MCP]
    main = [p + f for p, f in zip(palm, finger)]
    main.append(np.linalg.norm(off[4] - off[16]))
    main.append(L[0] + L[1] + L[2] + L[3])
    return np.array(main), np.array(palm + finger)


def mano_lengths(points_mm):
    """MANO 模板 21 点（OpenPose 顺序，mm）-> 同样的 (整体尺寸, 拆分)。"""
    p = np.asarray(points_mm, np.float64)

    def d(a, b):
        return float(np.linalg.norm(p[a] - p[b]))
    palm = [d(m, 0) for m in FINGER_MCP]
    finger = [d(m, m + 1) + d(m + 1, m + 2) + d(m + 2, m + 3) for m in FINGER_MCP]
    main = [a + b for a, b in zip(palm, finger)]
    main.append(d(5, 17))
    main.append(d(0, 1) + d(1, 2) + d(2, 3) + d(3, 4))
    return np.array(main), np.array(palm + finger)


def _rest_linear(model):
    """模板 21 点对 beta 线性：base (21,3) m、dirs (21,3,10)。"""
    base = rest_points(model)
    joint_dirs = np.einsum('jv,vck->jck', model['J_regressor'], model['shapedirs'])
    dirs = np.concatenate([joint_dirs, model['shapedirs'][TIP_VERTEX_IDS]])[MANO_TO_OPENPOSE]
    return base, dirs


def _named(names, values):
    return dict(zip(names, np.round(values, 2).tolist()))


def fit_beta(targets, splits=None, sigma_mm=LENGTH_SIGMA_MM, prior=BETA_PRIOR):
    """targets: [(side, (6,) mm), ...] -> (beta (10,), 诊断)。splits 仅用于诊断。"""
    linear = {side: _rest_linear(get_anatomy(side).model) for side in SIDES}

    def lengths(side, beta):
        base, dirs = linear[side]
        return mano_lengths((base + dirs @ beta) * 1000)

    side0 = targets[0][0]
    mean_target = np.mean([t for _, t in targets], axis=0)

    def residual(beta):
        return np.concatenate([(lengths(side0, beta)[0] - mean_target) / sigma_mm, np.sqrt(prior) * beta])

    beta = least_squares(residual, np.zeros(10), bounds=(-BETA_BOUND, BETA_BOUND)).x
    fitted, fitted_split = lengths(side0, beta)
    template, template_split = lengths(side0, np.zeros(10))
    diag = {
        'target_mean_mm': _named(LENGTH_NAMES, mean_target),
        'mano_fitted_mm': _named(LENGTH_NAMES, fitted),
        'mano_template_mm': _named(LENGTH_NAMES, template),
        'residual_mm': _named(LENGTH_NAMES, fitted - mean_target),
        'beta_norm': float(np.linalg.norm(beta)), 'beta_max_abs': float(np.abs(beta).max()),
    }
    if splits:
        mean_split = np.mean(splits, axis=0)
        diag['split_target_mm'] = _named(SPLIT_NAMES, mean_split)
        diag['split_mano_fitted_mm'] = _named(SPLIT_NAMES, fitted_split)
        diag['split_residual_mm'] = _named(SPLIT_NAMES, fitted_split - mean_split)
        diag['split_template_mm'] = _named(SPLIT_NAMES, template_split)
    return beta, diag


def recording_targets(folder):
    """一条录制两只手的 (被试, side, 整体尺寸, 拆分, 骨长表)；只读 CMA 手文件。"""
    _, human, _ = find_inputs(Path(folder))
    source = read_human(human)
    out = []
    for side in SIDES:
        hand = source['hands'][side]
        if not hand['present']:
            continue
        repaired = repair_hand(hand['points_mm'], hand['pos_valid'], hand['global_R'], hand['local_R'],
                               hand['rot_valid'])
        offsets = repaired['offsets_mm']
        main, split = cma_lengths(offsets)
        out.append((hand['schema'].get('subject') or 'unknown', side, main, split,
                    np.linalg.norm(offsets, axis=1)))
    return out


def subject_betas(raw_dir, names, cache=None):
    """全部录制 -> {subject: {'beta', 'recordings', 诊断}}；cache（json）存在、方法一致且覆盖这些录制时直接读。"""
    if cache is not None and Path(cache).is_file():
        stored = json.loads(Path(cache).read_text(encoding='utf-8'))
        if stored.get('method') == METHOD and set(names) <= set(stored.get('recordings', {})):
            return stored
    per_subject, recordings = {}, {}
    for name in names:
        for subject, side, main, split, bones in recording_targets(Path(raw_dir) / name):
            per_subject.setdefault(subject, []).append((side, main, split, name))
            recordings.setdefault(name, {})[side] = {
                'subject': subject, 'lengths_mm': _named(LENGTH_NAMES, main),
                'split_mm': _named(SPLIT_NAMES, split), 'cma_bone_lengths_mm': bones.round(2).tolist()}
    subjects = {}
    for subject, rows in sorted(per_subject.items()):
        beta, diag = fit_beta([(side, main) for side, main, _, _ in rows], [split for _, _, split, _ in rows])
        subjects[subject] = {'beta': beta.tolist(), 'recordings': sorted({row[-1] for row in rows}),
                             'hands_used': len(rows), **diag}
    result = {'method': METHOD, 'length_names': LENGTH_NAMES, 'sigma_mm': LENGTH_SIGMA_MM.tolist(),
              'beta_prior': BETA_PRIOR, 'beta_bound': BETA_BOUND, 'subjects': subjects,
              'recordings': recordings}
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding='utf-8')
    return result


def beta_for(result, name, side):
    subject = result['recordings'][name][side]['subject']
    return np.asarray(result['subjects'][subject]['beta'], np.float64), subject
