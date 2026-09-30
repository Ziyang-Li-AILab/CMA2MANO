# -*- coding: utf-8 -*-
"""无参考帧的 CMA -> MANO 逐骨标定。

旧流程（robust_convert.mano_model.initial_pose）在“参考帧”上把 MANO 的掌心法向对到 CMA 的局部 y 轴，
而 CMA 手指的局部 y 轴恰恰是屈曲轴，于是每根手指的 MANO 骨绕自身轴扭了约 90-115 度：
屈曲被写成了外展/扭转，网格手指侧着弯。

这里改为逐骨、只用骨架常量的标定（不挑参考帧）：
  CMA body c 的骨方向 o_c = offsets[c]（局部 -x），屈曲轴 e = 局部 y（手指）/ 局部 z（拇指）
  MANO body j 的骨方向 a_j（模板，末节指向指尖顶点），屈曲轴 f_j（anatomy.chain_axis，正转 = 屈曲）
  C_j = frame(o_c, s*e) @ frame(a_j, f_j)^T      G_j = W_c C_j（MANO body j 的全局旋转）
  手腕 C_0：MANO 模板 腕->四个 MCP 方向 与 CMA 腕局部系里同一组方向做 Kabsch（拇指 CMC 是模板点，不用）
符号 s：手指用几何判定（绕 s*e 正转时近节骨朝掌心移动），拇指用 MANO 姿态分布（Mahalanobis）在
{+1,-1}^2 里选。扭转偏置曲线 psi（整条链再绕骨轴附加扭转）用来验证：新标定的最优 psi 应接近 0。
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from robust_convert.cleaning import INVALID, MEASURED, modal_offsets
from robust_convert.cma_source import PARENTS as CMA_PARENTS
from robust_convert.mano_model import NATIVE_FROM_OP, _align, _frame
from robust_convert.timing import HOLD_NS, MAX_SPAN_NS, bracket

from .anatomy import CHAINS, FINGER_CHAINS, OP_MCP, PARENTS, THUMB, bone_dirs, unit

PALMAR_SIGN = {'left': -1.0, 'right': 1.0}     # 掌心 = sign * (P5-P0) x (P17-P0)
E_Y = np.array([0.0, 1.0, 0.0])
E_Z = np.array([0.0, 0.0, 1.0])
PSI_GRID_DEG = np.arange(-180, 180, 5)


# ---- CMA 骨架常量与重采样 ------------------------------------------------------------------------

def source_offsets(source):
    """每根 CMA 骨在父 body 局部系里的常量偏移 (20,3) mm，用 cleaning 的众数法在实测关节上重算。"""
    status = source['cma_status']
    points = np.nan_to_num(source['cma_points_mm'])
    world = source['cma_world_R']
    measured = status == MEASURED
    usable = status != INVALID
    bone_ok = measured[:, 1:] & measured[:, CMA_PARENTS[1:]]
    ref, support, count = modal_offsets(points, world, bone_ok)
    missing = ~np.isfinite(ref).all(1)
    if missing.any():   # 该骨从未被实测：退回到 FK 修复值（本身就等于常量偏移）
        fallback, _, _ = modal_offsets(points, world, usable[:, 1:] & usable[:, CMA_PARENTS[1:]])
        ref[missing] = fallback[missing]
    return ref, support, count


def _safe_rotations(matrices):
    out = np.array(matrices, np.float64, copy=True)
    bad = ~np.isfinite(out).all((-1, -2))
    out[bad] = np.eye(3)
    return out


def resample_world_R(source_ns, status, world_R, query_ns, hold_ns=HOLD_NS):
    """CMA 各 body 世界旋转在 RGB 时间戳上的值：前后两行都有效时 slerp（旧流程取最近行，位置却是线性插值）。"""
    source_ns = np.asarray(source_ns, np.int64)
    query_ns = np.asarray(query_ns, np.int64)
    before, after, weight = bracket(source_ns, query_ns)
    row_ok = (status != INVALID).all(1)
    ok_b, ok_a = row_ok[before], row_ok[after]
    span = source_ns[after] - source_ns[before]
    inside = (query_ns >= source_ns[0]) & (query_ns <= source_ns[-1]) & (span <= MAX_SPAN_NS)
    both = ok_b & ok_a & inside
    dist_b = np.abs(query_ns - source_ns[before])
    dist_a = np.abs(source_ns[after] - query_ns)
    hold_b = ~both & ok_b & (dist_b <= hold_ns) & ((dist_b <= dist_a) | ~ok_a)
    hold_a = ~both & ~hold_b & ok_a & (dist_a <= hold_ns)
    R_b = _safe_rotations(world_R[before])
    R_a = _safe_rotations(world_R[after])
    out = np.where((weight >= 0.5)[:, None, None, None], R_a, R_b)
    if both.any():
        rel = Rotation.from_matrix((np.swapaxes(R_b[both], -1, -2) @ R_a[both]).reshape(-1, 3, 3)).as_rotvec()
        rel = rel.reshape(-1, 21, 3) * weight[both][:, None, None]
        out[both] = R_b[both] @ Rotation.from_rotvec(rel.reshape(-1, 3)).as_matrix().reshape(-1, 21, 3, 3)
    out[hold_b] = R_b[hold_b]
    out[hold_a] = R_a[hold_a]
    return out


def palmar_normal(points_mm, side):
    """每帧 CMA 掌心方向（世界系单位向量）。"""
    p = np.asarray(points_mm, np.float64)
    return PALMAR_SIGN[side] * unit(np.cross(p[:, 5] - p[:, 0], p[:, 17] - p[:, 0]))


# ---- 旋转工具 -------------------------------------------------------------------------------------

def logvec(matrices):
    m = np.asarray(matrices, np.float64)
    return Rotation.from_matrix(m.reshape(-1, 3, 3)).as_rotvec().reshape(m.shape[:-2] + (3,))


def expvec(vectors):
    v = np.asarray(vectors, np.float64)
    return Rotation.from_rotvec(v.reshape(-1, 3)).as_matrix().reshape(v.shape[:-1] + (3, 3))


def local_from_global(G):
    """MANO 全局旋转 (N,16,3,3) -> 局部旋转（body 0 为全局朝向）。"""
    L = G.copy()
    L[:, 1:] = np.swapaxes(G[:, PARENTS[1:]], -1, -2) @ G[:, 1:]
    return L


# ---- 标定 ------------------------------------------------------------------------------------------

def mano_globals(C, world_R):
    """G_j = W_c C_j，(N,16,3,3)。"""
    W = np.asarray(world_R, np.float64)[:, NATIVE_FROM_OP]
    return W @ C[None]


def body_calibration(o_cma, e_cma, a_mano, f_mano):
    return _frame(o_cma, e_cma) @ _frame(a_mano, f_mano).T


def wrist_calibration(offsets, rest21):
    """C_0：MANO 模板 腕->MCP 方向 -> CMA 腕局部系。"""
    mano_dirs = unit(np.stack([rest21[c] - rest21[0] for c in OP_MCP]))
    cma_dirs = unit(np.stack([offsets[c - 1] for c in OP_MCP]))
    C0 = _align(mano_dirs, cma_dirs)
    residual = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', (C0 @ mano_dirs.T).T, cma_dirs), -1, 1)))
    return C0, residual


def finger_sign(world_R, offsets, points_mm, side, mcp_op):
    """绕 s*y 正转使近节骨朝掌心运动的 s。返回 (s, 一致帧比例)。"""
    W = world_R[:, mcp_op]
    hinge = W @ E_Y
    bone = W @ unit(offsets[mcp_op])
    n = palmar_normal(points_mm, side)
    score = np.einsum('ti,ti->t', np.cross(hinge, bone), n)
    score = score[np.isfinite(score)]
    s = 1.0 if np.mean(score) >= 0 else -1.0
    return s, float(np.mean(np.sign(score) == s)) if len(score) else float('nan')


def chain_local_twist_offset(L, anat, joints, psi):
    """整条链在各自骨轴上附加扭转 psi 后的局部旋转（只返回该链的关节）。"""
    out = {}
    extra = {j: expvec(anat.bones[j] * psi) for j in joints}
    for j in joints:
        parent = PARENTS[j]
        left = extra[parent].T if parent in extra else np.eye(3)
        out[j] = left @ L[:, j] @ extra[j]
    return out


def chain_scores(L, anat, joints, psis=PSI_GRID_DEG):
    """每个 psi 的 链内关节 Mahalanobis 中位数之和 与 |扭转| 中位数之和。"""
    maha, twist = [], []
    for psi in np.radians(psis):
        local = chain_local_twist_offset(L, anat, joints, psi)
        m = t = 0.0
        for j in joints:
            vec = logvec(local[j])
            d = vec - anat.mu[j - 1]
            m += float(np.median(np.einsum('ti,ij,tj->t', d, anat.cov_inv[j - 1], d)))
            t += float(np.median(np.abs(np.degrees(anat.twist_single(vec, j)))))
        maha.append(m)
        twist.append(t)
    return np.array(maha), np.array(twist)


def calibrate_hand(side, anat, offsets, world_R, points_mm, rest21=None):
    """一只手（一条录制）的标定。world_R / points_mm 只传有效帧。"""
    rest21 = anat.rest if rest21 is None else rest21
    bones = bone_dirs(rest21)
    C = np.zeros((16, 3, 3))
    C[0], wrist_residual = wrist_calibration(offsets, rest21)
    diagnostics = {'wrist_kabsch_residual_deg': wrist_residual.round(2).tolist(), 'finger_sign': {}}
    n = palmar_normal(points_mm, side)
    tips = np.asarray(points_mm)[:, [8, 12, 16, 20]] - np.asarray(points_mm)[:, :1]
    diagnostics['tips_on_palmar_side_fraction'] = float(np.mean(np.einsum('tki,ti->tk', tips, n) > 0))
    for name in FINGER_CHAINS:
        joints = CHAINS[name]
        s, agree = finger_sign(world_R, offsets, points_mm, side, NATIVE_FROM_OP[joints[0]])
        diagnostics['finger_sign'][name] = {'sign': s, 'agreement': round(agree, 4)}
        for j in joints:
            c = NATIVE_FROM_OP[j]
            C[j] = body_calibration(offsets[c], s * E_Y, bones[j], anat.chain_axis[j])
    # 拇指：CMC 用 T1 的 z，MCP/IP 用 T2/T3 的 z；两个符号在 MANO 姿态分布下选
    candidates = {}
    for s1 in (1.0, -1.0):
        for s23 in (1.0, -1.0):
            trial = C.copy()
            for j, s in zip(THUMB, (s1, s23, s23)):
                c = NATIVE_FROM_OP[j]
                trial[j] = body_calibration(offsets[c], s * E_Z, bones[j], anat.chain_axis[j])
            L = local_from_global(mano_globals(trial, world_R))
            vec = logvec(L[:, list(THUMB)])
            score = float(sum(np.median(anat.mahalanobis_joint(vec[:, k], j)) for k, j in enumerate(THUMB)))
            candidates[(s1, s23)] = (score, trial)
    best = min(candidates, key=lambda key: candidates[key][0])
    C = candidates[best][1]
    diagnostics['thumb_sign'] = {'cmc': best[0], 'mcp_ip': best[1],
                                 'mahalanobis_median_sum': {f'{a:+.0f}/{b:+.0f}': round(v[0], 2)
                                                            for (a, b), v in candidates.items()}}
    return C, diagnostics


def pose_from_calibration(C, world_R, anat):
    """标定后的 MANO 姿态：go (N,3) 世界系、physical (N,15,3)、hp 残差 (N,45)、局部旋转 L。"""
    G = mano_globals(C, world_R)
    L = local_from_global(G)
    go = logvec(L[:, 0])
    physical = logvec(L[:, 1:])
    hp = (physical.reshape(-1, 45) - anat.hands_mean)
    return {'go': go, 'physical': physical, 'hp': hp, 'G': G, 'L': L}


def psi_report(L, anat, psis=PSI_GRID_DEG):
    """每条链的扭转偏置曲线最优值（度），用于验证标定没有残留绕骨轴扭转。"""
    report = {}
    for name, joints in CHAINS.items():
        maha, twist = chain_scores(L, anat, joints, psis)
        report[name] = {'psi_best_mahalanobis_deg': float(psis[int(np.argmin(maha))]),
                        'psi_best_twist_deg': float(psis[int(np.argmin(twist))]),
                        'mahalanobis_at_0': float(maha[list(psis).index(0)]),
                        'mahalanobis_min': float(maha.min())}
    return report
