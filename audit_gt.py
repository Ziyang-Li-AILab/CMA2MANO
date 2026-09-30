# -*- coding: utf-8 -*-
"""GT 标签审计：按 ACE-Ego-Hand 训练约定检查 mano_fit.npz 里会伤害训练的标签。

读取一个结果目录（每条记录一个子目录，含 mano_fit.npz / mano_fit_report.json），
把标签换算成 ACE 训练时真正看到的样子（相机系 go/trans、exists、visible、joints2d），
逐帧逐手统计：

  presence   exists 空洞（手在、GT 说不在）、NaN 占位、画面可见却标 exists=0
  time       RGB 丢帧造成的时间断点，以及 81 帧训练窗口跨断点的比例
  temporal   关节加速度（相机系 / 手局部），与 CMA 输入自身的加速度对照；局部旋转逐帧跳变
  pose       残差轴角接近 pi（ACE 的 rotmat->轴角往返会改变姿态），手指绕骨轴扭转
  shape      beta 贴 ±3 边界、左右手 beta 不一致
  image      投影与 MediaPipe 2D 关键点的逐关节距离（MANO 关节和 CMA 关节分别算）
  raw CMA    追踪丢失 / 可疑跳变 / settle 帧、腕部世界速度、头显刚体速度、冻结行

用法（ace-ego-hand 解释器）：
  python audit_gt.py --results <结果目录> --out audit/<标签> [--raw <原始数据目录>]

输出：summary.json、summary.csv、report.md、per_frame/<记录>.npz（给画图脚本用）。
新版结果（含 ace_* 键）直接读 ace_* 数组；旧版结果按 ACE 约定现场换算。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from robust_convert.cleaning import INVALID, repair_hand  # noqa: E402
from robust_convert.cma_source import read_human, read_rig  # noqa: E402
from robust_convert.extrinsic import load_detections  # noqa: E402
from robust_convert.mano_model import DEFAULT_MANO, load_mano, mano_forward  # noqa: E402

SIDES = ('left', 'right')
RAW_DEFAULT = Path(r'C:\Users\11068\Desktop\项目材料\datasets_20\to_sftp')
MP_DEFAULT = Path(r'C:\Users\11068\Desktop\mano线推进\outputs\cache\mediapipe')
WINDOW = 81                      # ACE 视频训练窗口
FINGER_JOINTS = np.arange(1, 16)  # MANO 原生关节 1..15
PIP_DIP = np.array([2, 3, 5, 6, 8, 9, 11, 12, 14, 15])
MCP = np.array([1, 4, 7, 10])
TIP_VERTEX_IDS = np.array([744, 320, 443, 554, 671])  # 拇/食/中/无名/小 指尖顶点
TIP_OF_LAST = {15: 0, 3: 1, 6: 2, 12: 3, 9: 4}  # 原生末节 -> TIP_VERTEX_IDS 下标

# 阈值（报告里一并写出）
LIMITS = {'acc_spike_mm': 10.0, 'rot_step_spike_deg': 30.0, 'twist_pipdip_deg': 30.0, 'twist_mcp_deg': 45.0,
          'residual_risky_rad': 2.8, 'mp_mismatch_px': 40.0, 'beta_saturated': 2.95, 'wrist_teleport_mps': 3.0,
          'rig_jump_mps': 2.0, 'rig_spin_dps': 400.0}


# ---- 通用 -------------------------------------------------------------------------------------

def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else round(float(value), 4)
    if isinstance(value, np.integer):
        return int(value)
    return value


def runs(flags):
    """每段 True 的 (start, stop)。"""
    padded = np.concatenate([[False], np.asarray(flags, bool), [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def pct(values, q):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if len(values) else float('nan')


def aa_to_R(aa):
    aa = np.asarray(aa, np.float64)
    out = np.full(aa.shape[:-1] + (3, 3), np.nan)
    ok = np.isfinite(aa).all(-1)
    if ok.any():
        out[ok] = Rotation.from_rotvec(aa[ok]).as_matrix()
    return out


def R_to_aa(R):
    R = np.asarray(R, np.float64)
    out = np.full(R.shape[:-2] + (3,), np.nan)
    ok = np.isfinite(R).all((-1, -2))
    if ok.any():
        out[ok] = Rotation.from_matrix(R[ok]).as_rotvec()
    return out


def geodesic_deg(Ra, Rb):
    rel = np.swapaxes(Ra, -1, -2) @ Rb
    cos = (np.trace(rel, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


class Models:
    def __init__(self, path=DEFAULT_MANO):
        self.m = {side: load_mano(path, side) for side in SIDES}

    def rest(self, side, beta):
        m = self.m[side]
        shaped = m['v_template'] + m['shapedirs'] @ np.asarray(beta, np.float64)
        return m['J_regressor'] @ shaped, shaped

    def bone_axes(self, side, beta):
        """模板系下原生关节 1..15 的子骨方向（末节指向指尖顶点），(16,3)，第 0 行不用。"""
        rest, shaped = self.rest(side, beta)
        axes = np.zeros((16, 3))
        for j in range(1, 16):
            child = shaped[TIP_VERTEX_IDS[TIP_OF_LAST[j]]] if j in TIP_OF_LAST else rest[j + 1]
            d = child - rest[j]
            axes[j] = d / np.linalg.norm(d)
        return axes


# ---- 标签读取：统一成 ACE 训练看到的样子 -----------------------------------------------------------

def load_labels(folder, models):
    """返回 dict：每手 (N,...) 数组 + 公共数组。旧版结果按 ACE 约定换算到相机系。"""
    z = np.load(folder / 'mano_fit.npz')
    report = json.loads((folder / 'mano_fit_report.json').read_text(encoding='utf-8'))
    stamps = z['timestamps_ns'].astype(np.int64)
    count = len(stamps)
    width, height = report.get('source_resolution', [1920, 1080])
    K = np.asarray(z['K'], np.float64)
    labels = {'name': folder.name, 'count': count, 'stamps': stamps, 'K': K, 'W': int(width), 'H': int(height),
              'cma_rows': z['cma_row_indices'].astype(np.int64) if 'cma_row_indices' in z else None,
              'format': 'v2' if 'ace_exists' in z else 'legacy', 'report': report, 'hands': {}}
    if 'segment_id' in z:
        labels['segment_id'] = z['segment_id'].astype(np.int64)
    if 'window_ok' in z:
        labels['window_ok'] = z['window_ok'].astype(bool)
    for h, side in enumerate(SIDES):
        hand = {'present': f'{side}_beta' in z or 'ace_exists' in z}
        if labels['format'] == 'v2':
            beta = z['ace_betas'][h].astype(np.float64)
            hand.update(exists=z['ace_exists'][:, h].astype(bool), go_cam=z['ace_go'][:, h].astype(np.float64),
                        hp=z['ace_hp'][:, h].astype(np.float64), trans_cam=z['ace_trans'][:, h].astype(np.float64),
                        joints_cam=z['ace_joints_cam'][:, h].astype(np.float64), beta=beta,
                        has_mano=z['ace_has_mano'][:, h].astype(bool))
            hand['present'] = bool(z['hand_present'][h]) if 'hand_present' in z else True
            key = f'{side}_joints_cam_cma'
            hand['cma_cam'] = z[key].astype(np.float64) if key in z else np.full((count, 21, 3), np.nan)
            key = f'{side}_error_mm'
            hand['error_mm'] = z[key].astype(np.float64) if key in z else np.full((count, 21), np.nan)
            for extra in ('gt_source', 'gt_weight', 'exists_target', 'gt_quality', 'gt_state'):
                if extra in z:
                    hand[extra] = z[extra][:, h]
            hand['flags'] = z[f'{side}_frame_flags'] if f'{side}_frame_flags' in z else np.zeros(count, np.uint8)
        elif hand['present']:
            beta = z[f'{side}_beta'].astype(np.float64)
            go_w = z[f'{side}_global_orient'].astype(np.float64)
            trans_w = z[f'{side}_trans'].astype(np.float64)
            cam_R = z[f'{side}_cam_R'].astype(np.float64)
            cam_t = z[f'{side}_cam_t'].astype(np.float64)
            rest0 = models.rest(side, beta)[0][0]
            go_cam = R_to_aa(cam_R @ aa_to_R(go_w))
            trans_cam = np.einsum('tij,tj->ti', cam_R, rest0 + trans_w) + cam_t - rest0
            hand.update(exists=z['exists'][:, h].astype(bool), go_cam=go_cam,
                        hp=z[f'{side}_hand_pose'].astype(np.float64), trans_cam=trans_cam,
                        joints_cam=z[f'{side}_joints_cam'].astype(np.float64), beta=beta,
                        cma_cam=z[f'{side}_joints_cam_cma'].astype(np.float64),
                        error_mm=z[f'{side}_error_mm'].astype(np.float64), flags=z[f'{side}_frame_flags'])
            hand['has_mano'] = hand['exists'].copy()
        else:
            nan = np.full((count, 21, 3), np.nan)
            hand.update(exists=np.zeros(count, bool), go_cam=np.full((count, 3), np.nan),
                        hp=np.full((count, 45), np.nan), trans_cam=np.full((count, 3), np.nan), joints_cam=nan,
                        beta=np.full(10, np.nan), cma_cam=nan.copy(), error_mm=np.full((count, 21), np.nan),
                        flags=np.zeros(count, np.uint8), has_mano=np.zeros(count, bool))
        labels['hands'][side] = hand
    return labels


def project(points_m, K):
    """相机系米 -> 像素，z<=0.01 为 NaN。"""
    z = points_m[..., 2]
    with np.errstate(invalid='ignore', divide='ignore'):
        uv = np.stack([K[0, 0] * points_m[..., 0] / z + K[0, 2], K[1, 1] * points_m[..., 1] / z + K[1, 2]], -1)
    uv[~(z > 0.01)] = np.nan
    return uv


def ace_gate(joints_cam, K, W, H):
    """ACE App. A.1 可见性门（与 schema.visibility_gate 相同）：joint_vis (N,21), visible (N,)。"""
    uv = project(joints_cam, K)
    u, v = uv[..., 0] / W, uv[..., 1] / H
    with np.errstate(invalid='ignore'):
        inside = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
    joint_vis = inside & (joints_cam[..., 2] > 0.01)
    return joint_vis, joint_vis.any(-1)


def bridge_positions(joints, ok):
    """exists=0 帧用前后最近有效帧线性插值的关节位置（估计“手其实在哪”）。"""
    count = len(joints)
    out = np.full_like(joints, np.nan)
    good = np.flatnonzero(ok & np.isfinite(joints).all((-1, -2)))
    if not len(good):
        return out
    for t in np.flatnonzero(~ok):
        after = good[np.searchsorted(good, t):]
        before = good[:np.searchsorted(good, t)]
        if len(before) and len(after):
            a, b = before[-1], after[0]
            w = (t - a) / (b - a)
            out[t] = (1 - w) * joints[a] + w * joints[b]
        elif len(before):
            out[t] = joints[before[-1]]
        else:
            out[t] = joints[after[0]]
    return out


# ---- MediaPipe 2D 对照 ----------------------------------------------------------------------------

def match_mediapipe(uv_gt, det):
    """两只手与两个检测槽的最优配对；返回每手 (N,21) 距离（px）和配对槽的标签。"""
    count = uv_gt.shape[0]
    dist = np.full((count, 2, 21), np.nan)
    label = np.full((count, 2), -1, np.int8)
    uv_det = det['uv']
    for t in range(count):
        cost = np.full((2, 2), np.inf)
        per = {}
        for h in range(2):
            for s in range(2):
                if det['label'][t, s] < 0:
                    continue
                d = np.linalg.norm(uv_gt[t, h] - uv_det[t, s], axis=-1)
                if np.isfinite(d).sum() >= 10:
                    cost[h, s] = np.nanmedian(d)
                    per[h, s] = d
        options = [((0, 0), (1, 1)), ((0, 1), (1, 0))]
        best, best_cost = None, np.inf
        for option in options:
            finite = [cost[p] for p in option if np.isfinite(cost[p])]
            if not finite:
                continue
            # 优先配上更多手，其次总代价最小
            key = (-len(finite), sum(finite))
            if best is None or key < best_cost:
                best, best_cost = option, key
        if best is None:
            # 只有一只手能配：取它最近的槽
            for h in range(2):
                s = int(np.argmin(cost[h]))
                if np.isfinite(cost[h, s]):
                    dist[t, h] = per[h, s]
                    label[t, h] = det['label'][t, s]
            continue
        for h, s in best:
            if np.isfinite(cost[h, s]):
                dist[t, h] = per[h, s]
                label[t, h] = det['label'][t, s]
    return dist, label


# ---- 原始 CMA 检查 ---------------------------------------------------------------------------------

def raw_checks(raw_folder, cma_rows, count):
    """原始 120 Hz CMA 的问题映射到 RGB 帧。cause 位：1 追踪丢失 2 可疑跳变 4 settle 8 腕部瞬移 16 冻结。"""
    human_path = next(raw_folder.glob('*_Human.cma'))
    body_path = next(raw_folder.glob('*_Body.cma'))
    source = read_human(human_path)
    rig = read_rig(body_path)
    ns = source['timestamps_ns']
    rows_total = len(ns)
    out = {'hands': {}}
    window = np.clip(np.stack([cma_rows - 1, cma_rows, cma_rows + 1], 1), 0, rows_total - 1)

    def to_frames(row_flags):
        return row_flags[window].any(1)

    for side in SIDES:
        hand = source['hands'][side]
        if not hand['present']:
            out['hands'][side] = {'present': False, 'cause': np.zeros(count, np.uint8)}
            continue
        keep = repair_hand(hand['points_mm'], hand['pos_valid'], hand['global_R'], hand['local_R'],
                           hand['rot_valid'], drop_suspect=False)
        drop = repair_hand(hand['points_mm'], hand['pos_valid'], hand['global_R'], hand['local_R'],
                           hand['rot_valid'], drop_suspect=True)
        lost = (keep['status'] == INVALID).any(1)
        lost_core = np.zeros(rows_total, bool)
        for a, b in keep['diagnostics']['lost_runs']:
            lost_core[a:b] = True
        bad_drop = np.zeros(rows_total, bool)
        for a, b in drop['diagnostics']['lost_runs']:
            bad_drop[a:b] = True
        suspect = bad_drop & ~lost_core
        settle = ~drop['frame_valid'] & ~bad_drop
        points = hand['points_mm']
        wrist = np.where(hand['pos_valid'][:, :1], points[:, 0], np.nan)
        dt = np.diff(ns) / 1e9
        speed = np.full(rows_total, np.nan)
        speed[1:] = np.linalg.norm(np.diff(wrist, axis=0), axis=-1) / 1000 / dt
        teleport = speed > LIMITS['wrist_teleport_mps']
        teleport[:-1] |= teleport[1:]
        frozen = np.zeros(rows_total, bool)
        same = np.all(np.diff(np.nan_to_num(points), axis=0) == 0, axis=(1, 2)) & hand['pos_valid'][1:].all(1)
        frozen[1:] = same
        cause = (to_frames(lost_core).astype(np.uint8) * 1 | to_frames(suspect).astype(np.uint8) * 2
                 | to_frames(settle).astype(np.uint8) * 4 | to_frames(teleport).astype(np.uint8) * 8
                 | to_frames(frozen).astype(np.uint8) * 16)
        out['hands'][side] = {
            'present': True, 'cause': cause,
            'rows': rows_total, 'lost_rows': int(lost_core.sum()), 'suspect_rows': int(suspect.sum()),
            'settle_rows': int(settle.sum()), 'lost_runs_rows': keep['diagnostics']['lost_runs'],
            'wrist_speed_mps_p99': pct(speed, 99), 'wrist_speed_mps_max': float(np.nanmax(speed)) if np.isfinite(speed).any() else float('nan'),
            'teleport_rows': int(teleport.sum()), 'frozen_rows': int(frozen.sum()),
            'max_hand_local_step_mm': drop['diagnostics']['max_hand_local_step_mm'],
            'any_invalid_joint_rows': int(lost.sum())}
    origin = np.where(rig['valid'][:, None], rig['origin_mm'], np.nan)
    rdt = np.diff(rig['ptp_ns']) / 1e9
    rig_speed = np.linalg.norm(np.diff(origin, axis=0), axis=-1) / 1000 / rdt
    spin = geodesic_deg(rig['rig_R'][:-1], rig['rig_R'][1:]) / rdt
    out['rig'] = {'speed_mps_p99': pct(rig_speed, 99), 'speed_mps_max': float(np.nanmax(rig_speed)),
                  'spin_dps_p99': pct(spin, 99), 'spin_dps_max': float(np.nanmax(spin)),
                  'jump_rows': int((rig_speed > LIMITS['rig_jump_mps']).sum()),
                  'spin_rows': int((spin > LIMITS['rig_spin_dps']).sum()),
                  'invalid_rows': int((~rig['valid']).sum())}
    return out


# ---- 单条记录 ---------------------------------------------------------------------------------------

def audit_recording(folder, models, raw_dir, mp_dir):
    labels = load_labels(folder, models)
    name, count, K, W, H = labels['name'], labels['count'], labels['K'], labels['W'], labels['H']
    stamps = labels['stamps']
    dt = np.diff(stamps) / 1e6
    nominal = float(np.median(dt))
    gap_after = np.zeros(count, bool)          # t 与 t+1 之间有丢帧
    gap_after[:-1] = dt > 1.5 * nominal
    segment = labels.get('segment_id', np.concatenate([[0], np.cumsum(gap_after[:-1])]))
    per_frame = {'stamps': stamps, 'gap_after': gap_after, 'segment_id': segment}
    rec = {'name': name, 'format': labels['format'], 'frames': count, 'frame_ms': nominal,
           'time_gaps': [{'after_frame': int(t), 'ms': float(dt[t])} for t in np.flatnonzero(gap_after[:-1])],
           'hands': {}}

    raw = None
    if raw_dir is not None and labels['cma_rows'] is not None and (raw_dir / name).is_dir():
        raw = raw_checks(raw_dir / name, labels['cma_rows'], count)
        rec['raw_rig'] = raw['rig']

    det = None
    mp_path = mp_dir / f'{name}.npz'
    if mp_path.is_file():
        det = load_detections(mp_path, stamps)
    uv_gt = np.stack([project(labels['hands'][s]['joints_cam'], K) for s in SIDES], 1)
    uv_cma = np.stack([project(labels['hands'][s]['cma_cam'], K) for s in SIDES], 1)
    exists_all = np.stack([labels['hands'][s]['exists'] for s in SIDES], 1)
    # 只在 GT 存在的帧上配对（不存在的帧用桥接位置另算）
    uv_gt_masked = np.where(exists_all[..., None, None], uv_gt, np.nan)
    if det is not None:
        mp_dist, mp_label = match_mediapipe(uv_gt_masked, det)
        mp_dist_cma, _ = match_mediapipe(np.where(exists_all[..., None, None], uv_cma, np.nan), det)
    betas = {}

    # 训练窗口统计（ACE store：随机起点、连续 81 帧）
    starts = np.arange(max(count - WINDOW + 1, 0))
    seg_ok = np.array([segment[s] == segment[s + WINDOW - 1] for s in starts], bool) if len(starts) else np.zeros(0, bool)
    rec['windows'] = {'total': int(len(starts)), 'cross_time_gap': int((~seg_ok).sum())}
    window_bad_presence = np.zeros(len(starts), bool)
    window_nan = np.zeros(len(starts), bool)

    for h, side in enumerate(SIDES):
        hand = labels['hands'][side]
        exists = hand['exists']
        stats = {'present': bool(hand['present'])}
        pf = {}
        if not hand['present']:
            rec['hands'][side] = stats
            continue
        betas[side] = hand['beta']
        joints = hand['joints_cam']
        finite_all = (np.isfinite(hand['go_cam']).all(-1) & np.isfinite(hand['hp']).all(-1)
                      & np.isfinite(hand['trans_cam']).all(-1) & np.isfinite(joints).all((-1, -2)))
        # -- presence ---------------------------------------------------------------------------
        absent_runs = runs(~exists)
        holes = [(a, b) for a, b in absent_runs if a > 0 and b < count]
        edges = [(a, b) for a, b in absent_runs if a == 0 or b == count]
        stats['presence'] = {
            'exists_fraction': float(exists.mean()), 'exists_frames': int(exists.sum()),
            'absent_frames': int((~exists).sum()), 'transitions': int(np.abs(np.diff(exists.astype(int))).sum()),
            'holes': [[a, b] for a, b in holes], 'hole_frames': int(sum(b - a for a, b in holes)),
            'edge_absent': [[a, b] for a, b in edges], 'edge_absent_frames': int(sum(b - a for a, b in edges)),
            'nan_in_absent_frames': int((~exists & ~finite_all).sum()),
            'nan_in_exists_frames': int((exists & ~finite_all).sum()),
        }
        # 画面里其实有手：桥接位置投影在画面内 / MediaPipe 在附近检测到手
        bridged = bridge_positions(joints, exists)
        _, vis_bridge = ace_gate(bridged, K, W, H)
        vis_bridge &= ~exists
        stats['presence']['absent_but_on_screen'] = int(vis_bridge.sum())
        mp_near_absent = np.zeros(count, bool)
        if det is not None and (~exists).any():
            uv_b = project(bridged, K)
            for t in np.flatnonzero(~exists):
                for s in range(2):
                    if det['label'][t, s] < 0:
                        continue
                    d = np.linalg.norm(uv_b[t] - det['uv'][t, s], axis=-1)
                    if np.isfinite(d).sum() >= 10 and np.nanmedian(d) < 80:
                        mp_near_absent[t] = True
        stats['presence']['absent_but_mediapipe_sees_hand'] = int(mp_near_absent.sum())
        if 'exists_target' in hand:
            # 新版：exists=0 的帧分成“GT 断言手不在”（target 0）和“无可信 GT、presence 损失应忽略”（target -1）
            target = np.asarray(hand['exists_target'])
            stats['presence'].update(
                ignore_frames=int((target < 0).sum()),
                declared_absent_frames=int((target == 0).sum()),
                declared_absent_on_screen=int((vis_bridge & (target == 0)).sum()),
                declared_absent_mediapipe_sees_hand=int((mp_near_absent & (target == 0)).sum()))
        pf['absent_on_screen'] = vis_bridge
        pf['absent_mp_hand'] = mp_near_absent

        # -- visibility (ACE gate) --------------------------------------------------------------
        joint_vis, visible = ace_gate(joints, K, W, H)
        joint_vis &= exists[:, None]
        visible &= exists
        stats['visibility'] = {'visible_fraction_of_exists': float(visible[exists].mean()) if exists.any() else 0.0,
                               'visible_frames': int(visible.sum())}
        pf['visible'] = visible

        # -- temporal ---------------------------------------------------------------------------
        ok3 = np.zeros(count, bool)      # t-1, t, t+1 都有 GT 且不跨时间断点
        ok3[1:-1] = exists[:-2] & exists[1:-1] & exists[2:] & ~gap_after[:-2] & ~gap_after[1:-1]
        acc = np.full(count, np.nan)
        acc_rr = np.full(count, np.nan)
        acc_rr_cma = np.full(count, np.nan)
        rr = joints - joints[:, :1]
        second = joints[2:] - 2 * joints[1:-1] + joints[:-2]
        second_rr = rr[2:] - 2 * rr[1:-1] + rr[:-2]
        cma = hand['cma_cam']
        cma_rr = cma - cma[:, :1]
        second_cma = cma_rr[2:] - 2 * cma_rr[1:-1] + cma_rr[:-2]
        with np.errstate(invalid='ignore'):
            acc[1:-1] = np.linalg.norm(second, axis=-1).max(-1) * 1000
            acc_rr[1:-1] = np.linalg.norm(second_rr, axis=-1).max(-1) * 1000
            acc_rr_cma[1:-1] = np.linalg.norm(second_cma, axis=-1).max(-1) * 1000
        acc[~ok3], acc_rr[~ok3] = np.nan, np.nan
        acc_rr_cma[~ok3] = np.nan
        excess = acc_rr - acc_rr_cma
        # 局部旋转逐帧跳变（物理姿态 = 残差 + hands_mean）
        mean = models.m[side]['hands_mean']
        physical = hand['hp'] + mean
        local = aa_to_R(physical.reshape(count, 15, 3))
        ok2 = np.zeros(count, bool)
        ok2[1:] = exists[1:] & exists[:-1] & ~gap_after[:-1]
        rot_step = np.full(count, np.nan)
        go_step = np.full(count, np.nan)
        with np.errstate(invalid='ignore'):
            rot_step[1:] = np.nanmax(geodesic_deg(local[:-1], local[1:]), axis=-1)
            go_R = aa_to_R(hand['go_cam'])
            go_step[1:] = geodesic_deg(go_R[:-1], go_R[1:])
        rot_step[~ok2], go_step[~ok2] = np.nan, np.nan
        # 跳变点（边界）：exists 0->1 / 1->0 或时间断点两侧的位移
        stats['temporal'] = {
            'acc_cam_mm_p50': pct(acc, 50), 'acc_cam_mm_p99': pct(acc, 99), 'acc_cam_mm_max': float(np.nanmax(acc)) if np.isfinite(acc).any() else float('nan'),
            'acc_rootrel_mm_p50': pct(acc_rr, 50), 'acc_rootrel_mm_p99': pct(acc_rr, 99),
            'acc_rootrel_cma_mm_p50': pct(acc_rr_cma, 50), 'acc_rootrel_cma_mm_p99': pct(acc_rr_cma, 99),
            'fit_jitter_excess_mm_p99': pct(excess, 99),
            'fit_jitter_spike_frames': int((excess > LIMITS['acc_spike_mm']).sum()),
            'acc_spike_frames': int((acc_rr > LIMITS['acc_spike_mm']).sum()),
            'rot_step_deg_p50': pct(rot_step, 50), 'rot_step_deg_p99': pct(rot_step, 99),
            'rot_step_deg_max': float(np.nanmax(rot_step)) if np.isfinite(rot_step).any() else float('nan'),
            'rot_step_spike_frames': int((rot_step > LIMITS['rot_step_spike_deg']).sum()),
            'go_step_deg_p99': pct(go_step, 99),
        }
        pf.update(acc_cam_mm=acc, acc_rr_mm=acc_rr, acc_rr_cma_mm=acc_rr_cma, rot_step_deg=rot_step,
                  go_step_deg=go_step)

        # -- pose plausibility ------------------------------------------------------------------
        residual_norm = np.linalg.norm(hand['hp'].reshape(count, 15, 3), axis=-1)
        resid_max = np.nanmax(np.where(np.isfinite(residual_norm), residual_norm, -1), axis=-1)
        resid_max[resid_max < 0] = np.nan
        axes = models.bone_axes(side, hand['beta'] if np.isfinite(hand['beta']).all() else np.zeros(10))
        quat = np.full((count, 15, 4), np.nan)
        ok_pose = np.isfinite(physical).all(-1)
        if ok_pose.any():
            quat[ok_pose] = Rotation.from_rotvec(physical[ok_pose].reshape(-1, 3)).as_quat().reshape(-1, 15, 4)
        proj = np.einsum('tjk,jk->tj', quat[..., :3], axes[1:])
        twist = np.degrees(2 * np.arctan2(proj, quat[..., 3]))
        twist = (twist + 180) % 360 - 180
        twist_abs = np.abs(twist)
        pipdip = twist_abs[:, PIP_DIP - 1]
        mcp = twist_abs[:, MCP - 1]
        twist_bad = (np.nanmax(np.where(np.isfinite(pipdip), pipdip, 0), -1) > LIMITS['twist_pipdip_deg']) | \
                    (np.nanmax(np.where(np.isfinite(mcp), mcp, 0), -1) > LIMITS['twist_mcp_deg'])
        twist_bad &= exists
        stats['pose'] = {
            'residual_max_rad_p99': pct(np.where(exists, resid_max, np.nan), 99),
            'residual_risky_frames': int((exists & (resid_max > LIMITS['residual_risky_rad'])).sum()),
            'twist_pipdip_deg_p95': pct(np.where(exists[:, None], pipdip, np.nan), 95),
            'twist_mcp_deg_p95': pct(np.where(exists[:, None], mcp, np.nan), 95),
            'twist_per_joint_p95_deg': [pct(np.where(exists, twist_abs[:, j], np.nan), 95) for j in range(15)],
            'twist_bad_frames': int(twist_bad.sum()),
        }
        pf['resid_max'] = resid_max
        pf['twist_max_deg'] = np.nanmax(np.where(np.isfinite(twist_abs[:, :12]), twist_abs[:, :12], np.nan), axis=-1) \
            if np.isfinite(twist_abs).any() else np.full(count, np.nan)

        # -- fit error vs CMA -------------------------------------------------------------------
        err = hand['error_mm']
        if not np.isfinite(err).any():
            err = np.linalg.norm(joints - cma, axis=-1) * 1000
        err_e = np.where(exists[:, None], err, np.nan)
        stats['fit_vs_cma'] = {'mean_mm': float(np.nanmean(err_e)) if np.isfinite(err_e).any() else float('nan'),
                               'p95_mm': pct(err_e, 95),
                               'per_joint_mean_mm': [float(np.nanmean(err_e[:, j])) if np.isfinite(err_e[:, j]).any()
                                                     else float('nan') for j in range(21)]}
        pf['fit_err_mm'] = np.nanmean(err_e, axis=-1) if np.isfinite(err_e).any() else np.full(count, np.nan)

        # -- shape --------------------------------------------------------------------------------
        beta = hand['beta']
        stats['shape'] = {'beta': beta.tolist(), 'beta_norm': float(np.linalg.norm(beta)),
                          'saturated': int((np.abs(beta) >= LIMITS['beta_saturated']).sum()),
                          'max_abs': float(np.abs(beta).max())}

        # -- image agreement (MediaPipe) ----------------------------------------------------------
        if det is not None:
            d = mp_dist[:, h]
            dc = mp_dist_cma[:, h]
            frame_d = np.nanmedian(np.where(np.isfinite(d), d, np.nan), axis=-1) if np.isfinite(d).any() else np.full(count, np.nan)
            frame_dc = np.nanmedian(np.where(np.isfinite(dc), dc, np.nan), axis=-1) if np.isfinite(dc).any() else np.full(count, np.nan)
            matched = np.isfinite(frame_d)
            stats['image'] = {
                'mp_matched_fraction_of_visible': float((matched & visible).sum() / max(visible.sum(), 1)),
                'mp_err_px_p50': pct(frame_d, 50), 'mp_err_px_p90': pct(frame_d, 90),
                'mp_mismatch_frames': int((frame_d > LIMITS['mp_mismatch_px']).sum()),
                'mp_err_cma_px_p50': pct(frame_dc, 50), 'mp_err_cma_px_p90': pct(frame_dc, 90),
                'mp_per_joint_px_p50': [pct(d[:, j], 50) for j in range(21)],
                'mp_per_joint_cma_px_p50': [pct(dc[:, j], 50) for j in range(21)],
                'mp_label_agreement': float(np.mean(mp_label[matched, h] == h)) if matched.any() else float('nan'),
            }
            pf['mp_err_px'] = frame_d
            pf['mp_err_cma_px'] = frame_dc
            pf['mp_joint_px'] = d.astype(np.float32)
            pf['mp_joint_cma_px'] = dc.astype(np.float32)

        # -- raw CMA ----------------------------------------------------------------------------
        if raw is not None and raw['hands'][side]['present']:
            r = raw['hands'][side]
            cause = r['cause']
            stats['raw_cma'] = {k: v for k, v in r.items() if k not in ('cause', 'present')}
            stats['raw_cma']['frames_touched'] = {name_: int(((cause & bit) > 0).sum()) for bit, name_ in
                                                  ((1, 'lost'), (2, 'suspect'), (4, 'settle'), (8, 'teleport'),
                                                   (16, 'frozen'))}
            stats['raw_cma']['exists0_by_cause'] = {name_: int((~exists & ((cause & bit) > 0)).sum()) for bit, name_ in
                                                    ((1, 'lost'), (2, 'suspect'), (4, 'settle'), (8, 'teleport'),
                                                     (16, 'frozen'))}
            pf['raw_cause'] = cause
        pf['exists'] = exists
        pf['flags'] = hand['flags']
        for extra in ('gt_source', 'gt_weight', 'exists_target', 'gt_quality', 'gt_state'):
            if extra in hand:
                pf[extra] = hand[extra]

        # -- windows ----------------------------------------------------------------------------
        if len(starts):
            csum = np.concatenate([[0], np.cumsum(~exists)])
            absent_in = csum[starts + WINDOW] - csum[starts]
            flicker = (absent_in > 0) & (absent_in < WINDOW)     # 窗口里时有时无
            window_bad_presence |= flicker
            nan_frames = np.concatenate([[0], np.cumsum(~finite_all)])
            window_nan |= (nan_frames[starts + WINDOW] - nan_frames[starts]) > 0
        rec['hands'][side] = stats
        for key, value in pf.items():
            per_frame[f'{side}_{key}'] = value

    if 'left' in betas and 'right' in betas:
        rec['beta_left_right_l2'] = float(np.linalg.norm(betas['left'] - betas['right']))
    rec['windows'].update(presence_flicker=int(window_bad_presence.sum()), contains_nan=int(window_nan.sum()))
    if 'window_ok' in labels:
        ok = labels['window_ok'][starts] if len(starts) else np.zeros(0, bool)
        rec['windows'].update(sampler_ok=int(ok.sum()),
                              sampler_ok_but_bad=int((ok & (window_bad_presence | window_nan | ~seg_ok)).sum()))
    return rec, per_frame


# ---- 汇总 ---------------------------------------------------------------------------------------------

CSV_FIELDS = [
    ('exists%', lambda r, s: 100 * s['presence']['exists_fraction']),
    ('holes', lambda r, s: len(s['presence']['holes'])),
    ('hole_frm', lambda r, s: s['presence']['hole_frames']),
    ('edge_abs_frm', lambda r, s: s['presence']['edge_absent_frames']),
    ('abs_on_screen', lambda r, s: s['presence']['absent_but_on_screen']),
    ('abs_mp_sees', lambda r, s: s['presence']['absent_but_mediapipe_sees_hand']),
    ('nan_absent', lambda r, s: s['presence']['nan_in_absent_frames']),
    ('beta_sat', lambda r, s: s['shape']['saturated']),
    ('beta_max', lambda r, s: s['shape']['max_abs']),
    ('fit_mm', lambda r, s: s['fit_vs_cma']['mean_mm']),
    ('fit_p95', lambda r, s: s['fit_vs_cma']['p95_mm']),
    ('acc_rr_p99', lambda r, s: s['temporal']['acc_rootrel_mm_p99']),
    ('acc_rr_cma_p99', lambda r, s: s['temporal']['acc_rootrel_cma_mm_p99']),
    ('jit_spk', lambda r, s: s['temporal']['fit_jitter_spike_frames']),
    ('rot_p99', lambda r, s: s['temporal']['rot_step_deg_p99']),
    ('rot_spk', lambda r, s: s['temporal']['rot_step_spike_frames']),
    ('resid_risky', lambda r, s: s['pose']['residual_risky_frames']),
    ('twist_pd_p95', lambda r, s: s['pose']['twist_pipdip_deg_p95']),
    ('twist_bad', lambda r, s: s['pose']['twist_bad_frames']),
    ('mp_px_p50', lambda r, s: s.get('image', {}).get('mp_err_px_p50', float('nan'))),
    ('mp_px_p90', lambda r, s: s.get('image', {}).get('mp_err_px_p90', float('nan'))),
    ('mp_cma_p50', lambda r, s: s.get('image', {}).get('mp_err_cma_px_p50', float('nan'))),
    ('mp_mis', lambda r, s: s.get('image', {}).get('mp_mismatch_frames', 0)),
]


def fmt(value):
    if isinstance(value, float):
        return '' if not np.isfinite(value) else f'{value:.2f}'
    return str(value)


def aggregate(records):
    rows = []
    for rec in records:
        for side in SIDES:
            s = rec['hands'].get(side, {})
            if not s.get('present'):
                continue
            row = {'recording': rec['name'], 'hand': side}
            for key, fn in CSV_FIELDS:
                try:
                    row[key] = fn(rec, s)
                except (KeyError, TypeError):
                    row[key] = float('nan')
            rows.append(row)
    total = {
        'recordings': len(records),
        'frames': int(sum(r['frames'] for r in records)),
        'recordings_with_time_gaps': int(sum(bool(r['time_gaps']) for r in records)),
        'windows_total': int(sum(r['windows']['total'] for r in records)),
        'windows_cross_time_gap': int(sum(r['windows']['cross_time_gap'] for r in records)),
        'windows_presence_flicker': int(sum(r['windows']['presence_flicker'] for r in records)),
        'windows_contains_nan': int(sum(r['windows']['contains_nan'] for r in records)),
        'hole_frames': int(sum(row['hole_frm'] for row in rows)),
        'edge_absent_frames': int(sum(row['edge_abs_frm'] for row in rows)),
        'absent_on_screen_frames': int(sum(row['abs_on_screen'] for row in rows)),
        'absent_mediapipe_sees_frames': int(sum(row['abs_mp_sees'] for row in rows)),
        'nan_in_absent_frames': int(sum(row['nan_absent'] for row in rows)),
        'hands_with_saturated_beta': int(sum(row['beta_sat'] > 0 for row in rows)),
        'saturated_beta_components': int(sum(row['beta_sat'] for row in rows)),
        'beta_left_right_l2_mean': float(np.mean([r['beta_left_right_l2'] for r in records if 'beta_left_right_l2' in r])),
        'fit_vs_cma_mm_mean': float(np.nanmean([row['fit_mm'] for row in rows])),
        'fit_jitter_spike_frames': int(sum(row['jit_spk'] for row in rows)),
        'rot_step_spike_frames': int(sum(row['rot_spk'] for row in rows)),
        'residual_risky_frames': int(sum(row['resid_risky'] for row in rows)),
        'twist_bad_frames': int(sum(row['twist_bad'] for row in rows)),
        'mp_err_px_p50_median': float(np.nanmedian([row['mp_px_p50'] for row in rows])),
        'mp_err_cma_px_p50_median': float(np.nanmedian([row['mp_cma_p50'] for row in rows])),
        'mp_mismatch_frames': int(sum(row['mp_mis'] for row in rows)),
    }
    if any('sampler_ok' in r['windows'] for r in records):
        total['windows_sampler_ok'] = int(sum(r['windows'].get('sampler_ok', 0) for r in records))
        total['windows_sampler_ok_but_bad'] = int(sum(r['windows'].get('sampler_ok_but_bad', 0) for r in records))
    presence = [r['hands'][s]['presence'] for r in records for s in SIDES
                if r['hands'].get(s, {}).get('present') and 'ignore_frames' in r['hands'][s]['presence']]
    if presence:
        for key in ('ignore_frames', 'declared_absent_frames', 'declared_absent_on_screen',
                    'declared_absent_mediapipe_sees_hand'):
            total[key] = int(sum(p[key] for p in presence))
    return rows, total


def write_report(out, records, rows, total, results):
    lines = [f'# GT 标签审计：`{results}`', '',
             f"记录 {total['recordings']} 条，{total['frames']} 帧。阈值：`{json.dumps(LIMITS)}`", '',
             '## 总览', '', '| 指标 | 值 |', '| --- | ---: |']
    for key, value in total.items():
        lines.append(f'| {key} | {fmt(value)} |')
    lines += ['', '## 逐手', '', '| 记录 | 手 | ' + ' | '.join(k for k, _ in CSV_FIELDS) + ' |',
              '| --- | --- | ' + ' | '.join('---:' for _ in CSV_FIELDS) + ' |']
    for row in rows:
        lines.append(f"| {row['recording']} | {row['hand']} | " + ' | '.join(fmt(row[k]) for k, _ in CSV_FIELDS) + ' |')
    lines += ['', '## 时间断点（RGB 丢帧）', '']
    for rec in records:
        if rec['time_gaps']:
            gaps = ', '.join(f"帧{g['after_frame']}后 {g['ms']:.0f} ms" for g in rec['time_gaps'])
            lines.append(f"- {rec['name']}：{gaps}；跨断点的 81 帧窗口 {rec['windows']['cross_time_gap']}/{rec['windows']['total']}")
    lines += ['', '## exists=0 的帧', '']
    for rec in records:
        for side in SIDES:
            s = rec['hands'].get(side, {})
            if s.get('present') and s['presence']['absent_frames']:
                p = s['presence']
                cause = s.get('raw_cma', {}).get('exists0_by_cause', {})
                lines.append(f"- {rec['name']} {side}：{p['absent_frames']} 帧（空洞 {p['holes']}，边缘 {p['edge_absent']}），"
                             f"画面内 {p['absent_but_on_screen']}，MediaPipe 看到手 {p['absent_but_mediapipe_sees_hand']}；"
                             f"原始 CMA 原因 {cause}")
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=RAW_DEFAULT)
    parser.add_argument('--mediapipe', type=Path, default=MP_DEFAULT)
    parser.add_argument('--mano', type=Path, default=DEFAULT_MANO)
    parser.add_argument('--only', nargs='*', default=None)
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors='replace')
    models = Models(args.mano)
    out = args.out
    (out / 'per_frame').mkdir(parents=True, exist_ok=True)
    folders = sorted(p for p in args.results.iterdir() if (p / 'mano_fit.npz').is_file())
    if args.only:
        folders = [p for p in folders if p.name in args.only]
    raw_dir = args.raw if args.raw and args.raw.is_dir() else None
    records = []
    for folder in folders:
        rec, per_frame = audit_recording(folder, models, raw_dir, args.mediapipe)
        records.append(rec)
        np.savez_compressed(out / 'per_frame' / f'{folder.name}.npz', **per_frame)
        hands = ', '.join(f"{side[0]}: exists {100 * rec['hands'][side]['presence']['exists_fraction']:.1f}% "
                          f"fit {rec['hands'][side]['fit_vs_cma']['mean_mm']:.1f}mm "
                          f"mp {rec['hands'][side].get('image', {}).get('mp_err_px_p50', float('nan')):.1f}px"
                          for side in SIDES if rec['hands'].get(side, {}).get('present'))
        print(f"{folder.name}: {hands}; gaps {len(rec['time_gaps'])}", flush=True)
    rows, total = aggregate(records)
    (out / 'summary.json').write_text(json.dumps(jsonable({'results': str(args.results), 'limits': LIMITS,
                                                           'total': total, 'records': records}),
                                                 indent=1, ensure_ascii=False), encoding='utf-8')
    with open(out / 'summary.csv', 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=['recording', 'hand'] + [k for k, _ in CSV_FIELDS])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(v) if isinstance(v, float) else v for k, v in row.items()})
    write_report(out, records, rows, total, args.results)
    print(json.dumps(jsonable(total), indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
