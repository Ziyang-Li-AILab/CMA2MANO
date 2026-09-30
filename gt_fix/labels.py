# -*- coding: utf-8 -*-
"""GT 标签层：逐帧状态、短空洞补全、时间分段与训练片段，以及 ACE 训练约定的相机系数组。

旧交付结果里会直接伤害训练的标签（审计 audit/before）：
  * 手一直在画面里，只因 CMA 某几帧失效（追踪丢失 / 可疑跳变 / settle），GT 就写 exists=0：
    ACE 的 presence_3d 损失在这些帧学“手消失了”，81 帧窗口里手忽有忽无
    （125531 左手 [146,149)、164514 右手 [223,227)、112346 右手开头 [0,12)）；
  * exists=0 的帧参数全是 NaN：ACE 的 masked_mean 是 sum(x*m)/sum(m)，NaN*0=NaN，一个窗口就能让 loss 变 NaN；
  * RGB 丢帧（67-300 ms）两侧的帧在训练窗口里被当成相邻帧（ACE 按固定帧率取连续 81 帧）。

这里给每只手每帧一个状态：
  FITTED   0  CMA 完整有效，重拟合得到
  FILLED   1  CMA 短暂失效的内部空洞（<= MAX_FILL 帧，不跨 RGB 断点，相机位姿已知）：
              前后两个拟合帧之间按时间插值（全局朝向与各关节局部旋转 slerp，腕位置线性）。手确实在，exists=1
  UNKNOWN  2  手在录制里，但这一帧给不出可信 GT（录制开头 / 结尾的缺失、长空洞、跨断点的空洞）：
              不能说“手不在”，也给不出姿态。exists_target=-1，且不进入任何训练片段
  ABSENT   3  CMA 文件里根本没有这只手（本批数据没有出现）；是否在画面里未知，同样 exists_target=-1
按 RGB 断点把录制切成时间段（segment_id），再去掉含 UNKNOWN 的帧，得到训练片段 clip_ranges（>= 81 帧）：
ACE 的 ClipStore 一个 .pt 就是一段连续视频，按 clip_ranges 切开存，窗口就不会跨断点、也碰不到 UNKNOWN 帧。
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

FITTED, FILLED, UNKNOWN, ABSENT = 0, 1, 2, 3
STATE_NAMES = {FITTED: 'fitted', FILLED: 'filled', UNKNOWN: 'unknown', ABSENT: 'absent'}
STATE_WEIGHT = {FITTED: 1.0, FILLED: 0.5, UNKNOWN: 0.0, ABSENT: 0.0}      # 给带权重损失用的建议权重
STATE_EXISTS_TARGET = {FITTED: 1, FILLED: 1, UNKNOWN: -1, ABSENT: -1}    # -1 = presence 损失应忽略
MAX_FILL = 6          # 内部空洞最多补几帧（30 fps 下 200 ms）；依据见 fill_holdout 的留出实验
GAP_FACTOR = 1.5      # 相邻 RGB 帧间隔超过名义间隔的 1.5 倍即为断点（与 fit.adjacency 相同）
WINDOW = 81           # ACE 视频训练窗口（Wan VAE：1 + 4*20 帧 -> 21 个 latent）

# 逐帧标志位（旧版 1/2/4/8 含义不变）
FLAG_CMA_INVALID = 1      # 这一帧 CMA 没有完整有效的手：未拟合
FLAG_POSE_BRANCH = 2      # 某关节残差轴角 >= pi（ACE 的 rotmat <-> 轴角往返会改姿态）
FLAG_WRIST = 4            # MANO 腕离 CMA 腕 > 2.9 mm
FLAG_FIT_ERROR = 8        # 平均关节误差 > 15 mm
FLAG_FILLED = 16          # 插值补全
FLAG_UNKNOWN = 32         # 无可信 GT，排除出训练片段
FLAG_NAMES = {FLAG_CMA_INVALID: 'cma_invalid', FLAG_POSE_BRANCH: 'pose_branch', FLAG_WRIST: 'wrist_gt_2.9mm',
              FLAG_FIT_ERROR: 'mean_error_gt_15mm', FLAG_FILLED: 'filled', FLAG_UNKNOWN: 'unknown'}


def runs(flags):
    """每段 True 的 (start, stop)。"""
    padded = np.concatenate([[False], np.asarray(flags, bool), [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


# ---- 时间分段 -------------------------------------------------------------------------------------

def time_segments(stamps_ns, gap_factor=GAP_FACTOR):
    """gap_after[t]：t 与 t+1 之间丢了 RGB 帧；segment_id 在每个断点后加一。返回 (segment_id, gap_after, 名义间隔 ns)。"""
    stamps = np.asarray(stamps_ns, np.int64)
    dt = np.diff(stamps)
    nominal = float(np.median(dt)) if len(dt) else 0.0
    gap_after = np.zeros(len(stamps), bool)
    gap_after[:-1] = dt > gap_factor * nominal
    segment = np.concatenate([[0], np.cumsum(gap_after[:-1])]).astype(np.int32)
    return segment, gap_after, nominal


# ---- 状态 ------------------------------------------------------------------------------------------

def classify(valid, present, segment_id, camera_ok=None, max_fill=MAX_FILL):
    """一只手的逐帧状态 (T,) int8，以及每段缺失的说明。valid = CMA 完整有效（已拟合）的帧。"""
    count = len(segment_id)
    state = np.full(count, FITTED, np.int8)
    if not present:
        state[:] = ABSENT
        return state, [{'start': 0, 'stop': count, 'frames': count, 'state': 'absent', 'reason': 'hand_not_in_cma'}]
    events = []
    for a, b in runs(~np.asarray(valid, bool)):
        if a == 0 and b == count:
            reason = 'never_valid'
        elif a == 0:
            reason = 'recording_start'
        elif b == count:
            reason = 'recording_end'
        elif b - a > max_fill:
            reason = 'long_hole'
        elif segment_id[a - 1] != segment_id[b]:
            reason = 'crosses_time_gap'
        elif camera_ok is not None and not np.all(camera_ok[a:b]):
            reason = 'camera_unknown'
        else:
            reason = None
        state[a:b] = FILLED if reason is None else UNKNOWN
        events.append({'start': a, 'stop': b, 'frames': b - a, 'state': 'filled' if reason is None else 'unknown',
                       'reason': reason or 'short_interior_hole'})
    return state, events


# ---- 旋转插值 ---------------------------------------------------------------------------------------

def nearest_branch(r, ref):
    """同一个旋转的轴角有 r + 2*pi*k*axis 多种写法；取离 ref 最近的一种（保持与相邻帧的分支连续）。"""
    r = np.asarray(r, np.float64)
    theta = np.linalg.norm(r, axis=-1, keepdims=True)
    axis = r / np.maximum(theta, 1e-12)
    candidates = np.stack([r, r - 2 * np.pi * axis, r + 2 * np.pi * axis])
    pick = np.argmin(np.linalg.norm(candidates - np.asarray(ref, np.float64)[None], axis=-1), axis=0)
    return np.take_along_axis(candidates, pick[None, ..., None], axis=0)[0]


def slerp_pairs(r0, r1, w):
    """逐对球面插值：轴角 r0、r1 (...,3)，权重 w 可广播到 r0.shape[:-1]；结果分支贴近线性插值。"""
    r0 = np.asarray(r0, np.float64)
    r1 = np.asarray(r1, np.float64)
    w = np.broadcast_to(np.asarray(w, np.float64), r0.shape[:-1]).reshape(-1)
    a = r0.reshape(-1, 3)
    b = r1.reshape(-1, 3)
    start = Rotation.from_rotvec(a)
    rel = (start.inv() * Rotation.from_rotvec(b)).as_rotvec()
    out = (start * Rotation.from_rotvec(rel * w[:, None])).as_rotvec()
    return nearest_branch(out, (1 - w)[:, None] * a + w[:, None] * b).reshape(r0.shape)


def interpolate(go, physical, wrist, stamps_ns, before, after, frames):
    """用帧 before / after 在帧 frames 上插值（三个下标数组同长）。

    go (T,3) 世界系轴角，physical (T,15,3) 物理局部轴角，wrist (T,3) 米。返回三者在 frames 上的值。
    """
    stamps = np.asarray(stamps_ns, np.float64)
    w = (stamps[frames] - stamps[before]) / (stamps[after] - stamps[before])
    go_t = slerp_pairs(go[before], go[after], w)
    physical_t = slerp_pairs(physical[before], physical[after], w[:, None])
    wrist_t = (1 - w)[:, None] * wrist[before] + w[:, None] * wrist[after]
    return go_t, physical_t, wrist_t


def fill_holes(go, physical, wrist, stamps_ns, state):
    """在 FILLED 帧上原地插值（每段空洞用它前后紧邻的两个拟合帧）。"""
    for a, b in runs(state == FILLED):
        frames = np.arange(a, b)
        before = np.full(len(frames), a - 1)
        after = np.full(len(frames), b)
        go[a:b], physical[a:b], wrist[a:b] = interpolate(go, physical, wrist, stamps_ns, before, after, frames)


def fill_holdout(fitted, segment_id, go, physical, wrist, stamps_ns, joints_fn, reference_joints,
                 lengths=(1, 2, 3, 4, 6, 8, 10, 15), stride=3):
    """留出实验：在完整拟合段里人为挖掉 L 帧，用两端插值，和真实拟合的关节比（mm）。

    fitted (T,) 可用作端点和真值的帧；joints_fn(go, physical, wrist) -> (n,21,3) 米；reference_joints (T,21,3) 米。
    """
    fitted = np.asarray(fitted, bool)
    count = len(fitted)
    csum = np.concatenate([[0], np.cumsum(fitted)])
    out = {}
    for length in lengths:
        starts = np.arange(1, count - length)                    # 第一个被挖掉的帧
        if not len(starts):
            continue
        span_ok = csum[starts + length + 1] - csum[starts - 1] == length + 2
        same = segment_id[starts - 1] == segment_id[starts + length]
        starts = starts[span_ok & same][::stride]
        if not len(starts):
            continue
        frames = (starts[:, None] + np.arange(length)[None]).reshape(-1)
        before = np.repeat(starts - 1, length)
        after = np.repeat(starts + length, length)
        go_t, physical_t, wrist_t = interpolate(go, physical, wrist, stamps_ns, before, after, frames)
        joints = joints_fn(go_t, physical_t, wrist_t)
        truth = reference_joints[frames]
        err = np.linalg.norm(joints - truth, axis=-1) * 1000
        rootrel = np.linalg.norm((joints - joints[:, :1]) - (truth - truth[:, :1]), axis=-1) * 1000
        out[int(length)] = {'samples': int(len(starts)), 'mean_mm': float(err.mean()),
                            'p95_mm': float(np.percentile(err, 95)), 'max_mm': float(err.max()),
                            'wrist_mean_mm': float(err[:, 0].mean()), 'tips_mean_mm': float(err[:, [4, 8, 12, 16, 20]].mean()),
                            'rootrel_mean_mm': float(rootrel.mean()), 'rootrel_p95_mm': float(np.percentile(rootrel, 95))}
    return out


# ---- 训练窗口 / 片段 --------------------------------------------------------------------------------

def blocked_frames(states, present):
    """任一在场手处于 UNKNOWN 的帧（这些帧不能进训练片段）。"""
    states = np.asarray(states)
    blocked = np.zeros(len(states), bool)
    for h in range(states.shape[1]):
        if present[h]:
            blocked |= states[:, h] == UNKNOWN
    return blocked


def training_windows(segment_id, states, present, window=WINDOW):
    """window_ok[s]：从帧 s 开始的 window 帧可以直接训练（同一时间段、没有 UNKNOWN 帧）。(T,) bool。"""
    count = len(segment_id)
    ok = np.zeros(count, bool)
    if count < window:
        return ok
    csum = np.concatenate([[0], np.cumsum(blocked_frames(states, present))])
    starts = np.arange(count - window + 1)
    ok[starts] = (segment_id[starts] == segment_id[starts + window - 1]) & (csum[starts + window] == csum[starts])
    return ok


def clip_ranges(segment_id, states, present, min_frames=WINDOW):
    """最大连续可训练区间：同一时间段、无 UNKNOWN。返回 (>= min_frames 的训练片段, 更短的区间)，均为 [start, stop)。"""
    ranges = []
    for a, b in runs(~blocked_frames(states, present)):
        cuts = [a] + [int(c) for c in a + np.flatnonzero(np.diff(segment_id[a:b])) + 1] + [b]
        ranges += list(zip(cuts[:-1], cuts[1:]))
    train = [(a, b) for a, b in ranges if b - a >= min_frames]
    short = [(a, b) for a, b in ranges if b - a < min_frames]
    return train, short


# ---- ACE 相机系数组 -----------------------------------------------------------------------------------

def to_camera(go_w, trans_w, rest0, cam_R, cam_t):
    """世界系 go / trans -> 相机系（ACE build_clip 约定：joints_cam = FK(go_cam, hp, beta) + trans_cam）。

    MANO 绕模板腕点 J0 旋转：joints_w = J0 + trans_w + R(go_w)(X - J0)，
    所以 go_cam = cam_R R(go_w)，trans_cam = cam_R (J0 + trans_w) + cam_t - J0。
    """
    go_cam = (Rotation.from_matrix(cam_R) * Rotation.from_rotvec(go_w)).as_rotvec()
    trans_cam = np.einsum('tij,tj->ti', cam_R, rest0 + trans_w) + cam_t - rest0
    return go_cam, trans_cam


def visibility(joints_cam, exists, K, width, height):
    """ACE App. A.1 的可见性门：投影在画面内且 z > 0.01，再与 exists 取与。返回 (归一化 2D, joint_vis, visible)。"""
    z = joints_cam[..., 2]
    safe = np.where(z > 1e-4, z, 1e-4)
    u = (K[0, 0] * joints_cam[..., 0] / safe + K[0, 2]) / width
    v = (K[1, 1] * joints_cam[..., 1] / safe + K[1, 2]) / height
    joint_vis = (u >= 0) & (u < 1) & (v >= 0) & (v < 1) & (z > 0.01) & np.asarray(exists, bool)[..., None]
    uv = np.where(np.asarray(exists, bool)[..., None, None], np.stack([u, v], -1), 0.0)
    return uv, joint_vis, joint_vis.any(-1)


def label_arrays(states, present):
    """由状态得到 exists / has_mano / exists_target / gt_weight，(T,2)。"""
    states = np.asarray(states)
    exists = (states == FITTED) | (states == FILLED)
    target = np.vectorize(STATE_EXISTS_TARGET.get, otypes=[np.int8])(states)
    weight = np.vectorize(STATE_WEIGHT.get, otypes=[np.float32])(states)
    return {'ace_exists': exists, 'ace_has_mano': exists.copy(), 'exists_target': target, 'gt_weight': weight,
            'gt_state': states.astype(np.int8), 'hand_present': np.asarray(present, bool)}
