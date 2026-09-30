# -*- coding: utf-8 -*-
"""固定 beta 的 MANO 重拟合。

旧流程逐帧把 MANO 关节位置拟合到 CMA 关节位置，beta 自由（顶到 ±3），且只有一个很弱的“离初值不远”先验：
位置几乎约束不了“绕骨轴的扭转”，于是 40 只手的手指普遍扭转 90-115 度（MANO 训练姿态的 p95 只有 ~20-30 度）。

这里：
  * 初值来自 calibrate：手套逐骨旋转经无参考帧标定映射到 MANO 局部旋转（扭转已在正常范围）；
  * beta 固定为该被试的共享 beta（shape.py）；
  * 数据项：MANO 关节 vs CMA 关节（腕对齐、腕点可在 2.9 mm 球内移动），逐关节权重——
    MediaPipe 对照显示 CMA 的指尖 / MCP 与图像最吻合，拇指 CMC 很差（CMA 的 T0->T1 是常数 30 mm），
    PIP/DIP 位置受 CMA 畸形指节长度影响，权重降低；FK 修补出来的关节再减半；
  * 正则：
      anchor   物理轴角离标定初值的距离（位置约束不到的自由度——尤其末节扭转——留在手套测到的值）
      twist    每关节绕自身骨轴的扭转超过容差的部分（hinge 平方）
      hinge    PIP / DIP / 拇指 IP 偏离屈曲轴的分量超过容差的部分
      smooth   连续帧（无时间断档）上腕相对关节位置的二阶差分
全部帧一起用 Adam 优化（时间项把相邻帧耦合在一起）。
"""
from __future__ import annotations

import time

import numpy as np
import torch

from robust_convert.cleaning import INVALID, REPAIRED
from robust_convert.mano_fit import JointModel, _Adam, optimal_wrist
from robust_convert.mano_model import mano_forward

from .anatomy import DIP, MCP, PIP, THUMB, bone_dirs, rest_points

# OpenPose 21 点的数据项权重
JOINT_WEIGHTS = np.array([4.0,                    # 腕
                          0.1, 0.5, 0.5, 1.0,     # 拇指 CMC / MCP / IP / 尖
                          1.0, 0.5, 0.5, 1.0,     # 食指 MCP / PIP / DIP / 尖
                          1.0, 0.5, 0.5, 1.0,     # 中指
                          1.0, 0.5, 0.5, 1.0,     # 无名指
                          1.0, 0.5, 0.5, 1.0])    # 小指
REPAIRED_SCALE = 0.5
WRIST_RADIUS_MM = 2.899

# 原生关节 1..15 的扭转容差（度）：MCP 允许一点轴向旋转，PIP/DIP 基本是纯铰链，拇指 CMC 本来就有较大轴旋
TWIST_TOL_DEG = np.zeros(16)
TWIST_TOL_DEG[list(MCP)] = 15.0
TWIST_TOL_DEG[list(PIP) + list(DIP)] = 8.0
TWIST_TOL_DEG[13], TWIST_TOL_DEG[14], TWIST_TOL_DEG[15] = 35.0, 25.0, 12.0
# 偏离屈曲轴的容差（度），只约束铰链型关节
HINGE_JOINTS = list(PIP) + list(DIP) + [15]
HINGE_TOL_DEG = {j: 10.0 for j in list(PIP) + list(DIP)}
HINGE_TOL_DEG[15] = 15.0

DEFAULTS = {'steps': 300, 'lr': 0.02, 'anchor': 50.0, 'twist': 2000.0, 'hinge': 1000.0, 'smooth': 0.05,
            'threads': 4}


def adjacency(frames, stamps_ns, gap_factor=1.5):
    """相邻两个拟合帧是否是真正连续的 RGB 帧（帧号相邻且时间间隔不超过名义间隔的 gap_factor 倍），(T-1,)。"""
    frames = np.asarray(frames)
    stamps = np.asarray(stamps_ns, np.int64)
    if len(frames) < 2:
        return np.zeros(0, bool)
    nominal = np.median(np.diff(stamps))
    dt = stamps[frames[1:]] - stamps[frames[:-1]]
    return (np.diff(frames) == 1) & (dt <= gap_factor * nominal)


def data_weights(status, joint_weights=JOINT_WEIGHTS):
    """(T,21) 关节状态 -> 数据项权重。"""
    w = np.broadcast_to(np.asarray(joint_weights, np.float64), status.shape).copy()
    w[status == REPAIRED] *= REPAIRED_SCALE
    w[status == INVALID] = 0.0
    return w


def _twist(r, axis):
    """物理轴角 r (T,K,3) 绕单位轴 axis (K,3) 的扭转角（弧度，(-pi, pi]）。"""
    theta = (r.square().sum(-1) + 1e-12).sqrt()
    qw = torch.cos(theta / 2)
    qv = r * (0.5 * torch.sinc(theta / (2 * torch.pi)))[..., None]
    sign = torch.where(qw < 0, -1.0, 1.0).to(r.dtype).detach()
    return 2 * torch.atan2(sign * (qv * axis).sum(-1), sign * qw)


def refit_hand(anat, beta, points_mm, status, init, adjacent, **options):
    """一只手所有有效帧的重拟合。

    points_mm (T,21,3) CMA 世界系关节；status (T,21)；init = pose_from_calibration 的输出（go 世界系、hp 残差）；
    adjacent (T-1,) 相邻帧是否连续。返回 go / hp / delta_mm / joints_world_m 与诊断。
    """
    config = {**DEFAULTS, **options}
    started = time.perf_counter()
    torch.set_num_threads(int(config['threads']))
    dtype = torch.float64
    points = np.asarray(points_mm, np.float64)
    count = len(points)
    weights = data_weights(np.asarray(status), config.get('joint_weights', JOINT_WEIGHTS))
    radius = float(config.get('wrist_radius', WRIST_RADIUS_MM))
    target = torch.as_tensor(points - points[:, :1], dtype=dtype)
    weight_t = torch.as_tensor(weights, dtype=dtype)
    beta_t = torch.as_tensor(np.asarray(beta, np.float64), dtype=dtype)
    model = JointModel(anat.model, dtype=dtype)
    mean = torch.as_tensor(anat.hands_mean, dtype=dtype).reshape(15, 3)

    rest21 = rest_points(anat.model, beta)
    bones = torch.as_tensor(bone_dirs(rest21)[1:], dtype=dtype)                      # (15,3)
    twist_tol = torch.as_tensor(np.radians(TWIST_TOL_DEG[1:]), dtype=dtype)
    hinge_idx = [j - 1 for j in HINGE_JOINTS]
    hinge_axis = torch.as_tensor(anat.chain_axis[HINGE_JOINTS], dtype=dtype)
    hinge_tol = torch.as_tensor(np.radians([HINGE_TOL_DEG[j] for j in HINGE_JOINTS]), dtype=dtype)

    pose0 = np.concatenate([init['go'], init['hp']], -1)
    pose = torch.nn.Parameter(torch.as_tensor(pose0, dtype=dtype).clone())
    physical0 = torch.as_tensor(pose0[:, 3:], dtype=dtype).reshape(count, 15, 3) + mean
    go0 = torch.as_tensor(pose0[:, :3], dtype=dtype)
    adj = np.asarray(adjacent, bool)
    triple = torch.as_tensor(np.flatnonzero(adj[:-1] & adj[1:]) + 1 if count > 2 else np.zeros(0, int))

    def evaluate():
        predicted = model(beta_t, pose)
        error = predicted - target
        delta = optimal_wrist(error, weight_t, radius)
        residual = error + delta[:, None, :]
        data = (residual.square().sum(-1) * weight_t).sum(1).mean()
        physical = pose[:, 3:].reshape(count, 15, 3) + mean
        anchor = (physical - physical0).square().sum((1, 2)).mean() + (pose[:, :3] - go0).square().sum(1).mean()
        tw = _twist(physical, bones)
        twist = torch.relu(tw.abs() - twist_tol).square().sum(1).mean()
        r = physical[:, hinge_idx]
        off = r - (r * hinge_axis).sum(-1, keepdim=True) * hinge_axis
        off_norm = (off.square().sum(-1) + 1e-12).sqrt()
        hinge = torch.relu(off_norm - hinge_tol).square().sum(1).mean()
        if len(triple):
            placed = predicted + delta[:, None, :]
            acc = placed[triple + 1] - 2 * placed[triple] + placed[triple - 1]
            smooth = acc.square().sum((1, 2)).mean()
        else:
            smooth = torch.zeros((), dtype=dtype)
        terms = {'data': data, 'anchor': config['anchor'] * anchor, 'twist': config['twist'] * twist,
                 'hinge': config['hinge'] * hinge, 'smooth': config['smooth'] * smooth}
        return sum(terms.values()), delta, terms

    optimizer = _Adam([(pose, config['lr'])])
    history, best, best_loss = [], None, float('inf')
    steps = int(config['steps'])
    for step in range(steps + 1):
        optimizer.zero_grad()
        loss, _, terms = evaluate()
        value = float(loss.detach())
        if not np.isfinite(value):
            raise RuntimeError(f'nonfinite refit loss at step {step}')
        if value < best_loss:
            best_loss, best = value, pose.detach().clone()
        if step % 50 == 0 or step == steps:
            history.append({'step': step, 'loss': value, **{k: float(v.detach()) for k, v in terms.items()}})
        if step == steps:
            break
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pose.copy_(best)
        _, delta, terms = evaluate()
    go = pose.detach().numpy()[:, :3].copy()
    hp = pose.detach().numpy()[:, 3:].copy()
    delta_mm = delta.detach().numpy().copy()
    _, _, _, rest = mano_forward(anat.model, beta, np.zeros(3), np.zeros(45), np.zeros(3))
    trans = points[:, 0] / 1000 + delta_mm / 1000 - rest[0]
    _, joints, _, _ = mano_forward(anat.model, beta, go, hp, trans)
    errors = np.linalg.norm(joints * 1000 - points, axis=-1)
    physical = hp.reshape(count, 15, 3) + anat.hands_mean.reshape(15, 3)
    return {'go': go, 'hp': hp, 'delta_mm': delta_mm, 'trans': trans, 'joints_world_m': joints,
            'error_mm': errors, 'twist_deg': anat.twist(physical, bone_dirs(rest21)),
            'loss_terms': {k: float(v) for k, v in terms.items()}, 'history': history, 'config': config,
            'seconds': time.perf_counter() - started}
