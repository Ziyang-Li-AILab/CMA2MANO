# -*- coding: utf-8 -*-
"""MANO 手部解剖参考（左右手各一份），供标定、拟合正则和审计使用。

全部量在 MANO 模板（canonical）坐标系里，关节用 MANO 原生序号 j=1..15：
  body j 的骨   OpenPose 点 op(j) -> op(j)+1（末节指向指尖顶点），op = NATIVE_FROM_OP
  hinge[j]      关节 j 局部旋转的主转轴：MANO 训练姿态（hands_coeffs @ hands_components + hands_mean）
                轴角的无中心 SVD 主方向。符号取“正转 = 屈曲”：手指按掌心方向判定，拇指按样本均值判定
  palmar        掌心方向（屈曲时指尖移动的方向），用 hands_mean 姿态下指尖位移核对符号
  gauss         每关节物理轴角的均值 / 协方差（Mahalanobis 合理性分数）
  twist_ref     MANO 训练姿态里各关节绕自身骨轴扭转的 p95（“正常”扭转的参照）
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from robust_convert.mano_model import (DEFAULT_MANO, MANO_TO_OPENPOSE, NATIVE_FROM_OP, TIP_VERTEX_IDS, load_mano,
                                       mano_forward, read_model_arrays)

SIDES = ('left', 'right')
PARENTS = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14])
CHAINS = {'index': (1, 2, 3), 'middle': (4, 5, 6), 'pinky': (7, 8, 9), 'ring': (10, 11, 12), 'thumb': (13, 14, 15)}
FINGER_CHAINS = ('index', 'middle', 'ring', 'pinky')
MCP = (1, 4, 7, 10)
PIP = (2, 5, 8, 11)
DIP = (3, 6, 9, 12)
THUMB = (13, 14, 15)
JOINT_NAMES = ['I_MCP', 'I_PIP', 'I_DIP', 'M_MCP', 'M_PIP', 'M_DIP', 'P_MCP', 'P_PIP', 'P_DIP',
               'R_MCP', 'R_PIP', 'R_DIP', 'T_CMC', 'T_MCP', 'T_IP']   # 原生 1..15
OP_MCP = (5, 9, 13, 17)
GAUSS_REG_DEG = 5.0


def unit(v):
    v = np.asarray(v, np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def rest_points(model, beta=None):
    """模板（给定 beta）的 21 点，OpenPose 顺序，米。"""
    beta = np.zeros(10) if beta is None else np.asarray(beta, np.float64)
    shaped = model['v_template'] + model['shapedirs'] @ beta
    joints = model['J_regressor'] @ shaped
    return np.concatenate([joints, shaped[TIP_VERTEX_IDS]])[MANO_TO_OPENPOSE]


def bone_dirs(rest21):
    """原生 body j=1..15 的模板骨方向（单位），(16,3)，第 0 行为 腕->中指 MCP。"""
    out = np.zeros((16, 3))
    out[0] = unit(rest21[9] - rest21[0])
    for j in range(1, 16):
        op = NATIVE_FROM_OP[j]
        out[j] = unit(rest21[op + 1] - rest21[op])
    return out


def pose_samples(path, side):
    """MANO 训练姿态的物理局部轴角 (M,15,3)。"""
    arrays, _ = read_model_arrays(path, side)
    coeffs = np.asarray(arrays['hands_coeffs'], np.float64)
    components = np.asarray(arrays['hands_components'], np.float64)
    mean = np.asarray(arrays['hands_mean'], np.float64).reshape(45)
    return (coeffs @ components + mean).reshape(-1, 15, 3)


def twist_angle(rotvec, axis):
    """旋转（轴角）绕给定轴的扭转分量（swing-twist 分解），弧度，(-pi, pi]。"""
    quat = Rotation.from_rotvec(np.asarray(rotvec, np.float64).reshape(-1, 3)).as_quat()
    axis = np.broadcast_to(np.asarray(axis, np.float64), np.asarray(rotvec).shape).reshape(-1, 3)
    proj = np.einsum('ij,ij->i', quat[:, :3], axis)
    twist = 2 * np.arctan2(proj, quat[:, 3])
    twist = (twist + np.pi) % (2 * np.pi) - np.pi
    return twist.reshape(np.asarray(rotvec).shape[:-1])


class HandAnatomy:
    """一只手（left / right）的 MANO 参考量。"""

    def __init__(self, side, path=DEFAULT_MANO, model=None):
        self.side = side
        self.model = model if model is not None else load_mano(path, side)
        self.hands_mean = self.model['hands_mean']
        self.rest = rest_points(self.model)
        self.bones = bone_dirs(self.rest)
        self.samples = pose_samples(path, side)
        self.palmar, self.palmar_check = self._palmar()
        self.hinge, self.hinge_var = self._hinges()
        self.chain_axis = self._chain_axes()
        self.mu = self.samples.mean(0)
        reg = np.radians(GAUSS_REG_DEG) ** 2 * np.eye(3)
        self.cov = np.stack([np.cov(self.samples[:, k].T) + reg for k in range(15)])
        self.cov_inv = np.linalg.inv(self.cov)
        twist = np.stack([twist_angle(self.samples[:, j - 1], self.bones[j]) for j in range(1, 16)], 1)
        self.twist_ref_p95_deg = np.degrees(np.percentile(np.abs(twist), 95, axis=0))

    # -- 掌心方向 -------------------------------------------------------------------------------
    def _palmar(self):
        n = unit(np.cross(self.rest[5] - self.rest[0], self.rest[17] - self.rest[0]))
        # hands_mean 是轻度屈曲的姿态：指尖相对平伸模板的位移应朝掌心
        _, flexed, _, _ = mano_forward(self.model, np.zeros(10), np.zeros(3), np.zeros(45), np.zeros(3))
        _, flat, _, _ = mano_forward(self.model, np.zeros(10), np.zeros(3), -self.hands_mean, np.zeros(3))
        move = flexed[0, [8, 12, 16, 20]] - flat[0, [8, 12, 16, 20]]
        score = float(np.mean(move @ n))
        if score < 0:
            n = -n
        return n, {'cross_sign': 1 if score >= 0 else -1, 'tip_move_along_palmar_mm': abs(score) * 1000}

    # -- 屈曲轴 --------------------------------------------------------------------------------
    def _hinges(self):
        hinge = np.zeros((16, 3))
        var = np.zeros(16)
        for j in range(1, 16):
            x = self.samples[:, j - 1]
            _, s, vt = np.linalg.svd(x, full_matrices=False)
            axis = vt[0]
            var[j] = s[0] ** 2 / max((s ** 2).sum(), 1e-12)
            if j in THUMB:
                sign = np.sign(np.mean(x @ axis)) or 1.0
            else:
                sign = np.sign(np.cross(axis, self.bones[j]) @ self.palmar) or 1.0
            hinge[j] = axis * sign
        return hinge, var

    def _chain_axes(self):
        """标定用的每个 body 的屈曲轴：手指链三节共用 PIP/DIP 轴的平均；拇指 CMC 用 T1，MCP/IP 用 IP 轴。"""
        axis = np.zeros((16, 3))
        for name in FINGER_CHAINS:
            mcp, pip, dip = CHAINS[name]
            shared = unit(self.hinge[pip] + self.hinge[dip])
            axis[[mcp, pip, dip]] = shared
        axis[13] = self.hinge[13]
        axis[14] = self.hinge[15]
        axis[15] = self.hinge[15]
        return axis

    # -- 合理性 -------------------------------------------------------------------------------
    def mahalanobis(self, physical):
        """物理局部轴角 (...,15,3) -> 每关节 Mahalanobis 平方 (...,15)。"""
        d = np.asarray(physical, np.float64) - self.mu
        return np.einsum('...ki,kij,...kj->...k', d, self.cov_inv, d)

    def mahalanobis_joint(self, rotvec, j):
        """单个关节 j（原生 1..15）的物理轴角 (...,3) -> Mahalanobis 平方。"""
        d = np.asarray(rotvec, np.float64) - self.mu[j - 1]
        return np.einsum('...i,ij,...j->...', d, self.cov_inv[j - 1], d)

    def twist_single(self, rotvec, j):
        """单个关节 j 的物理轴角 (...,3) -> 绕模板骨轴的扭转（弧度）。"""
        return twist_angle(rotvec, self.bones[j])

    def twist(self, physical, bones=None):
        """物理局部轴角 (...,15,3) -> 每关节绕自身模板骨轴的扭转（度）(...,15)。"""
        bones = self.bones if bones is None else bones
        physical = np.asarray(physical, np.float64)
        out = np.stack([twist_angle(physical[..., j - 1, :], bones[j]) for j in range(1, 16)], -1)
        return np.degrees(out)

    def summary(self):
        return {
            'palmar': self.palmar.round(4).tolist(), 'palmar_check': self.palmar_check,
            'hinge': {JOINT_NAMES[j - 1]: self.hinge[j].round(4).tolist() for j in range(1, 16)},
            'hinge_var1': {JOINT_NAMES[j - 1]: round(float(self.hinge_var[j]), 3) for j in range(1, 16)},
            'hinge_vs_bone_deg': {JOINT_NAMES[j - 1]: round(float(np.degrees(np.arccos(abs(
                self.hinge[j] @ self.bones[j])))), 1) for j in range(1, 16)},
            'twist_ref_p95_deg': {JOINT_NAMES[j - 1]: round(float(self.twist_ref_p95_deg[j - 1]), 1)
                                  for j in range(1, 16)},
        }


_CACHE = {}


def get_anatomy(side, path=DEFAULT_MANO):
    key = (str(path), side)
    if key not in _CACHE:
        _CACHE[key] = HandAnatomy(side, path)
    return _CACHE[key]
