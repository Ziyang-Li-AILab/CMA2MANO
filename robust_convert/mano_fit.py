# -*- coding: utf-8 -*-
"""Batched MANO fit (adapted from the original fast_mano.fit_hand_fast).

The optimiser, losses, priors and defaults are unchanged. Two additions:
  * per-frame / per-joint weights (T, 21), so FK-repaired joints can be
    down-weighted and invalid joints ignored instead of aborting the fit;
  * the palm-width targets only use frames where those joints are valid.
Frames without any valid joint must be removed by the caller.

The optimiser is torch.optim.Adam re-implemented op for op (_Adam): the
results are bit-identical, but the first torch.optim call no longer imports
torch._dynamo (about 2 s per process).
"""
from __future__ import annotations

import time

import numpy as np
import torch

# rotation_matrices and JointModel: verbatim from the original fast_mano.py.
TIP_VERTEX_IDS = [744, 320, 443, 554, 671]
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


def rotation_matrices(vectors):
    """Rodrigues with finite derivatives at zero, using the axis-angle vector directly."""
    angle2 = vectors.square().sum(-1, keepdim=True)
    angle = angle2.clamp_min(1e-16).sqrt()
    a = torch.sinc(angle / torch.pi)
    b = .5 * torch.sinc(angle / (2 * torch.pi)).square()
    x, y, z = vectors.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(*vectors.shape[:-1], 3, 3)
    identity = torch.eye(3, dtype=vectors.dtype, device=vectors.device)
    return identity + a[..., None] * skew + b[..., None] * (skew @ skew)


class JointModel(torch.nn.Module):
    """Compute the same 16 native joints and five skinning vertices as full MANO, in mm."""
    def __init__(self, model, dtype=torch.float32):
        super().__init__()
        arrays = {
            'rest': model['J_regressor'] @ model['v_template'],
            'rest_dirs': np.einsum('jv,vck->jck', model['J_regressor'], model['shapedirs']),
            'tip_template': model['v_template'][TIP_VERTEX_IDS],
            'tip_dirs': model['shapedirs'][TIP_VERTEX_IDS],
            'tip_pose': model['posedirs'][TIP_VERTEX_IDS],
            'weights': model['weights'][TIP_VERTEX_IDS],
            'mean': model['hands_mean'],
        }
        for name, value in arrays.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=dtype))
        self.parents = model['parents'].tolist()

    def forward(self, beta, pose):
        rest = self.rest + self.rest_dirs @ beta
        physical = torch.cat([pose[:, :3], pose[:, 3:] + self.mean], -1).reshape(-1, 16, 3)
        local = rotation_matrices(physical)
        rotations, positions = [local[:, 0]], [rest[0].expand(len(pose), 3)]
        for joint in range(1, 16):
            parent = self.parents[joint]
            rotations.append(rotations[parent] @ local[:, joint])
            offset = rest[joint] - rest[parent]
            positions.append(positions[parent] + (rotations[parent] @ offset[:, None]).squeeze(-1))
        rotations = torch.stack(rotations, 1)
        positions = torch.stack(positions, 1)
        feature = (local[:, 1:] - torch.eye(3, dtype=pose.dtype, device=pose.device)).reshape(-1, 135)
        tip_posed = self.tip_template + self.tip_dirs @ beta
        tip_posed = tip_posed[None] + torch.einsum('vck,tk->tvc', self.tip_pose, feature)
        skin_offset = positions - torch.einsum('tjab,jb->tja', rotations, rest)
        blend_R = torch.einsum('vj,tjab->tvab', self.weights, rotations)
        blend_t = torch.einsum('vj,tja->tva', self.weights, skin_offset)
        tips = torch.einsum('tvab,tvb->tva', blend_R, tip_posed) + blend_t
        points = torch.cat([positions, tips], 1)[:, MANO_TO_OPENPOSE]
        return (points - rest[0]) * 1000


def optimal_wrist(errors, weights, radius_mm=2.899):
    """Per-frame weighted mean offset, projected into the wrist ball (weights are (T, 21))."""
    center = (errors * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-12)
    scale = (radius_mm / center.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1)
    return -center * scale


class _Adam:
    """torch.optim.Adam(params, lr) for plain tensors: default betas and eps, no weight decay.

    Same tensor operations in the same order as torch's single-tensor Adam (the
    default off CUDA), so the parameters are bit-identical on CPU.
    """
    def __init__(self, groups, betas=(0.9, 0.999), eps=1e-8):
        self.groups = [(param, float(lr), torch.zeros_like(param, memory_format=torch.preserve_format),
                        torch.zeros_like(param, memory_format=torch.preserve_format)) for param, lr in groups]
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.count = 0

    def zero_grad(self):
        for param, *_ in self.groups:
            param.grad = None

    @torch.no_grad()
    def step(self):
        self.count += 1
        step = float(self.count)
        beta1, beta2 = self.beta1, self.beta2
        for param, lr, exp_avg, exp_avg_sq in self.groups:
            grad = param.grad
            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr / bias_correction1
            bias_correction2_sqrt = bias_correction2 ** 0.5
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(self.eps)
            param.addcdiv_(exp_avg, denom, value=-step_size)


def fit_hand_masked(model, points_mm, init, weights=None, *, steps=200, shape_prior=0.25,
                    palm_weight=1.0, pose_prior=9.0, wrist_radius_mm=2.899, threads=4,
                    pose_lr=0.02, shape_lr=0.05, wrist_weight=4.0, device='cpu'):
    points = np.asarray(points_mm, np.float64)
    count = len(points)
    if points.ndim != 3 or points.shape[1:] != (21, 3) or not count:
        raise ValueError('points_mm must have nonempty shape (T,21,3)')
    base = np.ones((count, 21)) if weights is None else np.asarray(weights, np.float64).copy()
    base[~np.isfinite(points).all(-1)] = 0.0
    if not (base[:, 0] > 0).all():
        raise ValueError('every fitted frame needs a valid wrist')
    points = np.where(base[..., None] > 0, np.nan_to_num(points), 0.0)
    joint_weight = base.copy()
    joint_weight[:, 0] *= wrist_weight
    torch.set_num_threads(int(threads))
    device = torch.device(device)
    target = torch.as_tensor((points - points[:, :1]).astype(np.float32), device=device)
    weight_t = torch.as_tensor(joint_weight.astype(np.float32), device=device)
    pose_initial = np.concatenate([init['go'], init['hp']], axis=-1).astype(np.float32)
    pose = torch.nn.Parameter(torch.as_tensor(pose_initial.copy(), device=device))
    pose_center = pose.detach().clone()
    beta = torch.nn.Parameter(torch.as_tensor(np.clip(init['beta'], -3, 3), dtype=torch.float32, device=device))
    joint_model = JointModel(model, dtype=torch.float32).to(device)
    op_palm = np.array([0, 1, 5, 9, 13, 17])
    native_palm = np.array([0, 13, 1, 4, 10, 7])
    pair_i, pair_j = np.triu_indices(6, k=1)
    distances = np.linalg.norm(points[:, op_palm[pair_i]] - points[:, op_palm[pair_j]], axis=-1)
    pair_ok = (base[:, op_palm[pair_i]] > 0) & (base[:, op_palm[pair_j]] > 0)
    palm_target = torch.tensor([np.median(distances[pair_ok[:, k], k]) for k in range(len(pair_i))],
                               dtype=torch.float32, device=device)

    def evaluate():
        predicted = joint_model(beta, pose)
        error = predicted - target
        delta = optimal_wrist(error, weight_t, wrist_radius_mm)
        residual = error + delta[:, None, :]
        data = (residual.square().sum(-1) * weight_t).mean()
        rest = (joint_model.rest + joint_model.rest_dirs @ beta)[native_palm] * 1000
        palm = (torch.linalg.vector_norm(rest[pair_i] - rest[pair_j], dim=-1) - palm_target).square().mean()
        shape = float(shape_prior) * beta.square().sum()
        prior = float(pose_prior) * ((pose[:, :3] - pose_center[:, :3]).square().mean()
                                    + (pose[:, 3:] - pose_center[:, 3:]).square().mean())
        terms = {'data': data, 'shape': shape, 'pose': prior, 'palm': palm * palm_weight}
        return sum(terms.values()), delta, terms

    optimizer = _Adam([(pose, pose_lr), (beta, shape_lr)])
    history, best, best_loss, best_step = [], None, float('inf'), 0
    start = time.perf_counter()
    for step in range(int(steps) + 1):
        optimizer.zero_grad()
        loss, delta, terms = evaluate()
        value = float(loss.detach())
        if not np.isfinite(value):
            raise RuntimeError(f'Nonfinite MANO loss at step {step}')
        if value < best_loss:
            best_loss, best_step = value, step
            best = (beta.detach().clone(), pose.detach().clone())
        if step == 0 or step % 25 == 0 or step == steps:
            history.append({'step': step, 'loss': value, **{k: float(v.detach()) for k, v in terms.items()},
                            'beta_norm': float(beta.detach().norm())})
        if step == steps:
            break
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            beta.clamp_(-3., 3.)
    with torch.no_grad():
        beta.copy_(best[0])
        pose.copy_(best[1])
        total, delta, terms = evaluate()
    return {'beta': beta.detach().cpu().numpy().astype(np.float64),
            'go': pose.detach().cpu().numpy()[:, :3].astype(np.float64),
            'hp': pose.detach().cpu().numpy()[:, 3:].astype(np.float64),
            'delta_mm': delta.detach().cpu().numpy().astype(np.float64),
            'total': float(total), 'data': float(terms['data']),
            'loss_terms': {k: float(v) for k, v in terms.items()},
            'best_step': best_step, 'initial_total': history[0]['loss'], 'history': history,
            'solver': 'batched_torch_adam_palm_v1_masked', 'steps': int(steps),
            'seconds': time.perf_counter() - start}
