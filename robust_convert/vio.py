# -*- coding: utf-8 -*-
"""Mount rotation from the ZED visual-inertial odometry (hand-eye rotation).

The ZED pose Q(t) rotates with the camera, the CMA rig rotation rig_R(t) with
the head. For two instants, A = Q1^T Q2 and B = rig1^T rig2 are the same motion
seen in two frames: A = X B X^T with X = C mount_R, where C is the fixed axis
convention of the ZED pose (a signed permutation; diag(1,-1,-1) = ZED
"right-handed, y up" on these recordings). The rotation vectors satisfy
a = X b, so X follows from Kabsch; C is snapped against a reference mount.

This rotation does not depend on any image keypoint, so it is immune to the
rotation/translation trade-off of a hand-only PnP (hands at a nearly constant
depth let ~3 deg of rotation hide behind ~30 mm of translation).
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial.transform import Rotation

from .timing import slerp_rotations

LAGS = (10, 20, 30, 45)          # frame lags of the rotation pairs (0.33-1.5 s at 30 fps)
MIN_ANGLE_DEG = 4.0
MAX_GAP_DEG = 3.0                # |angle(A) - angle(B)| above this is a glitch in one stream
MIN_PAIRS = 40
MIN_AXIS_SPREAD = 0.15           # smallest normalised singular value of the rotation axes


def angle_deg(a, b):
    return float(np.degrees(Rotation.from_matrix(np.asarray(a).T @ np.asarray(b)).magnitude()))


def signed_permutations():
    """The 24 proper signed permutation matrices (axis-convention changes)."""
    out = []
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            matrix = np.zeros((3, 3))
            matrix[np.arange(3), order] = signs
            if np.linalg.det(matrix) > 0:
                out.append(matrix)
    return out


def snap_convention(matrix):
    """Nearest proper signed permutation and the remaining angle (deg)."""
    candidates = signed_permutations()
    angles = [angle_deg(C, matrix) for C in candidates]
    best = int(np.argmin(angles))
    return candidates[best], float(angles[best])


def kabsch(source, target, weights=None):
    """Rotation X minimising sum w |target - X source|^2."""
    weights = np.ones(len(source)) if weights is None else weights
    H = (source * weights[:, None]).T @ target
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def rotation_pairs(session, vio, lags=LAGS, stride=2):
    """Rotation vectors a (VIO, both quaternion conventions) and b (rig) for frame pairs."""
    stamps = session['stamps']
    usable = vio['valid'] & ~(vio['confidence'] <= 0)
    if usable.sum() < 30:
        return None
    vio_stamps = vio['stamps'][usable]
    Q = slerp_rotations(vio_stamps, Rotation.from_quat(vio['quat_xyzw'][usable]).as_matrix(), stamps)
    position = np.clip(np.searchsorted(vio_stamps, stamps), 1, len(vio_stamps) - 1)
    gap = np.minimum(np.abs(vio_stamps[position] - stamps), np.abs(vio_stamps[position - 1] - stamps))
    ok = session['rig_ok'] & (gap < 20_000_000) & (stamps >= vio_stamps[0]) & (stamps <= vio_stamps[-1])
    first, second = [], []
    for lag in lags:
        i = np.arange(0, max(len(stamps) - lag, 0), stride)
        j = i + lag
        good = ok[i] & ok[j]
        first.append(i[good])
        second.append(j[good])
    i, j = np.concatenate(first), np.concatenate(second)
    if len(i) < MIN_PAIRS:
        return None
    rig = session['rig_R']
    b = Rotation.from_matrix(np.swapaxes(rig[i], -1, -2) @ rig[j]).as_rotvec()
    a = {'camera_to_world': Rotation.from_matrix(np.swapaxes(Q[i], -1, -2) @ Q[j]).as_rotvec(),
         'world_to_camera': Rotation.from_matrix(Q[i] @ np.swapaxes(Q[j], -1, -2)).as_rotvec()}
    return {'i': i, 'j': j, 'a': a, 'b': b}


def _solve(a, b, use):
    X = kabsch(b[use], a[use])
    for _ in range(3):
        residual = np.degrees(np.linalg.norm(a - b @ X.T, axis=1))
        limit = max(3.0 * float(np.median(residual[use])), 0.5)
        refined = use & (residual < limit)
        if refined.sum() < MIN_PAIRS // 2:
            break
        X = kabsch(b[refined], a[refined])
    residual = np.degrees(np.linalg.norm(a - b @ X.T, axis=1))
    return X, residual


def handeye_rotation(session, vio, reference_R, blocks=5):
    """Mount rotation implied by the VIO, with diagnostics and a jackknife uncertainty.

    reference_R only fixes the axis convention C (any mount within 45 deg works).
    """
    pairs = rotation_pairs(session, vio)
    if pairs is None:
        return {'status': 'no_vio'}
    b = pairs['b']
    best = None
    for convention, a in pairs['a'].items():
        angle_a, angle_b = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        use = (angle_b > np.radians(MIN_ANGLE_DEG)) & (np.abs(angle_a - angle_b) < np.radians(MAX_GAP_DEG))
        if use.sum() < MIN_PAIRS:
            continue
        X, residual = _solve(a, b, use)
        score = float(np.median(residual[use]))
        if best is None or score < best['score']:
            best = {'convention': convention, 'X': X, 'score': score, 'use': use, 'a': a,
                    'gap': float(np.degrees(np.median(np.abs(angle_a - angle_b)[use]))),
                    'range': float(np.degrees(np.percentile(angle_b[use], 90)))}
    if best is None:
        return {'status': 'insufficient_rotation', 'pairs': int(len(b))}
    X, use, a = best['X'], best['use'], best['a']
    C, snap_angle = snap_convention(X @ np.asarray(reference_R).T)
    mount_R = C.T @ X
    axes = b[use] / np.linalg.norm(b[use], axis=1, keepdims=True)
    spread = np.linalg.svd(axes, compute_uv=False)
    spread = spread / spread[0]
    # Jackknife over temporal blocks of the first frame of every pair.
    edges = np.array_split(np.arange(len(session['stamps'])), blocks)
    leave_out = []
    for block in edges:
        subset = use & ~np.isin(pairs['i'], block)
        if subset.sum() >= MIN_PAIRS // 2:
            leave_out.append(C.T @ _solve(a, b, subset)[0])
    jack = [angle_deg(R, mount_R) for R in leave_out]
    n = len(jack)
    jack_se = float(np.sqrt((n - 1) / n * np.sum(np.square(jack)))) if n > 1 else float('nan')
    covariance = _covariance(a, b, X, use, C, mount_R, leave_out)
    usable = spread[-1] >= MIN_AXIS_SPREAD and best['score'] < 1.0
    return {'status': 'ok' if usable else 'weak', 'mount_R': mount_R, 'X': X, 'axis_convention': C.astype(int).tolist(),
            'convention': best['convention'], 'pairs_used': int(use.sum()),
            'pair_residual_median_deg': best['score'], 'angle_gap_median_deg': best['gap'],
            'rotation_p90_deg': best['range'], 'axis_spread': spread.tolist(),
            'reference_snap_deg': snap_angle, 'jackknife_se_deg': jack_se,
            'jackknife_max_deg': float(max(jack)) if jack else float('nan'),
            'covariance_deg2': covariance}


def _covariance(a, b, X, use, C, mount_R, leave_out):
    """Rotation covariance of mount_R (camera-frame rotation vector, deg^2).

    Shape from the Kabsch Fisher information (a rotation about the dominant
    head-rotation axis is the poorly observed one), scale from the block
    jackknife because overlapping pairs are strongly correlated.
    """
    v = b[use] @ X.T
    residual = a[use] - v
    sigma2 = float(np.mean(np.square(residual)))            # per component, rad^2
    information = np.sum(np.square(v)) * np.eye(3) - v.T @ v   # sum of [v]x^T [v]x
    shape = C.T @ (sigma2 * np.linalg.inv(information)) @ C
    shape = np.degrees(np.degrees(shape))
    if len(leave_out) > 1:
        theta = np.degrees(np.array([Rotation.from_matrix(R @ mount_R.T).as_rotvec() for R in leave_out]))
        n = len(theta)
        jack = (n - 1) / n * (theta - theta.mean(0)).T @ (theta - theta.mean(0))
        shape = shape * max(np.trace(jack) / np.trace(shape), 1.0)
    return shape.tolist()


def pair_residual_curve(session, vio, handeye, offsets_ms, resample_fn):
    """VIO/rig rotation consistency with the CMA shifted by each offset (a MediaPipe-free sync check).

    The pair set and the VIO convention are fixed at offset 0 so the curve is comparable.
    Restores the session at offset 0.
    """
    base = rotation_pairs(session, vio)
    if base is None or handeye.get('status') not in ('ok', 'weak'):
        return []
    a = base['a'][handeye['convention']]
    angle_a, angle_b = np.linalg.norm(a, axis=1), np.linalg.norm(base['b'], axis=1)
    use = (angle_b > np.radians(MIN_ANGLE_DEG)) & (np.abs(angle_a - angle_b) < np.radians(MAX_GAP_DEG))
    rows = []
    for offset in offsets_ms:
        resample_fn(session, int(round(offset * 1e6)))
        rig = session['rig_R']
        b = Rotation.from_matrix(np.swapaxes(rig[base['i']], -1, -2) @ rig[base['j']]).as_rotvec()
        valid = use & session['rig_ok'][base['i']] & session['rig_ok'][base['j']]
        if valid.sum() < MIN_PAIRS // 2:
            rows.append({'offset_ms': float(offset), 'residual_deg': None})
            continue
        X = kabsch(b[valid], a[valid])
        residual = np.degrees(np.linalg.norm(a - b @ X.T, axis=1))[valid]
        rows.append({'offset_ms': float(offset), 'residual_deg': float(np.mean(residual)),
                     'residual_median_deg': float(np.median(residual))})
    resample_fn(session, 0)
    return rows
