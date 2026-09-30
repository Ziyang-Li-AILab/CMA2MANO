# -*- coding: utf-8 -*-
"""Same robust reprojection refinement as robust_convert.extrinsic.refine, with an analytic Jacobian.

scipy's default finite-difference Jacobian repeats the projection once per pose
parameter. The residual, loss, gates and iteration limits are unchanged, so the
mount stays at the solver tolerance of the original fit.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from robust_convert.extrinsic import pose_from_vector, pose_vector, project


def _skew(w):
    x, y, z = w
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _right_jacobian(w):
    """Right Jacobian of SO(3) at rotvec w. ∂(Exp(w) p)/∂w = -R [p]_x Jr."""
    theta2 = float(np.dot(w, w))
    skew_w = _skew(w)
    if theta2 < 1e-12:
        return np.eye(3) - 0.5 * skew_w + skew_w @ skew_w / 6.0
    theta = np.sqrt(theta2)
    a = (1.0 - np.cos(theta)) / theta2
    b = (theta - np.sin(theta)) / (theta2 * theta)
    return np.eye(3) - a * skew_w + b * (skew_w @ skew_w)


def _projection_jacobian(obj, rotation, translation, K, rotvec, fixed_rotation):
    """Jacobian of (u, v) rows, shape (N, 2, n_params). Rows with z<=1 are zero."""
    camera = obj @ rotation.T + translation
    z = camera[:, 2]
    ok = z > 1.0
    inv_z = np.zeros_like(z)
    inv_z[ok] = 1.0 / z[ok]
    x = camera[:, 0] * inv_z
    y = camera[:, 1] * inv_z
    fx, fy = K[0, 0], K[1, 1]
    # du/d(Xc), dv/d(Xc)
    du = np.stack([fx * inv_z, np.zeros_like(z), -fx * x * inv_z], axis=1)
    dv = np.stack([np.zeros_like(z), fy * inv_z, -fy * y * inv_z], axis=1)
    count = len(obj)
    if fixed_rotation:
        jac = np.zeros((count, 2, 3))
        jac[:, 0, :] = du
        jac[:, 1, :] = dv
        jac[~ok] = 0.0
        return jac
    # d(Xc)/dw = -R [p]_x Jr , p = obj
    jr = _right_jacobian(rotvec)
    # d(R p)/dw = -R [p]_x Jr
    skew_p = np.zeros((count, 3, 3))
    px, py, pz = obj[:, 0], obj[:, 1], obj[:, 2]
    skew_p[:, 0, 1] = -pz
    skew_p[:, 0, 2] = py
    skew_p[:, 1, 0] = pz
    skew_p[:, 1, 2] = -px
    skew_p[:, 2, 0] = -py
    skew_p[:, 2, 1] = px
    # (N,3,3) = -R @ skew(p) @ Jr
    d_camera = -np.einsum('ab,nbc,cd->nad', rotation, skew_p, jr)
    d_u = np.einsum('na,nad->nd', du, d_camera)
    d_v = np.einsum('na,nad->nd', dv, d_camera)
    jac = np.zeros((count, 2, 6))
    jac[:, 0, :3] = d_u
    jac[:, 1, :3] = d_v
    jac[:, 0, 3:] = du
    jac[:, 1, 3:] = dv
    jac[~ok] = 0.0
    return jac


def refine(obj, img, K, rotation, translation, weights=None, f_scale=5.0, drop_px=30.0, max_nfev=80,
           fixed_rotation=False):
    """robust_convert.extrinsic.refine with an analytic Jacobian."""
    weights = np.ones(len(obj)) if weights is None else np.asarray(weights, np.float64)
    rotation = np.asarray(rotation, np.float64)
    sqrt_w = np.sqrt(weights)

    def unpack(x):
        return (rotation, np.asarray(x, np.float64)) if fixed_rotation else pose_from_vector(x)

    def residual(x, keep):
        r, t = unpack(x)
        uv, _ = project(obj[keep], r, t, K)
        out = (uv - img[keep]) * sqrt_w[keep, None]
        return np.nan_to_num(out, nan=1e3).ravel()

    def jacobian(x, keep):
        chosen = obj[keep]
        image = img[keep]
        scale = sqrt_w[keep]
        if fixed_rotation:
            r, t = rotation, np.asarray(x, np.float64)
            rotvec = None
        else:
            r, t = pose_from_vector(x)
            rotvec = np.asarray(x[:3], np.float64)
        jac = _projection_jacobian(chosen, r, t, K, rotvec, fixed_rotation)
        # residual is nan_to_num: zero the derivative where the projection is not finite
        uv, z = project(chosen, r, t, K)
        bad = ~np.isfinite(uv).all(1) | ~(z > 1.0)
        jac *= scale[:, None, None]
        jac[bad] = 0.0
        return jac.reshape(-1, jac.shape[-1])

    x = np.asarray(translation, np.float64).copy() if fixed_rotation else pose_vector(rotation, translation)
    scale = np.full(3, 5.0) if fixed_rotation else np.array([0.01, 0.01, 0.01, 5.0, 5.0, 5.0])
    keep = np.ones(len(obj), bool)
    for _ in range(2):
        x = least_squares(residual, x, jac=jacobian, args=(keep,), loss='soft_l1', f_scale=f_scale,
                          max_nfev=max_nfev, x_scale=scale).x
        r, t = unpack(x)
        error = np.linalg.norm(project(obj, r, t, K)[0] - img, axis=-1)
        keep = np.nan_to_num(error, nan=np.inf) < drop_px
        if keep.sum() < 12:
            break
    return r, t, keep, error


def _ransac_sequential(local, detections, K, frames=None, seed=0):
    """Same two hypotheses as ransac_hypotheses, one after the other.

    A batch of recordings already fills the machine. Extra threads inside each
    recording only wait on each other.
    """
    import numpy as np
    from robust_convert.extrinsic import _correspondences, label_pairs, pnp_ransac

    frames = np.arange(len(local)) if frames is None else np.asarray(frames, np.int64)

    def one(hypothesis):
        data = _correspondences(local, detections['uv'], label_pairs(detections, frames, hypothesis))
        result = pnp_ransac(data['obj'], data['img'], K, seed=seed)
        return {'hypothesis': hypothesis, 'points': int(len(data['obj'])),
                'inliers': 0 if result is None else result['inliers'], 'result': result, 'data': data}

    return [one(0), one(1)]


def install():
    """Use the analytic refine inside robust_convert.extrinsic without editing that file."""
    import robust_convert.extrinsic as extrinsic
    extrinsic.refine = refine
    extrinsic.ransac_hypotheses = _ransac_sequential


def check_jacobian(seed=0):
    rng = np.random.default_rng(seed)
    obj = rng.normal(size=(40, 3)) * 20
    obj[:, 2] += 400
    K = np.array([[700.0, 0, 960.0], [0, 700.0, 540.0], [0, 0, 1]])
    rotvec = rng.normal(size=3) * 0.05
    translation = rng.normal(size=3) * 5
    rotation = Rotation.from_rotvec(rotvec).as_matrix()
    analytic = _projection_jacobian(obj, rotation, translation, K, rotvec, False)
    numeric = np.zeros_like(analytic)
    eps = 1e-6
    base, _ = project(obj, rotation, translation, K)
    for k in range(3):
        step = np.zeros(3)
        step[k] = eps
        r = Rotation.from_rotvec(rotvec + step).as_matrix()
        numeric[:, :, k] = (project(obj, r, translation, K)[0] - base) / eps
    for k in range(3):
        step = np.zeros(3)
        step[k] = eps
        numeric[:, :, 3 + k] = (project(obj, rotation, translation + step, K)[0] - base) / eps
    err = np.max(np.abs(analytic - numeric))
    return float(err)
