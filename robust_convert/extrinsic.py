# -*- coding: utf-8 -*-
"""Camera mount (head rig -> camera) from MediaPipe 2D keypoints and CMA 3D joints.

The original pipeline estimates the mount from depth skin masks and convex-hull
fingertips: it needs both hands fully open, five separated tips per hand and
depth for the same frame, so most recordings yield 0 observations and fail.

Here every RGB frame contributes. MediaPipe gives 21 keypoints per detected
hand; the CMA gives the same 21 OpenPose joints in the rig frame. With object
points in the rig frame, the PnP pose (rvec, tvec) is directly (mount_R,
mount_t) of the original convention cam = local @ mount_R.T + mount_t.

  1. Handedness: MediaPipe labels may be mirrored, so both mappings are tried
     with pooled RANSAC PnP; the one with more inliers wins.
  2. Association: with the current mount every detection is re-assigned to the
     CMA hand it overlaps best (per frame, both permutations compared), and
     assignments whose median error exceeds a gate are dropped.
  3. Refinement: robust least squares (soft L1, f_scale 5 px), points worse
     than 30 px dropped, refit. Steps 2-3 are iterated.
  4. Rotation fusion (fit_mount): the MediaPipe rotation is combined with an
     independent rotation, the ZED VIO hand-eye (vio.py) or else the fleet
     prior, weighted by their covariances; the translation is then refitted.
Too little evidence for 4: the rotation is taken from VIO or from the fleet prior,
whichever fits the few points better, and only the translation is fitted; with
fewer points still the fleet prior mount (or the template) is used.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .session import SIDES, rig_local

DEFAULT_PRIOR = Path(__file__).resolve().parent / 'fleet_prior.json'

TEMPLATE_R = np.array([
    [0.998887680216014, -0.04693668629799374, -0.004511074376952759],
    [-0.03947189303945952, -0.7800013494092103, -0.624531716232005],
    [0.02579480514662097, 0.6240150978936131, -0.7809864183379069],
])
TEMPLATE_T = np.array([18.883115428553992, -42.61734060378433, -9.1789565010356])
MIN_JOINTS = 6          # joints needed to score one detection against one CMA hand
MIN_POINTS = 300        # inlier correspondences needed to accept a calibration
MIN_FRAMES = 15         # distinct frames needed to accept a calibration
MIN_POINTS_FIXED_R = 60     # translation only (rotation from VIO / fleet prior): far less evidence needed
# Sigma scales of the two rotation covariances, fitted by maximum likelihood on the
# MediaPipe-vs-VIO differences of 15 recordings (tools/calibrate_fusion.py, mean d2 = 3.0).
FUSION_MP_SCALE = 36.2
FUSION_VIO_SCALE = 2.1


def load_detections(path, stamps):
    """MediaPipe cache aligned to the session stamps (missing frames are empty slots)."""
    cache = np.load(path)
    count = len(stamps)
    uv = np.full((count, 2, 21, 2), np.nan)
    label = np.full((count, 2), -1, np.int8)
    score = np.zeros((count, 2))
    position = {int(stamp): i for i, stamp in enumerate(cache['stamps'])}
    rows = np.array([position.get(int(stamp), -1) for stamp in stamps])
    found = rows >= 0
    uv[found] = cache['uv'][rows[found]]
    label[found] = cache['label'][rows[found]]
    score[found] = cache['score'][rows[found]]
    width, height = (int(v) for v in cache['image_size'])
    # Keypoints MediaPipe extrapolated outside the image are guesses, not observations.
    inside = (uv[..., 0] >= 0) & (uv[..., 0] < width) & (uv[..., 1] >= 0) & (uv[..., 1] < height)
    uv[~inside] = np.nan
    return {'uv': uv, 'label': label, 'score': score, 'frames_matched': int(found.sum()),
            'image_size': (width, height)}


def local_tracks(session):
    """Rig-frame CMA joints of both hands, (N, 2, 21, 3) mm; NaN where the joint is invalid."""
    count = len(session['stamps'])
    local = np.full((count, 2, 21, 3), np.nan)
    for h, side in enumerate(SIDES):
        hand = session['hands'][side]
        if hand['present']:
            local[:, h] = rig_local(hand['points_mm'], session['origin_mm'], session['rig_R'])
    return local


def project(local, rotation, translation, K):
    """Pixels and camera depth of rig-frame points (any leading shape); NaN behind the camera."""
    camera = local @ rotation.T + translation
    z = camera[..., 2]
    with np.errstate(invalid='ignore', divide='ignore'):
        uv = np.stack([K[0, 0] * camera[..., 0] / z + K[0, 2], K[1, 1] * camera[..., 1] / z + K[1, 2]], -1)
    uv[~(z > 1.0)] = np.nan
    return uv, z


def median_ignoring_nan(values, axis=-1):
    """nanmedian without the all-NaN warning (all-NaN slices give NaN)."""
    values = np.asarray(values, np.float64)
    count = np.isfinite(values).sum(axis)
    ordered = np.sort(np.where(np.isfinite(values), values, np.inf), axis=axis)
    ordered = np.moveaxis(ordered, axis, -1)
    low = np.take_along_axis(ordered, np.maximum((count - 1) // 2, 0)[..., None], -1)[..., 0]
    high = np.take_along_axis(ordered, np.maximum(count // 2, 0)[..., None], -1)[..., 0]
    median = (low + high) / 2
    median[count == 0] = np.nan
    return median


def pose_vector(rotation, translation):
    return np.concatenate([Rotation.from_matrix(rotation).as_rotvec(), np.asarray(translation, np.float64)])


def pose_from_vector(x):
    return Rotation.from_rotvec(x[:3]).as_matrix(), np.asarray(x[3:], np.float64)


def _correspondences(local, uv, pairs):
    """Stack rig-frame points and detections for (frame, slot, side) pairs, finite in both."""
    if not len(pairs):
        empty = np.zeros(0, np.int64)
        return {'obj': np.zeros((0, 3)), 'img': np.zeros((0, 2)), 'frame': empty, 'side': empty,
                'joint': empty, 'slot': empty}
    frame, slot, side = (np.asarray(v, np.int64) for v in zip(*pairs))
    obj = local[frame, side]                    # (P, 21, 3)
    img = uv[frame, slot]                       # (P, 21, 2)
    ok = np.isfinite(obj).all(-1) & np.isfinite(img).all(-1)
    p, j = np.nonzero(ok)
    return {'obj': obj[p, j], 'img': img[p, j], 'frame': frame[p], 'side': side[p], 'joint': j,
            'slot': slot[p]}


def label_pairs(detections, frames, hypothesis):
    """(frame, slot, side) from MediaPipe labels: hypothesis 0 keeps them, 1 swaps them."""
    label = detections['label']
    pairs = []
    for f in frames:
        for slot in (0, 1):
            if label[f, slot] >= 0:
                pairs.append((f, slot, int(label[f, slot]) ^ int(hypothesis)))
    return pairs


def pnp_ransac(obj, img, K, threshold_px=12.0, iterations=4000, max_points=30000, seed=0):
    """Pooled PnP over all correspondences; rvec/tvec map rig-frame mm to camera mm."""
    if len(obj) < 12:
        return None
    rng = np.random.default_rng(seed)
    index = np.arange(len(obj)) if len(obj) <= max_points else np.sort(rng.choice(len(obj), max_points, False))
    cv2.setRNGSeed(int(seed))
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj[index].astype(np.float64), img[index].astype(np.float64), K, None,
        iterationsCount=int(iterations), reprojectionError=float(threshold_px), confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 12:
        return None
    rotation = cv2.Rodrigues(rvec)[0]
    translation = tvec.ravel()
    error = np.linalg.norm(project(obj, rotation, translation, K)[0] - img, axis=-1)
    return {'R': rotation, 't': translation, 'inliers': int((error < threshold_px).sum()),
            'inlier_fraction': float(np.mean(error < threshold_px))}


def ransac_hypotheses(local, detections, K, frames=None, seed=0):
    """Pooled RANSAC PnP for both handedness mappings (label hypothesis 0 and 1).

    It does not depend on a fixed rotation, so fit_mount / fixed_rotation_mount compute it once and
    pass it to every calibrate() call on the same frames. The two RANSACs run in two threads
    (OpenCV releases the GIL; the result does not depend on it).
    """
    frames = np.arange(len(local)) if frames is None else np.asarray(frames, np.int64)

    def one(hypothesis):
        data = _correspondences(local, detections['uv'], label_pairs(detections, frames, hypothesis))
        result = pnp_ransac(data['obj'], data['img'], K, seed=seed)
        return {'hypothesis': hypothesis, 'points': int(len(data['obj'])),
                'inliers': 0 if result is None else result['inliers'], 'result': result, 'data': data}

    with ThreadPoolExecutor(2) as pool:
        return list(pool.map(one, (0, 1)))


def associate(local, detections, frames, rotation, translation, K, gate_px):
    """Assign every detection slot to the CMA hand it matches best, per frame.

    Returns pairs (frame, slot, side), the per-pair median error and the number
    of frames where at least one detection existed.
    """
    projected = project(local[frames], rotation, translation, K)[0]          # (F, side, 21, 2)
    detected = detections['uv'][frames]                                      # (F, slot, 21, 2)
    diff = np.linalg.norm(detected[:, :, None] - projected[:, None], axis=-1)   # (F, slot, side, 21)
    cost = median_ignoring_nan(diff)
    cost[np.isfinite(diff).sum(-1) < MIN_JOINTS] = np.inf
    pairs, costs = [], []
    for i, f in enumerate(frames):
        c = cost[i]
        straight, crossed = c[0, 0] + c[1, 1], c[0, 1] + c[1, 0]
        if np.isfinite(straight) or np.isfinite(crossed):
            options = [(0, 0), (1, 1)] if straight <= crossed else [(0, 1), (1, 0)]
        elif np.isfinite(c).any():
            options = [np.unravel_index(int(np.argmin(c)), c.shape)]
        else:
            continue
        for slot, side in options:
            if c[slot, side] <= gate_px:
                pairs.append((int(f), int(slot), int(side)))
                costs.append(float(c[slot, side]))
    has_detection = int((detections['label'][frames] >= 0).any(1).sum())
    return pairs, np.asarray(costs), has_detection


def refine(obj, img, K, rotation, translation, weights=None, f_scale=5.0, drop_px=30.0, max_nfev=80,
           fixed_rotation=False):
    """Robust reprojection refinement, drop gross outliers, refit. Returns R, t, keep mask, errors.

    With fixed_rotation only the translation is estimated.
    """
    weights = np.ones(len(obj)) if weights is None else np.asarray(weights, np.float64)
    rotation = np.asarray(rotation, np.float64)

    def unpack(x):
        return (rotation, np.asarray(x, np.float64)) if fixed_rotation else pose_from_vector(x)

    def residual(x, keep):
        r, t = unpack(x)
        uv, _ = project(obj[keep], r, t, K)
        out = (uv - img[keep]) * np.sqrt(weights[keep])[:, None]
        return np.nan_to_num(out, nan=1e3).ravel()

    x = np.asarray(translation, np.float64).copy() if fixed_rotation else pose_vector(rotation, translation)
    scale = np.full(3, 5.0) if fixed_rotation else np.array([0.01, 0.01, 0.01, 5.0, 5.0, 5.0])
    keep = np.ones(len(obj), bool)
    for _ in range(2):
        x = least_squares(residual, x, args=(keep,), loss='soft_l1', f_scale=f_scale, max_nfev=max_nfev,
                          x_scale=scale).x
        r, t = unpack(x)
        error = np.linalg.norm(project(obj, r, t, K)[0] - img, axis=-1)
        keep = np.nan_to_num(error, nan=np.inf) < drop_px
        if keep.sum() < 12:
            break
    return r, t, keep, error


def calibrate(local, detections, K, frames=None, *, fixed_R=None, rounds=3, gate_px=40.0, drop_px=30.0,
              f_scale=5.0, joint_weights=None, seed=0, hypotheses=None):
    """Estimate the mount from the given frames only (all frames by default).

    With fixed_R (e.g. the VIO hand-eye rotation) only the translation is fitted.
    hypotheses: ransac_hypotheses() of the same frames and seed, when already computed.
    Returns None when there is not enough evidence, else a dict with R, t,
    the correspondences used and fit statistics.
    """
    count = len(local)
    frames = np.arange(count) if frames is None else np.asarray(frames, np.int64)
    if hypotheses is None:
        hypotheses = ransac_hypotheses(local, detections, K, frames, seed)
    best = max(hypotheses, key=lambda item: item['inliers'])
    if best['result'] is None:
        return None
    rotation, translation = best['result']['R'], best['result']['t']
    if fixed_R is not None:
        # Move the RANSAC pose onto the fixed rotation: keep the mean hand point where it was,
        # then refit the translation on the RANSAC inliers.
        data = best['data']
        error = np.linalg.norm(project(data['obj'], rotation, translation, K)[0] - data['img'], axis=-1)
        inliers = np.nan_to_num(error, nan=np.inf) < 12.0
        center = data['obj'][inliers].mean(0)
        translation = translation + (rotation - fixed_R) @ center
        rotation = np.asarray(fixed_R, np.float64)
        _, translation, _, _ = refine(data['obj'][inliers], data['img'][inliers], K, rotation, translation,
                                      f_scale=f_scale, drop_px=2.5 * gate_px, fixed_rotation=True)
    history = []
    for round_index in range(rounds):
        # The first association uses a loose gate: the RANSAC pose is still coarse.
        gate = 2.5 * gate_px if round_index == 0 else gate_px
        pairs, costs, with_detection = associate(local, detections, frames, rotation, translation, K, gate)
        data = _correspondences(local, detections['uv'], pairs)
        if len(data['obj']) < 12:
            return None
        weights = None if joint_weights is None else np.asarray(joint_weights)[data['joint']]
        rotation, translation, keep, error = refine(data['obj'], data['img'], K, rotation, translation,
                                                    weights, f_scale, drop_px, fixed_rotation=fixed_R is not None)
        history.append({'round': round_index, 'gate_px': gate, 'pairs': len(pairs),
                        'points': int(len(keep)), 'inliers': int(keep.sum()),
                        'median_px': float(np.median(error[keep])) if keep.any() else float('nan')})
        # Adaptive gate for the next rounds: never tighter than 40 px, never looser than 3x the typical error.
        typical = float(np.median(costs)) if len(costs) else gate_px
        gate_px = max(40.0, min(3.0 * typical, 80.0))
    label = detections['label']
    agree = [int(label[f, s]) ^ best['hypothesis'] == h for f, s, h in pairs]
    frames_used = np.unique(data['frame'][keep])
    return {'R': rotation, 't': translation, 'data': data, 'keep': keep, 'error_px': error,
            'rotation_source': 'fixed' if fixed_R is not None else 'pnp', 'hypothesis': best['hypothesis'],
            'hypotheses': [{k: v for k, v in item.items() if k not in ('result', 'data')} for item in hypotheses],
            'history': history, 'frames_with_detection': with_detection,
            'frames_used': int(len(frames_used)), 'pairs': len(pairs),
            'label_agreement': float(np.mean(agree)) if agree else float('nan')}


def rotation_covariance(fit, K):
    """Shape of the rotation covariance of a 6-DoF fit (camera-frame rotation vector, deg^2).

    Gauss-Newton on the inliers with the translation profiled out: the weak
    direction is the rotation about the hand workspace that a translation can
    almost compensate. The per-point noise is a robust residual scale; points
    are strongly correlated (per-joint keypoint biases), so the absolute scale
    is corrected by FUSION_MP_SCALE, calibrated on the fleet.
    """
    data, keep = fit['data'], fit['keep']
    obj, img = data['obj'][keep], data['img'][keep]
    R, t = fit['R'], fit['t']
    base = (project(obj, R, t, K)[0] - img).ravel()
    jacobian = np.zeros((len(base), 6))
    for k in range(3):
        step = np.zeros(3)
        step[k] = 1e-5
        jacobian[:, k] = ((project(obj, Rotation.from_rotvec(step).as_matrix() @ R, t, K)[0] - img).ravel() - base) / 1e-5
        step = np.zeros(3)
        step[k] = 1e-2
        jacobian[:, 3 + k] = ((project(obj, R, t + step, K)[0] - img).ravel() - base) / 1e-2
    F = jacobian.T @ jacobian
    profiled = F[:3, :3] - F[:3, 3:] @ np.linalg.solve(F[3:, 3:], F[3:, :3])
    sigma = 1.4826 * float(np.median(np.abs(base)))
    return np.degrees(np.degrees(sigma ** 2 * np.linalg.inv(profiled)))


def fuse_rotations(R_image, cov_image, R_vio, cov_vio):
    """Information-weighted combination of two mount rotations (covariances in deg^2).

    Returns the fused rotation, its covariance, the squared Mahalanobis
    distance between the inputs (chi^2 with 3 dof if both covariances are right)
    and the per-axis split of the correction.
    """
    cov_image, cov_vio = np.asarray(cov_image), np.asarray(cov_vio)
    delta = np.degrees(Rotation.from_matrix(np.asarray(R_vio) @ np.asarray(R_image).T).as_rotvec())
    info_image, info_vio = np.linalg.inv(cov_image), np.linalg.inv(cov_vio)
    cov = np.linalg.inv(info_image + info_vio)
    step = cov @ info_vio @ delta
    fused = Rotation.from_rotvec(np.radians(step)).as_matrix() @ np.asarray(R_image)
    d2 = float(delta @ np.linalg.solve(cov_image + cov_vio, delta))
    return {'R': fused, 'covariance_deg2': cov, 'mahalanobis2': d2, 'delta_deg': delta, 'step_deg': step}


def joint_bias(fit, K):
    """Per-joint median residual (detection - projection) in px, from the inliers of a fit."""
    data, keep = fit['data'], fit['keep']
    uv, _ = project(data['obj'], fit['R'], fit['t'], K)
    residual = data['img'] - uv
    bias = np.full((2, 21, 2), np.nan)
    counts = np.zeros((2, 21), np.int64)
    for side in (0, 1):
        for joint in range(21):
            chosen = keep & (data['side'] == side) & (data['joint'] == joint)
            counts[side, joint] = int(chosen.sum())
            if chosen.sum() >= 10:
                bias[side, joint] = np.median(residual[chosen], axis=0)
    return bias, counts


def summarize_fit(fit):
    error = fit['error_px'][fit['keep']]
    return {'points': int(len(fit['keep'])), 'inliers': int(fit['keep'].sum()),
            'inlier_fraction': float(fit['keep'].mean()) if len(fit['keep']) else 0.0,
            'frames_with_detection': fit['frames_with_detection'], 'frames_used': fit['frames_used'],
            'pairs': fit['pairs'], 'hypothesis': 'as_labelled' if fit['hypothesis'] == 0 else 'swapped',
            'hypotheses': fit['hypotheses'], 'label_agreement': fit['label_agreement'],
            'train_median_px': float(np.median(error)) if len(error) else float('nan'),
            'train_p90_px': float(np.percentile(error, 90)) if len(error) else float('nan'),
            'history': fit['history']}


def vio_usable(handeye):
    return handeye is not None and handeye.get('status') in ('ok', 'weak')


def load_prior(path=DEFAULT_PRIOR):
    """Fleet mount prior written by tools/build_fleet_prior.py; None when the file is missing."""
    path = Path(path)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    return {'R': np.asarray(data['R'], np.float64), 't': np.asarray(data['t_mm'], np.float64),
            'covariance_deg2': np.asarray(data['covariance_deg2'], np.float64),
            'source': 'fleet_prior', 'path': str(path), 'recordings': len(data.get('recordings', []))}


def rotation_reference(handeye=None, prior=None):
    """Independent rotation to fuse with MediaPipe: the VIO hand-eye, else the fleet prior, else None."""
    if vio_usable(handeye):
        return 'vio', np.asarray(handeye['mount_R']), FUSION_VIO_SCALE ** 2 * np.asarray(handeye['covariance_deg2'])
    if prior is not None:
        return 'fleet_prior', prior['R'], prior['covariance_deg2']
    return None


def fit_mount(local, detections, K, frames=None, handeye=None, prior=None, hypotheses=None, **options):
    """Final mount estimator: MediaPipe 6-DoF, rotation fused with an independent reference, translation refitted.

    The reference is the VIO hand-eye rotation when available, else the fleet
    prior. MediaPipe and VIO have complementary weak directions (MediaPipe:
    rotation about the hand workspace, i.e. mostly camera pitch; VIO: rotation
    about the dominant head-rotation axis), so the information-weighted fusion
    is better than either. The fleet prior only bounds the weak MediaPipe
    direction of recordings without VIO. Without any reference the MediaPipe
    fit is returned unchanged. Returns None when MediaPipe gives no usable
    correspondences.
    """
    if hypotheses is None:
        hypotheses = ransac_hypotheses(local, detections, K, frames, options.get('seed', 0))
    image = calibrate(local, detections, K, frames, hypotheses=hypotheses, **options)
    if image is None:
        return None
    cov_image = FUSION_MP_SCALE ** 2 * rotation_covariance(image, K)
    fusion = {'image_R': image['R'], 'image_t': image['t'], 'image_covariance_deg2': cov_image, 'reference': None}
    reference = rotation_reference(handeye, prior)
    if reference is None:
        image['fusion'] = fusion
        return image
    source, reference_R, reference_cov = reference
    fused = fuse_rotations(image['R'], cov_image, reference_R, reference_cov)
    final = calibrate(local, detections, K, frames, fixed_R=fused['R'], hypotheses=hypotheses, **options)
    if final is None:
        image['fusion'] = {**fusion, 'reason': 'translation refit with the fused rotation failed'}
        return image
    final['rotation_source'] = f'fused_{source}'
    final['fusion'] = {**fusion, 'reference': source, 'reference_R': reference_R,
                       'reference_covariance_deg2': reference_cov, 'fused_covariance_deg2': fused['covariance_deg2'],
                       'mahalanobis2': fused['mahalanobis2'], 'delta_deg': fused['delta_deg'],
                       'step_deg': fused['step_deg']}
    return final


def reprojection_median(local, detections, K, R, t, frames=None, gate_px=80.0):
    """Median reprojection error (px) of the detections in `frames` that associate with the mount."""
    frames = np.arange(len(local)) if frames is None else np.asarray(frames, np.int64)
    pairs, _, _ = associate(local, detections, frames, R, t, K, gate_px)
    data = _correspondences(local, detections['uv'], pairs)
    if not len(data['obj']):
        return float('inf')
    return float(np.median(np.linalg.norm(project(data['obj'], R, t, K)[0] - data['img'], axis=-1)))


def fixed_rotation_mount(local, detections, K, frames=None, handeye=None, prior=None, hypotheses=None, **options):
    """Few MediaPipe points: rotation from VIO or from the fleet prior, translation fitted; the better fit wins.

    Both candidates have the same 3 free parameters, so their errors on the same frames are
    comparable. The fleet rotation fits the images better than the VIO rotation on most recordings,
    the VIO rotation wins where the camera sits differently from the fleet (tools/degradation_check.py).
    Returns None when no candidate can be fitted.
    """
    candidates = []
    if vio_usable(handeye):
        candidates.append(('vio', np.asarray(handeye['mount_R'], np.float64)))
    if prior is not None:
        candidates.append(('fleet_prior', np.asarray(prior['R'], np.float64)))
    best, scores = None, {}
    if candidates and hypotheses is None:
        hypotheses = ransac_hypotheses(local, detections, K, frames, options.get('seed', 0))
    for source, R in candidates:
        fit = calibrate(local, detections, K, frames, fixed_R=R, hypotheses=hypotheses, **options)
        scores[source] = None if fit is None else reprojection_median(local, detections, K, fit['R'], fit['t'], frames)
        if fit is not None and (best is None or scores[source] < scores[best[0]]):
            best = (source, fit)
    if best is None:
        return None
    source, fit = best
    fit['rotation_source'] = source
    fit['fusion'] = {'reference': source, 'reference_only': True, 'candidates_median_px': scores}
    return fit


def fallback_mount(handeye=None, prior=None):
    """Mount when nothing can be fitted on the recording: the fleet prior mount, else the original template.

    Rotation and translation are kept together: the VIO rotation with the fleet translation fits the
    images worse than the fleet mount on most recordings (tools/degradation_check.py). The angle between
    the VIO rotation and the fallback rotation is reported: a large one hints at a camera mounted
    differently, where the fallback mount is far off.
    """
    R, t, source = (prior['R'], prior['t'], 'fleet_prior') if prior is not None else (TEMPLATE_R, TEMPLATE_T, 'template')
    info = {'status': 'FALLBACK', 'rotation_source': source, 'fallback_source': source}
    if vio_usable(handeye):
        info['vio_vs_fallback_deg'] = float(np.degrees(
            Rotation.from_matrix(np.asarray(handeye['mount_R']) @ np.asarray(R).T).magnitude()))
    return R, t, info


def estimate_mount(session, detections, handeye=None, prior=None, **options):
    """Mount for one recording, with graceful degradation.

    CALIBRATED        enough MediaPipe evidence (rotation fused with VIO, else with the fleet prior)
    VIO_ROTATION      little MediaPipe evidence: rotation from VIO or from the fleet prior, whichever
    PRIOR_ROTATION    fits the few points better, translation from the few points (fixed_rotation_mount)
    FALLBACK          fleet prior mount (or the original template) - the metrics flag it
    Returns (mount_R, mount_t, report, fit_or_None).
    """
    local = local_tracks(session)
    K = session['K']
    hypotheses = ransac_hypotheses(local, detections, K, seed=options.get('seed', 0))
    fit = fit_mount(local, detections, K, handeye=handeye, prior=prior, hypotheses=hypotheses, **options)
    report = {'method': 'mediapipe_pnp_fusion_v1'}
    if fit is not None:
        report.update(summarize_fit(fit))
        report['rotation_source'] = fit['rotation_source']
    if fit is not None and fit['keep'].sum() >= MIN_POINTS and fit['frames_used'] >= MIN_FRAMES:
        report['status'] = 'CALIBRATED'
        return fit['R'], fit['t'], report, fit
    fixed = fixed_rotation_mount(local, detections, K, handeye=handeye, prior=prior, hypotheses=hypotheses,
                                 **options)
    if fixed is not None and fixed['keep'].sum() >= MIN_POINTS_FIXED_R and fixed['frames_used'] >= 5:
        report.update(summarize_fit(fixed))
        report.update({'status': 'VIO_ROTATION' if fixed['rotation_source'] == 'vio' else 'PRIOR_ROTATION',
                       'rotation_source': fixed['rotation_source'],
                       'rotation_candidates_median_px': fixed['fusion']['candidates_median_px'],
                       'reason': 'too little MediaPipe evidence for a 6-DoF fit'})
        return fixed['R'], fixed['t'], report, fixed
    R, t, info = fallback_mount(handeye, prior)
    report.update(info)
    report['reason'] = 'no usable MediaPipe correspondences' if fit is None else (
        f"only {int(fit['keep'].sum())} inlier points in {fit['frames_used']} frames")
    return R, t, report, fit
