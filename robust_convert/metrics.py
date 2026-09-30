# -*- coding: utf-8 -*-
"""Accuracy metrics for the camera mount (extrinsic).

E1  held-out reprojection: temporal K-fold, the mount is re-estimated without
    the test block (and a buffer around it), then MediaPipe keypoints of the
    test block are compared with the projected CMA joints. px and mm at the hand.
E2  depth agreement: the fitted MANO surface rendered with the mount vs the
    ZED depth map (median signed / absolute residual in mm). Checks the depth
    direction, to which reprojection is least sensitive. Computed in convert.py.
E3  stability: spread of the K fold mounts (deg, mm, induced px shift).
E4  rotation consistency: Mahalanobis distance (chi^2, 3 dof) between the
    MediaPipe-only rotation and the reference rotation it is fused with, the
    ZED VIO hand-eye rotation (vio.py, uses no image keypoint) or, without
    usable VIO, the fleet prior.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .extrinsic import associate, calibrate, _correspondences, project, refine

FOLDS = 5
BUFFER_FRAMES = 15
TEST_GATE_PX = 80.0


def rotation_angle_deg(a, b):
    return float(np.degrees(Rotation.from_matrix(np.asarray(a).T @ np.asarray(b)).magnitude()))


def _error_stats(error_px, z_mm, K):
    if not len(error_px):
        return {'points': 0}
    mm = error_px * z_mm / K[0, 0]
    return {'points': int(len(error_px)), 'median_px': float(np.median(error_px)),
            'p90_px': float(np.percentile(error_px, 90)), 'mean_px': float(np.mean(error_px)),
            'median_mm': float(np.median(mm)), 'p90_mm': float(np.percentile(mm, 90)),
            'median_depth_mm': float(np.median(z_mm))}


def held_out(local, detections, K, full_fit, folds=FOLDS, buffer=BUFFER_FRAMES, estimator=None):
    """E1 and E3 by temporal K-fold cross-validation.

    estimator(frames) -> fit re-runs the full mount estimator on the training
    frames (default: MediaPipe-only calibrate).
    """
    if estimator is None:
        def estimator(frames):
            return calibrate(local, detections, K, frames)
    count = len(local)
    blocks = np.array_split(np.arange(count), folds)
    errors, depths, joints, fold_reports, fold_poses = [], [], [], [], []
    for k, block in enumerate(blocks):
        if not len(block):
            continue
        lo, hi = block[0] - buffer, block[-1] + buffer
        train = np.flatnonzero((np.arange(count) < lo) | (np.arange(count) > hi))
        fit = estimator(train)
        if fit is None:
            fold_reports.append({'fold': k, 'status': 'no_fit', 'test_frames': int(len(block))})
            continue
        pairs, _, with_detection = associate(local, detections, block, fit['R'], fit['t'], K, TEST_GATE_PX)
        data = _correspondences(local, detections['uv'], pairs)
        uv, z = project(data['obj'], fit['R'], fit['t'], K)
        error = np.linalg.norm(uv - data['img'], axis=-1)
        errors.append(error)
        depths.append(z)
        joints.append(data['joint'])
        fold_poses.append((fit['R'], fit['t']))
        fold_reports.append({'fold': k, 'status': 'ok', 'test_frames': int(len(block)),
                             'test_frames_with_detection': with_detection,
                             'test_pairs': len(pairs), 'train_frames_used': fit['frames_used'],
                             **_error_stats(error, z, K),
                             'rotation_vs_full_deg': rotation_angle_deg(fit['R'], full_fit['R']),
                             'translation_vs_full_mm': float(np.linalg.norm(fit['t'] - full_fit['t']))})
    error = np.concatenate(errors) if errors else np.zeros(0)
    depth = np.concatenate(depths) if depths else np.zeros(0)
    joint = np.concatenate(joints) if joints else np.zeros(0, np.int64)
    e1 = _error_stats(error, depth, K)
    e1['per_joint_median_px'] = [float(np.median(error[joint == j])) if (joint == j).any() else None
                                 for j in range(21)]
    e1['folds_ok'] = int(len(fold_poses))
    e1['folds'] = fold_reports
    e3 = stability(fold_poses, full_fit, K)
    return e1, e3


def stability(fold_poses, full_fit, K):
    """E3: how far the fold mounts are from the full-data mount, measured on the real hand points."""
    if not fold_poses:
        return {'folds_ok': 0}
    data, keep = full_fit['data'], full_fit['keep']
    obj = data['obj'][keep]
    reference = project(obj, full_fit['R'], full_fit['t'], K)[0]
    rotation = [rotation_angle_deg(R, full_fit['R']) for R, _ in fold_poses]
    translation = [float(np.linalg.norm(t - full_fit['t'])) for _, t in fold_poses]
    shift = [float(np.median(np.linalg.norm(project(obj, R, t, K)[0] - reference, axis=-1))) for R, t in fold_poses]
    return {'folds_ok': len(fold_poses), 'rotation_max_deg': max(rotation), 'rotation_rms_deg': float(np.sqrt(np.mean(np.square(rotation)))),
            'translation_max_mm': max(translation), 'translation_rms_mm': float(np.sqrt(np.mean(np.square(translation)))),
            'pixel_shift_max_px': max(shift), 'pixel_shift_median_px': float(np.median(shift))}


def vio_agreement(handeye, mount_R):
    """E4: angle between a mount rotation and the VIO hand-eye rotation (MediaPipe-independent)."""
    if handeye.get('status') not in ('ok', 'weak'):
        return {'status': handeye.get('status', 'no_vio')}
    return {'status': handeye['status'], 'rotation_error_deg': rotation_angle_deg(mount_R, handeye['mount_R']),
            'vio_uncertainty_deg': handeye['jackknife_se_deg'],
            'pair_residual_median_deg': handeye['pair_residual_median_deg'],
            'axis_spread_min': handeye['axis_spread'][-1]}


def rotation_consistency(fit, handeye):
    """E4: do the MediaPipe-only rotation and the independent reference rotation agree within their uncertainty?

    The reference is the VIO hand-eye rotation (a real independent sensor
    check) or, without VIO, the fleet prior (only "is this mount like the
    others"). mahalanobis2 is chi^2 with 3 dof when both covariances are right
    (95 % quantile 7.81, 99 % quantile 11.34).
    """
    fusion = (fit or {}).get('fusion') or {}
    if 'mahalanobis2' not in fusion:
        return {'status': 'no_reference', 'vio_status': (handeye or {}).get('status', 'no_vio'),
                'reference': fusion.get('reference')}
    from scipy.stats import chi2
    image_sigma = np.sqrt(np.linalg.eigvalsh(fusion['image_covariance_deg2']))
    reference_sigma = np.sqrt(np.linalg.eigvalsh(fusion['reference_covariance_deg2']))
    fused_sigma = np.sqrt(np.linalg.eigvalsh(fusion['fused_covariance_deg2']))
    out = {'status': 'ok', 'reference': fusion['reference'], 'vio_status': (handeye or {}).get('status', 'no_vio'),
           'angle_deg': float(np.linalg.norm(fusion['delta_deg'])),
           'delta_camera_xyz_deg': np.asarray(fusion['delta_deg']).tolist(),
           'mahalanobis2': float(fusion['mahalanobis2']),
           'p_value': float(chi2.sf(fusion['mahalanobis2'], 3)),
           'image_sigma_deg': image_sigma.tolist(), 'reference_sigma_deg': reference_sigma.tolist(),
           'fused_sigma_deg': fused_sigma.tolist(),
           'final_vs_image_deg': rotation_angle_deg(fit['R'], fusion['image_R']),
           'final_vs_reference_deg': rotation_angle_deg(fit['R'], fusion['reference_R'])}
    if fusion['reference'] == 'vio':
        out['pair_residual_median_deg'] = handeye['pair_residual_median_deg']
    return out


def time_offset_scan(session, detections, K, R, t, offsets_ms, local_fn, resample_fn, gate_px=40.0,
                     fixed_rotation=False):
    """Median reprojection error after refitting the mount, for CMA time shifted by each offset.

    Restores the session at offset 0 before returning.
    """
    rows = []
    frames = np.arange(len(session['stamps']))
    for offset in offsets_ms:
        resample_fn(session, int(round(offset * 1e6)))
        local = local_fn(session)
        pairs, _, _ = associate(local, detections, frames, R, t, K, gate_px)
        data = _correspondences(local, detections['uv'], pairs)
        if len(data['obj']) < 50:
            rows.append({'offset_ms': float(offset), 'median_px': None})
            continue
        _, _, keep, error = refine(data['obj'], data['img'], K, R, t, fixed_rotation=fixed_rotation)
        rows.append({'offset_ms': float(offset), 'median_px': float(np.median(error[keep])),
                     'mean_inlier_px': float(np.mean(error[keep])), 'inliers': int(keep.sum())})
    resample_fn(session, 0)
    valid = [row for row in rows if row['median_px'] is not None]
    best = min(valid, key=lambda row: row['mean_inlier_px']) if valid else None
    zero = next((row for row in valid if row['offset_ms'] == 0), None)
    return {'curve': rows, 'best_offset_ms': None if best is None else best['offset_ms'],
            'best_mean_inlier_px': None if best is None else best['mean_inlier_px'],
            'zero_mean_inlier_px': None if zero is None else zero['mean_inlier_px']}


def fixed_mount_reprojection(local, detections, K, R, t, gate_px=TEST_GATE_PX):
    """E1 for a mount that was not estimated from this recording (fleet prior / template).

    Every frame is a test frame: nothing of this recording was used to choose the mount.
    """
    frames = np.arange(len(local))
    pairs, _, with_detection = associate(local, detections, frames, R, t, K, gate_px)
    data = _correspondences(local, detections['uv'], pairs)
    uv, z = project(data['obj'], R, t, K)
    error = np.linalg.norm(uv - data['img'], axis=-1)
    stats = _error_stats(error, z, K)
    stats.update(test_frames_with_detection=with_detection,
                 protocol='all frames (mount independent of this recording)')
    stats['per_joint_median_px'] = [float(np.median(error[data['joint'] == j])) if (data['joint'] == j).any()
                                    else None for j in range(21)]
    return stats


# ---- E2: MANO surface vs ZED depth -------------------------------------------------------------

DEPTH_FRAMES = 120       # frames sampled per recording
DEPTH_ERODE_PX = 4       # mask erosion: the ZED depth bleeds over silhouette edges
DEPTH_MIN_PIXELS = 50    # per hand and frame
DEPTH_INLIER_MM = 20.0


def depth_frames(valid, count=DEPTH_FRAMES):
    """Evenly spaced frames among those where at least one hand has a valid MANO fit."""
    candidates = np.flatnonzero(valid)
    if len(candidates) <= count:
        return candidates
    return candidates[np.unique(np.linspace(0, len(candidates) - 1, count).round().astype(int))]


def depth_agreement(session, mount_R, mount_t, meshes, frames, depth_maps):
    """E2: ZED depth minus rendered MANO depth inside each (eroded, unoccluded) hand silhouette.

    meshes: {side: {'vertices_mm': (N, 778, 3) world mm, 'faces', 'valid': (N,) bool}}.
    Positive bias = the real surface is farther than the MANO surface placed with this mount.
    Per frame the signed median is used (robust to objects in front of the hand);
    the recording bias is the median of the per-frame medians.
    """
    import cv2
    from .render import render_depth
    from .session import SIDES

    K = session['K']
    width, height = session['width'], session['height']
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DEPTH_ERODE_PX + 1, 2 * DEPTH_ERODE_PX + 1))
    rows = {side: [] for side in meshes}
    pooled = {side: [] for side in meshes}
    missing = 0
    for index in frames:
        zed = depth_maps.get(int(session['stamps'][index]))
        if zed is None:
            missing += 1
            continue
        K_depth = K.copy()
        K_depth[0] *= zed.shape[1] / width
        K_depth[1] *= zed.shape[0] / height
        origin, rig = session['origin_mm'][index], session['rig_R'][index]
        items = []
        for side in SIDES:
            mesh = meshes.get(side)
            if mesh is None or not mesh['valid'][index]:
                items.append(None)
                continue
            camera = ((mesh['vertices_mm'][index] - origin) @ rig) @ np.asarray(mount_R).T + mount_t
            items.append((camera, mesh['faces']))
        rendered, label = render_depth(items, K_depth, (zed.shape[1], zed.shape[0]))
        for tag, side in enumerate(SIDES):
            if items[tag] is None:
                continue
            inner = cv2.erode((label == tag).astype(np.uint8), kernel).astype(bool)
            use = inner & (zed > 0)
            row = {'frame': int(index), 'pixels': int(use.sum()), 'mask_pixels': int(inner.sum())}
            if use.sum() >= DEPTH_MIN_PIXELS:
                residual = zed[use].astype(np.float64) - rendered[use]
                pooled[side].append(residual)
                row.update(median_mm=float(np.median(residual)), abs_median_mm=float(np.median(np.abs(residual))),
                           mano_depth_mm=float(np.median(rendered[use])))
            rows[side].append(row)
    out = {'frames_requested': int(len(frames)), 'frames_without_depth': missing, 'erode_px': DEPTH_ERODE_PX,
           'inlier_mm': DEPTH_INLIER_MM, 'hands': {}}
    medians_all = []
    for side, entries in rows.items():
        used = [row for row in entries if 'median_mm' in row]
        if not used:
            out['hands'][side] = {'frames': 0, 'frames_attempted': len(entries)}
            continue
        medians = np.array([row['median_mm'] for row in used])
        residual = np.concatenate(pooled[side])
        coverage = sum(row['pixels'] for row in entries) / max(sum(row['mask_pixels'] for row in entries), 1)
        medians_all.extend(medians.tolist())
        out['hands'][side] = {'frames': len(used), 'frames_attempted': len(entries),
                              'pixels': int(len(residual)), 'depth_coverage': float(coverage),
                              'bias_mm': float(np.median(medians)),
                              'bias_iqr_mm': float(np.subtract(*np.percentile(medians, [75, 25]))),
                              'abs_median_mm': float(np.median(np.abs(residual))),
                              'inlier_fraction': float(np.mean(np.abs(residual) < DEPTH_INLIER_MM)),
                              'median_hand_depth_mm': float(np.median([row['mano_depth_mm'] for row in used])),
                              'per_frame': entries}
    if medians_all:
        out['bias_mm'] = float(np.median(medians_all))
        out['frames_used'] = len(medians_all)
    return out


# ---- verdict -----------------------------------------------------------------------------------

# (pass_limit, fail_limit): value <= pass_limit passes, value > fail_limit fails, in between warns.
# Calibrated with tools/perturbation_study.py (outputs/analysis/perturbation/summary.json).
THRESHOLDS = {
    'E1_median_px': (13.0, 20.0),
    'E1_p90_px': (40.0, 60.0),
    'E2_abs_bias_mm': (15.0, 30.0),
    'E3_rotation_max_deg': (1.0, 2.0),
    'E3_translation_max_mm': (10.0, 20.0),
    'E4_mahalanobis2': (11.34, 30.0),
}


def verdict(status, e1, e2, e3, e4, thresholds=None):
    """PASS / WARN / FAIL for the mount of one recording, with the result of every check."""
    limits = {**THRESHOLDS, **(thresholds or {})}
    bias = (e2 or {}).get('bias_mm')
    values = {'E1_median_px': (e1 or {}).get('median_px'), 'E1_p90_px': (e1 or {}).get('p90_px'),
              'E2_abs_bias_mm': None if bias is None else abs(bias),
              'E3_rotation_max_deg': (e3 or {}).get('rotation_max_deg'),
              'E3_translation_max_mm': (e3 or {}).get('translation_max_mm'),
              'E4_mahalanobis2': (e4 or {}).get('mahalanobis2')}
    checks, level = {}, 0
    for key, value in values.items():
        if value is None or not np.isfinite(value):
            checks[key] = {'value': None, 'result': 'n/a'}
            continue
        low, high = limits[key]
        result = 'PASS' if value <= low else ('FAIL' if value > high else 'WARN')
        level = max(level, {'PASS': 0, 'WARN': 1, 'FAIL': 2}[result])
        checks[key] = {'value': float(value), 'pass_limit': low, 'fail_limit': high, 'result': result}
    notes = []
    if status == 'FALLBACK':
        level = max(level, 1)
        notes.append('mount not calibrated on this recording (fallback)')
    if values['E1_median_px'] is None:
        level = 2
        notes.append('no MediaPipe evidence to check the mount against')
    if bias is None:
        notes.append('no depth check (no valid MANO frame with ZED depth)')
    return {'verdict': ('PASS', 'WARN', 'FAIL')[level], 'checks': checks, 'notes': notes}
