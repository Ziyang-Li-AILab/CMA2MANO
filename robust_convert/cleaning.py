# -*- coding: utf-8 -*-
"""Repair CMA hand skeletons from their own rotation stream.

Findings on the 20 test recordings (outputs/analysis/*):
  * On clean frames the solved skeleton is exactly rigid: every child joint sits
    at a constant offset in its parent's world-rotation frame (< 0.01 mm).
  * Glitches hit the position stream only; rotations stay smooth through them.
So forward kinematics (anchored wrist + world rotations + per-bone modal offset)
reproduces clean frames exactly and replaces glitched positions. A repaired
joint is flagged, never silently mixed with measured data; a joint whose
kinematic chain has no trustworthy rotation is marked invalid.

Status codes per joint and frame: 0 measured, 1 repaired by FK, 2 invalid.
"""
from __future__ import annotations

import numpy as np

from .cma_source import PARENTS

MEASURED, REPAIRED, INVALID = 0, 1, 2
MODES = ('raw_world', 'root_relative_fingers')


def angle_deg(matrices):
    cos = (np.trace(matrices, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def canonical_world_rotations(global_R, local_R, rot_ok, samples=240):
    """Pick the GlobalQ convention that agrees with LocalQ; never raises.

    Returns world rotations, their validity and diagnostics. In the
    root_relative_fingers convention every finger rotation depends on the wrist,
    so an invalid wrist quaternion invalidates the whole frame.
    """
    count = len(global_R)
    idx = np.unique(np.linspace(0, count - 1, min(samples, count)).astype(int))
    child, parent = np.arange(1, 21), PARENTS[1:]
    scores = {}
    for mode in MODES:
        world = global_R[idx].copy()
        if mode == 'root_relative_fingers':
            world[:, 1:] = world[:, :1] @ world[:, 1:]
        relative = np.swapaxes(world[:, parent], -1, -2) @ world[:, child]
        error = angle_deg(np.swapaxes(local_R[idx][:, child], -1, -2) @ relative)
        ok = rot_ok[idx][:, child] & rot_ok[idx][:, parent]
        if mode == 'root_relative_fingers':
            ok &= rot_ok[idx][:, :1]
        scores[mode] = float(np.percentile(error[ok], 95)) if ok.any() else float('inf')
    mode = min(scores, key=scores.get)
    world = global_R.copy()
    world_ok = rot_ok.copy()
    if mode == 'root_relative_fingers':
        world[:, 1:] = world[:, :1] @ world[:, 1:]
        world_ok[:, 1:] &= rot_ok[:, :1]
    warnings = []
    if scores[mode] > 0.5:
        warnings.append(f'LocalQ consistency p95 {scores[mode]:.2f} deg exceeds 0.5 deg')
    return world, world_ok, {'mode': mode, 'local_consistency_p95_deg': scores, 'warnings': warnings}


def bone_offsets(points, world_R):
    """Child position in the parent's world-rotation frame, (T, 20, 3) mm."""
    parent = PARENTS[1:]
    return np.einsum('tkji,tkj->tki', world_R[:, parent], points[:, 1:] - points[:, parent])


def _distances(a, b):
    """np.linalg.norm(b[None] - a[:, None], axis=-1) with the same float operations ((dx^2 + dy^2) + dz^2,
    then the root), one coordinate at a time: no (len(a), len(b), 3) temporaries, about 3x faster."""
    total = None
    for axis in range(3):
        step = b[:, axis][None] - a[:, axis][:, None]
        step *= step
        if total is None:
            total = step
        else:
            total += step
    return np.sqrt(total, out=total)


def modal_offsets(points, world_R, bone_ok, tol_mm=0.5, max_candidates=200):
    """Per bone: the offset shared by most frames (mode), refined by the inlier median.

    Glitched frames scatter away from the mode, so they do not bias it.
    Returns offsets (20, 3), inlier support fraction (20,) and sample count (20,).
    """
    offsets = bone_offsets(points, world_R)
    ref = np.full((20, 3), np.nan)
    support = np.zeros(20)
    count = np.zeros(20, np.int64)
    for k in range(20):
        values = offsets[bone_ok[:, k], k]
        count[k] = len(values)
        if not len(values):
            continue
        candidates = values[::max(1, len(values) // max_candidates)]
        near = _distances(candidates, values) < tol_mm
        best = candidates[int(np.argmax(near.sum(1)))]
        inliers = np.linalg.norm(values - best, axis=1) < tol_mm
        ref[k] = np.median(values[inliers], axis=0)
        support[k] = float((np.linalg.norm(values - ref[k], axis=1) < tol_mm).mean())
    return ref, support, count


def fk_relative(world_R, ref):
    """Joint positions relative to the wrist from rotations and fixed offsets."""
    rel = np.zeros((len(world_R), 21, 3))
    for j in range(1, 21):
        p = PARENTS[j]
        rel[:, j] = rel[:, p] + np.einsum('tab,b->ta', world_R[:, p], ref[j - 1])
    return rel


def chain_validity(world_ok, ref_ok):
    """A joint's FK position needs every ancestor rotation and every bone offset on the way."""
    chain = np.ones_like(world_ok)
    for j in range(1, 21):
        p = PARENTS[j]
        chain[:, j] = chain[:, p] & world_ok[:, p] & ref_ok[j - 1]
    return chain


def anchor_roots(points, rel, candidate_ok, tol_mm=1.0, min_agree=4, min_members=3):
    """Choose one wrist position per frame.

    Every valid joint proposes a wrist (its position minus its FK offset). The
    measured wrist is kept when enough proposals agree with it; otherwise the
    largest agreeing cluster wins. Frames without a cluster have no anchor.
    Returns root (T, 3), anchor_ok (T,), used_fallback (T,).
    """
    proposals = points - rel
    count = len(points)
    root = np.full((count, 3), np.nan)
    ok = np.zeros(count, bool)
    fallback = np.zeros(count, bool)
    others = candidate_ok[:, 1:].sum(1)
    agree = ((np.linalg.norm(proposals[:, 1:] - points[:, :1], axis=-1) < tol_mm)
             & candidate_ok[:, 1:]).sum(1)
    keep_wrist = candidate_ok[:, 0] & (agree >= np.minimum(min_agree, others))
    root[keep_wrist] = points[keep_wrist, 0]
    ok[keep_wrist] = True
    for t in np.flatnonzero(~keep_wrist):
        valid = np.flatnonzero(candidate_ok[t])
        if len(valid) < min_members:
            continue
        values = proposals[t, valid]
        near = np.linalg.norm(values[:, None] - values[None], axis=-1) < tol_mm
        best = int(np.argmax(near.sum(1)))
        members = near[best]
        if members.sum() >= min_members:
            root[t] = np.median(values[members], axis=0)
            ok[t] = True
            fallback[t] = True
    return root, ok, fallback


def _runs(flags):
    """(start, stop) of every True run."""
    padded = np.concatenate([[False], flags, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))


def repair_hand(points_mm, pos_valid, global_R, local_R, rot_valid, *, tol_mm=1.0,
                offset_tol_mm=0.5, settle_frames=12, suspect_step_mm=30.0, drop_suspect=True):
    """Clean one hand track at the CMA rate.

    Returns a dict with points_mm (T,21,3; NaN where invalid), world_R, status
    (T,21), frame_valid (T,) and diagnostics. Rotation of a joint is trusted only
    where its quaternion is valid and its position is not a zero placeholder.
    """
    points = np.asarray(points_mm, np.float64)
    count = len(points)
    rot_ok = rot_valid & pos_valid
    world, world_ok, rotation_info = canonical_world_rotations(global_R, local_R, rot_ok)
    bone_ok = pos_valid[:, 1:] & pos_valid[:, PARENTS[1:]] & world_ok[:, PARENTS[1:]]
    ref, support, samples = modal_offsets(points, world, bone_ok, offset_tol_mm)
    ref_ok = np.isfinite(ref).all(1)
    safe_ref = np.where(ref_ok[:, None], ref, 0.0)
    rel = fk_relative(world, safe_ref)
    chain = chain_validity(world_ok, ref_ok)
    candidate_ok = pos_valid & chain
    root, anchor_ok, anchor_fallback = anchor_roots(np.nan_to_num(points), rel, candidate_ok, tol_mm)
    fk = root[:, None] + rel
    distance = np.linalg.norm(fk - np.nan_to_num(points), axis=-1)
    status = np.full((count, 21), REPAIRED, np.int8)
    status[pos_valid & (distance <= tol_mm)] = MEASURED
    status[~chain | ~anchor_ok[:, None]] = INVALID

    # Hand-local motion of the repaired track: a jump here means the rotation
    # stream itself glitched, which FK cannot fix.
    local = np.einsum('tba,tjb->tja', world[:, 0], rel)
    step = np.full(count, 0.0)
    step[1:] = np.nanmax(np.linalg.norm(np.diff(local, axis=0), axis=-1), axis=1)
    step[~np.isfinite(step)] = 0.0
    suspect = np.zeros(count, bool)
    suspect[1:] = step[1:] > suspect_step_mm
    suspect[:-1] |= suspect[1:]          # both sides of a jump
    bad = (status == INVALID).any(1)
    if drop_suspect:
        bad |= suspect
    lost_runs = _runs(bad)
    settled = bad.copy()
    for _, stop in lost_runs:
        settled[stop:stop + settle_frames] = True
    status[settled] = INVALID
    frame_valid = ~settled
    output = np.where((status == MEASURED)[..., None], points, fk)
    output[status == INVALID] = np.nan

    valid_frames = max(int(frame_valid.sum()), 1)
    repaired = (status == REPAIRED) & frame_valid[:, None]
    diagnostics = {
        'frames': count,
        'rotation': rotation_info,
        'offset_support_min': float(support[ref_ok].min()) if ref_ok.any() else 0.0,
        'offset_support_per_bone': np.round(support, 4).tolist(),
        'offset_low_support_bones': (np.flatnonzero(support < 0.3) + 1).tolist(),
        'frames_invalid': int((~frame_valid).sum()),
        'frames_invalid_fraction': float((~frame_valid).mean()),
        'frames_with_repair': int(repaired.any(1).sum()),
        'frames_with_repair_fraction': float(repaired.any(1).sum() / valid_frames),
        'joints_repaired_fraction': float(repaired.sum() / (valid_frames * 21)),
        'repair_max_mm': float(np.nanmax(np.where(repaired & pos_valid, distance, np.nan)))
        if (repaired & pos_valid).any() else 0.0,
        'anchor_fallback_frames': int((anchor_fallback & frame_valid).sum()),
        'suspect_frames': int(suspect.sum()),
        'suspect_dropped': bool(drop_suspect),
        'lost_runs': [[int(a), int(b)] for a, b in lost_runs],
        'settle_frames': int(settle_frames),
        'max_hand_local_step_mm': float(step[frame_valid].max()) if frame_valid.any() else 0.0,
    }
    return {'points_mm': output, 'world_R': world, 'world_ok': world_ok, 'status': status,
            'frame_valid': frame_valid, 'offsets_mm': ref, 'diagnostics': diagnostics}
