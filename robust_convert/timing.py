# -*- coding: utf-8 -*-
"""Resample the 120 Hz CMA streams at RGB camera stamps.

The original pipeline takes the nearest CMA row (up to 4.2 ms away) and raises
when none is close enough. Here positions are linearly interpolated between the
two surrounding rows and the rig rotation is slerped, which removes the up to
4 ms sampling jitter. When one neighbour is invalid, the other is held if it is
within `hold_ns`; otherwise the joint is invalid at that stamp.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .cleaning import INVALID

HOLD_NS = 4_200_000        # the original nearest-row tolerance
MAX_SPAN_NS = 12_500_000   # never interpolate across a CMA gap wider than 1.5 rows


def bracket(source_ns, query_ns):
    """Indices of the rows before/after every query, and the interpolation weight of the later row."""
    source_ns = np.asarray(source_ns, np.int64)
    query_ns = np.asarray(query_ns, np.int64)
    after = np.clip(np.searchsorted(source_ns, query_ns, side='left'), 1, len(source_ns) - 1)
    before = after - 1
    span = (source_ns[after] - source_ns[before]).astype(np.float64)
    weight = np.clip((query_ns - source_ns[before]) / np.maximum(span, 1.0), 0.0, 1.0)
    return before, after, weight


def nearest_rows(source_ns, query_ns):
    """Nearest source row for each query and the signed offset (row - query) in ns."""
    before, after, weight = bracket(source_ns, query_ns)
    rows = np.where(weight >= 0.5, after, before)
    return rows, np.asarray(source_ns, np.int64)[rows] - np.asarray(query_ns, np.int64)


def resample_hand(source_ns, points_mm, status, world_R, query_ns, hold_ns=HOLD_NS):
    """Hand track at query stamps. Returns points (N,21,3; NaN invalid), status (N,21), world_R (N,21,3,3)."""
    source_ns = np.asarray(source_ns, np.int64)
    query_ns = np.asarray(query_ns, np.int64)
    before, after, weight = bracket(source_ns, query_ns)
    ok_before = status[before] != INVALID
    ok_after = status[after] != INVALID
    span = source_ns[after] - source_ns[before]
    inside = (query_ns >= source_ns[0]) & (query_ns <= source_ns[-1]) & (span <= MAX_SPAN_NS)
    both = ok_before & ok_after & inside[:, None]
    w = weight[:, None, None]
    lerp = (1 - w) * np.nan_to_num(points_mm[before]) + w * np.nan_to_num(points_mm[after])
    distance_before = np.abs(query_ns - source_ns[before])[:, None]
    distance_after = np.abs(source_ns[after] - query_ns)[:, None]
    hold_before = ~both & ok_before & (distance_before <= hold_ns) & ((distance_before <= distance_after) | ~ok_after)
    hold_after = ~both & ~hold_before & ok_after & (distance_after <= hold_ns)
    points = np.full(lerp.shape, np.nan)
    points[both] = lerp[both]
    points[hold_before] = points_mm[before][hold_before]
    points[hold_after] = points_mm[after][hold_after]
    resampled = np.full(status[before].shape, INVALID, np.int8)
    resampled[both] = np.maximum(status[before], status[after])[both]
    resampled[hold_before] = status[before][hold_before]
    resampled[hold_after] = status[after][hold_after]
    nearest = np.where(weight >= 0.5, after, before)
    return points, resampled, world_R[nearest]


def resample_rig(source_ns, origin_mm, rig_R, valid, query_ns, hold_ns=HOLD_NS):
    """Rig origin (linear) and rotation (slerp) at query stamps, with validity."""
    source_ns = np.asarray(source_ns, np.int64)
    query_ns = np.asarray(query_ns, np.int64)
    before, after, weight = bracket(source_ns, query_ns)
    span = source_ns[after] - source_ns[before]
    inside = (query_ns >= source_ns[0]) & (query_ns <= source_ns[-1]) & (span <= MAX_SPAN_NS)
    both = valid[before] & valid[after] & inside
    origin = (1 - weight[:, None]) * origin_mm[before] + weight[:, None] * origin_mm[after]
    relative = Rotation.from_matrix(np.swapaxes(rig_R[before], -1, -2) @ rig_R[after]).as_rotvec()
    rotation = rig_R[before] @ Rotation.from_rotvec(relative * weight[:, None]).as_matrix()
    # Single valid neighbour close enough: hold it.
    near_before = ~both & valid[before] & (np.abs(query_ns - source_ns[before]) <= hold_ns)
    near_after = ~both & ~near_before & valid[after] & (np.abs(source_ns[after] - query_ns) <= hold_ns)
    origin[near_before], rotation[near_before] = origin_mm[before][near_before], rig_R[before][near_before]
    origin[near_after], rotation[near_after] = origin_mm[after][near_after], rig_R[after][near_after]
    ok = both | near_before | near_after
    origin[~ok] = np.nan
    return origin, rotation, ok


def slerp_rotations(source_ns, rotations, query_ns):
    """Plain slerp of a rotation stream (used for the VIO pose)."""
    source_ns = np.asarray(source_ns, np.int64)
    query = np.clip(np.asarray(query_ns, np.int64), source_ns[0], source_ns[-1])
    base = source_ns[0]
    return Slerp((source_ns - base).astype(np.float64), Rotation.from_matrix(rotations))(
        (query - base).astype(np.float64)).as_matrix()
