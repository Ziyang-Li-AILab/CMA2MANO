# -*- coding: utf-8 -*-
"""Load one recording: tolerant CMA reading, FK cleaning, resampling at RGB stamps."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .cleaning import INVALID, repair_hand
from .cma_source import read_human, read_rig
from .mcap_io import find_inputs, load_intrinsics, rgb_index
from .timing import nearest_rows, resample_hand, resample_rig

SIDES = ('left', 'right')


def load_session(folder, cleaning=None):
    """Everything needed for calibration, fitting and rendering, before any extrinsic is known."""
    folder = Path(folder)
    mcap, human, body = find_inputs(folder)
    K, width, height, intrinsics_info = load_intrinsics(mcap)
    rgb = rgb_index(mcap)
    usable = rgb['valid']
    stamps = rgb['stamps'][usable]
    source = read_human(human)
    rig = read_rig(body)
    cma_ns = source['timestamps_ns']
    if np.any(np.diff(cma_ns) <= 0) or np.any(np.diff(rig['ptp_ns']) <= 0):
        raise ValueError('CMA timestamps must be strictly increasing')
    hands = {}
    for side in SIDES:
        hand = source['hands'][side]
        if not hand['present']:
            hands[side] = {'present': False, 'reason': hand.get('reason', 'absent')}
            continue
        repaired = repair_hand(hand['points_mm'], hand['pos_valid'], hand['global_R'], hand['local_R'],
                               hand['rot_valid'], **(cleaning or {}))
        hands[side] = {'present': True, 'schema': hand['schema'], 'defects': hand['defects'],
                       'cleaning': repaired['diagnostics'],
                       'cma_points_mm': repaired['points_mm'], 'cma_status': repaired['status'],
                       'cma_world_R': repaired['world_R'],
                       'raw_points_mm': hand['points_mm']}
    session = {'folder': folder, 'name': folder.name, 'mcap': mcap, 'human': human, 'body': body,
               'K': K, 'width': width, 'height': height, 'intrinsics_info': intrinsics_info,
               'rgb': rgb, 'stamps': stamps, 'source_indices': np.flatnonzero(usable),
               'cma_ns': cma_ns, 'rig_source': rig, 'hands_source': hands,
               'malformed_rows': {'human': source['malformed_rows'], 'body': rig['malformed_rows']}}
    resample(session, 0)
    return session


def resample(session, offset_ns=0):
    """(Re)compute the per-RGB-frame tracks with CMA time = camera time + offset_ns."""
    query = session['stamps'] + int(offset_ns)
    rig = session['rig_source']
    origin, rig_R, rig_ok = resample_rig(rig['ptp_ns'], rig['origin_mm'], rig['rig_R'], rig['valid'], query)
    rows, delta = nearest_rows(session['cma_ns'], query)
    hands = {}
    for side in SIDES:
        source = session['hands_source'][side]
        if not source['present']:
            hands[side] = {'present': False}
            continue
        points, status, world = resample_hand(session['cma_ns'], source['cma_points_mm'], source['cma_status'],
                                              source['cma_world_R'], query)
        status[~rig_ok] = INVALID
        points[~rig_ok] = np.nan
        valid = (status != INVALID).all(1)
        hands[side] = {'present': True, 'points_mm': points, 'status': status, 'world_R': world,
                       'frame_valid': valid}
    session.update({'offset_ns': int(offset_ns), 'origin_mm': origin, 'rig_R': rig_R, 'rig_ok': rig_ok,
                    'cma_rows': rows, 'cma_delta_ns': delta, 'hands': hands})
    return session


def rig_local(points_mm, origin_mm, rig_R):
    """World CMA points into the head-rig frame: (P - origin) @ rig_R, as in the original."""
    return np.einsum('tkj,tji->tki', points_mm - origin_mm[:, None], rig_R)
