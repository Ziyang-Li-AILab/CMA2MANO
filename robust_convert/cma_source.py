# -*- coding: utf-8 -*-
"""Tolerant CMA reading.

The original source_schema.read_human raises on the first defect (partial zero,
bad quaternion, malformed row). Here every defect becomes a mask entry, so one
bad joint in one frame no longer throws away the whole recording.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation

FINGERS = ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky')
# OpenPose-21 parent of every joint; the wrist is its own parent.
PARENTS = np.array([0] + [0 if j % 4 == 1 else j - 1 for j in range(1, 21)])
QUAT_NORM_TOL = 0.02


def _header(path):
    with Path(path).open('r', encoding='utf-8-sig') as source:
        names = source.readline().rstrip('\r\n').split('\t')
    if len(names) != len(set(names)):
        raise ValueError('Duplicate CMA column names')
    return names


def _hand_columns(bases):
    return [name for base in bases for name in (
        *[f'{base}_Pos_{axis}(mm)' for axis in 'XYZ'],
        *[f'{base}_{kind}_{axis}' for kind in ('GlobalQ', 'LocalQ') for axis in 'XYZR'],
    )]


def discover_hands(human_path):
    """Same schema rule as the original: one subject, one wrist alias per side.

    A side whose columns are missing is reported as absent instead of raising.
    """
    names = set(_header(human_path))
    if 'PtpTimeStamp' not in names:
        raise ValueError('Missing PtpTimeStamp')
    expression = re.compile(r'^(.+)_(left|right)_(LeftHand|RightHand)(Thumb|Index|Middle|Ring|Pinky)[1-4]_Pos_X\(mm\)$')
    subjects = {match.group(1) for name in names if (match := expression.match(name))}
    if len(subjects) != 1:
        raise ValueError(f'Expected one unambiguous subject, found {sorted(subjects)}')
    subject = next(iter(subjects))
    result = {}
    for side in ('left', 'right'):
        family = 'LeftHand' if side == 'left' else 'RightHand'
        prefix = f'{subject}_{side}_'
        aliases = [prefix + alias for alias in ('Hand', 'LeftHand', 'RightHand')
                   if prefix + alias + '_Pos_X(mm)' in names]
        bases = aliases[:1] + [f'{prefix}{family}{finger}{joint}'
                               for finger in FINGERS for joint in range(1, 5)]
        missing = [name for name in _hand_columns(bases) if name not in names]
        if len(aliases) != 1 or missing:
            result[side] = {'bases': bases, 'subject': subject, 'physical_side': side,
                            'usable': False,
                            'reason': f'wrist aliases {aliases}; missing columns {missing[:4]}'}
        else:
            result[side] = {'bases': bases, 'subject': subject, 'physical_side': side, 'usable': True}
    return result


def _timestamp_ns(text):
    try:
        scaled = Decimal(text.strip()) * 1_000_000_000
    except InvalidOperation:
        return None
    if not scaled.is_finite():
        return None
    return int(scaled.to_integral_value())


def _float(text):
    try:
        return float(text)
    except ValueError:
        return np.nan


def _rows_by_field(lines, width, chosen, stamp_col):
    """Reference reader: every chosen field through _float (text that is no number gives NaN)."""
    stamps, rows, malformed = [], [], 0
    for line in lines:
        if not line.strip():
            continue
        fields = line.rstrip('\r\n').split('\t')
        stamp = _timestamp_ns(fields[stamp_col]) if len(fields) > stamp_col else None
        if stamp is None:
            malformed += 1
            continue
        if len(fields) != width:
            malformed += 1
            rows.append([np.nan] * len(chosen))
        else:
            rows.append([_float(fields[i]) for i in chosen])
        stamps.append(stamp)
    return np.asarray(stamps, np.int64), np.asarray(rows, np.float64).reshape(len(stamps), len(chosen)), malformed


def _rows_by_loadtxt(lines, width, chosen, stamp_col):
    """The result of _rows_by_field with the complete rows parsed by np.loadtxt (about 2x faster).

    loadtxt converts with the routine of float() (PyOS_string_to_double) and raises on any text it does
    not take (empty field, underscore, non-ASCII digit, ...); read_table then uses _rows_by_field.
    """
    stamps, complete, good, malformed = [], [], [], 0
    for line in lines:
        if not line or line.isspace():  # = not line.strip()
            continue
        line = line.rstrip('\r\n')
        tabs = line.count('\t')
        stamp = _timestamp_ns(line.split('\t', stamp_col + 1)[stamp_col]) if tabs >= stamp_col else None
        if stamp is None:
            malformed += 1
            continue
        complete.append(tabs + 1 == width)
        if complete[-1]:
            good.append(line)
        else:
            malformed += 1
        stamps.append(stamp)
    rows = np.full((len(stamps), len(chosen)), np.nan)
    if good:
        rows[np.asarray(complete, bool)] = np.loadtxt(good, dtype=np.float64, delimiter='\t', comments=None,
                                                      quotechar=None, usecols=chosen, ndmin=2)
    return np.asarray(stamps, np.int64), rows, malformed


def read_table(path, columns):
    """Read selected columns. Malformed rows become NaN rows and are reported."""
    header = _header(path)
    index = {name: i for i, name in enumerate(header)}
    missing = [name for name in columns if name not in index]
    if missing:
        raise ValueError(f'{Path(path).name}: missing columns {missing[:6]}')
    chosen = [index[name] for name in columns]
    stamp_col = index['PtpTimeStamp']
    with Path(path).open('r', encoding='utf-8-sig', errors='replace') as source:
        next(source)
        lines = source.readlines()
    try:
        return _rows_by_loadtxt(lines, len(header), chosen, stamp_col)
    except Exception:  # noqa: BLE001 - some field loadtxt does not take: field by field, as before
        return _rows_by_field(lines, len(header), chosen, stamp_col)


def _quat_valid(q):
    norm = np.linalg.norm(q, axis=-1)
    return np.isfinite(q).all(-1) & (norm > 1e-8) & (np.abs(norm - 1) <= QUAT_NORM_TOL)


def _as_matrix(q, valid):
    safe = np.where(valid[..., None], q, np.array([0.0, 0.0, 0.0, 1.0]))
    return Rotation.from_quat(safe.reshape(-1, 4)).as_matrix().reshape(*q.shape[:-1], 3, 3)


def read_human(human_path):
    """Both hands with per-frame/per-joint validity masks. Nothing raises for local defects."""
    schema = discover_hands(human_path)
    sides = [side for side in ('left', 'right') if schema[side]['usable']]
    columns = [name for side in sides for name in _hand_columns(schema[side]['bases'])]
    stamps, table, malformed = read_table(human_path, columns)
    hands, offset = {}, 0
    for side in ('left', 'right'):
        spec = schema[side]
        if not spec['usable']:
            hands[side] = {'present': False, 'schema': spec, 'reason': spec['reason']}
            continue
        values = table[:, offset:offset + 21 * 11].reshape(-1, 21, 11)
        offset += 21 * 11
        points, global_q, local_q = values[..., :3], values[..., 3:7], values[..., 7:11]
        finite = np.isfinite(points).all(-1)
        zero = np.all(points == 0, axis=-1)
        pos_valid = finite & ~zero
        present = bool(pos_valid.any())
        global_valid, local_valid = _quat_valid(global_q), _quat_valid(local_q)
        hands[side] = {
            'present': present,
            'schema': spec,
            'points_mm': np.where(pos_valid[..., None], points, np.nan),
            'pos_valid': pos_valid,
            'global_R': _as_matrix(global_q, global_valid),
            'local_R': _as_matrix(local_q, local_valid),
            'rot_valid': global_valid & local_valid,
            'defects': {
                'zero_or_nonfinite_joints': int((~pos_valid).sum()),
                'frames_with_zero_or_nonfinite': int((~pos_valid).any(1).sum()),
                'invalid_quaternions': int((~(global_valid & local_valid)).sum()),
            },
        }
        if not present:
            hands[side]['reason'] = 'whole track is a zero placeholder'
    return {'timestamps_ns': stamps, 'hands': hands, 'malformed_rows': malformed}


def read_rig(body_path, name='camera'):
    """Head-camera rigid body. Rows with a lost body (zeros, bad quaternion) are masked."""
    columns = [f'{name}_Pos_{axis}(mm)' for axis in 'XYZ'] + [f'{name}_Qua_{axis}' for axis in 'XYZR']
    stamps, table, malformed = read_table(body_path, columns)
    origin, quat = table[:, :3], table[:, 3:7]
    valid = np.isfinite(origin).all(-1) & ~np.all(origin == 0, axis=-1) & _quat_valid(quat)
    return {'ptp_ns': stamps, 'origin_mm': origin, 'quat_xyzw': quat,
            'rig_R': _as_matrix(quat, valid), 'valid': valid, 'malformed_rows': malformed}
