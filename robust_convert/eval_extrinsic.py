# -*- coding: utf-8 -*-
"""Extrinsic (camera mount) accuracy evaluation: the checks, the per-recording result and the batch report.

Part of the unified entry (the conversion evaluates the mount it has just estimated):
    python mano_convert.py --data <input> --output <output> --eval_fit_ex true
Standalone, on results converted without the evaluation (nothing is refitted: the mount is derived again
and checked against the saved one, the MANO meshes are rebuilt from mano_fit.npz):
    python -m robust_convert.eval_extrinsic --data <input> --result <output> [--only NAME ...] [--redo]
    python -m robust_convert.eval_extrinsic --result <output> --report-only

Checks (metrics.py; limits calibrated with tools/perturbation_study.py):
  E1  held-out reprojection: temporal 5-fold; the mount estimator is re-run without the test block and a
      15-frame buffer, then the CMA joints of the test block projected with that mount are compared with
      the MediaPipe keypoints (median / p90, px and mm at the hand)
  E2  depth: ZED depth minus the depth of the fitted MANO surface placed with the mount (median, mm)
  E3  stability: largest rotation / translation difference between a fold mount and the full mount
  E4  rotation consistency: Mahalanobis d^2 (chi^2, 3 dof) between the MediaPipe-only rotation and the
      independent reference rotation (ZED VIO hand-eye, else the fleet prior)
  sync, diagnostic only: CMA time shift that minimises the reprojection / VIO residual
Verdict PASS / WARN / FAIL: metrics.verdict.

Outputs:
  <output>/<recording>/extrinsic_eval.json    verdict, check values and limits, summary, all metrics
  <output>/<recording>/mano_fit_report.json   'metrics', 'extrinsic_verdict' and 'sample_usable' filled in
  <output>/外参评估报告.md                      batch report
  <output>/extrinsic_eval_summary.csv|.json   one row per recording | counts, statistics and rows
  <output>/extrinsic_eval_summary.png         the six checks of every recording with their limits

Only the standard library is imported at module level: the batch report is written by any Python 3
(the figure needs matplotlib; another interpreter is used when this one has none).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .batch import (CONVERT_MODULES, EVAL_ERROR, EVAL_NAME, GRADES, MEDIAPIPE_PYTHON, NOT_EVALUATED, REPORT_NAME,
                    ROOT, TORCH_PYTHON, python_with, result_folders)

EVALUATOR = 'eval_extrinsic_v1'
SUMMARY_NAME = 'extrinsic_eval_summary'
MARKDOWN_NAME = '外参评估报告.md'
# check of metrics.verdict -> (label, unit, digits)
CHECKS = {'E1_median_px': ('E1 留出重投影误差 中位数', 'px', 1), 'E1_p90_px': ('E1 留出重投影误差 p90', 'px', 1),
          'E2_abs_bias_mm': ('E2 深度偏差 |ZED−MANO|', 'mm', 1), 'E3_rotation_max_deg': ('E3 折间旋转最大差', '°', 2),
          'E3_translation_max_mm': ('E3 折间平移最大差', 'mm', 1), 'E4_mahalanobis2': ('E4 旋转一致性 d²', '', 2)}
SOURCE_TEXT = {'fused_vio': 'MediaPipe+VIO', 'fused_fleet_prior': 'MediaPipe+车队先验', 'pnp': 'MediaPipe',
               'vio': 'VIO', 'fleet_prior': '车队先验', 'template': '原始模板'}
VERDICT_TEXT = {NOT_EVALUATED: '未评估', EVAL_ERROR: '评估出错'}
NOTE_TEXT = {'mount not calibrated on this recording (fallback)': '外参未在本录制上标定（FALLBACK，直接用车队先验），至少判 WARN',
             'no MediaPipe evidence to check the mount against': '没有可用的 MediaPipe 关键点，无法检验外参，判 FAIL',
             'no depth check (no valid MANO frame with ZED depth)': '无深度检查（没有带 ZED 深度的有效 MANO 帧）'}


# ---- the checks of one recording (numpy, scipy, opencv are imported by the functions) ----------------

def extrinsic_metrics(session, detections, handeye, prior, R, t, report, fit):
    """E1/E3 (held-out re-estimation) and E4 (rotation consistency) for the estimator that chose the mount."""
    from .extrinsic import fit_mount, fixed_rotation_mount, local_tracks
    from .metrics import fixed_mount_reprojection, held_out, rotation_consistency
    K = session['K']
    local = local_tracks(session)
    e1 = e3 = None
    e4 = {'status': 'no_reference', 'vio_status': handeye.get('status', 'no_vio')}
    if detections is None:
        return e1, e3, e4
    status = report['status']
    if status == 'CALIBRATED':
        e1, e3 = held_out(local, detections, K, fit,
                          estimator=lambda frames: fit_mount(local, detections, K, frames, handeye, prior))
        e1['protocol'] = '5-fold temporal, 15-frame buffer, full estimator re-run without the test block'
        e4 = rotation_consistency(fit, handeye)
    elif status in ('VIO_ROTATION', 'PRIOR_ROTATION'):
        e1, e3 = held_out(local, detections, K, fit,
                          estimator=lambda frames: fixed_rotation_mount(local, detections, K, frames, handeye, prior))
        e1['protocol'] = ('5-fold temporal, rotation choice (VIO / fleet prior) and translation re-run '
                          'without the test block')
    else:
        e1 = fixed_mount_reprojection(local, detections, K, R, t)
    return e1, e3, e4


def sync_diagnostics(session, detections, vio, handeye, R, t, status):
    """Is the CMA clock aligned with the camera? Diagnostic only, no offset is applied."""
    from .extrinsic import local_tracks, vio_usable
    from .metrics import time_offset_scan
    from .session import resample
    from .vio import pair_residual_curve
    out = {}
    if detections is not None and status != 'FALLBACK':
        scan = time_offset_scan(session, detections, session['K'], R, t, list(range(-30, 35, 10)), local_tracks,
                                resample)
        out['mediapipe'] = scan
    if vio is not None and vio_usable(handeye):
        curve = pair_residual_curve(session, vio, handeye, list(range(-40, 45, 5)), resample)
        valid = [row for row in curve if row.get('residual_deg') is not None]
        best = min(valid, key=lambda row: row['residual_deg']) if valid else None
        out['vio'] = {'curve': curve, 'best_offset_ms': None if best is None else best['offset_ms']}
    resample(session, 0)
    offsets = [value['best_offset_ms'] for value in out.values() if value.get('best_offset_ms') is not None]
    out['max_abs_best_offset_ms'] = max((abs(v) for v in offsets), default=None)
    return out


def depth_check(session, mount_R, mount_t, meshes, exists, frame_count):
    """E2 on up to frame_count evenly spaced frames with a valid MANO hand; None without any."""
    from .mcap_io import read_depth
    from .metrics import depth_agreement, depth_frames
    if not meshes or frame_count <= 0:
        return None
    frames = depth_frames(exists.any(1), frame_count)
    if not len(frames):
        return None
    depth_maps = read_depth(session['mcap'], session['stamps'][frames])
    return depth_agreement(session, mount_R, mount_t, meshes, frames, depth_maps)


def evaluate_mount(session, detections, vio, handeye, prior, mount_R, mount_t, ext_report, fit, meshes, exists,
                   depth_frames=120, sync=True, mode='inline'):
    """All checks and the verdict of the mount of one recording: the 'metrics' of mano_fit_report.json.

    meshes: {side: {'vertices_mm', 'faces', 'valid'}} of the fitted MANO hands, exists: (N, 2) valid frames.
    """
    from .metrics import THRESHOLDS, verdict
    seconds = {}
    started = time.perf_counter()
    e1, e3, e4 = extrinsic_metrics(session, detections, handeye, prior, mount_R, mount_t, ext_report, fit)
    seconds['E1_E3_E4'] = time.perf_counter() - started
    started = time.perf_counter()
    timing = (sync_diagnostics(session, detections, vio, handeye, mount_R, mount_t, ext_report['status'])
              if sync else {'status': 'skipped'})
    seconds['sync'] = time.perf_counter() - started
    started = time.perf_counter()
    e2 = depth_check(session, mount_R, mount_t, meshes, exists, depth_frames)
    seconds['E2'] = time.perf_counter() - started
    return {'evaluated': True, 'evaluator': EVALUATOR, 'mode': mode,
            'E1_heldout_reprojection': e1, 'E2_depth': e2, 'E3_stability': e3, 'E4_rotation_consistency': e4,
            'sync': timing, 'verdict': verdict(ext_report['status'], e1, e2, e3, e4),
            'thresholds': {key: list(value) for key, value in THRESHOLDS.items()}, 'seconds': seconds}


def _placeholder(**fields):
    return {'evaluated': False, 'evaluator': EVALUATOR, 'E1_heldout_reprojection': None, 'E2_depth': None,
            'E3_stability': None, 'E4_rotation_consistency': None, 'sync': None, **fields}


def not_evaluated():
    """'metrics' of a recording converted without --eval_fit_ex."""
    return _placeholder(verdict={'verdict': NOT_EVALUATED, 'checks': {}, 'notes': [
        'extrinsic accuracy not evaluated: run with --eval_fit_ex true, or python -m robust_convert.eval_extrinsic']})


def evaluation_failed(error):
    """'metrics' when the evaluation raised: the conversion result is kept, the sample is not graded."""
    import traceback
    text = f'{type(error).__name__}: {error}'
    return _placeholder(status='failed', error=text, traceback=traceback.format_exc(),
                        verdict={'verdict': EVAL_ERROR, 'checks': {}, 'notes': [f'evaluation failed: {text}']})


def sample_usable(verdict_name, any_valid):
    """True / False once graded (mount PASS or WARN and at least one valid hand frame), None when not graded."""
    if verdict_name not in GRADES:
        return None
    return verdict_name in ('PASS', 'WARN') and bool(any_valid)


# ---- per-recording file ----------------------------------------------------------------------------

def _write_json(path, data, indent=1):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def _load(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _get(data, *keys):
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def sample_record(report):
    """Content of <output>/<recording>/extrinsic_eval.json (report: mano_fit_report.json, JSON-ready)."""
    metrics = report.get('metrics') or {}
    grade = metrics.get('verdict') or {}
    ext = report.get('extrinsic') or {}
    e1 = metrics.get('E1_heldout_reprojection') or {}
    e2 = metrics.get('E2_depth') or {}
    e3 = metrics.get('E3_stability') or {}
    e4 = metrics.get('E4_rotation_consistency') or {}
    sync = metrics.get('sync') or {}
    hands = {side: {key: value for key, value in hand.items() if key != 'per_frame'}
             for side, hand in (e2.get('hands') or {}).items()}
    return {'recording': report.get('recording_name'), 'evaluator': metrics.get('evaluator', EVALUATOR),
            'mode': metrics.get('mode'), 'status': 'ok' if metrics.get('evaluated') else metrics.get('status'),
            'error': metrics.get('error'), 'verdict': report.get('extrinsic_verdict'),
            'sample_usable': report.get('sample_usable'), 'checks': grade.get('checks', {}),
            'notes': grade.get('notes', []), 'thresholds': metrics.get('thresholds'),
            'mount': {'status': ext.get('status'), 'rotation_source': ext.get('rotation_source'),
                      'mount_R': ext.get('mount_R'), 'mount_t_mm': ext.get('mount_t_mm'),
                      'inliers': ext.get('inliers'), 'frames_used': ext.get('frames_used'),
                      'train_median_px': ext.get('train_median_px'), 'vio': _get(ext, 'handeye', 'status')},
            'summary': {'E1_median_px': e1.get('median_px'), 'E1_median_mm': e1.get('median_mm'),
                        'E1_p90_px': e1.get('p90_px'), 'E1_points': e1.get('points'), 'E1_folds_ok': e1.get('folds_ok'),
                        'E1_protocol': e1.get('protocol'), 'E2_bias_mm': e2.get('bias_mm'),
                        'E2_frames': e2.get('frames_used'), 'E2_hands': hands,
                        'E3_rotation_max_deg': e3.get('rotation_max_deg'),
                        'E3_translation_max_mm': e3.get('translation_max_mm'),
                        'E3_pixel_shift_max_px': e3.get('pixel_shift_max_px'),
                        'E4_mahalanobis2': e4.get('mahalanobis2'), 'E4_angle_deg': e4.get('angle_deg'),
                        'E4_p_value': e4.get('p_value'), 'E4_reference': e4.get('reference'),
                        'sync_mediapipe_best_ms': _get(sync, 'mediapipe', 'best_offset_ms'),
                        'sync_vio_best_ms': _get(sync, 'vio', 'best_offset_ms'),
                        'sync_max_abs_ms': sync.get('max_abs_best_offset_ms')},
            'frames': report.get('n_frames'), 'frames_any_hand_valid': _get(report, 'yield', 'frames_any_hand_valid'),
            'seconds': metrics.get('seconds'), 'metrics': metrics}


def write_sample_eval(folder, report):
    _write_json(Path(folder) / EVAL_NAME, sample_record(report))


# ---- evaluation of an existing result ----------------------------------------------------------------

def rebuild_meshes(arrays, mano):
    """MANO meshes of a saved fit, identical to the conversion (same float32 parameters, same batch)."""
    import numpy as np
    from .convert import get_model
    from .mano_model import mano_forward
    from .session import SIDES
    count = len(arrays['timestamps_ns'])
    meshes = {}
    for side in SIDES:
        if f'{side}_fitted' not in arrays:
            continue
        fitted = arrays[f'{side}_fitted']
        frames = np.flatnonzero(fitted)
        model = get_model(mano, side)

        def parameter(key):
            return arrays[f'{side}_{key}'][frames].astype(np.float64)

        vertices = mano_forward(model, arrays[f'{side}_beta'].astype(np.float64), parameter('global_orient'),
                                parameter('hand_pose'), parameter('trans'))[0]
        full = np.full((count,) + vertices.shape[1:], np.nan, np.float32)
        full[frames] = (vertices * 1000).astype(np.float32)
        meshes[side] = {'vertices_mm': full, 'faces': np.asarray(model['f'], np.int32), 'fitted': fitted,
                        'valid': arrays[f'{side}_valid']}
    return meshes


def evaluate_existing(data, result, *, mano=None, mediapipe_cache=None, mp_python=None, log=print):
    """Evaluate the mount of one converted recording in place; data: recording folder, result: its output."""
    import numpy as np
    from .convert import ensure_detections, handeye_safe, jsonable, solve_extrinsic
    from .extrinsic import load_prior
    from .mano_model import DEFAULT_MANO
    from .metrics import rotation_angle_deg
    from .session import load_session
    started = time.perf_counter()
    result = Path(result)
    report = json.loads((result / REPORT_NAME).read_text(encoding='utf-8'))
    session = load_session(data)
    name, count = session['name'], len(session['stamps'])
    with np.load(result / 'mano_fit.npz') as npz:
        arrays = {key: npz[key] for key in npz.files}
    if name != report.get('recording_name') or not np.array_equal(arrays['timestamps_ns'], session['stamps']):
        raise ValueError(f'{result} holds the result of {report.get("recording_name")} '
                         f'({report.get("n_frames")} frames), not of {data} ({count} frames)')
    ext = report['extrinsic']
    mount_R, mount_t = arrays['mount_R'].astype(np.float64), arrays['mount_t_mm'].astype(np.float64)
    recorded = ext.get('detections') or {}
    cache = Path(mediapipe_cache) if mediapipe_cache else (Path(recorded['path']).parent if recorded.get('path')
                                                           else None)
    detections, detection_info = (ensure_detections(session, cache, mp_python) if cache is not None
                                  else (None, {'status': 'missing', 'reason': 'no keypoint cache known'}))
    vio, handeye = handeye_safe(session)
    prior = None
    if ext.get('prior'):
        prior = load_prior(Path(ext['prior']['path']))
        if prior is None:
            raise FileNotFoundError(f"fleet prior used by the conversion not found: {ext['prior']['path']}")
    if ext['status'] == 'FALLBACK':
        # the mount does not depend on this recording: every frame with keypoints is a test frame
        ext_report, fit, reproduction = ext, None, {'status': 'fallback mount, not derived from the recording'}
    else:
        if detections is None:
            raise RuntimeError(f"the MediaPipe keypoints of the conversion are missing ({detection_info.get('path')}): "
                               f"{detection_info.get('reason', '')}"[:600])
        R, t, ext_report, fit = solve_extrinsic(session, detections, handeye, prior)
        reproduction = {'status': ext_report['status'], 'rotation_deg': rotation_angle_deg(R, mount_R),
                        'translation_mm': float(np.linalg.norm(np.asarray(t, np.float64) - mount_t))}
        if (ext_report['status'] != ext['status'] or reproduction['rotation_deg'] > 0.01
                or reproduction['translation_mm'] > 0.1):
            raise RuntimeError(f"the saved mount is not reproduced ({ext['status']} -> {ext_report['status']}, "
                               f"{reproduction['rotation_deg']:.3f} deg, {reproduction['translation_mm']:.2f} mm): "
                               'keypoints or fleet prior changed since the conversion; convert again with '
                               '--eval_fit_ex true')
    config = report.get('config') or {}
    meshes = rebuild_meshes(arrays, mano or config.get('mano') or DEFAULT_MANO)
    exists = arrays['exists']
    metrics = evaluate_mount(session, detections, vio, handeye, prior, mount_R, mount_t, ext_report, fit, meshes,
                             exists, depth_frames=config.get('depth_frames', 120), sync=config.get('sync', True),
                             mode='standalone')
    metrics.update(mount_reproduction=reproduction, detections=detection_info)
    metrics['seconds']['total'] = time.perf_counter() - started
    report['metrics'] = metrics
    report['extrinsic_verdict'] = metrics['verdict']['verdict']
    report['sample_usable'] = sample_usable(report['extrinsic_verdict'], exists.any())
    report = jsonable(report)
    _write_json(result / REPORT_NAME, report)
    write_sample_eval(result, report)
    log(f"[{name}] extrinsic {report['extrinsic_verdict']}, "
        f"E1 {_get(metrics, 'E1_heldout_reprojection', 'median_px') or float('nan'):.1f} px, "
        f"{metrics['seconds']['total']:.0f} s")
    return report


def write_eval_failure(result, name, error):
    """extrinsic_eval.json of a standalone evaluation that raised (the conversion result is left as it is)."""
    import traceback
    _write_json(Path(result) / EVAL_NAME, {'recording': name, 'evaluator': EVALUATOR, 'mode': 'standalone',
                                          'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                                          'traceback': traceback.format_exc()})


# ---- batch report ------------------------------------------------------------------------------------

def _row(folder):
    report = _load(folder / REPORT_NAME)
    if report is None:
        failure = _load(folder / 'failure.json') or {}
        error = (f"{failure.get('stage')}: {failure.get('error_type')}: {failure.get('error')}" if failure
                 else 'no result')
        return {'recording': folder.name, 'status': 'failed', 'evaluated': False, 'verdict': None,
                'sample_usable': False, 'checks': {}, 'notes': [], 'error': error}
    metrics = report.get('metrics') or {}
    grade = metrics.get('verdict') or {}
    evaluation = _load(folder / EVAL_NAME) or {}
    error = metrics.get('error') or (evaluation.get('error') if evaluation.get('status') == 'failed' else None)
    verdict_name = report.get('extrinsic_verdict')
    checks = grade.get('checks') or {}
    return {'recording': folder.name, 'status': 'ok', 'evaluated': verdict_name in GRADES, 'verdict': verdict_name,
            'sample_usable': report.get('sample_usable'), 'extrinsic_status': _get(report, 'extrinsic', 'status'),
            'rotation_source': _get(report, 'extrinsic', 'rotation_source'),
            'vio': _get(report, 'extrinsic', 'handeye', 'status'),
            'E1_median_mm': _get(metrics, 'E1_heldout_reprojection', 'median_mm'),
            'E1_points': _get(metrics, 'E1_heldout_reprojection', 'points'),
            'E2_bias_mm': _get(metrics, 'E2_depth', 'bias_mm'),
            'E4_reference': _get(metrics, 'E4_rotation_consistency', 'reference'),
            'E4_angle_deg': _get(metrics, 'E4_rotation_consistency', 'angle_deg'),
            'sync_offset_ms': _get(metrics, 'sync', 'max_abs_best_offset_ms'),
            'frames': report.get('n_frames'), 'frames_any_hand_valid': _get(report, 'yield', 'frames_any_hand_valid'),
            'checks': {key: checks.get(key) or {'value': None, 'result': 'n/a'} for key in CHECKS},
            'notes': [NOTE_TEXT.get(note, note) for note in grade.get('notes', [])] if verdict_name in GRADES else [],
            'thresholds': metrics.get('thresholds'), 'error': error}


def collect_rows(result):
    """Rows of every <result>/<recording> with a conversion result or failure.json, and the check limits."""
    rows = [_row(folder) for folder in result_folders(result)]
    thresholds = {}
    for row in rows:
        for key, limits in (row.pop('thresholds', None) or {}).items():
            thresholds.setdefault(key, list(limits))
        for key, check in row['checks'].items():
            if check.get('pass_limit') is not None:
                thresholds.setdefault(key, [check['pass_limit'], check['fail_limit']])
    return rows, thresholds


def check_statistics(rows, thresholds):
    evaluated = [row for row in rows if row['evaluated']]
    stats = {}
    for key in CHECKS:
        values = [row['checks'][key]['value'] for row in evaluated if row['checks'][key].get('value') is not None]
        counts = {grade: sum(row['checks'][key].get('result') == grade for row in evaluated)
                  for grade in (*GRADES, 'n/a')}
        low, high = thresholds.get(key, (None, None))
        stats[key] = {'median': statistics.median(values) if values else None, 'min': min(values, default=None),
                      'max': max(values, default=None), **counts, 'pass_limit': low, 'fail_limit': high}
    return stats


def _short(name):
    return name[4:] if len(name) == 15 and name[:8].isdigit() else name


def plot_summary(rows, thresholds, path):
    """Bars per recording (histograms above 40 recordings) of the six checks, with the PASS / FAIL limits."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    rows = [row for row in rows if row['evaluated']]
    if not rows:
        return False
    colors = {'PASS': '#2e7d32', 'WARN': '#ef6c00', 'FAIL': '#c62828', 'n/a': '#9e9e9e'}
    many = len(rows) > 40
    fig, axes = plt.subplots(2, 3, figsize=(15 if many else max(13, 0.5 * len(rows) + 4), 8.5))
    for ax, (key, (label, unit, _)) in zip(axes.flat, CHECKS.items()):
        values = [row['checks'][key].get('value') for row in rows]
        results = [row['checks'][key].get('result', 'n/a') for row in rows]
        finite = [value for value in values if value is not None]
        low, high = thresholds.get(key, (None, None))
        if many:
            if finite:
                ax.hist(finite, bins=30, color='#607d8b')
            line = ax.axvline
            ax.set_xlabel(unit)
            ax.set_ylabel('录制数')
        else:
            x = list(range(len(rows)))
            ax.bar(x, [value if value is not None else 0 for value in values],
                   color=[colors.get(result, colors['n/a']) for result in results])
            for index, value in enumerate(values):
                if value is None:
                    ax.text(index, 0, 'n/a', rotation=90, ha='center', va='bottom', fontsize=7, color='#616161')
            ax.set_xticks(x)
            ax.set_xticklabels([_short(row['recording']) for row in rows], rotation=90, fontsize=7)
            ax.set_ylabel(unit)
            ax.set_ylim(0, max([high * 1.15 if high is not None else 1.0] + [value * 1.08 for value in finite]))
            line = ax.axhline
        if low is not None:
            line(low, color='#ef6c00', linestyle='--', linewidth=1.2, label=f'PASS ≤ {low:g}')
            line(high, color='#c62828', linestyle='--', linewidth=1.2, label=f'FAIL > {high:g}')
            ax.legend(fontsize=7, loc='upper right')
        counts = {grade: results.count(grade) for grade in (*GRADES, 'n/a')}
        title = f"{label}\nPASS {counts['PASS']} · WARN {counts['WARN']} · FAIL {counts['FAIL']}"
        ax.set_title(title + (f" · n/a {counts['n/a']}" if counts['n/a'] else ''), fontsize=10)
        ax.grid(axis='x' if many else 'y', alpha=0.3)
    fig.suptitle(f'外参精度评估：{len(rows)} 个已评估录制（绿 PASS / 橙 WARN / 红 FAIL，虚线为 PASS 与 FAIL 阈值）',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def _f(value, digits=1):
    return '—' if value is None else f'{value:.{digits}f}'


def _limit(value):
    return '—' if value is None else f'{value:g}'


def _mark(check):
    result = (check or {}).get('result')
    return f' **{result}**' if result in ('WARN', 'FAIL') else ''


def _value(row, key):
    check = row['checks'][key]
    return _f(check.get('value'), CHECKS[key][2]) + _mark(check)


def _verdict_text(value):
    return VERDICT_TEXT.get(value, value or '—')


def _reasons(row, thresholds):
    reasons = []
    for key, (label, unit, digits) in CHECKS.items():
        check = row['checks'][key]
        if check.get('result') in ('WARN', 'FAIL'):
            low, high = thresholds.get(key, (None, None))
            reasons.append(f"{label} {_f(check['value'], digits)} {unit}（PASS ≤ {_limit(low)}，FAIL > {_limit(high)}）"
                           f"→ {check['result']}".replace(' （', '（'))
    return reasons + row['notes']


def write_markdown(path, result, rows, stats, thresholds, figure):
    ok = [row for row in rows if row['status'] == 'ok']
    failed = [row for row in rows if row['status'] == 'failed']
    evaluated = [row for row in ok if row['evaluated']]
    waiting = [row for row in ok if not row['evaluated']]
    verdicts = {grade: sum(row['verdict'] == grade for row in evaluated) for grade in GRADES}
    usable = sum(row['sample_usable'] is True for row in evaluated)
    e4_pass = _limit(thresholds.get('E4_mahalanobis2', (None,))[0])
    lines = []
    add = lines.append
    add('# 外参（相机安装位姿）精度评估报告')
    add('')
    add(f'结果目录：`{result}`  ')
    add(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')} · 评估器 `{EVALUATOR}`")
    add('')
    add('## 1. 结论')
    add('')
    add(f'- 录制 {len(rows)} 个：转换成功 {len(ok)}，转换失败 {len(failed)}；已评估外参 {len(evaluated)}，'
        f'未评估 {len(waiting)}。')
    if evaluated:
        add(f"- 外参判定：**PASS {verdicts['PASS']} · WARN {verdicts['WARN']} · FAIL {verdicts['FAIL']}**；"
            f'可用样本（外参 PASS/WARN 且至少一帧有效手）{usable} / {len(evaluated)}。')
        parts = []
        e1, mm = stats['E1_median_px'], [row['E1_median_mm'] for row in evaluated if row['E1_median_mm'] is not None]
        if e1['median'] is not None:
            parts.append(f"E1 留出重投影误差中位数 {e1['median']:.1f} px（各录制 {e1['min']:.1f}–{e1['max']:.1f} px"
                         + (f'，在手上约 {statistics.median(mm):.1f} mm' if mm else '') + '）')
        e2 = stats['E2_abs_bias_mm']
        if e2['median'] is not None:
            parts.append(f"E2 深度偏差绝对值中位数 {e2['median']:.1f} mm，最大 {e2['max']:.1f} mm")
        rotation, translation = stats['E3_rotation_max_deg'], stats['E3_translation_max_mm']
        if rotation['max'] is not None:
            parts.append(f"E3 折间差最大 {rotation['max']:.2f}° / {translation['max']:.1f} mm")
        e4 = stats['E4_mahalanobis2']
        if e4['max'] is not None:
            parts.append(f"E4 d² 最大 {e4['max']:.2f}（χ²(3) 的 99% 分位为 {e4_pass}）")
        if parts:
            add('- ' + '；'.join(parts) + '。')
    elif ok:
        add('- 没有已评估的录制：转换时加 `--eval_fit_ex true`，或对已有结果补做评估（见第 6 节）。')
    else:
        add('- 没有转换成功的录制，无法评估外参（失败原因见第 5 节）。')
    add('')
    if evaluated:
        add('| 检查项 | PASS ≤ | FAIL > | 中位数 | 最小 | 最大 | PASS / WARN / FAIL / 缺失 |')
        add('|---|---|---|---|---|---|---|')
        for key, (label, unit, digits) in CHECKS.items():
            item = stats[key]
            name = label.replace('|', r'\|') + (f' ({unit})' if unit else '')  # a | would split the cell
            add(f"| {name} | {_limit(item['pass_limit'])} | {_limit(item['fail_limit'])} | "
                f"{_f(item['median'], digits)} | {_f(item['min'], digits)} | {_f(item['max'], digits)} | "
                f"{item['PASS']} / {item['WARN']} / {item['FAIL']} / {item['n/a']} |")
        add('')
        if figure:
            add(f'![各录制的外参指标]({figure.name})')
            add('')
    add('## 2. 评估方法')
    add('')
    add('外参是相机相对头部动捕刚体的安装位姿 (R, t)，它决定 CMA 手部关节投到图像上的位置，以及 MANO 结果中的 '
        '`joints_cam`、`cam_R`、`cam_t`。四项检查使用的都是没有参与该外参拟合的证据：')
    add('')
    add('- **E1 留出重投影（主指标）**：把录制按时间分成 5 段，每次去掉 1 段及其前后 15 帧，用剩余帧重新运行完整的'
        '外参估计，再用得到的外参把被去掉那段的 CMA 关节投影到图像，与 MediaPipe 检测的 21 个手部关键点比较，'
        '报告中位数与 p90（px，并按手的深度换算为 mm）。测试帧没有参与拟合，因此它衡量的是外参的真实误差而不是拟合残差。'
        '外参为 FALLBACK（车队先验，与本录制无关）时，所有帧都是测试帧。')
    add('- **E2 深度一致性**：取最多 120 帧，把拟合好的 MANO 手表面按该外参渲染成深度图，在手部轮廓内（内缩 4 px）'
        '与 ZED 深度逐像素相减，每帧取带符号的中位数，再取各帧中位数的中位数作为偏差。重投影对沿视线方向的误差最不敏感，'
        'E2 补上这个方向。')
    add('- **E3 稳定性**：5 个留出外参与全量外参之间的最大旋转差和最大平移差；证据不足或病态时会变大。')
    add('- **E4 旋转一致性**：只用 MediaPipe 得到的旋转与独立参照旋转（ZED VIO 手眼标定；没有可用 VIO 时为车队先验）'
        f'之间的马氏距离 d²，外参正确时服从 χ²(3)，99% 分位为 {e4_pass}。')
    add('- **时间同步诊断**（不参与判定）：把 CMA 时间整体平移 −30…+30 ms（MediaPipe）/ −40…+40 ms（VIO），'
        '报告误差最小的偏移。')
    add('')
    add('判定：每项检查的值 ≤ PASS 阈值为 PASS，> FAIL 阈值为 FAIL，其间为 WARN，录制的判定取最差一项；'
        'FALLBACK 外参至少为 WARN；没有 MediaPipe 证据（没有 E1）判 FAIL；缺少某项证据时该项记为“缺失”，不参与判定。'
        '可用样本 = 外参 PASS 或 WARN 且至少一帧有有效的 MANO 手。')
    add('')
    add('阈值来源（`tools/perturbation_study.py`，20 个测试录制）：外参正确时 E1 中位数的下限约 7–12 px'
        '（MediaPipe 与 CMA 关键点的定义差异和检测噪声）；在正确外参上加 1° 俯仰/偏航、2° 横滚或 10 mm 横向平移，'
        '≥90% 的录制报警；不加扰动时误报 0%。详见 README.md 第 4 节。')
    add('')
    add('## 3. 逐录制结果')
    add('')
    if ok:
        add('| 录制 | 外参等级 | 旋转来源 | 判定 | E1 中位 px (mm) | E1 p90 px | E2 偏差 mm | E3 旋转° / 平移 mm | '
            'E4 d² | 同步 ms | 有效帧 |')
        add('|---|---|---|---|---|---|---|---|---|---|---|')
        for row in ok:
            if row['evaluated']:
                e2 = _f(row['E2_bias_mm']) + _mark(row['checks']['E2_abs_bias_mm'])
                add(f"| {row['recording']} | {row['extrinsic_status']} | "
                    f"{SOURCE_TEXT.get(row['rotation_source'], row['rotation_source'])} | **{row['verdict']}** | "
                    f"{_value(row, 'E1_median_px')} ({_f(row['E1_median_mm'])}) | {_value(row, 'E1_p90_px')} | {e2} | "
                    f"{_value(row, 'E3_rotation_max_deg')} / {_value(row, 'E3_translation_max_mm')} | "
                    f"{_value(row, 'E4_mahalanobis2')} | {_f(row['sync_offset_ms'], 0)} | "
                    f"{row['frames_any_hand_valid']}/{row['frames']} |")
            else:
                add(f"| {row['recording']} | {row['extrinsic_status']} | "
                    f"{SOURCE_TEXT.get(row['rotation_source'], row['rotation_source'])} | {_verdict_text(row['verdict'])} | "
                    f"— | — | — | — | — | — | {row['frames_any_hand_valid']}/{row['frames']} |")
        add('')
        add('外参等级：CALIBRATED = 在本录制上完整标定（旋转与 VIO 或车队先验融合）；VIO_ROTATION / PRIOR_ROTATION = '
            'MediaPipe 证据较少，旋转取 VIO / 车队先验、平移在本录制上拟合；FALLBACK = 直接使用车队先验外参。'
            'E2 偏差为正表示真实手表面比 MANO 表面更远。有效帧 = 至少一只手有有效 MANO 拟合的帧数 / 总帧数。')
    else:
        add('没有转换成功的录制。')
    add('')
    add('## 4. 需要关注的录制')
    add('')
    attention = [row for row in evaluated if row['verdict'] != 'PASS' or row['notes']]
    if not attention and not waiting:
        add('无：所有已评估录制均为 PASS，且没有附加提示。' if evaluated else '无。')
    for row in attention:
        add(f"- `{row['recording']}` **{row['verdict']}**：" + '；'.join(_reasons(row, thresholds)))
    for row in waiting:
        reason = f"：{row['error']}" if row['error'] else ''
        add(f"- `{row['recording']}` {_verdict_text(row['verdict'])}{reason}")
    if waiting:
        add('')
        add('未评估的录制可在不重新转换的情况下补做评估：`python mano_convert.py --data <输入> --output <本目录> '
            '--eval_fit_ex true --skip-done`。')
    add('')
    add('## 5. 转换失败的录制')
    add('')
    if failed:
        for row in failed:
            add(f"- `{row['recording']}`：{row['error']}")
    else:
        add('无。')
    add('')
    add('## 6. 文件说明')
    add('')
    add(f'- `<录制>/{EVAL_NAME}`：单个录制的评估结果（判定、各检查项的数值与阈值、摘要、完整指标）。')
    add(f'- `<录制>/{REPORT_NAME}` 中的 `metrics`：完整指标（逐折、逐关节、逐帧深度），`extrinsic_verdict`、'
        '`sample_usable`：判定与是否可用。')
    add(f'- `{SUMMARY_NAME}.csv` / `{SUMMARY_NAME}.json`：逐录制一行 / 统计与全部行。')
    if figure:
        add(f'- `{SUMMARY_NAME}.png`：各检查项（虚线为 PASS 与 FAIL 阈值）。')
    add('- `batch_summary.csv`：转换汇总（有效帧、MANO 拟合误差、是否有对比视频、耗时）。')
    add('- 补做评估：`python mano_convert.py --data <输入> --output <本目录> --eval_fit_ex true --skip-done`；'
        '只重写本报告：`python -m robust_convert.eval_extrinsic --result <本目录> --report-only`。')
    add('')
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


CSV_COLUMNS = ['recording', 'status', 'verdict', 'sample_usable', 'extrinsic_status', 'rotation_source', 'vio',
               'E1_median_px', 'E1_median_px_result', 'E1_median_mm', 'E1_p90_px', 'E1_p90_px_result', 'E1_points',
               'E2_bias_mm', 'E2_abs_bias_mm_result', 'E3_rotation_max_deg', 'E3_rotation_max_deg_result',
               'E3_translation_max_mm', 'E3_translation_max_mm_result', 'E4_mahalanobis2', 'E4_mahalanobis2_result',
               'E4_reference', 'E4_angle_deg', 'sync_offset_ms', 'frames', 'frames_any_hand_valid', 'notes', 'error']


def _csv_row(row):
    flat = {key: value for key, value in row.items() if key not in ('checks', 'notes')}
    for key in CHECKS:
        check = row['checks'].get(key) or {}
        if key != 'E2_abs_bias_mm':
            flat[key] = check.get('value')
        flat[f'{key}_result'] = check.get('result') if row['evaluated'] else None
    flat['notes'] = '; '.join(row['notes'])
    return flat


def write_report(result, figure_python=None):
    """外参评估报告.md and extrinsic_eval_summary.csv/.json/.png of every recording in `result`; returns the paths."""
    result = Path(result).resolve()
    rows, thresholds = collect_rows(result)
    stats = check_statistics(rows, thresholds)
    figure = result / f'{SUMMARY_NAME}.png'
    figure.unlink(missing_ok=True)
    if any(row['evaluated'] for row in rows):
        if importlib.util.find_spec('matplotlib') is not None:
            plot_summary(rows, thresholds, figure)
        elif figure_python is not None:
            subprocess.run([str(figure_python), '-m', 'robust_convert.eval_extrinsic', '--result', str(result),
                            '--figure-only'], capture_output=True, env={**os.environ, 'PYTHONPATH': str(ROOT)})
    ok = [row for row in rows if row['status'] == 'ok']
    evaluated = [row for row in ok if row['evaluated']]
    summary = {'evaluator': EVALUATOR, 'result': str(result), 'written': datetime.now().isoformat(timespec='seconds'),
               'recordings': len(rows), 'converted': len(ok), 'failed': len(rows) - len(ok),
               'evaluated': len(evaluated), 'not_evaluated': len(ok) - len(evaluated),
               'verdicts': {grade: sum(row['verdict'] == grade for row in evaluated) for grade in GRADES},
               'usable': sum(row['sample_usable'] is True for row in evaluated), 'checks': stats,
               'thresholds': thresholds, 'rows': rows}
    paths = {'markdown': result / MARKDOWN_NAME, 'csv': result / f'{SUMMARY_NAME}.csv',
             'json': result / f'{SUMMARY_NAME}.json'}
    _write_json(paths['json'], summary)
    with paths['csv'].open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(_csv_row(row) for row in rows)
    write_markdown(paths['markdown'], result, rows, stats, thresholds, figure if figure.is_file() else None)
    if figure.is_file():
        paths['figure'] = figure
    return summary, paths


# ---- command line -------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description='Extrinsic accuracy evaluation of converted recordings '
                                                 '(module docstring: python -m pydoc robust_convert.eval_extrinsic)')
    parser.add_argument('--data', type=Path, help='input given to mano_convert.py: a recording folder or a folder '
                                                  'with recordings')
    parser.add_argument('--result', type=Path, required=True, help='output folder of mano_convert.py')
    parser.add_argument('--only', nargs='+', metavar='NAME', help='only these recordings')
    parser.add_argument('--jobs', type=int, default=4, help='recordings evaluated in parallel')
    parser.add_argument('--redo', action='store_true', help='also recordings that already have an evaluation')
    parser.add_argument('--report-only', action='store_true', help='only write the batch report again')
    parser.add_argument('--mediapipe-cache', type=Path, help='keypoint cache (default: the one of the conversion)')
    parser.add_argument('--mediapipe-python', help='interpreter with mediapipe, only used when the keypoints of the '
                                                   'conversion are gone (default: the one of this machine; none: never)')
    parser.add_argument('--mano', type=Path, help='MANO model folder (default: the one of the conversion)')
    parser.add_argument('--no-report', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--figure-only', action='store_true', help=argparse.SUPPRESS)
    return parser


def figure_python():
    """Interpreter drawing the figure: None when this one has matplotlib (it draws in-process)."""
    if importlib.util.find_spec('matplotlib') is not None:
        return None
    return python_with(['matplotlib'], [MEDIAPIPE_PYTHON, TORCH_PYTHON])


def _report(result):
    summary, paths = write_report(result, figure_python())
    verdicts = summary['verdicts']
    print(f"evaluated {summary['evaluated']}/{summary['converted']} converted recordings: PASS {verdicts['PASS']}, "
          f"WARN {verdicts['WARN']}, FAIL {verdicts['FAIL']}, usable {summary['usable']}")
    for path in paths.values():
        print(f'  {path}')


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors='replace')
    parser = build_parser()
    args = parser.parse_args(argv)
    result = args.result.resolve()
    if not result.is_dir():
        parser.error(f'result folder not found: {result}')
    if args.figure_only:
        rows, thresholds = collect_rows(result)
        plot_summary(rows, thresholds, result / f'{SUMMARY_NAME}.png')
        return 0
    if args.report_only:
        _report(result)
        return 0
    if args.data is None:
        parser.error('--data is required, unless --report-only')
    from .batch import find_recordings, is_evaluated, run_parallel, worker_env, write_summary
    if not all(importlib.util.find_spec(module) for module in CONVERT_MODULES):
        python = python_with(CONVERT_MODULES, [TORCH_PYTHON])
        if python is None:
            raise SystemExit(f'this interpreter lacks {", ".join(CONVERT_MODULES)} and {TORCH_PYTHON} was not found')
        print(f'running with {python}', flush=True)
        return subprocess.call([str(python), '-m', 'robust_convert.eval_extrinsic', *(argv or sys.argv[1:])],
                               env={**worker_env(), 'PYTHONPATH': str(ROOT)})
    folders = find_recordings(args.data)
    if args.only:
        unknown = sorted(set(args.only) - {folder.name for folder in folders})
        if unknown:
            parser.error(f'not found in --data: {", ".join(unknown)}')
        folders = [folder for folder in folders if folder.name in args.only]
    todo, absent, done = [], [], []
    for folder in folders:
        target = result / folder.name
        if not (target / REPORT_NAME).is_file():
            absent.append(folder.name)
        elif is_evaluated(target) and not args.redo:
            done.append(folder.name)
        else:
            todo.append(folder)
    for names, text in ((absent, f'no conversion result in {result}'), (done, 'already evaluated (--redo: again)')):
        if names:
            print(f'skipped {len(names)} recordings, {text}: {", ".join(names[:5])}{" ..." if len(names) > 5 else ""}')
    common = []
    if args.mediapipe_cache:
        common += ['--mediapipe-cache', str(args.mediapipe_cache.resolve())]
    if args.mediapipe_python:
        common += ['--mediapipe-python', args.mediapipe_python]
    if args.mano:
        common += ['--mano', str(args.mano.resolve())]
    failures = 0
    if len(todo) > 1 and args.jobs > 1:
        commands = {folder.name: [sys.executable, '-m', 'robust_convert.eval_extrinsic', '--data',
                                  str(folder.resolve()), '--result', str(result), '--no-report', '--redo', *common]
                    for folder in todo}

        def progress(name, code, seconds, last, done, total):
            print(f'[{done}/{total}] {name} {"ok" if code == 0 else "FAILED"} {seconds:.0f} s  {last}', flush=True)

        outcome = run_parallel(commands, args.jobs, result / '_logs', suffix='_eval', on_done=progress)
        failures = sum(code != 0 for code, _, _ in outcome.values())
    else:
        from threadpoolctl import threadpool_limits
        import cv2
        threadpool_limits(1)
        cv2.setNumThreads(1)
        mp_python = args.mediapipe_python or str(MEDIAPIPE_PYTHON)
        for folder in todo:
            try:
                evaluate_existing(folder, result / folder.name, mano=args.mano, mediapipe_cache=args.mediapipe_cache,
                                  mp_python=None if mp_python.lower() == 'none' else mp_python)
            except Exception as error:  # noqa: BLE001 - one recording must not stop the others
                failures += 1
                write_eval_failure(result / folder.name, folder.name, error)
                print(f'FAILED: {folder.name}: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
    if not args.no_report:
        write_summary(result, this_run=[folder.name for folder in todo])
        _report(result)
    return 1 if todo and failures == len(todo) else 0


if __name__ == '__main__':
    sys.exit(main())
