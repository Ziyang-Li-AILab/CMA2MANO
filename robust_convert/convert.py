# -*- coding: utf-8 -*-
"""Robust conversion of one recording: CMA + MCAP -> MANO parameters, optional extrinsic evaluation and video.

The entry for users is mano_convert.py in the project root (one recording or a folder of recordings):
    python mano_convert.py --data <input> --output <output> [--gen_video true] [--eval_fit_ex true]
This module converts one recording into one folder; it is the worker process of mano_convert.py:
    python -m robust_convert.convert --data <recording folder> --output <new or empty folder>
                                     [--gen_video true] [--eval_fit_ex true]

What changes compared with the original run_pipeline.py, which aborts on the
first defect of any kind:
  * invalid RGB messages are skipped; CMA rows are read tolerantly, glitched
    hand joints are repaired by forward kinematics from the CMA rotation stream
    or marked invalid (session.py, cleaning.py); CMA is interpolated at the RGB
    stamps instead of taking the nearest row;
  * the mount is estimated from MediaPipe keypoints with its rotation fused with
    the ZED VIO hand-eye rotation (else with the fleet prior), and degrades to
    the VIO or fleet prior rotation (whichever fits better) with a fitted
    translation, then to the fleet prior mount, instead of failing (extrinsic.py);
  * MANO is fitted on the valid frames only; frames that break the original
    acceptance rules (pose outside the residual branch, wrist > 2.9 mm) are
    flagged per frame instead of discarding the recording;
  * --eval_fit_ex true: the mount accuracy is measured (eval_extrinsic.py: E1 held-out
    reprojection, E2 depth agreement, E3 fold stability, E4 rotation consistency) and
    graded PASS/WARN/FAIL; a failing evaluation never loses the conversion result.

Outputs in --output (file names as in the original):
  mano_fit.npz          every key of the original file (full length, NaN where no fit)
                        plus validity masks (see README)
  mano_fit_report.json  extrinsic, per-hand fit statistics, yield; with --eval_fit_ex the
                        metrics, the verdict and sample_usable (null without evaluation)
  extrinsic_eval.json   --eval_fit_ex true: the extrinsic evaluation of this recording
  comparison.mp4        --gen_video true: original layout, CMA skeleton projection | MANO mesh overlay
  failure.json          only when the conversion itself failed
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .batch import add_switch
from .cleaning import MEASURED, REPAIRED
from .eval_extrinsic import (EVAL_NAME, evaluate_mount, evaluation_failed, not_evaluated, sample_usable,
                             write_sample_eval)
from .extrinsic import DEFAULT_PRIOR, TEMPLATE_R, estimate_mount, fallback_mount, load_detections, load_prior
from .mano_fit import fit_hand_masked
from .mano_model import DEFAULT_MANO, TIP_INDEX, export_hand, initial_pose, load_mano, point_error_mm
from .mcap_io import read_vio_pose
from .session import SIDES, load_session
from .vio import handeye_rotation

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / 'outputs' / 'cache' / 'mediapipe'
MP_PYTHON = Path(r'C:\Users\11068\AppData\Local\Programs\Python\Python312\python.exe')
VERSION = 'robust_convert_v1.2'
REPORT_NAME = 'mano_fit_report.json'

# Per-frame flags of a hand (bit mask). A frame is valid when it was fitted and no flag is set.
FLAG_CMA_INVALID = 1       # no complete valid CMA hand at this frame: not fitted
FLAG_POSE_BRANCH = 2       # hand pose outside the residual branch (original abort rule)
FLAG_WRIST = 4             # MANO wrist more than 2.9 mm from the CMA wrist (original abort rule)
FLAG_FIT_ERROR = 8         # mean joint error above FIT_ERROR_LIMIT_MM
FLAG_NAMES = {FLAG_CMA_INVALID: 'cma_invalid', FLAG_POSE_BRANCH: 'pose_branch', FLAG_WRIST: 'wrist_gt_2.9mm',
              FLAG_FIT_ERROR: 'mean_error_gt_limit'}
WRIST_LIMIT_MM = 2.9
FIT_ERROR_LIMIT_MM = 15.0  # the original accepted fits average ~6 mm per joint (p95 ~12 mm)
MIN_FIT_FRAMES = 10

# threads: torch threads of the MANO fit (the float result depends on it; the delivered results used 3)
DEFAULTS = {'steps': 200, 'shape_prior': 0.25, 'palm_weight': 1.0, 'shape_lr': 0.05, 'threads': 3, 'workers': 3,
            'preview_width': 1920, 'repaired_weight': 0.5, 'gen_video': False, 'eval_fit_ex': False, 'sync': True,
            'depth_frames': 120}

_MODELS = {}


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def get_model(path, side):
    key = (str(path), side)
    if key not in _MODELS:
        _MODELS[key] = load_mano(path, side)
    return _MODELS[key]


# ---- inputs ------------------------------------------------------------------------------------

def ensure_detections(session, cache_dir=DEFAULT_CACHE, mp_python=MP_PYTHON):
    """MediaPipe keypoints of every usable RGB frame; computed with the MediaPipe interpreter when not cached.

    (None, info) when they cannot be had: the mount then falls back to the fleet prior.
    """
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{session['name']}.npz"
    info = {'path': str(path)}
    if not path.is_file():
        python = Path(mp_python) if mp_python else None
        if python is None or not python.is_file():
            return None, {**info, 'status': 'missing',
                          'reason': f'no cached keypoints and no MediaPipe interpreter ({python})'}
        started = time.perf_counter()
        command = [str(python), str(ROOT / 'tools' / 'mp_detect.py'), '--data', str(session['folder']),
                   '--out', str(cache_dir), '--workers', '1']
        try:
            done = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace',
                                  env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        except OSError as error:
            return None, {**info, 'status': 'failed', 'reason': f'cannot run {python}: {error}'}
        if done.returncode != 0 or not path.is_file():
            return None, {**info, 'status': 'failed', 'reason': (done.stderr or done.stdout)[-2000:]}
        info.update(status='computed', seconds=time.perf_counter() - started)
    else:
        info['status'] = 'cached'
    try:
        detections = load_detections(path, session['stamps'])
    except Exception as error:  # noqa: BLE001 - a damaged cache file only removes the MediaPipe evidence
        return None, {**info, 'status': 'failed', 'reason': f'unreadable keypoint file: {type(error).__name__}: {error}'}
    info.update(frames_matched=detections['frames_matched'], frames=int(len(session['stamps'])))
    return detections, info


def handeye_safe(session):
    """VIO hand-eye rotation; a broken or missing VIO stream only disables the VIO reference."""
    try:
        vio = read_vio_pose(session['mcap'])
        return vio, handeye_rotation(session, vio, TEMPLATE_R)
    except Exception as error:  # noqa: BLE001 - VIO is optional evidence
        return None, {'status': 'no_vio', 'error': f'{type(error).__name__}: {error}'}


# ---- extrinsic -----------------------------------------------------------------------------------

def solve_extrinsic(session, detections, handeye, prior):
    """Mount (R, t mm), report and the fit object (None for a fallback)."""
    if detections is not None:
        return estimate_mount(session, detections, handeye, prior)
    R, t, info = fallback_mount(handeye, prior)
    return R, t, {'method': 'mediapipe_pnp_fusion_v1', **info, 'reason': 'no MediaPipe detections'}, None


# ---- MANO ------------------------------------------------------------------------------------

def camera_transform(session, mount_R, mount_t):
    """Per-frame world -> camera transform in metres, as the original export (NaN where the rig is invalid)."""
    rig_R = session['rig_R']
    cam_R = np.asarray(mount_R) @ np.swapaxes(rig_R, -1, -2)
    cam_t = np.asarray(mount_t) / 1000 - np.einsum('tij,tj->ti', cam_R, session['origin_mm'] / 1000)
    cam_R[~session['rig_ok']] = np.nan
    cam_t[~session['rig_ok']] = np.nan
    return cam_R, cam_t


def fit_side(session, side, model, mount_R, mount_t, config):
    """MANO fit of one hand on its valid frames; returns (arrays, mesh, report)."""
    started = time.perf_counter()
    hand = session['hands'][side]
    source = session['hands_source'][side]
    count = len(session['stamps'])
    if not hand['present']:
        return None, None, {'fitted': False, 'reason': source.get('reason', 'hand absent from the CMA file')}
    frames = np.flatnonzero(hand['frame_valid'])
    base = {'frames_total': count, 'frames_cma_valid': int(len(frames)), 'source_schema': source['schema'],
            'cleaning': source['cleaning'], 'source_defects': source['defects']}
    if len(frames) < MIN_FIT_FRAMES:
        return None, None, {**base, 'fitted': False,
                            'reason': f'only {len(frames)} frames with a complete valid CMA hand'}
    points = hand['points_mm'][frames]
    status = hand['status'][frames]
    weights = np.where(status == MEASURED, 1.0, config['repaired_weight'])
    init = initial_pose(model, points / 1000, hand['world_R'][frames])
    fit = fit_hand_masked(model, points, init, weights, steps=config['steps'], shape_prior=config['shape_prior'],
                          palm_weight=config['palm_weight'], threads=config['threads'], shape_lr=config['shape_lr'])
    saved = export_hand(model, fit, points[:, 0] / 1000, session['origin_mm'][frames], session['rig_R'][frames],
                        mount_R, mount_t)
    errors = point_error_mm(saved['joints_world'].astype(np.float64), points)
    branch = np.linalg.norm(fit['hp'].reshape(len(frames), 15, 3), axis=-1).max(1) >= np.pi - 1e-4
    flags = np.zeros(count, np.uint8)
    flags[~hand['frame_valid']] |= FLAG_CMA_INVALID
    flags[frames[branch]] |= FLAG_POSE_BRANCH
    flags[frames[errors[:, 0] > WRIST_LIMIT_MM]] |= FLAG_WRIST
    flags[frames[errors.mean(1) > FIT_ERROR_LIMIT_MM]] |= FLAG_FIT_ERROR
    fitted = np.zeros(count, bool)
    fitted[frames] = True
    valid = fitted & (flags == 0)

    def full(values):
        out = np.full((count,) + values.shape[1:], np.nan, np.float32)
        out[frames] = values
        return out

    cam_R, cam_t = camera_transform(session, mount_R, mount_t)
    cma_cam = np.einsum('tij,tkj->tki', cam_R, hand['points_mm'] / 1000) + cam_t[:, None]
    arrays = {'beta': saved['beta'], 'global_orient': full(saved['global_orient']),
              'hand_pose': full(saved['hand_pose']), 'trans': full(saved['trans']),
              'joints_cam': full(saved['joints_cam']), 'joints_world': full(saved['joints_world']),
              'joints_cam_cma': cma_cam.astype(np.float32), 'cam_R': cam_R.astype(np.float32),
              'cam_t': cam_t.astype(np.float32), 'error_mm': full(errors),
              'fitted': fitted, 'valid': valid, 'frame_valid': hand['frame_valid'].copy(),
              'joint_status': hand['status'].astype(np.int8), 'frame_flags': flags}
    mesh = {'vertices_mm': full(saved['vertices_mm']), 'faces': saved['faces'], 'fitted': fitted, 'valid': valid}
    ok = valid[frames]
    chosen = errors[ok] if ok.any() else errors[:0]
    world = saved['joints_world'].astype(np.float64)
    source_width = np.linalg.norm(points[:, 5] - points[:, 17], axis=-1)
    model_width = np.linalg.norm(world[:, 5] - world[:, 17], axis=-1) * 1000

    def stats(e):
        if not len(e):
            return {}
        return {'all_points_mm_mean': float(e.mean()), 'all_points_mm_p95': float(np.percentile(e, 95)),
                'fingertip_mm_mean': float(e[:, TIP_INDEX].mean()),
                'finger_bases_mm_mean': float(e[:, [1, 5, 9, 13, 17]].mean()),
                'mcp4_mm_mean': float(e[:, [5, 9, 13, 17]].mean()),
                'wrist_mm_max': float(e[:, 0].max()), 'wrist_violations': int((e[:, 0] > WRIST_LIMIT_MM).sum()),
                'per_joint_mean_mm': e.mean(0).tolist()}

    report = {**base, 'fitted': True, 'frames_fitted': int(len(frames)), 'frames_valid': int(valid.sum()),
              'valid_fraction': float(valid.mean()),
              'flag_counts': {name: int(((flags & bit) > 0).sum()) for bit, name in FLAG_NAMES.items()},
              'repaired_joint_fraction_in_fit': float((status == REPAIRED).mean()),
              'valid_frames': stats(chosen), 'all_fitted_frames': stats(errors),
              'frame_mean_error_mm_p50_p95_max': np.percentile(errors.mean(1), [50, 95, 100]).tolist(),
              'source_palm_width_mm': float(source_width.mean()), 'mano_palm_width_mm': float(model_width.mean()),
              'beta': saved['beta'].tolist(), 'objective_optimizer_float32': fit['total'],
              'loss_terms': fit['loss_terms'], 'best_step': fit['best_step'], 'history': fit['history'],
              'solver': fit['solver'], 'model_sha256': model['metadata']['pkl_sha256'],
              'mirror_applied': model['metadata']['mirror_applied'], 'init_reference_frame': int(frames[init['reference_frame']]),
              'seconds': time.perf_counter() - started}
    return arrays, mesh, report


def fit_side_safe(session, side, mano, mount_R, mount_t, config):
    """fit_side that never raises: one hand failing must not lose the other."""
    try:
        return fit_side(session, side, get_model(mano, side), mount_R, mount_t, config)
    except Exception as error:  # noqa: BLE001
        return None, None, {'fitted': False, 'reason': f'{type(error).__name__}: {error}',
                            'traceback': traceback.format_exc()}


def fit_hands(session, mano, mount_R, mount_t, config):
    """Both hands, fitted concurrently in two threads (torch releases the GIL; results do not depend on it)."""
    with ThreadPoolExecutor(len(SIDES)) as pool:
        futures = [pool.submit(fit_side_safe, session, side, mano, mount_R, mount_t, config) for side in SIDES]
        return dict(zip(SIDES, (future.result() for future in futures)))


# ---- one recording ---------------------------------------------------------------------------

def _prepare_output(output, overwrite):
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise SystemExit(f'{output} is not empty; use --overwrite to replace its results')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('mano_fit.npz', REPORT_NAME, EVAL_NAME, 'comparison.mp4', 'failure.json'):
        (output / name).unlink(missing_ok=True)
    return output


def convert(folder, output, *, mano=DEFAULT_MANO, cache_dir=DEFAULT_CACHE, mp_python=MP_PYTHON,
            prior_path=DEFAULT_PRIOR, overwrite=False, log=print, **options):
    """Convert one recording; returns the report. Raises (after writing failure.json) on a fatal error.

    options: DEFAULTS keys, among them gen_video (comparison.mp4) and eval_fit_ex (extrinsic evaluation).
    """
    config = {**DEFAULTS, **options}
    started = time.perf_counter()
    output = _prepare_output(output, overwrite)
    stage_name, timings = 'load', {}

    def stage(name):
        nonlocal stage_name, stage_start
        now = time.perf_counter()
        timings[f'{stage_name}_seconds'] = now - stage_start
        stage_name, stage_start = name, now

    stage_start = started
    try:
        session = load_session(folder)
        name = session['name']
        count = len(session['stamps'])
        log(f'[{name}] {count} usable RGB frames of {len(session["rgb"]["stamps"])}')
        stage('detections')
        detections, detection_info = ensure_detections(session, cache_dir, mp_python)
        if detections is None:
            log(f"[{name}] no MediaPipe keypoints ({detection_info['status']}): {detection_info.get('reason', '')}"[:400])
        stage('vio')
        vio, handeye = handeye_safe(session)
        prior = load_prior(prior_path)
        stage('extrinsic')
        mount_R, mount_t, ext_report, fit = solve_extrinsic(session, detections, handeye, prior)
        mount_R, mount_t = np.asarray(mount_R, np.float64), np.asarray(mount_t, np.float64)
        log(f"[{name}] mount {ext_report['status']} ({ext_report.get('rotation_source')})")
        stage('mano')
        arrays = {'K': session['K'], 'timestamps_ns': session['stamps'], 'source_indices': session['source_indices'],
                  'cma_row_indices': session['cma_rows'], 'cma_delta_ns': session['cma_delta_ns'],
                  'rig_ok': session['rig_ok'], 'mount_R': mount_R, 'mount_t_mm': mount_t,
                  'parameter_coordinate_frame': np.array('world'),
                  'hp_convention': np.array('smplx_residual_flat_hand_mean_false'),
                  'frame_flag_bits': np.array(json.dumps({str(bit): n for bit, n in FLAG_NAMES.items()}))}
        exists = np.zeros((count, 2), bool)
        meshes, hands = {}, {}
        fitted = fit_hands(session, mano, mount_R, mount_t, config)
        for h, side in enumerate(SIDES):
            side_arrays, mesh, hands[side] = fitted[side]
            if side_arrays is None:
                continue
            exists[:, h] = side_arrays['valid']
            meshes[side] = mesh
            for key, value in side_arrays.items():
                arrays[f'{side}_{key}'] = value
            report_hand = hands[side]
            log(f"[{name}] {side}: {report_hand['frames_valid']}/{count} valid frames, "
                f"mean {report_hand['valid_frames'].get('all_points_mm_mean', float('nan')):.2f} mm")
        arrays['exists'] = exists
        valid_any = exists.any(1)
        if config['eval_fit_ex']:
            stage('evaluation')
            try:
                metrics = evaluate_mount(session, detections, vio, handeye, prior, mount_R, mount_t, ext_report, fit,
                                         meshes, exists, depth_frames=config['depth_frames'], sync=config['sync'])
            except Exception as error:  # noqa: BLE001 - the conversion result is kept, without a grade
                metrics = evaluation_failed(error)
                log(f'[{name}] extrinsic evaluation failed: {type(error).__name__}: {error}')
        else:
            metrics = not_evaluated()
        stage('save')
        grade = metrics['verdict']['verdict']
        present = [h for h, side in enumerate(SIDES) if session['hands'][side]['present']]
        yield_info = {'rgb_messages': int(len(session['rgb']['stamps'])), 'frames': count,
                      'frames_any_hand_valid': int(valid_any.sum()),
                      'frames_all_present_hands_valid': int(exists[:, present].all(1).sum()) if present else 0,
                      'hands': {side: hands[side].get('valid_fraction', 0.0) for side in SIDES}}
        fusion = fit.get('fusion') if fit is not None and ext_report['status'] != 'FALLBACK' else None
        report = {'converter': VERSION, 'solver': 'batched_torch_adam_palm_v1_masked', 'loss': 'l2_with_palm_pairs_v1',
                  'recording_name': name, 'recording': str(session['mcap']),
                  'recording_bytes': session['mcap'].stat().st_size, 'K': session['K'], 'n_frames': count,
                  'source_resolution': [session['width'], session['height']],
                  'parameter_coordinate_frame': 'world', 'joints_cam_coordinate_frame': 'main_left_rectified_camera',
                  'sample_usable': sample_usable(grade, valid_any.any()), 'extrinsic_verdict': grade,
                  'yield': yield_info,
                  'extrinsic': {**ext_report, 'mount_R': mount_R, 'mount_t_mm': mount_t,
                                'fusion': fusion, 'handeye': handeye,
                                'detections': detection_info,
                                'prior': None if prior is None else {k: prior[k] for k in ('source', 'path', 'recordings')}},
                  'metrics': metrics,
                  'hands': hands, 'input_quality': {'malformed_rows': session['malformed_rows'],
                                                    'rgb_invalid_messages': int((~session['rgb']['valid']).sum()),
                                                    'rig_invalid_frames': int((~session['rig_ok']).sum())},
                  'config': {**config, 'mano': str(mano), 'fit_error_limit_mm': FIT_ERROR_LIMIT_MM,
                             'wrist_limit_mm': WRIST_LIMIT_MM}, 'timings': timings}
        np.savez(output / 'mano_fit.npz', **arrays)
        if config['gen_video']:
            stage('video')
            try:
                from .render import write_comparison
                report['comparison_video'] = write_comparison(session, mount_R, mount_t, meshes,
                                                              output / 'comparison.mp4', workers=config['workers'],
                                                              preview_width=config['preview_width'])
            except Exception as error:  # noqa: BLE001 - the MANO result is kept without its video
                (output / 'comparison.mp4').unlink(missing_ok=True)
                report['comparison_video'] = {'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                                              'traceback': traceback.format_exc()}
                log(f'[{name}] comparison video failed: {type(error).__name__}: {error}')
        stage('done')
        timings['total_seconds'] = time.perf_counter() - started
        report = jsonable(report)
        if config['eval_fit_ex']:
            write_sample_eval(output, report)
        (output / REPORT_NAME).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding='utf-8')
        e1 = (metrics.get('E1_heldout_reprojection') or {}).get('median_px')
        log(f"[{name}] {int(valid_any.sum())}/{count} frames with a valid hand, extrinsic {grade}"
            + (f', E1 {e1:.1f} px' if e1 is not None else '') + f", {timings['total_seconds']:.0f} s")
        return report
    except (Exception, SystemExit) as error:
        failure = {'status': 'failed', 'recording_input': str(folder), 'stage': stage_name,
                   'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc(),
                   'seconds_before_failure': time.perf_counter() - started}
        (output / 'failure.json').write_text(json.dumps(failure, indent=2, ensure_ascii=False), encoding='utf-8')
        raise


def build_parser():
    parser = argparse.ArgumentParser(description='Convert one recording (the entry for folders of recordings is '
                                                 'mano_convert.py)')
    parser.add_argument('--data', type=Path, required=True, help='recording folder (*.mcap, *_Human.cma, *_Body.cma)')
    parser.add_argument('--output', type=Path, required=True, help='result folder of this recording')
    add_switch(parser, 'gen_video', 'write comparison.mp4 (left CMA projection, right MANO overlay)')
    add_switch(parser, 'eval_fit_ex', 'evaluate the extrinsic (mount) accuracy: extrinsic_eval.json and the '
                                      'metrics of mano_fit_report.json')
    parser.add_argument('--mano', type=Path, default=DEFAULT_MANO)
    parser.add_argument('--mediapipe-cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--mediapipe-python', default=str(MP_PYTHON),
                        help='interpreter with mediapipe, used when the keypoint cache is missing (none: never)')
    parser.add_argument('--prior', type=Path, default=DEFAULT_PRIOR, help='fleet mount prior (tools/build_fleet_prior.py)')
    parser.add_argument('--steps', type=int, default=DEFAULTS['steps'])
    parser.add_argument('--shape-prior', type=float, default=DEFAULTS['shape_prior'])
    parser.add_argument('--palm-weight', type=float, default=DEFAULTS['palm_weight'])
    parser.add_argument('--shape-lr', type=float, default=DEFAULTS['shape_lr'])
    parser.add_argument('--repaired-weight', type=float, default=DEFAULTS['repaired_weight'],
                        help='MANO data weight of FK-repaired joints (measured joints: 1)')
    parser.add_argument('--threads', type=int, default=DEFAULTS['threads'], help='torch threads of the MANO fit')
    parser.add_argument('--workers', type=int, default=DEFAULTS['workers'], help='video render threads')
    parser.add_argument('--preview-width', type=int, default=DEFAULTS['preview_width'])
    parser.add_argument('--depth-frames', type=int, default=DEFAULTS['depth_frames'], help='frames of the E2 check')
    parser.add_argument('--no-sync', action='store_true', help='skip the time-sync diagnostic of the evaluation')
    parser.add_argument('--overwrite', action='store_true')
    return parser


def options_from_args(args):
    return {'steps': args.steps, 'shape_prior': args.shape_prior, 'palm_weight': args.palm_weight,
            'shape_lr': args.shape_lr, 'repaired_weight': args.repaired_weight, 'threads': args.threads,
            'workers': args.workers, 'preview_width': args.preview_width, 'depth_frames': args.depth_frames,
            'gen_video': args.gen_video, 'eval_fit_ex': args.eval_fit_ex, 'sync': not args.no_sync}


def main():
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors='replace')
    args = build_parser().parse_args()
    from threadpoolctl import threadpool_limits
    import cv2
    threadpool_limits(1)
    cv2.setNumThreads(1)
    mp_python = None if args.mediapipe_python.strip().lower() in ('', 'none') else args.mediapipe_python
    try:
        convert(args.data, args.output, mano=args.mano, cache_dir=args.mediapipe_cache, mp_python=mp_python,
                prior_path=args.prior, overwrite=args.overwrite, **options_from_args(args))
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001
        print(f'FAILED: {type(error).__name__}: {error}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
