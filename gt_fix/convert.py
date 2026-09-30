# -*- coding: utf-8 -*-
"""convert_last_version：CMA -> MANO GT 的修复版转换（逐录制）。

用法（ace-ego-hand 解释器，在 convert_last_version 目录下）：
    python -m gt_fix.convert [--only 20260820_125531 ...] [--workers 4] [--out results]

每条录制：
  1. 读原始录制（robust_convert.session：与旧流程相同的清洗 / 插值 / 帧有效性）；相机外参沿用旧交付结果
     （outputs/unified_20/<rec>/mano_fit.npz 的 mount_R / mount_t_mm，旧流程已做过 MediaPipe 融合与评估，这里不重估）；
  2. 每只手：被试共享 beta（shape.py）-> 手套旋转无参考帧标定（calibrate.py）-> 固定 beta 重拟合（fit.py）；
  3. 标签层（labels.py）：逐帧状态、短空洞插值补全、时间分段、训练片段、ACE 相机系数组与可见性；
  4. 自检：ACE 参数做前向运动学复现 ace_joints_cam、残差轴角 < 2.8、所有 ace_* 数组有限。
输出 <out>/<rec>/mano_fit.npz + mano_fit_report.json，以及 <out>/summary.json。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gt_fix import labels as L  # noqa: E402
from gt_fix.anatomy import bone_dirs, get_anatomy, rest_points  # noqa: E402
from gt_fix.calibrate import calibrate_hand, pose_from_calibration, resample_world_R, source_offsets  # noqa: E402
from gt_fix.fit import adjacency, refit_hand  # noqa: E402
from gt_fix.shape import beta_for, subject_betas  # noqa: E402
from robust_convert.convert import camera_transform, jsonable  # noqa: E402
from robust_convert.mano_model import mano_forward  # noqa: E402
from robust_convert.session import load_session  # noqa: E402

VERSION = 'convert_last_version_v2'
RAW = Path(r'C:\Users\11068\Desktop\项目材料\datasets_20\to_sftp')
OLD = Path(r'C:\Users\11068\Desktop\mano线推进\outputs\unified_20')
RESULTS = ROOT / 'results'
REPORT_NAME = 'mano_fit_report.json'
SIDES = ('left', 'right')
WRIST_LIMIT_MM = 2.9
FIT_ERROR_LIMIT_MM = 15.0
FIT_ERROR_WEIGHT = 0.5         # 平均关节误差 > 15 mm 的拟合帧：手在（exists=1），但建议权重降到 0.5
RISKY_RESIDUAL_RAD = 2.8
MIN_FIT_FRAMES = 10
FK_CHUNK = 256
ACE_CONVENTION = {
    'frame': 'main_left_rectified_camera (x right, y down, z forward), metres',
    'slots': ['left', 'right'],
    'ace_go': 'global orient axis-angle in the camera frame (T,2,3)',
    'ace_hp': 'hand pose residual axis-angle (T,2,45); physical = hp + hands_mean (smplx flat_hand_mean=False)',
    'ace_trans': 'camera-frame translation (T,2,3); joints_cam = FK(go, hp, beta) + trans, rest wrist at J0 + trans',
    'ace_betas': 'shape (2,10), one shared beta per subject (both hands, all recordings)',
    'ace_joints_cam': 'OpenPose-21 joints (MANO 16 + 5 tip vertices) in the camera frame (T,2,21,3)',
    'ace_joints2d': 'projection normalised by the source resolution (T,2,21,2)',
    'placeholders': 'all ace_* values are 0 where ace_exists == 0 (never NaN)',
    'exists_target': '1 hand exists (fitted / filled); -1 unknown, ignore in presence losses; 0 never used here',
    'training': 'cut each recording by clip_ranges ([start, stop), >= 81 frames) before building ACE clips',
}


def fk_joints(model, beta, go, hp, trans, chunk=FK_CHUNK):
    """分块前向运动学，只返回 (n,21,3) 米关节（避免一次分配 n x 778 x 4 x 4）。"""
    out = np.zeros((len(go), 21, 3))
    for a in range(0, len(go), chunk):
        out[a:a + chunk] = mano_forward(model, beta, go[a:a + chunk], hp[a:a + chunk], trans[a:a + chunk])[1]
    return out


def error_stats(e):
    """(n,21) mm 误差的摘要。"""
    if not len(e):
        return {}
    frame = e.mean(1)
    return {'mean_mm': float(e.mean()), 'p95_mm': float(np.percentile(e, 95)),
            'tips_mean_mm': float(e[:, [4, 8, 12, 16, 20]].mean()), 'mcp_mean_mm': float(e[:, [5, 9, 13, 17]].mean()),
            'wrist_max_mm': float(e[:, 0].max()), 'per_joint_mean_mm': e.mean(0).round(2).tolist(),
            'frame_mean_p50_p95_max_mm': np.percentile(frame, [50, 95, 100]).round(2).tolist()}


# ---- 一只手 -------------------------------------------------------------------------------------------

def convert_hand(session, side, beta, cam_R, cam_t, segment_id, camera_ok, fit_options=None):
    """一只手：拟合 + 状态 + 补全 + 相机系参数。返回 (逐帧数组 dict, 报告 dict)。"""
    started = time.perf_counter()
    stamps = session['stamps']
    count = len(stamps)
    source = session['hands_source'][side]
    hand = session['hands'][side]
    if not (source['present'] and hand['present']):
        state, events = L.classify(np.zeros(count, bool), False, segment_id)
        return {'state': state, 'present': False}, {'present': False, 'events': events,
                                                     'reason': source.get('reason', 'hand absent from the CMA file')}
    anat = get_anatomy(side)
    model = anat.model
    mean = anat.hands_mean.reshape(15, 3)
    valid = np.asarray(hand['frame_valid'], bool)
    frames = np.flatnonzero(valid)
    if len(frames) < MIN_FIT_FRAMES:        # 本批数据没有这种手；整只手记 UNKNOWN
        state, events = L.classify(np.zeros(count, bool), True, segment_id)
        return {'state': state, 'present': True}, {'present': True, 'events': events,
                                                    'reason': f'only {len(frames)} valid CMA frames'}

    # 标定 + 重拟合（只在 CMA 有效帧上）
    points = hand['points_mm'][valid]
    offsets, _, _ = source_offsets(source)
    world_R = resample_world_R(session['cma_ns'], source['cma_status'], source['cma_world_R'], stamps)
    rest21 = rest_points(model, beta)
    C, calibration = calibrate_hand(side, anat, offsets, world_R[valid], points, rest21=rest21)
    init = pose_from_calibration(C, world_R[valid], anat)
    fit = refit_hand(anat, beta, points, hand['status'][valid], init, adjacency(frames, stamps), **(fit_options or {}))
    rest0 = mano_forward(model, beta, np.zeros(3), np.zeros(45), np.zeros(3))[3][0]

    # 整段数组（世界系），再按状态补全
    go_w = np.zeros((count, 3))
    physical = np.zeros((count, 15, 3))
    wrist = np.zeros((count, 3))
    go_w[frames] = fit['go']
    physical[frames] = fit['hp'].reshape(-1, 15, 3) + mean
    wrist[frames] = fit['trans'] + rest0
    state, events = L.classify(valid, True, segment_id, camera_ok)
    L.fill_holes(go_w, physical, wrist, stamps, state)
    exists = (state == L.FITTED) | (state == L.FILLED)
    ex = np.flatnonzero(exists)

    hp = np.zeros((count, 45))
    trans_w = np.zeros((count, 3))
    joints_w = np.zeros((count, 21, 3))
    hp[ex] = (physical[ex] - mean).reshape(-1, 45)
    trans_w[ex] = wrist[ex] - rest0
    joints_w[ex] = fk_joints(model, beta, go_w[ex], hp[ex], trans_w[ex])
    refit_consistency_mm = float(np.abs(joints_w[frames] - fit['joints_world_m']).max() * 1000)

    # ACE 相机系：参数先按 float32 存，关节由 float32 参数再做一次 FK，保证 FK(ace 参数) == ace_joints_cam
    go_c = np.zeros((count, 3))
    trans_c = np.zeros((count, 3))
    go_c[ex], trans_c[ex] = L.to_camera(go_w[ex], trans_w[ex], rest0, cam_R[ex], cam_t[ex])
    go_c32, hp32, trans_c32, beta32 = (a.astype(np.float32) for a in (go_c, hp, trans_c, beta))
    joints_c = np.zeros((count, 21, 3))
    joints_c[ex] = fk_joints(model, beta32.astype(np.float64), go_c32[ex].astype(np.float64),
                             hp32[ex].astype(np.float64), trans_c32[ex].astype(np.float64))
    world_to_cam = np.einsum('tij,tkj->tki', cam_R[ex], joints_w[ex]) + cam_t[ex][:, None]
    camera_consistency_mm = float(np.abs(joints_c[ex] - world_to_cam).max() * 1000) if len(ex) else 0.0

    # 逐帧标志
    error = np.full((count, 21), np.nan)
    error[frames] = fit['error_mm']
    frame_error = fit['error_mm'].mean(1)
    residual = np.linalg.norm(hp.reshape(count, 15, 3), axis=-1).max(1)
    flags = np.zeros(count, np.uint8)
    flags[~valid] |= L.FLAG_CMA_INVALID
    flags[exists & (residual >= np.pi - 1e-4)] |= L.FLAG_POSE_BRANCH
    flags[frames[fit['error_mm'][:, 0] > WRIST_LIMIT_MM]] |= L.FLAG_WRIST
    flags[frames[frame_error > FIT_ERROR_LIMIT_MM]] |= L.FLAG_FIT_ERROR
    flags[state == L.FILLED] |= L.FLAG_FILLED
    flags[state == L.UNKNOWN] |= L.FLAG_UNKNOWN

    # 补全的留出实验：在完整拟合段里挖掉 L 帧再插值，和真实拟合比
    def joints_fn(go, phys, wr):
        return fk_joints(model, beta, go, (phys - mean).reshape(len(go), 45), wr - rest0)
    holdout = L.fill_holdout(state == L.FITTED, segment_id, go_w, physical, wrist, stamps, joints_fn, joints_w)

    twist = np.abs(anat.twist(physical[ex], bone_dirs(rest21)))
    fitted_rows = state[ex] == L.FITTED
    cma_cam = np.einsum('tij,tkj->tki', cam_R, hand['points_mm'] / 1000) + cam_t[:, None]
    arrays = {
        'state': state, 'present': True, 'flags': flags, 'go_cam': go_c32, 'hp': hp32, 'trans_cam': trans_c32,
        'joints_cam': joints_c.astype(np.float32), 'beta': beta32,
        # 旧版键（世界系，NaN = 无 GT）：保持与 robust_convert 交付结果相同的含义，ACE 训练请用 ace_*
        'legacy': {'beta': beta32, 'global_orient': go_w, 'hand_pose': hp, 'trans': trans_w, 'joints_world': joints_w,
                   'joints_cam': joints_c, 'joints_cam_cma': cma_cam, 'cam_R': cam_R, 'cam_t': cam_t,
                   'error_mm': error, 'fitted': valid.copy(), 'frame_valid': valid.copy(),
                   'joint_status': hand['status'].astype(np.int8), 'frame_flags': flags},
    }
    report = {
        'present': True, 'frames_total': count, 'frames_cma_valid': int(len(frames)),
        'states': {name: int((state == code).sum()) for code, name in L.STATE_NAMES.items()},
        'events': events,
        'flag_counts': {name: int(((flags & bit) > 0).sum()) for bit, name in L.FLAG_NAMES.items()},
        'fit_vs_cma': error_stats(fit['error_mm']),
        'twist_p95_deg': {'fitted': np.percentile(twist[fitted_rows], 95, 0).round(1).tolist(),
                          'filled': (np.percentile(twist[~fitted_rows], 95, 0).round(1).tolist()
                                     if (~fitted_rows).any() else None),
                          'mano_reference': np.asarray(anat.twist_ref_p95_deg).round(1).tolist()},
        'residual_max_rad': float(residual[ex].max()),
        'residual_risky_frames': int((exists & (residual > RISKY_RESIDUAL_RAD)).sum()),
        'wrist_offset_mm_max': float(np.linalg.norm(fit['delta_mm'], axis=-1).max()),
        'calibration': calibration, 'loss_terms': fit['loss_terms'], 'history': fit['history'],
        'fit_config': fit['config'], 'fill_holdout_mm': holdout,
        'checks': {'refit_fk_consistency_mm': refit_consistency_mm, 'camera_fk_consistency_mm': camera_consistency_mm},
        'seconds': time.perf_counter() - started,
    }
    return arrays, report


# ---- 一条录制 -----------------------------------------------------------------------------------------

def convert_recording(name, out_root, betas, fit_options=None):
    started = time.perf_counter()
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / 'failure.json').unlink(missing_ok=True)
    session = load_session(RAW / name)
    stamps = session['stamps']
    count = len(stamps)
    old = np.load(OLD / name / 'mano_fit.npz')
    old_report = json.loads((OLD / name / REPORT_NAME).read_text(encoding='utf-8'))
    if not np.array_equal(old['timestamps_ns'].astype(np.int64), stamps.astype(np.int64)):
        raise RuntimeError(f'{name}: session stamps differ from the delivered npz')
    K = old['K'].astype(np.float64)
    width, height = int(session['width']), int(session['height'])
    mount_R = old['mount_R'].astype(np.float64)
    mount_t = old['mount_t_mm'].astype(np.float64)
    cam_R, cam_t = camera_transform(session, mount_R, mount_t)
    camera_ok = np.isfinite(cam_R).all((1, 2)) & np.isfinite(cam_t).all(1)
    cam_check = {side: float(np.nanmax(np.abs(cam_R - old[f'{side}_cam_R'].astype(np.float64))))
                 for side in SIDES if f'{side}_cam_R' in old}
    segment_id, gap_after, nominal = L.time_segments(stamps)

    hands, reports, subjects = {}, {}, {}
    for side in SIDES:
        beta, subjects[side] = beta_for(betas, name, side)
        hands[side], reports[side] = convert_hand(session, side, beta, cam_R, cam_t, segment_id, camera_ok,
                                                  fit_options)
        reports[side]['subject'] = subjects[side]
        reports[side]['beta'] = beta.tolist()
        if f'{side}_beta' in old:
            reports[side]['old_beta'] = old[f'{side}_beta'].astype(np.float64).tolist()

    states = np.stack([hands[s]['state'] for s in SIDES], 1)
    present = np.array([hands[s]['present'] for s in SIDES])
    arrays = {'K': K, 'timestamps_ns': stamps, 'source_indices': session['source_indices'],
              'cma_row_indices': session['cma_rows'], 'cma_delta_ns': session['cma_delta_ns'],
              'rig_ok': session['rig_ok'], 'mount_R': mount_R, 'mount_t_mm': mount_t,
              'source_resolution': np.array([width, height]),
              'converter': np.array(VERSION), 'ace_convention': np.array(json.dumps(ACE_CONVENTION)),
              'hp_convention': np.array('smplx_residual_flat_hand_mean_false'),
              'parameter_coordinate_frame': np.array('ace_*: camera; {side}_*: world'),
              'state_names': np.array(json.dumps({str(k): v for k, v in L.STATE_NAMES.items()})),
              'frame_flag_bits': np.array(json.dumps({str(bit): n for bit, n in L.FLAG_NAMES.items()}))}
    label = L.label_arrays(states, present)
    for h, side in enumerate(SIDES):          # 拟合误差大的帧：手在，但建议权重降低
        if present[h] and 'flags' in hands[side]:
            bad = (hands[side]['flags'] & L.FLAG_FIT_ERROR) > 0
            label['gt_weight'][bad, h] = np.minimum(label['gt_weight'][bad, h], FIT_ERROR_WEIGHT)
    arrays.update(label)
    exists = label['ace_exists']

    ace = {'ace_go': np.zeros((count, 2, 3), np.float32), 'ace_hp': np.zeros((count, 2, 45), np.float32),
           'ace_trans': np.zeros((count, 2, 3), np.float32), 'ace_joints_cam': np.zeros((count, 2, 21, 3), np.float32),
           'ace_betas': np.zeros((2, 10), np.float32)}
    for h, side in enumerate(SIDES):
        hand = hands[side]
        if 'go_cam' not in hand:
            continue
        e = exists[:, h]
        ace['ace_go'][e, h] = hand['go_cam'][e]
        ace['ace_hp'][e, h] = hand['hp'][e]
        ace['ace_trans'][e, h] = hand['trans_cam'][e]
        ace['ace_joints_cam'][e, h] = hand['joints_cam'][e]
        ace['ace_betas'][h] = hand['beta']
        legacy = hand['legacy']
        for key, value in legacy.items():
            value = np.asarray(value)
            if value.ndim >= 1 and len(value) == count and value.dtype.kind == 'f' and key not in (
                    'cam_R', 'cam_t', 'joints_cam_cma', 'error_mm'):
                value = np.where(e.reshape((-1,) + (1,) * (value.ndim - 1)), value, np.nan)
            arrays[f'{side}_{key}'] = value.astype(np.float32) if value.dtype.kind == 'f' else value
        arrays[f'{side}_valid'] = e & ((hand['flags'] & (L.FLAG_POSE_BRANCH | L.FLAG_WRIST | L.FLAG_FIT_ERROR)) == 0)
    uv, joint_vis, visible = L.visibility(ace['ace_joints_cam'].astype(np.float64), exists, K, width, height)
    ace.update(ace_joints2d=uv.astype(np.float32), ace_joint_vis=joint_vis, ace_visible=visible)
    arrays.update(ace)
    arrays['exists'] = exists.copy()

    window_ok = L.training_windows(segment_id, states, present)
    train, short = L.clip_ranges(segment_id, states, present)
    arrays.update(segment_id=segment_id, gap_after=gap_after, window_ok=window_ok,
                  clip_ranges=np.array(train, np.int32).reshape(-1, 2),
                  short_ranges=np.array(short, np.int32).reshape(-1, 2))

    # 自检
    finite = {k: bool(np.isfinite(v).all()) for k, v in ace.items() if np.asarray(v).dtype.kind == 'f'}
    if not all(finite.values()):
        raise RuntimeError(f'{name}: non-finite ACE arrays {finite}')
    residual = np.linalg.norm(ace['ace_hp'].reshape(count, 2, 15, 3), axis=-1).max(-1)
    checks = {'ace_arrays_finite': finite, 'ace_hp_residual_max_rad': float(residual[exists].max()),
              'fk_reproduces_ace_joints_cam_mm': max((reports[s]['checks']['camera_fk_consistency_mm']
                                                      for s in SIDES if 'checks' in reports[s]), default=0.0),
              'cam_R_vs_delivered_max_abs': cam_check,
              'placeholders_zero': bool(all(np.all(np.asarray(v)[~exists] == 0) for k, v in ace.items()
                                            if k not in ('ace_betas',)))}
    if checks['ace_hp_residual_max_rad'] >= RISKY_RESIDUAL_RAD:
        raise RuntimeError(f'{name}: residual pose {checks["ace_hp_residual_max_rad"]:.3f} rad too close to pi')
    if checks['fk_reproduces_ace_joints_cam_mm'] > 0.05:
        raise RuntimeError(f'{name}: ace params do not reproduce ace_joints_cam '
                           f'({checks["fk_reproduces_ace_joints_cam_mm"]:.4f} mm)')

    starts = max(count - L.WINDOW + 1, 0)
    report = {
        'converter': VERSION, 'recording_name': name, 'recording': str(session['mcap']), 'n_frames': count,
        'K': K, 'source_resolution': [width, height], 'frame_ms': nominal / 1e6,
        'time_gaps': [{'after_frame': int(t), 'ms': float((stamps[t + 1] - stamps[t]) / 1e6)}
                      for t in np.flatnonzero(gap_after[:-1])],
        'segments': int(segment_id.max() + 1) if count else 0,
        'clip_ranges': train, 'short_ranges': short,
        'frames_in_clips': int(sum(b - a for a, b in train)),
        'windows': {'total': int(starts), 'ok': int(window_ok.sum())},
        'mount': {'source': str(OLD / name / 'mano_fit.npz'), 'converter': old_report.get('converter'),
                  'status': old_report.get('extrinsic', {}).get('status'),
                  'rotation_source': old_report.get('extrinsic', {}).get('rotation_source'),
                  'verdict': old_report.get('extrinsic_verdict')},
        'labels': {'state_names': L.STATE_NAMES, 'state_weight': L.STATE_WEIGHT,
                   'state_exists_target': L.STATE_EXISTS_TARGET, 'max_fill_frames': L.MAX_FILL,
                   'gap_factor': L.GAP_FACTOR, 'window': L.WINDOW, 'fit_error_weight': FIT_ERROR_WEIGHT,
                   'flag_bits': L.FLAG_NAMES},
        'ace_convention': ACE_CONVENTION, 'checks': checks, 'hands': reports,
        'seconds': time.perf_counter() - started,
    }
    np.savez(out / 'mano_fit.npz', **arrays)
    (out / REPORT_NAME).write_text(json.dumps(jsonable(report), indent=1, ensure_ascii=False), encoding='utf-8')
    return summarize(report)


def summarize(report):
    hands = {}
    for side, r in report['hands'].items():
        if not r.get('present') or 'fit_vs_cma' not in r:
            hands[side] = {'present': bool(r.get('present')), 'reason': r.get('reason')}
            continue
        hold = r['fill_holdout_mm']
        hands[side] = {'subject': r['subject'], 'states': r['states'],
                       'events': [(e['start'], e['stop'], e['state'], e['reason']) for e in r['events']],
                       'fit_mean_mm': r['fit_vs_cma']['mean_mm'], 'fit_p95_mm': r['fit_vs_cma']['p95_mm'],
                       'twist_p95_max_deg': max(r['twist_p95_deg']['fitted']),
                       'residual_max_rad': r['residual_max_rad'], 'flag_counts': r['flag_counts'],
                       'holdout_mean_mm': {k: round(v['mean_mm'], 2) for k, v in hold.items()},
                       'holdout_rootrel_mm': {k: round(v['rootrel_mean_mm'], 2) for k, v in hold.items()},
                       'seconds': r['seconds']}
    return {'recording': report['recording_name'], 'frames': report['n_frames'], 'time_gaps': report['time_gaps'],
            'clip_ranges': report['clip_ranges'], 'short_ranges': report['short_ranges'],
            'frames_in_clips': report['frames_in_clips'], 'windows': report['windows'], 'checks': report['checks'],
            'hands': hands, 'seconds': report['seconds']}


def _worker(name, out_root, betas, fit_options):
    try:
        return name, convert_recording(name, out_root, betas, fit_options), None
    except Exception as error:  # noqa: BLE001 - 一条失败不影响其它录制
        failure = {'status': 'failed', 'recording': name, 'error_type': type(error).__name__, 'error': str(error),
                   'traceback': traceback.format_exc()}
        out = Path(out_root) / name
        out.mkdir(parents=True, exist_ok=True)
        (out / 'failure.json').write_text(json.dumps(failure, indent=2, ensure_ascii=False), encoding='utf-8')
        return name, None, failure


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', nargs='*', default=None)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', type=Path, default=RESULTS)
    parser.add_argument('--options', default='{}', help='gt_fix.fit.refit_hand 关键字参数（JSON）')
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors='replace')
    names = sorted(p.name for p in RAW.iterdir() if (OLD / p.name / 'mano_fit.npz').is_file())
    if args.only:
        names = [n for n in names if n in args.only]
    betas = subject_betas(RAW, sorted(p.name for p in RAW.iterdir() if (OLD / p.name / 'mano_fit.npz').is_file()),
                          cache=RESULTS / 'subject_betas.json')
    fit_options = json.loads(args.options)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    done, failures = {}, {}
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(names)))) as pool:
        futures = [pool.submit(_worker, name, args.out, betas, fit_options) for name in names]
        for future in as_completed(futures):
            name, summary, failure = future.result()
            if failure is not None:
                failures[name] = failure
                print(f'[{name}] FAILED {failure["error_type"]}: {failure["error"]}', flush=True)
                continue
            done[name] = summary
            hands = '; '.join(
                f"{side[0]} {h['states']} fit {h['fit_mean_mm']:.2f} mm twist95 {h['twist_p95_max_deg']:.0f}"
                for side, h in summary['hands'].items() if 'states' in h)
            print(f"[{name}] clips {summary['clip_ranges']} windows {summary['windows']['ok']}/"
                  f"{summary['windows']['total']}; {hands}; {summary['seconds']:.0f} s", flush=True)
    merged = {}
    summary_path = args.out / 'summary.json'
    if summary_path.is_file() and args.only:
        merged = json.loads(summary_path.read_text(encoding='utf-8')).get('recordings', {})
    merged.update(jsonable(done))
    for name in failures:
        merged.pop(name, None)
    summary_path.write_text(json.dumps({'converter': VERSION, 'recordings': dict(sorted(merged.items())),
                                        'failures': jsonable(failures)}, indent=1, ensure_ascii=False),
                            encoding='utf-8')
    print(f'{len(done)} converted, {len(failures)} failed, {time.perf_counter() - started:.0f} s', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
