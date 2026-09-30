# -*- coding: utf-8 -*-
"""用 MediaPipe 2D 关键点做独立裁判，比较几种 GT 的图像对齐（逐关节像素误差）。

变体：
  cma    CMA 手套 FK 关节（旧 npz 的 {side}_joints_cam_cma）
  old    旧交付的 MANO 关节（旧 npz 的 {side}_joints_cam）
  init   新标定的局部旋转 + 被试 beta，手腕放在 CMA 手腕（不做位置拟合）
  refit  gt_fix.fit 的重拟合（--refit 时）
MediaPipe 槽位只按 CMA 手掌（腕 + 4 个 MCP）关联一次，所有变体在同一组帧 / 关节上比较。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gt_fix.anatomy import bone_dirs, get_anatomy, rest_points  # noqa: E402
from gt_fix.calibrate import calibrate_hand, pose_from_calibration, resample_world_R, source_offsets  # noqa: E402
from gt_fix.shape import beta_for  # noqa: E402
from robust_convert.extrinsic import load_detections  # noqa: E402
from robust_convert.mano_model import mano_forward  # noqa: E402
from robust_convert.session import load_session  # noqa: E402

RAW = Path(r'C:\Users\11068\Desktop\项目材料\datasets_20\to_sftp')
OLD = Path(r'C:\Users\11068\Desktop\mano线推进\outputs\unified_20')
MP = Path(r'C:\Users\11068\Desktop\mano线推进\outputs\cache\mediapipe')
SIDES = ('left', 'right')
PALM = [0, 5, 9, 13, 17]
GATE_PX = 80.0
OP_NAMES = ['W', 'T1', 'T2', 'T3', 'T4', 'I1', 'I2', 'I3', 'I4', 'M1', 'M2', 'M3', 'M4',
            'R1', 'R2', 'R3', 'R4', 'P1', 'P2', 'P3', 'P4']


def project(points_m, K):
    z = points_m[..., 2]
    with np.errstate(invalid='ignore', divide='ignore'):
        uv = np.stack([K[0, 0] * points_m[..., 0] / z + K[0, 2], K[1, 1] * points_m[..., 1] / z + K[1, 2]], -1)
    uv[~(z > 0.01)] = np.nan
    return uv


def associate(uv_ref, det):
    """每帧每手按手掌中位距离选 MediaPipe 槽（两手不共用一槽），超过门限记 -1。"""
    count = len(uv_ref)
    slot = np.full((count, 2), -1)
    for t in range(count):
        cost = np.full((2, 2), np.inf)
        for h in range(2):
            for s in range(2):
                if det['label'][t, s] < 0:
                    continue
                d = np.linalg.norm(uv_ref[t, h, PALM] - det['uv'][t, s, PALM], axis=-1)
                if np.isfinite(d).sum() >= 3:
                    cost[h, s] = np.nanmedian(d)
        a = cost[0, 0] + cost[1, 1]
        b = cost[0, 1] + cost[1, 0]
        pairs = [(0, 0), (1, 1)] if (a <= b or not np.isfinite(b)) and np.isfinite(a) else [(0, 1), (1, 0)]
        if not np.isfinite(min(a, b)):
            pairs = [(h, int(np.argmin(cost[h]))) for h in range(2)]
            if pairs[0][1] == pairs[1][1]:
                keep = int(np.argmin([cost[0, pairs[0][1]], cost[1, pairs[1][1]]]))
                pairs = [pairs[keep]]
        for h, s in pairs:
            if cost[h, s] < GATE_PX:
                slot[t, h] = s
    return slot


def variants_for(name, betas, refit=False, fit_options=None):
    session = load_session(RAW / name)
    old = np.load(OLD / name / 'mano_fit.npz')
    stamps = session['stamps']
    if not np.array_equal(old['timestamps_ns'].astype(np.int64), stamps.astype(np.int64)):
        raise RuntimeError(f'{name}: session stamps differ from the delivered npz')
    K = old['K'].astype(np.float64)
    count = len(stamps)
    out = {side: {} for side in SIDES}
    extra = {}
    for side in SIDES:
        source = session['hands_source'][side]
        if not source['present']:
            continue
        hand = session['hands'][side]
        anat = get_anatomy(side)
        beta, subject = beta_for(betas, name, side)
        cam_R = old[f'{side}_cam_R'].astype(np.float64)
        cam_t = old[f'{side}_cam_t'].astype(np.float64)
        valid = hand['frame_valid']
        P = hand['points_mm'][valid]
        offsets, _, _ = source_offsets(source)
        W = resample_world_R(session['cma_ns'], source['cma_status'], source['cma_world_R'], stamps)
        rest21 = rest_points(anat.model, beta)
        C, diag = calibrate_hand(side, anat, offsets, W[valid], P, rest21=rest21)
        pose = pose_from_calibration(C, W[valid], anat)
        _, _, _, rest = mano_forward(anat.model, beta, np.zeros(3), np.zeros(45), np.zeros(3))
        trans = P[:, 0] / 1000 - rest[0]
        _, joints, _, _ = mano_forward(anat.model, beta, pose['go'], pose['hp'], trans)

        def to_cam(world_m, frames):
            full = np.full((count, 21, 3), np.nan)
            full[frames] = np.einsum('tij,tkj->tki', cam_R[frames], world_m) + cam_t[frames][:, None]
            return full
        frames = np.flatnonzero(valid)
        out[side]['cma'] = old[f'{side}_joints_cam_cma'].astype(np.float64)
        old_cam = old[f'{side}_joints_cam'].astype(np.float64)
        out[side]['old'] = old_cam
        out[side]['init'] = to_cam(joints, frames)
        if refit:
            from gt_fix.fit import adjacency, refit_hand
            result = refit_hand(anat, beta, P, hand['status'][valid], pose, adjacency(frames, stamps),
                                **(fit_options or {}))
            out[side]['refit'] = to_cam(result['joints_world_m'], frames)
            extra[side] = result
            tw = np.abs(result['twist_deg'])
            tw_init = np.abs(anat.twist(pose['physical'], bone_dirs(rest21)))
            e_init = np.linalg.norm(joints * 1000 - P, axis=-1)
            print(f'[{name} {side}] 3D err init {e_init.mean():.2f} -> refit {result["error_mm"].mean():.2f} mm; '
                  f'tips {e_init[:, [4, 8, 12, 16, 20]].mean():.2f} -> {result["error_mm"][:, [4, 8, 12, 16, 20]].mean():.2f}; '
                  f'twist p95 max init {np.percentile(tw_init, 95, 0).max():.1f} -> {np.percentile(tw, 95, 0).max():.1f} deg; '
                  f'{result["seconds"]:.1f}s terms {({k: round(v, 1) for k, v in result["loss_terms"].items()})}')
            print('   twist p95 per joint refit', np.percentile(tw, 95, 0).round(0))
        extra.setdefault('subjects', {})[side] = subject
    return session, K, out, extra


def evaluate(name, betas, refit=False, fit_options=None, show=True):
    session, K, out, extra = variants_for(name, betas, refit, fit_options)
    det = load_detections(MP / f'{name}.npz', session['stamps'])
    count = len(session['stamps'])
    uv = {side: {k: project(v, K) for k, v in variants.items()} for side, variants in out.items()}
    ref = np.full((count, 2, 21, 2), np.nan)
    for h, side in enumerate(SIDES):
        if 'cma' in uv[side]:
            ref[:, h] = uv[side]['cma']
    slot = associate(ref, det)
    summary = {}
    for h, side in enumerate(SIDES):
        if not uv[side]:
            continue
        names = list(uv[side])
        rows = np.flatnonzero(slot[:, h] >= 0)
        target = det['uv'][rows, slot[rows, h]]
        errs = {k: np.linalg.norm(uv[side][k][rows] - target, axis=-1) for k in names}
        common = np.all([np.isfinite(e) for e in errs.values()], axis=0)
        per = {k: np.array([np.median(e[common[:, j], j]) if common[:, j].any() else np.nan for j in range(21)])
               for k, e in errs.items()}
        overall = {k: float(np.median(e[common])) for k, e in errs.items()}
        summary[side] = {'frames': int(len(rows)), 'overall_median_px': overall,
                         'per_joint_median_px': {k: v.round(2).tolist() for k, v in per.items()}}
        if show:
            print(f'\n{name} {side} subject={extra["subjects"][side]} frames={len(rows)} '
                  + ' '.join(f'{k}={v:.2f}' for k, v in overall.items()))
            print('      ' + ' '.join(f'{n:>5}' for n in OP_NAMES))
            for k, v in per.items():
                print(f'{k:>5} ' + ' '.join(f'{x:5.1f}' for x in v))
    return summary, extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('recordings', nargs='+')
    parser.add_argument('--refit', action='store_true')
    parser.add_argument('--options', default='{}', help='gt_fix.fit.refit_hand 关键字参数（JSON）')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    betas = json.loads((ROOT / 'results' / 'subject_betas.json').read_text(encoding='utf-8'))
    results = {}
    for name in args.recordings:
        results[name], _ = evaluate(name, betas, args.refit, json.loads(args.options))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
