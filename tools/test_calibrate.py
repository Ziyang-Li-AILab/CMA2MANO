# -*- coding: utf-8 -*-
"""标定自检：新标定（无参考帧逐骨）vs 旧交付结果，逐关节扭转 / Mahalanobis / 扭转偏置曲线 / 位置误差。"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gt_fix.anatomy import JOINT_NAMES, get_anatomy  # noqa: E402
from gt_fix.calibrate import (calibrate_hand, expvec, pose_from_calibration, psi_report,  # noqa: E402
                              resample_world_R, source_offsets)
from robust_convert.mano_model import mano_forward  # noqa: E402
from robust_convert.session import load_session  # noqa: E402

RAW = Path(r'C:\Users\11068\Desktop\项目材料\datasets_20\to_sftp')
OLD = Path(r'C:\Users\11068\Desktop\mano线推进\outputs\unified_20')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('recordings', nargs='+')
    args = parser.parse_args()
    np.set_printoptions(precision=1, suppress=True, linewidth=200)
    for name in args.recordings:
        session = load_session(RAW / name)
        old = np.load(OLD / name / 'mano_fit.npz')
        query = session['stamps']
        for side in ('left', 'right'):
            source = session['hands_source'][side]
            if not source['present']:
                continue
            hand = session['hands'][side]
            anat = get_anatomy(side)
            offsets, support, _ = source_offsets(source)
            W = resample_world_R(session['cma_ns'], source['cma_status'], source['cma_world_R'], query)
            valid = hand['frame_valid']
            P = hand['points_mm'][valid]
            C, diag = calibrate_hand(side, anat, offsets, W[valid], P)
            pose = pose_from_calibration(C, W[valid], anat)
            print(f'\n=== {name} {side} subject={source["schema"].get("subject")} frames={valid.sum()}')
            print(' wrist kabsch residual deg', diag['wrist_kabsch_residual_deg'],
                  ' tips palmar frac', round(diag['tips_on_palmar_side_fraction'], 3))
            print(' finger signs', {k: (v['sign'], v['agreement']) for k, v in diag['finger_sign'].items()})
            print(' thumb', diag['thumb_sign'])
            twist_new = np.abs(anat.twist(pose['physical']))
            maha_new = anat.mahalanobis(pose['physical'])
            old_fit = old[f'{side}_fitted'].astype(bool)
            same = np.array_equal(old_fit, valid)
            hp_old = old[f'{side}_hand_pose'][valid].astype(np.float64)
            phys_old = hp_old.reshape(-1, 15, 3) + anat.hands_mean.reshape(15, 3)
            twist_old = np.abs(anat.twist(phys_old))
            maha_old = anat.mahalanobis(phys_old)
            print(' old fitted mask == new valid:', same)
            print(' joint        ' + ' '.join(f'{n:>6}' for n in JOINT_NAMES))
            print(' twist50 old  ' + ' '.join(f'{v:6.0f}' for v in np.median(twist_old, 0)))
            print(' twist50 new  ' + ' '.join(f'{v:6.0f}' for v in np.median(twist_new, 0)))
            print(' twist95 new  ' + ' '.join(f'{v:6.0f}' for v in np.percentile(twist_new, 95, 0)))
            print(' maha50 old   ' + ' '.join(f'{v:6.1f}' for v in np.median(maha_old, 0)))
            print(' maha50 new   ' + ' '.join(f'{v:6.1f}' for v in np.median(maha_new, 0)))
            print(' |hp|max new  ', np.linalg.norm(pose['hp'].reshape(-1, 15, 3), axis=-1).max().round(2))
            L_old = np.zeros((valid.sum(), 16, 3, 3))
            L_old[:, 0] = expvec(old[f'{side}_global_orient'][valid].astype(np.float64))
            L_old[:, 1:] = expvec(phys_old)
            print(' psi old', {k: v['psi_best_mahalanobis_deg'] for k, v in psi_report(L_old, anat).items()})
            print(' psi new', {k: v['psi_best_mahalanobis_deg'] for k, v in psi_report(pose['L'], anat).items()})
            beta = old[f'{side}_beta'].astype(np.float64)
            _, _, _, rest = mano_forward(anat.model, beta, np.zeros(3), np.zeros(45), np.zeros(3))
            trans = P[:, 0] / 1000 - rest[0]
            _, joints, _, _ = mano_forward(anat.model, beta, pose['go'], pose['hp'], trans)
            err_new = np.linalg.norm(joints * 1000 - P, axis=-1)
            err_old = old[f'{side}_error_mm'][valid]
            print(' err old per-joint', np.nanmean(err_old, 0).round(1), 'mean', np.nanmean(err_old).round(2))
            print(' err new(init, old beta)', err_new.mean(0).round(1), 'mean', err_new.mean().round(2))


if __name__ == '__main__':
    main()
