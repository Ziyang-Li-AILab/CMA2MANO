# -*- coding: utf-8 -*-
"""检查 CMA 手套各关节局部旋转的主轴（屈曲轴）约定，确认 y=手指屈曲轴、z=拇指屈曲轴 在全部录制上成立。"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robust_convert.cleaning import repair_hand  # noqa: E402
from robust_convert.cma_source import read_human  # noqa: E402
from robust_convert.mcap_io import find_inputs  # noqa: E402

RAW = Path(r'C:\Users\11068\Desktop\项目材料\datasets_20\to_sftp')
PARENT = [0] + [0 if j % 4 == 1 else j - 1 for j in range(1, 21)]
JOINTS = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
NAMES = ['T1', 'T2', 'T3', 'I1', 'I2', 'I3', 'M1', 'M2', 'M3', 'R1', 'R2', 'R3', 'P1', 'P2', 'P3']


def main():
    table = {}
    for folder in sorted(RAW.iterdir()):
        _, human, _ = find_inputs(folder)
        source = read_human(human)
        for side in ('left', 'right'):
            hand = source['hands'][side]
            if not hand['present']:
                continue
            rep = repair_hand(hand['points_mm'], hand['pos_valid'], hand['global_R'], hand['local_R'], hand['rot_valid'])
            W = rep['world_R'][rep['frame_valid']]
            row = []
            for j in JOINTS:
                local = np.swapaxes(W[:, PARENT[j]], -1, -2) @ W[:, j]
                vec = Rotation.from_matrix(local).as_rotvec()
                angle = np.linalg.norm(vec, axis=-1)
                _, sv, vt = np.linalg.svd(vec, full_matrices=False)
                axis = vt[0] * np.sign(vt[0][np.argmax(np.abs(vt[0]))])
                row.append((axis, sv[0] ** 2 / max((sv ** 2).sum(), 1e-12), np.percentile(angle, 95),
                            (angle < np.radians(0.5)).mean()))
            table[(folder.name, side)] = row
            print(folder.name, side, hand['schema']['subject'], ' '.join(
                f"{n}:{'xyz'[int(np.argmax(np.abs(a)))]}{np.abs(a).max():.2f}/{np.degrees(p95):.0f}d"
                + (f"/zero{z:.0%}" if z > 0.2 else '') for n, (a, f, p95, z) in zip(NAMES, row)))


if __name__ == '__main__':
    main()
