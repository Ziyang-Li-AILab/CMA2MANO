# -*- coding: utf-8 -*-
"""网格特写、空洞胶片和三条对比视频。用 ace-ego-hand 解释器运行：

    python tools/make_views.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gt_fix.anatomy import get_anatomy  # noqa: E402
from robust_convert.mano_model import mano_forward  # noqa: E402
from robust_convert.mcap_io import iter_rgb_items  # noqa: E402
from robust_convert.render import (  # noqa: E402
    COLORS, _header, _paint_compiled_mesh, decode_preview_item, draw_skeleton, encode_ordered_frames,
    ordered_parallel_map,
)

OLD = Path(r'C:\Users\11068\Desktop\mano线推进\outputs\unified_20')
NEW = ROOT / 'results'
OUT = ROOT / 'figures'
SIDES = ('left', 'right')
MCP = np.array([0, 3, 6, 9])
STATE = ('fitted', 'filled', 'unknown', 'absent')
SKIN = np.array([196, 214, 232], np.float64)  # BGR
FILM = (
    ('20260820_125531', 'left', 144, 151),
    ('20260820_164514', 'right', 221, 228),
    ('20260820_112346', 'right', 0, 14),
)
VIDEOS = ('20260820_112346', '20260820_125531', '20260820_164514')


def project_mm(points_mm, K):
    z = np.asarray(points_mm[..., 2], np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        uv = np.stack([K[0, 0] * points_mm[..., 0] / z + K[0, 2],
                       K[1, 1] * points_mm[..., 1] / z + K[1, 2]], -1)
    uv = uv.astype(np.float64)
    uv[~(z > 1.0)] = np.nan
    return uv, z


def forward(model, beta, go, hp, trans, mask):
    count = len(go)
    out = np.full((count, 778, 3), np.nan, np.float32)
    ok = np.asarray(mask, bool) & np.isfinite(go).all(1) & np.isfinite(hp).all(1) & np.isfinite(trans).all(1)
    idx = np.flatnonzero(ok)
    beta = np.asarray(beta, np.float64)
    for start in range(0, len(idx), 64):
        sl = idx[start:start + 64]
        out[sl] = mano_forward(model, beta, go[sl], hp[sl], trans[sl])[0]
    return out


def cam_mm(verts_m, cam_R, cam_t):
    return np.einsum('tij,tvj->tvi', cam_R, verts_m.astype(np.float64)) + cam_t[:, None]


def load_pack(name):
    """一只录制的旧/新网格、CMA 投影和状态文字。顶点在相机系、毫米。"""
    new = np.load(NEW / name / 'mano_fit.npz')
    old = np.load(OLD / name / 'mano_fit.npz')
    report = json.loads((NEW / name / 'mano_fit_report.json').read_text(encoding='utf-8'))
    stamps = new['timestamps_ns'].astype(np.int64)
    if not np.array_equal(stamps, old['timestamps_ns'].astype(np.int64)):
        raise RuntimeError(f'{name}: old and new stamps differ')
    width, height = (int(v) for v in new['source_resolution'])
    K = new['K'].astype(np.float64)
    count = len(stamps)
    cam_R = new['left_cam_R'].astype(np.float64)
    cam_t = new['left_cam_t'].astype(np.float64)
    meshes = {}
    twist = {}
    for side in SIDES:
        anat = get_anatomy(side)
        faces = anat.model['f']
        for tag, z, exists in (
            ('new', new, new['ace_exists'][:, 0 if side == 'left' else 1]),
            ('old', old, old['exists'][:, 0 if side == 'left' else 1]),
        ):
            go = z[f'{side}_global_orient'].astype(np.float64)
            hp = z[f'{side}_hand_pose'].astype(np.float64)
            trans = z[f'{side}_trans'].astype(np.float64)
            beta = z[f'{side}_beta'].astype(np.float64)
            verts = forward(anat.model, beta, go, hp, trans, exists)
            pts = cam_mm(verts, cam_R, cam_t) * 1000
            uv, depth = project_mm(pts, K)
            meshes[(tag, side)] = {'uv': uv.astype(np.float32), 'depth': depth.astype(np.float32),
                                    'faces': faces, 'cam': pts.astype(np.float32)}
            if tag == 'old':
                physical = hp.reshape(count, 15, 3) + anat.hands_mean.reshape(15, 3)
                ok = np.flatnonzero(exists & np.isfinite(hp).all(1))
                angles = np.zeros((count, 15))
                if len(ok):
                    angles[ok] = np.abs(anat.twist(physical[ok]))
                twist[side] = angles
            else:
                physical = hp.reshape(count, 15, 3) + anat.hands_mean.reshape(15, 3)
                ok = np.flatnonzero(exists & np.isfinite(hp).all(1))
                angles = np.zeros((count, 15))
                if len(ok):
                    angles[ok] = np.abs(anat.twist(physical[ok]))
                twist[side + '_new'] = angles
    skel = np.full((count, 42, 2), np.nan, np.float32)
    status = np.full((count, 42), 2, np.int8)
    for h, side in enumerate(SIDES):
        pts = new[f'{side}_joints_cam_cma'].astype(np.float64) * 1000
        uv, _ = project_mm(pts, K)
        joint_status = new[f'{side}_joint_status'].astype(np.int8)
        uv[joint_status == 2] = np.nan
        skel[:, h * 21:(h + 1) * 21] = uv
        status[:, h * 21:(h + 1) * 21] = joint_status
    state = new['gt_state'].astype(np.int8)
    old_exists = old['exists'].astype(bool)
    new_text = [f"L {STATE[int(a)]}   R {STATE[int(b)]}" for a, b in state]
    old_text = [f"L {'exists' if a else 'ABSENT'}   R {'exists' if b else 'ABSENT'}" for a, b in old_exists]
    return {'name': name, 'stamps': stamps, 'K': K, 'wh': (width, height), 'mcap': report['recording'],
            'meshes': meshes, 'skel': skel, 'status': status, 'new_text': new_text, 'old_text': old_text,
            'twist': twist, 'state': state, 'subject': report['hands']['left']['subject']}


def render_video(pack):
    width, height = pack['wh']
    panel = (960, 540)
    scale = np.array([panel[0] / width, panel[1] / height])
    count = len(pack['stamps'])
    source = pack['wh']

    def one(item):
        index, rgb = item
        image = decode_preview_item(rgb, source, panel)
        cma = draw_skeleton(image, pack['skel'][index] * scale, pack['status'][index], f'CMA   frame {index}')
        old = image.copy()
        new = image.copy()
        for side in SIDES:
            for canvas, tag in ((old, 'old'), (new, 'new')):
                mesh = pack['meshes'][(tag, side)]
                uv, depth = mesh['uv'][index], mesh['depth'][index]
                if np.isfinite(uv).any():
                    _paint_compiled_mesh(canvas, uv, depth, mesh['faces'], COLORS[side], source)
        _header(old, f"old MANO   {pack['old_text'][index]}")
        _header(new, f"new MANO   {pack['new_text'][index]}")
        return np.concatenate((cma, old, new), axis=1)

    def items():
        wanted = [int(s) for s in pack['stamps']]
        seen = 0
        for stamp, item in iter_rgb_items(pack['mcap'], wanted):
            if int(stamp) != wanted[seen]:
                raise RuntimeError(f"{pack['name']}: RGB stamp {stamp} is not frame {seen}")
            yield seen, item
            seen += 1
        if seen != count:
            raise RuntimeError(f"{pack['name']}: expected {count} RGB frames, got {seen}")

    path = OUT / f"video_{pack['name']}.mp4"
    print(f"video {pack['name']} -> {path.name}", flush=True)
    encode_ordered_frames(ordered_parallel_map(one, items(), workers=4), path, (panel[0] * 3, panel[1]), count)
    print(f"video {pack['name']} done", flush=True)


def load_frames(pack, frames, reduce=1):
    wanted = {int(pack['stamps'][i]) for i in frames}
    found = {}
    by_stamp = {int(pack['stamps'][i]): i for i in frames}
    target = (pack['wh'][0] // reduce, pack['wh'][1] // reduce)
    for stamp, item in iter_rgb_items(pack['mcap'], wanted):
        found[by_stamp[int(stamp)]] = decode_preview_item(item, pack['wh'], target)
    missing = [i for i in frames if i not in found]
    if missing:
        raise RuntimeError(f"{pack['name']}: missing RGB frames {missing[:5]}")
    return found


def shade(image, cam_mm_pts, faces, K, scale=1.0):
    """画家算法 + Lambert。cam_mm_pts (V,3) 相机系毫米；image 会被改写。"""
    pts = np.asarray(cam_mm_pts, np.float64)
    if scale != 1.0:
        pts = pts.copy()
    uv, z = project_mm(pts, K)
    if scale != 1.0:
        uv = uv * scale
    tri = pts[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(normal, axis=1)
    normal /= np.maximum(length, 1e-8)[:, None]
    light = np.array([0.15, -0.35, -1.0])
    light /= np.linalg.norm(light)
    shade_v = np.clip(normal @ light, 0, 1)
    depth = tri[:, :, 2].mean(1)
    order = np.argsort(-depth)
    for i in order:
        if not (z[faces[i]] > 1).all() or not np.isfinite(uv[faces[i]]).all():
            continue
        if normal[i, 2] > 0.05:
            continue
        tone = 0.28 + 0.72 * float(shade_v[i])
        color = tuple(int(c) for c in np.clip(SKIN * tone, 0, 255))
        poly = np.rint(uv[faces[i]]).astype(np.int32)
        cv2.fillConvexPoly(image, poly, color, cv2.LINE_AA)


def visible_uv(uv, shape, pad=80):
    """丢掉落到画面外很远的投影，避免裁剪框被个别顶点拉爆。"""
    pts = np.asarray(uv, np.float64).reshape(-1, 2)
    height, width = shape[:2]
    ok = np.isfinite(pts).all(1)
    ok &= (pts[:, 0] > -pad) & (pts[:, 0] < width + pad) & (pts[:, 1] > -pad) & (pts[:, 1] < height + pad)
    return pts[ok]


def crop_box(pixels, shape, margin=36):
    pts = np.concatenate([np.asarray(p, np.float64).reshape(-1, 2) for p in pixels])
    pts = pts[np.isfinite(pts).all(1)]
    if not len(pts):
        return 0, 0, shape[1], shape[0]
    x0, y0 = np.floor(pts.min(0)).astype(int) - margin
    x1, y1 = np.ceil(pts.max(0)).astype(int) + margin
    height, width = shape[:2]
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def resize_h(image, height):
    scale = height / image.shape[0]
    return cv2.resize(image, (max(1, int(round(image.shape[1] * scale))), height), interpolation=cv2.INTER_AREA)


def imwrite(path, image):
    """cv2.imwrite 打不开含非 ASCII 的 Windows 路径，改为自己写字节。"""
    ok, buf = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError(f'failed to encode {path}')
    Path(path).write_bytes(buf.tobytes())


def banner(image, text):
    bar = np.full((28, image.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([bar, image])


def filmstrips(packs):
    for name, side, start, stop in FILM:
        pack = packs[name]
        frames = list(range(start, stop))
        images = load_frames(pack, frames, reduce=2)
        scale = images[frames[0]].shape[1] / pack['wh'][0]
        boxes = []
        for frame in frames:
            for tag in ('old', 'new'):
                boxes.append(pack['meshes'][(tag, side)]['uv'][frame] * scale)
        x0, y0, x1, y1 = crop_box(boxes, images[frames[0]].shape, margin=28)
        rows = []
        for tag, label in (('old', 'old'), ('new', 'new')):
            cells = []
            for frame in frames:
                canvas = images[frame].copy()
                mesh = pack['meshes'][(tag, side)]
                uv = mesh['uv'][frame]
                depth = mesh['depth'][frame]
                if np.isfinite(uv).any():
                    _paint_compiled_mesh(canvas, uv, depth, mesh['faces'], COLORS[side], pack['wh'])
                canvas = canvas[y0:y1, x0:x1]
                note = pack['old_text'][frame] if tag == 'old' else pack['new_text'][frame]
                cells.append(banner(resize_h(canvas, 220), f'f{frame} {note}'))
            row = np.hstack(cells)
            rows.append(banner(row, f'{name} {side}  {label}'))
        # rows may differ by 1 px
        width = min(r.shape[1] for r in rows)
        sheet = np.vstack([r[:, :width] for r in rows])
        path = OUT / f'film_{name}_{side}.png'
        imwrite(path, sheet)
        print(path.name, flush=True)


def main(skip_video=False):
    OUT.mkdir(parents=True, exist_ok=True)
    names = sorted(set(VIDEOS) | {n for n, *_ in FILM})
    # closeups may need every recording; scan twist without building video meshes first
    subjects = json.loads((NEW / 'subject_betas.json').read_text(encoding='utf-8'))['subjects']
    chosen = {}
    for subject, info in subjects.items():
        best = None
        for name in info['recordings']:
            old = np.load(OLD / name / 'mano_fit.npz')
            new = np.load(NEW / name / 'mano_fit.npz')
            width, height = (int(v) for v in new['source_resolution'])
            K = old['K'].astype(np.float64)
            for h, side in enumerate(SIDES):
                anat = get_anatomy(side)
                hp = old[f'{side}_hand_pose'].astype(np.float64)
                exists = old['exists'][:, h].astype(bool) & np.isfinite(hp).all(1)
                if not exists.any():
                    continue
                physical = hp.reshape(len(hp), 15, 3) + anat.hands_mean.reshape(15, 3)
                twist = np.zeros(len(hp))
                twist[exists] = np.abs(anat.twist(physical[exists]))[:, MCP].max(1)
                joints = old[f'{side}_joints_cam'].astype(np.float64)
                uv, depth = project_mm(joints * 1000, K)
                inside = ((uv[..., 0] >= 0) & (uv[..., 0] < width) & (uv[..., 1] >= 0)
                          & (uv[..., 1] < height) & (depth > 1))
                visible = inside.mean(1)
                reach = np.linalg.norm(joints[:, [4, 8, 12, 16, 20]] - joints[:, :1], axis=-1).mean(1)
                score = np.where(exists & (visible > 0.9) & (twist > 60), reach * twist, -1)
                frame = int(np.argmax(score))
                if score[frame] < 0:
                    continue
                if best is None or score[frame] > best[0]:
                    best = (float(twist[frame]), name, side, frame)
        chosen[subject] = best
        print('open-hand', subject, best, flush=True)
    needed = set(names) | {item[1] for item in chosen.values()}
    packs = {}
    for name in sorted(needed):
        print('pack', name, flush=True)
        packs[name] = load_pack(name)
    closeups_from(packs, chosen)
    filmstrips(packs)
    if not skip_video:
        for name in VIDEOS:
            render_video(packs[name])


def closeups_from(packs, chosen):
    for subject, (score, name, side, frame) in sorted(chosen.items()):
        pack = packs[name]
        image = load_frames(pack, [frame], reduce=1)[frame]
        K = pack['K']
        old = image.copy()
        new = image.copy()
        shade(old, pack['meshes'][('old', side)]['cam'][frame], pack['meshes'][('old', side)]['faces'], K)
        shade(new, pack['meshes'][('new', side)]['cam'][frame], pack['meshes'][('new', side)]['faces'], K)
        box = crop_box([visible_uv(pack['meshes'][('old', side)]['uv'][frame], image.shape),
                        visible_uv(pack['meshes'][('new', side)]['uv'][frame], image.shape)],
                       image.shape, margin=48)
        x0, y0, x1, y1 = box
        pair = np.hstack([
            banner(resize_h(old[y0:y1, x0:x1], 480), f'old  MCP twist {score:.0f} deg'),
            banner(resize_h(new[y0:y1, x0:x1], 480),
                   f"new  MCP twist {pack['twist'][side + '_new'][frame, MCP].max():.0f} deg"),
        ])
        pair = banner(pair, f'{subject}   {name}   {side}   frame {frame}')
        path = OUT / f'closeup_{subject}_{name}_{side}_f{frame}.png'
        imwrite(path, pair)
        print(path.name, flush=True)


if __name__ == '__main__':
    main(skip_video='--skip-video' in sys.argv)
