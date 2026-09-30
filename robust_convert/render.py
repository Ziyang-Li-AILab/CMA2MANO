# -*- coding: utf-8 -*-
"""Comparison video in the original format, and MANO depth rendering for metric E2.

Video: same layout, colours, texts, mesh opacity (0.62), encoder (libx264
ultrafast CRF 20, 30 fps) and default 1920x540 size as the original
fast_render.write_comparison_fast - left panel the CMA skeleton projected with
the mount, right panel the MANO mesh overlay. Differences:
  * one video frame per usable RGB frame (invalid RGB messages are skipped
    instead of aborting the conversion);
  * the skeleton is the cleaned, resampled CMA track: FK-repaired joints get a
    white outline, invalid joints are not drawn;
  * a hand gets a mesh only in frames where its MANO fit exists; a short
    per-hand status is appended to the header text.
The rasteriser, JPEG preview decoder, ordered map and encoder are copied
verbatim from baseline_original_code/fast_render.py.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np
from numba import njit

try:
    import pybase64 as base64
except ImportError:
    import base64

from .cleaning import REPAIRED
from .extrinsic import project
from .mano_model import BONES, TIP_INDEX
from .mcap_io import iter_rgb_items
from .session import SIDES, rig_local

COLORS = {'left': (255, 180, 0), 'right': (0, 140, 255)}
REPAIRED_OUTLINE = (255, 255, 255)
MESH_OPACITY = 0.62


# ---- verbatim from the original fast_render.py -------------------------------------------------

@njit(cache=True, nogil=True)
def _fill_triangle_union(mask, triangles, supersample):
    height, width = mask.shape
    for triangle in triangles:
        x0, y0 = triangle[0, 0] * supersample, triangle[0, 1] * supersample
        x1, y1 = triangle[1, 0] * supersample, triangle[1, 1] * supersample
        x2, y2 = triangle[2, 0] * supersample, triangle[2, 1] * supersample
        low = max(0, int(np.floor(min(y0, y1, y2))))
        high = min(height - 1, int(np.ceil(max(y0, y1, y2))))
        xs = np.empty(3, dtype=np.float64)
        for row in range(low, high + 1):
            scan_y = row + 0.5
            count = 0
            for edge in range(3):
                ax = triangle[edge, 0] * supersample
                ay = triangle[edge, 1] * supersample
                bx = triangle[(edge + 1) % 3, 0] * supersample
                by = triangle[(edge + 1) % 3, 1] * supersample
                if (ay <= scan_y < by) or (by <= scan_y < ay):
                    xs[count] = ax + (scan_y - ay) * (bx - ax) / (by - ay)
                    count += 1
            if count >= 2:
                left = max(0, int(np.ceil(min(xs[0], xs[1]) - 0.5)))
                right = min(width - 1, int(np.floor(max(xs[0], xs[1]) - 0.5)))
                for col in range(left, right + 1):
                    mask[row, col] = 255
        for edge in range(3):
            ax = triangle[edge, 0] * supersample
            ay = triangle[edge, 1] * supersample
            bx = triangle[(edge + 1) % 3, 0] * supersample
            by = triangle[(edge + 1) % 3, 1] * supersample
            steps = max(1, int(np.ceil(max(abs(bx-ax), abs(by-ay)))))
            for point in range(steps + 1):
                cx = int(np.rint(ax + (bx-ax) * point / steps))
                cy = int(np.rint(ay + (by-ay) * point / steps))
                for yy in range(max(0, cy-2), min(height, cy+3)):
                    for xx in range(max(0, cx-2), min(width, cx+3)):
                        mask[yy, xx] = 255


def _mesh_mask_roi(size, pixels, depth, faces, source_size):
    width, height = size
    source_width, source_height = source_size
    face_pixels = pixels[faces]
    valid = np.all(depth[faces] > 1.0, axis=1)
    valid &= np.isfinite(face_pixels).all(axis=(1, 2))
    valid &= np.all((face_pixels[:, :, 0] >= -80) & (face_pixels[:, :, 0] <= source_width + 80), axis=1)
    valid &= np.all((face_pixels[:, :, 1] >= -80) & (face_pixels[:, :, 1] <= source_height + 80), axis=1)
    triangles = np.rint(face_pixels[valid] * np.array([width / source_width, height / source_height]))
    if len(triangles) == 0:
        return np.zeros((0, 0), np.uint8), 0, 0
    minimum = np.floor(triangles.min(axis=(0, 1))).astype(np.int64) - 2
    maximum = np.ceil(triangles.max(axis=(0, 1))).astype(np.int64) + 3
    x0, y0 = np.maximum(minimum, 0)
    x1, y1 = np.minimum(maximum, [width, height])
    if x0 >= x1 or y0 >= y1:
        return np.zeros((0, 0), np.uint8), 0, 0
    triangles = np.ascontiguousarray(triangles - np.array([x0, y0]), dtype=np.float64)
    high_resolution = np.zeros(((y1-y0)*2, (x1-x0)*2), np.uint8)
    _fill_triangle_union(high_resolution, triangles, 2)
    mask = cv2.resize(high_resolution, (int(x1-x0), int(y1-y0)), interpolation=cv2.INTER_AREA)
    return mask, int(x0), int(y0)


def _paint_compiled_mesh(image, pixels, depth, faces, color, source_size):
    mask, x0, y0 = _mesh_mask_roi((image.shape[1], image.shape[0]), pixels, depth, faces, source_size)
    if mask.size:
        roi = image[y0:y0+mask.shape[0], x0:x0+mask.shape[1]]
        opacity = mask[:, :, None].astype(np.float32) * (MESH_OPACITY / 255.0)
        result = roi.astype(np.float32) * (1.0 - opacity) + np.asarray(color, np.float32) * opacity
        roi[:] = np.clip(np.rint(result), 0, 255).astype(np.uint8)


def _jpeg_source_dimensions(data):
    """Read baseline/progressive JPEG SOF dimensions without decoding pixels."""
    if data[:2] != b'\xff\xd8':
        raise ValueError('JPEG dimensions unavailable: missing SOI')
    offset = 2
    while offset < len(data):
        if data[offset] != 255:
            raise ValueError('JPEG dimensions unavailable: malformed marker')
        while offset < len(data) and data[offset] == 255:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in (0xDA, 0xD9):
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:
            continue
        if offset + 2 > len(data):
            break
        length = (data[offset] << 8) | data[offset+1]
        if length < 2 or offset + length > len(data):
            raise ValueError('JPEG dimensions unavailable: truncated segment')
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if length < 8:
                raise ValueError('JPEG dimensions unavailable: malformed SOF')
            height = (data[offset+3] << 8) | data[offset+4]
            width = (data[offset+5] << 8) | data[offset+6]
            return width, height
        offset += length
    raise ValueError('JPEG dimensions unavailable: no supported SOF')


def decode_preview_item(item, source_size, target_size):
    """Use JPEG decoder downsampling for exact half/quarter/eighth previews."""
    source_width, source_height = source_size
    width, height = target_size
    flag, decode_divisor = cv2.IMREAD_COLOR, 1
    for divisor, candidate in [(8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4),
                               (2, cv2.IMREAD_REDUCED_COLOR_2)]:
        if width <= source_width // divisor and height <= source_height // divisor:
            flag, decode_divisor = candidate, divisor
            break
    jpeg = base64.b64decode(item['data'], validate=True)
    if _jpeg_source_dimensions(jpeg) != (source_width, source_height):
        raise ValueError('JPEG source dimensions differ from intrinsics')
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), flag)
    if image is None:
        raise ValueError('JPEG decode failed')
    expected_shape = ((source_height + decode_divisor - 1) // decode_divisor,
                      (source_width + decode_divisor - 1) // decode_divisor)
    if image.shape[:2] != expected_shape:
        raise ValueError('JPEG source dimensions differ from intrinsics')
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def ordered_parallel_map(function, items, workers=4):
    """Bound outstanding images to 2*workers and return submission order."""
    if workers < 1:
        raise ValueError('workers must be positive')
    if workers == 1:
        yield from map(function, items)
        return
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='mano-render') as pool:
        pending = deque()
        for _ in range(2 * workers):
            item = next(iterator, None)
            if item is None:
                break
            pending.append(pool.submit(function, item))
        while pending:
            yield pending.popleft().result()
            item = next(iterator, None)
            if item is not None:
                pending.append(pool.submit(function, item))


def _find_ffmpeg(explicit=None):
    candidates = [explicit, os.environ.get('FFMPEG_EXE'), shutil.which('ffmpeg')]
    try:
        import imageio_ffmpeg
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        pass
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return None


class _VideoEncoder:
    def __init__(self, path, size, fps, ffmpeg_path):
        self.process, self.writer, self.error_file = None, None, None
        if ffmpeg_path:
            command = [ffmpeg_path, '-hide_banner', '-loglevel', 'error', '-y',
                       '-f', 'rawvideo', '-pixel_format', 'bgr24',
                       '-video_size', f'{size[0]}x{size[1]}', '-framerate', str(fps),
                       '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'ultrafast',
                       '-crf', '20', '-pix_fmt', 'yuv420p', '-threads', '4',
                       '-movflags', '+faststart', str(path)]
            self.error_file = tempfile.TemporaryFile(mode='w+b')
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                            stdout=subprocess.DEVNULL, stderr=self.error_file)
            self.name = 'libx264-ultrafast-crf20'
        else:
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
            if not self.writer.isOpened():
                raise RuntimeError(f'VideoWriter failed to open {path}')
            self.name = 'opencv-mp4v'

    def write(self, image):
        if self.process is not None:
            self.process.stdin.write(memoryview(np.ascontiguousarray(image)))
        else:
            self.writer.write(image)

    def close(self, abort=False):
        if self.process is None:
            if self.writer is not None:
                self.writer.release()
            return
        if abort and self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        return_code = self.process.wait()
        self.error_file.seek(0)
        error = self.error_file.read().decode('utf-8', errors='replace')
        self.error_file.close()
        self.process = None
        if return_code and not abort:
            raise RuntimeError(f'FFmpeg exited with {return_code}: {error.strip()}')


def encode_ordered_frames(frames, path, size, expected_frames, fps=30.0, ffmpeg_path=None):
    """Encode every frame, replacing the result only after count validation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.stem}.{uuid4().hex}.tmp.mp4')
    encoder = _VideoEncoder(temporary, size, fps, _find_ffmpeg(ffmpeg_path))
    count, encode_seconds = 0, 0.0
    try:
        for image in frames:
            if image.shape != (size[1], size[0], 3) or image.dtype != np.uint8:
                raise ValueError(f'Unexpected video frame: {image.shape}, {image.dtype}')
            start = perf_counter()
            encoder.write(image)
            encode_seconds += perf_counter() - start
            count += 1
        if count != expected_frames:
            raise ValueError(f'Expected {expected_frames} video frames, got {count}')
        start = perf_counter()
        encoder.close()
        encode_seconds += perf_counter() - start
        temporary.replace(path)
    except BaseException:
        encoder.close(abort=True)
        temporary.unlink(missing_ok=True)
        raise
    return {'frames': count, 'encode_seconds': encode_seconds, 'encoder': encoder.name}


# ---- adapted drawing -------------------------------------------------------------------------

def draw_skeleton(image, pixels, status, text, bones=BONES, tip_index=TIP_INDEX):
    """Original draw_skeleton_fast (order, sizes, AA) plus the joint status.

    pixels (42, 2) left hand then right hand, NaN = not drawn; status (42,)
    cleaning codes, REPAIRED joints get a white instead of a dark outline.
    """
    canvas = image.copy()
    finite = np.isfinite(pixels).all(axis=1)
    rounded = np.rint(np.clip(np.where(np.isfinite(pixels), pixels, 0), -1e5, 1e5)).astype(np.int32)
    edges = np.asarray(bones, np.int32)
    tips = set(int(t) for t in tip_index)
    for hand, side in enumerate(SIDES):
        color = COLORS[side]
        points = rounded[hand*21:(hand+1)*21]
        present = finite[hand*21:(hand+1)*21]
        repaired = status[hand*21:(hand+1)*21] == REPAIRED
        selected = edges[present[edges].all(axis=1)]
        for a, b in selected:
            cv2.line(canvas, tuple(int(v) for v in points[a]), tuple(int(v) for v in points[b]), color, 2,
                     cv2.LINE_AA)
        for index in np.flatnonzero(present):
            center = tuple(int(v) for v in points[index])
            radius = 7 if index == 0 or index in tips else 4
            cv2.circle(canvas, center, radius, color, -1, cv2.LINE_AA)
            outline = REPAIRED_OUTLINE if repaired[index] else (20, 20, 20)
            cv2.circle(canvas, center, radius, outline, 1, cv2.LINE_AA)
    _header(canvas, text)
    return canvas


def _header(canvas, text):
    height, width = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (min(width, 780), min(height, 72)), (0, 0, 0), -1)
    cv2.putText(canvas, text, (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)


def _cma_status_text(session, index):
    parts = []
    for side in SIDES:
        hand = session['hands'][side]
        if not hand['present']:
            parts.append(f'{side[0].upper()} -')
            continue
        status = hand['status'][index]
        invalid = int((status == 2).sum())
        repaired = int((status == REPAIRED).sum())
        label = 'ok' if not invalid and not repaired else (
            'invalid' if invalid == 21 else ' '.join(filter(None, [f'{invalid} inv' if invalid else '',
                                                                  f'{repaired} FK' if repaired else ''])))
        parts.append(f'{side[0].upper()} {label}')
    return '  '.join(parts)


def _mano_status_text(meshes, index):
    parts = []
    for side in SIDES:
        mesh = meshes.get(side)
        if mesh is None or not mesh['fitted'][index]:
            parts.append(f'{side[0].upper()} -')
        else:
            parts.append(f'{side[0].upper()} {"ok" if mesh["valid"][index] else "flag"}')
    return '  '.join(parts)


def preview_size(width, height, preview_width=1920):
    """Original rule: total width preview_width (0 = 2x source), even height."""
    if preview_width == 0:
        return width * 2, height
    return preview_width, int(round(height * preview_width / (2 * width) / 2)) * 2


def _usable_items(session):
    stamps = session['stamps']
    index = 0
    for stamp, item in iter_rgb_items(session['mcap']):
        if index >= len(stamps) or stamp != int(stamps[index]):
            raise ValueError(f'RGB frame {index} does not match the session stamps')
        yield index, item
        index += 1
    if index != len(stamps):
        raise ValueError(f'Expected {len(stamps)} usable RGB messages, got {index}')


def write_comparison(session, mount_R, mount_t, meshes, path, workers=4, preview_width=1920, ffmpeg_path=None):
    """Left: CMA skeleton projected with the mount. Right: MANO mesh overlay.

    meshes: {side: {'vertices_mm': (N, 778, 3) world mm, NaN rows where not
    fitted, 'faces': (F, 3), 'fitted': (N,) bool, 'valid': (N,) bool}}
    """
    start = perf_counter()
    width, height, K = session['width'], session['height'], session['K']
    output_width, output_height = preview_size(width, height, preview_width)
    panel_width, panel_height = output_width // 2, output_height
    _fill_triangle_union(np.zeros((2, 2), np.uint8), np.zeros((0, 3, 2), np.float64), 2)
    scale = np.array([panel_width / width, panel_height / height])
    count = len(session['stamps'])
    points = np.full((count, 42, 3), np.nan)
    status = np.full((count, 42), 2, np.int8)
    for h, side in enumerate(SIDES):
        hand = session['hands'][side]
        if hand['present']:
            points[:, h * 21:(h + 1) * 21] = hand['points_mm']
            status[:, h * 21:(h + 1) * 21] = hand['status']
    local = rig_local(points, session['origin_mm'], session['rig_R'])
    pixels_all = project(local, mount_R, mount_t, K)[0]
    delta = session['cma_delta_ns']
    for side, mesh in meshes.items():
        if len(mesh['vertices_mm']) != count:
            raise ValueError(f'{side} mesh frame count differs from the usable RGB frames')

    def render(item):
        index, rgb_item = item
        image = decode_preview_item(rgb_item, (width, height), (panel_width, panel_height))
        left = draw_skeleton(image, pixels_all[index] * scale, status[index],
                             f'CMA projection   frame {index}   {_cma_status_text(session, index)}')
        right = image.copy()
        origin, rig = session['origin_mm'][index], session['rig_R'][index]
        for side in SIDES:
            mesh = meshes.get(side)
            if mesh is None or not mesh['fitted'][index]:
                continue
            vertices = (mesh['vertices_mm'][index] - origin) @ rig
            pixels, depth = project(vertices, mount_R, mount_t, K)
            _paint_compiled_mesh(right, pixels, depth, mesh['faces'], COLORS[side], (width, height))
        _header(right, f'MANO mesh   frame {index}   {delta[index] / 1e6:+.2f} ms   {_mano_status_text(meshes, index)}')
        return np.concatenate((left, right), axis=1)

    frames = ordered_parallel_map(render, _usable_items(session), workers)
    metrics = encode_ordered_frames(frames, path, (output_width, output_height), count, ffmpeg_path=ffmpeg_path)
    metrics.update(wall_seconds=perf_counter() - start, workers=workers, width=output_width, height=output_height,
                   fps=30.0, layout='left CMA projection | right MANO mesh (alpha 0.62)')
    return metrics


# ---- depth rendering for E2 -----------------------------------------------------------------

@njit(cache=True, nogil=True)
def _zbuffer(depth, label, pixels, z, faces, x0, y0, tag):
    """Nearest-surface depth of a triangle mesh (perspective-correct 1/z), pixel centres at integers."""
    height, width = depth.shape
    for f in range(faces.shape[0]):
        a, b, c = faces[f, 0], faces[f, 1], faces[f, 2]
        za, zb, zc = z[a], z[b], z[c]
        if not (za > 1.0 and zb > 1.0 and zc > 1.0):
            continue
        ax, ay = pixels[a, 0] - x0, pixels[a, 1] - y0
        bx, by = pixels[b, 0] - x0, pixels[b, 1] - y0
        cx, cy = pixels[c, 0] - x0, pixels[c, 1] - y0
        if not (np.isfinite(ax) and np.isfinite(ay) and np.isfinite(bx) and np.isfinite(by)
                and np.isfinite(cx) and np.isfinite(cy)):
            continue
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(area) < 1e-9:
            continue
        low_x = max(0, int(np.ceil(min(ax, bx, cx))))
        high_x = min(width - 1, int(np.floor(max(ax, bx, cx))))
        low_y = max(0, int(np.ceil(min(ay, by, cy))))
        high_y = min(height - 1, int(np.floor(max(ay, by, cy))))
        for yy in range(low_y, high_y + 1):
            for xx in range(low_x, high_x + 1):
                w0 = ((bx - xx) * (cy - yy) - (by - yy) * (cx - xx)) / area
                w1 = ((cx - xx) * (ay - yy) - (cy - yy) * (ax - xx)) / area
                w2 = 1.0 - w0 - w1
                if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                    continue
                value = 1.0 / (w0 / za + w1 / zb + w2 / zc)
                if value < depth[yy, xx]:
                    depth[yy, xx] = value
                    label[yy, xx] = tag


def render_depth(meshes, K, size):
    """Z-buffer of one or more triangle meshes given in camera mm.

    meshes: list of (vertices (V, 3), faces (F, 3)) or None (skipped).
    Returns depth (H, W) float64 with inf where empty, and label (H, W) int8
    with the list index of the nearest mesh (-1 empty), so hands occlude each other.
    """
    width, height = size
    depth = np.full((height, width), np.inf)
    label = np.full((height, width), -1, np.int8)
    for tag, mesh in enumerate(meshes):
        if mesh is None:
            continue
        vertices, faces = mesh
        z = np.ascontiguousarray(vertices[:, 2], np.float64)
        with np.errstate(invalid='ignore', divide='ignore'):
            uv = np.stack([K[0, 0] * vertices[:, 0] / z + K[0, 2], K[1, 1] * vertices[:, 1] / z + K[1, 2]], -1)
        uv[~(z > 1.0)] = np.nan
        ok = np.isfinite(uv).all(1)
        if not ok.any():
            continue
        lo = np.maximum(np.floor(uv[ok].min(0)).astype(int) - 1, 0)
        hi = np.minimum(np.ceil(uv[ok].max(0)).astype(int) + 2, [width, height])
        if (lo >= hi).any():
            continue
        _zbuffer(depth[lo[1]:hi[1], lo[0]:hi[0]], label[lo[1]:hi[1], lo[0]:hi[0]],
                 np.ascontiguousarray(uv, np.float64), z, np.ascontiguousarray(faces, np.int64),
                 float(lo[0]), float(lo[1]), tag)
    return depth, label
