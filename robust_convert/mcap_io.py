# -*- coding: utf-8 -*-
"""Tolerant MCAP access: intrinsics, RGB/depth by timestamp, ZED VIO pose and IMU.

The original pipeline aborts when an RGB message inside the recording is
invalid ("Internal missing RGB slots"). Here every RGB message is indexed with
its validity, and consumers simply skip the invalid ones.

This module only needs numpy, OpenCV and mcap, so the MediaPipe interpreter
(Python 3.12) can import it as well.
"""
from __future__ import annotations

import json
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader

try:
    import pybase64 as base64
except ImportError:  # the MediaPipe interpreter only has the standard module
    import base64

RGB_TOPIC = '/main/left/image'
DEPTH_TOPIC = '/main/depth_raw'
POSE_TOPIC = '/main/pose'
IMU_TOPIC = '/main/imu'
# Threads of one topic read (read_topic): the zstd decompression and the file reads release the GIL.
READ_THREADS = 4


def find_inputs(folder):
    """One recording folder: exactly one .mcap, *_Human.cma and *_Body.cma."""
    folder = Path(folder)
    found = {}
    for key, pattern in (('mcap', '*.mcap'), ('human', '*_Human.cma'), ('body', '*_Body.cma')):
        hits = sorted(folder.glob(pattern))
        if len(hits) != 1:
            raise FileNotFoundError(f'{folder}: expected one {pattern}, found {len(hits)}')
        found[key] = hits[0]
    return found['mcap'], found['human'], found['body']


def _stamp(value):
    return int(value['sec']) * 1_000_000_000 + int(value['nsec'])


def load_intrinsics(mcap):
    """Rectified main-left pinhole K and resolution. Non-zero distortion is reported, not fatal."""
    with Path(mcap).open('rb') as handle:
        meta = next((item for item in make_reader(handle).iter_metadata()
                     if item.name == 'telemetry.camera_intrinsics'), None)
    if meta is None:
        raise ValueError('telemetry.camera_intrinsics metadata is missing')
    camera = json.loads(meta.metadata['json'])['cameras']['main']
    left = camera['left']
    K = np.array([[left['fx'], 0, left['cx']], [0, left['fy'], left['cy']], [0, 0, 1]], np.float64)
    distortion = np.asarray(left.get('disto', []), np.float64)
    info = {'distortion_nonzero': bool(np.any(distortion != 0))}
    return K, int(camera['resolution']['width']), int(camera['resolution']['height']), info


def parse_rgb(message):
    """(camera stamp, flag, parsed JSON item) of one RGB message.

    flag: the JSON parses, the message is flagged valid and carries data. An
    unparsable message gets its log time as stamp and no item.
    """
    try:
        item = json.loads(message.data)
        return _stamp(item['timestamp']), bool(item.get('valid', True) and item.get('data')), item
    except (ValueError, KeyError, TypeError):
        return int(message.log_time), False, None


class UsableRule:
    """Ordering rule of the RGB stream, message by message in file order: a flagged message is
    usable when its stamp is strictly later than the previous usable one."""

    def __init__(self):
        self.last = None

    def __call__(self, stamp, flag):
        usable = bool(flag) and (self.last is None or stamp > self.last)
        if usable:
            self.last = stamp
        return usable


def usable_flags(stamps, flags):
    """UsableRule over whole arrays (every RGB message in file order)."""
    rule = UsableRule()
    return np.array([rule(int(stamp), flag) for stamp, flag in zip(stamps, flags)], bool).reshape(-1)


def topic_ranges(mcap, topic, parts, min_chunks=1):
    """Log-time ranges [(start, end), ...] (start inclusive, end exclusive, None = open) splitting the
    messages of `topic` into at most `parts` contiguous pieces of about the same number of chunks, each
    at least `min_chunks` chunks long.

    Any split reads the same messages: iter_messages orders by log time, and a log time falls in exactly
    one range. One open range when the file has no usable summary (it is then read linearly).
    """
    try:
        with Path(mcap).open('rb') as handle:
            summary = make_reader(handle).get_summary()
    except Exception:  # noqa: BLE001 - a damaged summary: one linear read raises or copes as before
        return [(None, None)]
    if summary is None:
        return [(None, None)]
    ids = {cid for cid, channel in summary.channels.items() if channel.topic == topic}
    starts = sorted(chunk.message_start_time for chunk in summary.chunk_indexes
                    if ids.intersection(chunk.message_index_offsets))
    parts = max(1, min(int(parts), len(starts) // max(1, int(min_chunks))))
    bounds = sorted({starts[round(i * len(starts) / parts)] for i in range(1, parts)})
    edges = [None, *bounds, None]
    return list(zip(edges[:-1], edges[1:]))


def read_topic(mcap, topic, parse, threads=READ_THREADS, min_chunks=16):
    """[parse(message) for every message of `topic`] in the order of one iter_messages pass, read by
    `threads` threads over contiguous log-time ranges (topic_ranges) and joined in time order."""
    spans = topic_ranges(mcap, topic, threads, min_chunks) if threads > 1 else [(None, None)]

    def one(span):
        with Path(mcap).open('rb') as handle:
            return [parse(message) for _, _, message in
                    make_reader(handle).iter_messages(topics=[topic], start_time=span[0], end_time=span[1])]

    if len(spans) == 1:
        return one(spans[0])
    with ThreadPoolExecutor(len(spans)) as pool:
        return [item for part in pool.map(one, spans) for item in part]


def _index_entry(message):
    stamp, flag, _ = parse_rgb(message)
    return stamp, int(message.log_time), flag


def rgb_index(mcap, threads=READ_THREADS):
    """Every RGB message in file order: camera stamp, log time and validity.

    A message is usable when its JSON parses, it is flagged valid, carries data
    and its stamp is strictly later than the previous usable one. tools/mp_detect.py
    stores the same index in the MediaPipe cache (see cached_rgb_index).
    """
    rows = read_topic(mcap, RGB_TOPIC, _index_entry, threads)
    stamps = np.asarray([row[0] for row in rows], np.int64)
    return {'stamps': stamps, 'log_times': np.asarray([row[1] for row in rows], np.int64),
            'valid': usable_flags(stamps, [row[2] for row in rows])}


def cached_rgb_index(cache_path, mcap):
    """The RGB index stored by tools/mp_detect.py, or None (older cache, other file size, unreadable)."""
    try:
        with np.load(cache_path) as cache:
            if 'index_valid' not in cache.files or int(cache['mcap_bytes']) != Path(mcap).stat().st_size:
                return None
            index = {'stamps': cache['index_stamps'].astype(np.int64), 'log_times':
                     cache['index_log_times'].astype(np.int64), 'valid': cache['index_valid'].astype(bool)}
    except Exception:  # noqa: BLE001 - a damaged cache: the index is read from the MCAP
        return None
    if not len(index['stamps']) == len(index['log_times']) == len(index['valid']):
        return None
    return index


def decode_jpeg_item(item, reduce=1):
    """BGR image from a parsed RGB JSON message; reduce in (1, 2, 4, 8) decodes at lower resolution."""
    flags = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
             4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}
    data = base64.b64decode(item['data'])
    image = cv2.imdecode(np.frombuffer(data, np.uint8), flags[reduce])
    if image is None:
        raise ValueError('JPEG decode failed')
    return image


def iter_rgb_items(mcap, wanted=None):
    """Yield (stamp, parsed JSON item) for usable RGB messages, optionally only stamps in `wanted`."""
    wanted = None if wanted is None else {int(value) for value in wanted}
    rule = UsableRule()
    with Path(mcap).open('rb') as handle:
        for _, _, message in make_reader(handle).iter_messages(topics=[RGB_TOPIC]):
            stamp, flag, item = parse_rgb(message)
            if rule(stamp, flag) and (wanted is None or stamp in wanted):
                yield stamp, item


def _varint(data, offset):
    value = shift = 0
    while offset < len(data) and shift < 70:
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, offset
        shift += 7
    raise ValueError('Malformed protobuf varint')


def _protobuf_fields(data):
    fields, offset = {}, 0
    while offset < len(data):
        key, offset = _varint(data, offset)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, offset = _varint(data, offset)
        elif wire == 1:
            value = struct.unpack_from('<d', data, offset)[0]
            offset += 8
        elif wire == 2:
            size, offset = _varint(data, offset)
            value, offset = data[offset:offset + size], offset + size
        elif wire == 5:
            value = struct.unpack_from('<f', data, offset)[0]
            offset += 4
        else:
            raise ValueError(f'Unsupported protobuf wire type {wire}')
        fields[number] = value
    return fields


def decode_depth(data):
    """(stamp, depth in mm as float32 or None). Same field layout as the original decoder."""
    fields = _protobuf_fields(data)
    stamp = int(fields.get(1, 0))
    if not fields.get(11, 0) or not fields.get(9):
        return stamp, None
    width, height, stride = (int(fields[key]) for key in (3, 4, 8))
    encoding = fields[5].decode()
    dtype = np.dtype('<u2' if encoding == '16UC1' else '<f4')
    image = np.frombuffer(fields[9], dtype).reshape(height, stride // dtype.itemsize)[:, :width]
    unit = fields[6].decode()
    depth = image.astype(np.float32) * float(fields.get(7, 1.0)) * (1000 if unit == 'meter' else 1)
    depth[image == fields.get(10, 0)] = 0
    depth[~np.isfinite(depth)] = 0
    return stamp, depth


def read_frames(mcap, stamps, depth=True, reduce=1):
    """Seek RGB (and depth) messages logged exactly at the requested stamps.

    Returns {stamp: (image or None, depth or None)}; missing pieces stay None.
    """
    topics = [RGB_TOPIC] + ([DEPTH_TOPIC] if depth else [])
    result = {}
    with Path(mcap).open('rb') as handle:
        reader = make_reader(handle)
        for requested in sorted({int(value) for value in stamps}):
            image = depth_mm = None
            for _, channel, message in reader.iter_messages(topics=topics, start_time=requested,
                                                            end_time=requested + 1):
                try:
                    if channel.topic == RGB_TOPIC:
                        item = json.loads(message.data)
                        if _stamp(item['timestamp']) == requested and item.get('valid', True) and item.get('data'):
                            image = decode_jpeg_item(item, reduce)
                    else:
                        _, depth_mm = decode_depth(message.data)
                except (ValueError, KeyError, TypeError):
                    continue
            result[requested] = (image, depth_mm)
    return result


def read_depth(mcap, stamps):
    """Only the depth maps logged exactly at the requested stamps: {stamp: depth mm or None}.

    The ZED depth is aligned to the rectified left image and stamped with the
    same camera stamp; a map whose own stamp differs is not used.
    """
    result = {}
    with Path(mcap).open('rb') as handle:
        reader = make_reader(handle)
        for requested in sorted({int(value) for value in stamps}):
            depth_mm = None
            for _, _, message in reader.iter_messages(topics=[DEPTH_TOPIC], start_time=requested,
                                                      end_time=requested + 1):
                try:
                    stamp, candidate = decode_depth(message.data)
                except (ValueError, KeyError, TypeError, struct.error):
                    continue
                if stamp == requested and candidate is not None:
                    depth_mm = candidate
            result[requested] = depth_mm
    return result


def _pose_entry(message):
    """(stamp, position, quaternion, ok, confidence) of one VIO message; None when it does not parse."""
    try:
        item = json.loads(message.data)
        pose = item['pose']
        p = [float(pose['position'][axis]) for axis in 'xyz']
        q = [float(pose['orientation'][axis]) for axis in 'xyzw']
        ok = bool(item.get('valid', True)) and item.get('tracking_state', 'OK') == 'OK'
        stamp = _stamp(item['timestamp'])
        score = float(item.get('confidence', np.nan))
    except (ValueError, KeyError, TypeError):
        return None
    norm = np.linalg.norm(q)
    ok &= bool(np.isfinite(p).all() and np.isfinite(q).all() and abs(norm - 1) < 0.02)
    return stamp, p, q, ok, score


def read_vio_pose(mcap, threads=READ_THREADS):
    """ZED visual-inertial camera pose: stamps, position (m), quaternion xyzw, validity."""
    rows = [row for row in read_topic(mcap, POSE_TOPIC, _pose_entry, threads) if row is not None]
    stamps, position, quat, valid, confidence = ([row[i] for row in rows] for i in range(5))
    order = np.argsort(stamps, kind='stable')
    return {'stamps': np.asarray(stamps, np.int64)[order],
            'position_m': np.asarray(position, np.float64).reshape(-1, 3)[order],
            'quat_xyzw': np.asarray(quat, np.float64).reshape(-1, 4)[order],
            'valid': np.asarray(valid, bool)[order],
            'confidence': np.asarray(confidence, np.float64)[order]}


def read_imu(mcap):
    """IMU gyro in rad/s (the stream is deg/s) and acceleration, sorted by stamp."""
    stamps, gyro, accel = [], [], []
    with Path(mcap).open('rb') as handle:
        for _, _, message in make_reader(handle).iter_messages(topics=[IMU_TOPIC]):
            try:
                item = json.loads(message.data)
                w = item['angular_velocity']
                a = item.get('linear_acceleration', {'x': np.nan, 'y': np.nan, 'z': np.nan})
                stamp = _stamp(item['timestamp']) if 'timestamp' in item else int(message.log_time)
                gyro.append([float(w[axis]) for axis in 'xyz'])
                accel.append([float(a[axis]) for axis in 'xyz'])
                stamps.append(stamp)
            except (ValueError, KeyError, TypeError):
                continue
    order = np.argsort(stamps, kind='stable')
    return {'stamps': np.asarray(stamps, np.int64)[order],
            'gyro_rad_s': np.radians(np.asarray(gyro, np.float64).reshape(-1, 3)[order]),
            'accel': np.asarray(accel, np.float64).reshape(-1, 3)[order]}
