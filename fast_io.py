# -*- coding: utf-8 -*-
"""One-pass pose read plus a cached RGB index.

RGB and VIO live in different MCAP chunks. The camera stamp equals the message
index time, but the valid flag is inside the image JSON, so that flag is scanned
once into optim_speed/cache/rgb. Conversion then decompresses only pose chunks.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap.records import Chunk
from mcap.stream_reader import ReadDataStream, get_chunk_data_stream

from robust_convert.cma_source import read_human, read_rig
from robust_convert.cleaning import repair_hand
from robust_convert.mcap_io import RGB_TOPIC, POSE_TOPIC, find_inputs, load_intrinsics
from robust_convert.session import SIDES, resample

RGB_CACHE = Path(__file__).resolve().parent / 'cache' / 'rgb'
# Filled by load_session. One recording at a time inside each worker.
VIO_BY_MCAP = {}
# Optional multiprocessing.Semaphore. Caps how many pose chunks decompress at once
# across the worker pool, so a lone long file can use several cores.
ZSTD_SLOTS = None
DECODE_THREADS = 4
_DECODE_POOL = None
_MESSAGE = 5


def _rgb_stamp(data, log_time):
    """(stamp, usable) matching rgb_index. Unparseable messages are invalid and keep log_time."""
    try:
        if not data.startswith(b'{"timestamp":{"sec":'):
            raise ValueError('header')
        comma = data.index(b',', 20)
        sec = int(data[20:comma])
        key = data.index(b'"nsec":', comma, comma + 24)
        start = key + 7
        end = data.index(b'}', start, start + 16)
        nsec = int(data[start:end])
        stamp = sec * 1_000_000_000 + nsec
        data_key = data.find(b'"data":', 0, 96)
        if data_key < 0:
            has_data = False
        else:
            quote = data_key + 7
            has_data = quote < len(data) and data[quote:quote + 1] == b'"' and data[quote + 1:quote + 2] != b'"'
        tail = data[-192:]
        flag = tail.rfind(b'"valid":')
        if flag < 0:
            valid = True
        else:
            token = tail[flag + 8:flag + 13]
            valid = token.startswith(b'true')
        return stamp, bool(valid and has_data)
    except (ValueError, IndexError):
        return int(log_time), False


def _pose_record(data):
    """One pose JSON object, or None when it does not match read_vio_pose's try-block."""
    try:
        item = json.loads(data)
        pose = item['pose']
        position = [float(pose['position'][axis]) for axis in 'xyz']
        quat = [float(pose['orientation'][axis]) for axis in 'xyzw']
        ok = bool(item.get('valid', True)) and item.get('tracking_state', 'OK') == 'OK'
        stamp = int(item['timestamp']['sec']) * 1_000_000_000 + int(item['timestamp']['nsec'])
        score = float(item.get('confidence', np.nan))
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    norm = float(np.linalg.norm(quat))
    ok = ok and bool(np.isfinite(position).all() and np.isfinite(quat).all() and abs(norm - 1) < 0.02)
    return stamp, position, quat, ok, score


def _skip(stream, nbytes):
    """Advance a decompressed chunk without copying the skipped bytes."""
    if nbytes <= 0:
        return
    stream._stream.seek(nbytes, 1)
    stream._count += nbytes


def _decode_chunk(task):
    """Pose records inside one chunk. Non-pose payloads are skipped, not copied."""
    pose_id, chunk = task
    slots = ZSTD_SLOTS
    if slots is not None:
        slots.acquire()
    try:
        stream, length = get_chunk_data_stream(chunk, validate_crc=False)
        found = []
        while stream.count < length:
            opcode = stream.read1()
            rec_len = stream.read8()
            if opcode != _MESSAGE:
                _skip(stream, rec_len)
                continue
            channel_id = stream.read2()
            rest = rec_len - 2
            if channel_id != pose_id:
                _skip(stream, rest)
                continue
            _skip(stream, 20)  # sequence, log time, publish time
            record = _pose_record(stream.read(rest - 20))
            if record is not None:
                found.append(record)
        return found
    finally:
        if slots is not None:
            slots.release()


def _decode_pool():
    global _DECODE_POOL
    if _DECODE_POOL is None:
        _DECODE_POOL = ThreadPoolExecutor(max_workers=DECODE_THREADS, thread_name_prefix='zstd')
    return _DECODE_POOL


def _pose_chunks(mcap):
    with Path(mcap).open('rb') as handle:
        summary = make_reader(handle).get_summary()
        pose_id = next(channel.id for channel in summary.channels.values() if channel.topic == POSE_TOPIC)
        for chunk_index in summary.chunk_indexes:
            if pose_id not in chunk_index.message_index_offsets:
                continue
            handle.seek(chunk_index.chunk_start_offset + 1 + 8)
            yield pose_id, Chunk.read(ReadDataStream(handle))


def read_pose(mcap):
    """VIO dict matching read_vio_pose, decoding pose chunks on a few threads."""
    pose_stamps, position, quat, pose_valid, confidence = [], [], [], [], []
    batch = []
    for task in _pose_chunks(mcap):
        batch.append(task)
        if len(batch) >= 48:
            _consume_pose(batch, pose_stamps, position, quat, pose_valid, confidence)
            batch = []
    if batch:
        _consume_pose(batch, pose_stamps, position, quat, pose_valid, confidence)
    order = np.argsort(pose_stamps, kind='stable')
    return {'stamps': np.asarray(pose_stamps, np.int64)[order],
            'position_m': np.asarray(position, np.float64).reshape(-1, 3)[order],
            'quat_xyzw': np.asarray(quat, np.float64).reshape(-1, 4)[order],
            'valid': np.asarray(pose_valid, bool)[order],
            'confidence': np.asarray(confidence, np.float64)[order]}


def _consume_pose(batch, pose_stamps, position, quat, pose_valid, confidence):
    for found in _decode_pool().map(_decode_chunk, batch):
        for stamp, pos, rot, ok, score in found:
            pose_stamps.append(stamp)
            position.append(pos)
            quat.append(rot)
            pose_valid.append(ok)
            confidence.append(score)


def _rgb_from_chunk(task):
    """(stamp, log_time, flag) for RGB messages. The JPEG body is not copied."""
    rgb_id, chunk = task
    stream, length = get_chunk_data_stream(chunk, validate_crc=False)
    found = []
    while stream.count < length:
        opcode = stream.read1()
        rec_len = stream.read8()
        if opcode != _MESSAGE:
            _skip(stream, rec_len)
            continue
        channel_id = stream.read2()
        rest = rec_len - 2
        if channel_id != rgb_id:
            _skip(stream, rest)
            continue
        _skip(stream, 4)  # sequence
        log_time = stream.read8()
        _skip(stream, 8)  # publish time
        data_len = rest - 20
        if data_len <= 288:
            data = stream.read(data_len)
        else:
            head = stream.read(96)
            _skip(stream, data_len - 288)
            data = head + stream.read(192)
        stamp, ok = _rgb_stamp(data, log_time)
        found.append((stamp, int(log_time), ok))
    return found


def _rgb_chunks(mcap):
    with Path(mcap).open('rb') as handle:
        summary = make_reader(handle).get_summary()
        rgb_id = next(channel.id for channel in summary.channels.values() if channel.topic == RGB_TOPIC)
        for chunk_index in summary.chunk_indexes:
            if rgb_id not in chunk_index.message_index_offsets:
                continue
            handle.seek(chunk_index.chunk_start_offset + 1 + 8)
            yield rgb_id, Chunk.read(ReadDataStream(handle))


def scan_rgb(mcap):
    """RGB index arrays. Image bodies are skipped; the valid flag is in the JSON tail."""
    rows = []
    batch = []
    for task in _rgb_chunks(mcap):
        batch.append(task)
        if len(batch) >= 48:
            for found in _decode_pool().map(_rgb_from_chunk, batch):
                rows.extend(found)
            batch = []
    if batch:
        for found in _decode_pool().map(_rgb_from_chunk, batch):
            rows.extend(found)
    stamps, log_times, valid = [], [], []
    last = None
    for stamp, log_time, ok in rows:
        if ok and last is not None and stamp <= last:
            ok = False
        if ok:
            last = stamp
        stamps.append(stamp)
        log_times.append(log_time)
        valid.append(ok)
    return {'stamps': np.asarray(stamps, np.int64), 'log_times': np.asarray(log_times, np.int64),
            'valid': np.asarray(valid, bool)}


def _cache_path(mcap):
    return RGB_CACHE / f'{Path(mcap).parent.name}.npz'


def load_rgb_index(mcap):
    """Cached RGB index, scanning the image chunks only when the cache is missing."""
    mcap = Path(mcap)
    path = _cache_path(mcap)
    size = mcap.stat().st_size
    if path.is_file():
        try:
            with np.load(path) as blob:
                if int(blob['mcap_bytes']) == size:
                    return {'stamps': blob['stamps'].astype(np.int64, copy=False),
                            'log_times': blob['log_times'].astype(np.int64, copy=False),
                            'valid': blob['valid'].astype(bool, copy=False)}
        except (OSError, ValueError, KeyError):
            pass
    rgb = scan_rgb(mcap)
    RGB_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, stamps=rgb['stamps'], log_times=rgb['log_times'], valid=rgb['valid'],
             mcap_bytes=np.int64(size))
    return rgb


def read_mcap(mcap):
    """RGB index and VIO dict. Image chunks are not decompressed when the cache exists."""
    return load_rgb_index(mcap), read_pose(mcap)


def load_session(folder, cleaning=None):
    """load_session drop-in. Pose decode overlaps the CMA parse."""
    folder = Path(folder)
    mcap, human, body = find_inputs(folder)
    K, width, height, intrinsics_info = load_intrinsics(mcap)
    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_mcap = pool.submit(read_mcap, mcap)
        fut_human = pool.submit(read_human, human)
        fut_rig = pool.submit(read_rig, body)
        rgb, vio = fut_mcap.result()
        source = fut_human.result()
        rig = fut_rig.result()
    VIO_BY_MCAP[str(mcap)] = vio
    usable = rgb['valid']
    stamps = rgb['stamps'][usable]
    cma_ns = source['timestamps_ns']
    if np.any(np.diff(cma_ns) <= 0) or np.any(np.diff(rig['ptp_ns']) <= 0):
        raise ValueError('CMA timestamps must be strictly increasing')
    hands = {}
    for side in SIDES:
        hand = source['hands'][side]
        if not hand['present']:
            hands[side] = {'present': False, 'reason': hand.get('reason', 'absent')}
            continue
        repaired = repair_hand(hand['points_mm'], hand['pos_valid'], hand['global_R'], hand['local_R'],
                               hand['rot_valid'], **(cleaning or {}))
        hands[side] = {'present': True, 'schema': hand['schema'], 'defects': hand['defects'],
                       'cleaning': repaired['diagnostics'],
                       'cma_points_mm': repaired['points_mm'], 'cma_status': repaired['status'],
                       'cma_world_R': repaired['world_R'],
                       'raw_points_mm': hand['points_mm']}
    session = {'folder': folder, 'name': folder.name, 'mcap': mcap, 'human': human, 'body': body,
               'K': K, 'width': width, 'height': height, 'intrinsics_info': intrinsics_info,
               'rgb': rgb, 'stamps': stamps, 'source_indices': np.flatnonzero(usable),
               'cma_ns': cma_ns, 'rig_source': rig, 'hands_source': hands,
               'malformed_rows': {'human': source['malformed_rows'], 'body': rig['malformed_rows']}}
    resample(session, 0)
    return session


def cached_vio_pose(mcap):
    """read_vio_pose drop-in. The pose stream was stored by load_session."""
    key = str(Path(mcap))
    if key not in VIO_BY_MCAP:
        VIO_BY_MCAP[key] = read_pose(mcap)
    return VIO_BY_MCAP[key]


def _build_rgb_one(folder):
    import time
    folder = Path(folder)
    mcap = next(folder.glob('*.mcap'))
    started = time.perf_counter()
    rgb = load_rgb_index(mcap)
    return folder.name, int(rgb['stamps'].shape[0]), int((~rgb['valid']).sum()), time.perf_counter() - started


def build_missing_rgb_cache(folders, jobs=8):
    """Scan image headers for recordings whose cache is absent or the wrong size."""
    import multiprocessing as mp
    pending = []
    for folder in folders:
        folder = Path(folder)
        mcap = next(folder.glob('*.mcap'))
        path = _cache_path(mcap)
        if path.is_file():
            try:
                with np.load(path) as blob:
                    if int(blob['mcap_bytes']) == mcap.stat().st_size and blob['stamps'].shape == blob['valid'].shape:
                        continue
            except (OSError, ValueError, KeyError):
                pass
        pending.append(str(folder))
    if not pending:
        print(f'rgb cache ready ({len(folders)} files)', flush=True)
        return
    print(f'building rgb cache for {len(pending)} files', flush=True)
    ctx = mp.get_context('spawn')
    with ctx.Pool(min(jobs, len(pending))) as pool:
        for name, count, invalid, seconds in pool.imap_unordered(_build_rgb_one, pending):
            print(f'rgb {name}  frames {count}  invalid {invalid}  {seconds:.1f}s', flush=True)
