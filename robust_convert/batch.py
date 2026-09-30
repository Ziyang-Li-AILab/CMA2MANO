# -*- coding: utf-8 -*-
"""Batch helpers of the unified entry mano_convert.py (standard library only: they run with any Python 3).

  find_recordings  recording folders (with a .mcap) at or below an input path
  run_parallel     one worker process per task, a log file each, results as they finish
  write_summary    <output>/batch_summary.json and batch_summary.csv, one row per recording in <output>
  python_with      first interpreter that can import some modules
  add_switch       the --gen_video / --eval_fit_ex style options (true / false)

Every recording runs in its own process, so one failure never stops the batch.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# conversion (torch, scipy, opencv, mcap, numba) and MediaPipe / matplotlib interpreters of this machine
TORCH_PYTHON = Path(r'C:\Users\11068\miniconda3\envs\ace-ego-hand\python.exe')
MEDIAPIPE_PYTHON = Path(r'C:\Users\11068\AppData\Local\Programs\Python\Python312\python.exe')
CONVERT_MODULES = ['numpy', 'scipy', 'cv2', 'mcap', 'numba', 'torch', 'threadpoolctl']
MEDIAPIPE_MODULES = ['mediapipe', 'cv2', 'mcap', 'numpy']
REPORT_NAME = 'mano_fit_report.json'
EVAL_NAME = 'extrinsic_eval.json'
GRADES = ('PASS', 'WARN', 'FAIL')
NOT_EVALUATED = 'NOT_EVALUATED'
EVAL_ERROR = 'EVAL_ERROR'

SUMMARY_COLUMNS = ['recording', 'status', 'sample_usable', 'extrinsic_status', 'rotation_source', 'verdict',
                   'frames', 'frames_any_hand_valid', 'left_valid', 'right_valid', 'E1_median_px', 'E1_p90_px',
                   'E2_bias_mm', 'E3_rotation_max_deg', 'E3_translation_max_mm', 'E4_mahalanobis2', 'sync_offset_ms',
                   'left_mean_mm', 'right_mean_mm', 'video', 'seconds', 'error']


def str2bool(value):
    """true / false, 1 / 0, yes / no, on / off."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ('true', 't', '1', 'yes', 'y', 'on'):
        return True
    if text in ('false', 'f', '0', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError(f'expected true or false, got {value!r}')


def add_switch(parser, name, help_text):
    """--name and --name-with-dashes taking true / false; the bare option means true, the default is false."""
    flags = [f'--{name}'] + ([f'--{name.replace("_", "-")}'] if '_' in name else [])
    parser.add_argument(*flags, dest=name, type=str2bool, nargs='?', const=True, default=False,
                        metavar='true|false', help=help_text)


def find_recordings(data):
    """Recording folders (with a .mcap file): `data` itself when it is one, else every such folder below it."""
    data = Path(data)
    if not data.is_dir():
        raise SystemExit(f'input folder not found: {data}')
    if any(data.glob('*.mcap')):
        return [data]
    folders = sorted({path.parent for path in data.rglob('*.mcap')}, key=lambda path: (path.name, str(path)))
    names = [folder.name for folder in folders]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise SystemExit(f'recording folder names must be unique, the results go to <output>/<name>: '
                         f'{", ".join(repeated[:5])}')
    return folders


def python_with(modules, candidates):
    """First existing interpreter of `candidates` that imports every module (None when there is none)."""
    seen = set()
    for python in candidates:
        if not python or not Path(python).is_file():
            continue
        python = Path(python).resolve()
        if python in seen:
            continue
        seen.add(python)
        if python == Path(sys.executable).resolve():
            if all(importlib.util.find_spec(module) for module in modules):
                return python
            continue
        try:
            done = subprocess.run([str(python), '-c', 'import ' + ', '.join(modules)], capture_output=True,
                                  timeout=600)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if done.returncode == 0:
            return python
    return None


def worker_env():
    return {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1'}


def run_worker(name, command, log_path):
    started = time.perf_counter()
    try:
        done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                              errors='replace', env=worker_env())
        code, lines = done.returncode, (done.stdout + done.stderr).splitlines()
    except OSError as error:
        code, lines = -1, [f'cannot start {command[0]}: {error}']
    seconds = time.perf_counter() - started
    log_path.write_text('\n'.join([' '.join(command), ''] + lines) + '\n', encoding='utf-8')
    messages = [line for line in lines if line.strip() and 'Warn' not in line]
    return name, code, seconds, messages[-1] if messages else ''


def run_parallel(commands, jobs, log_dir, suffix='', on_done=None):
    """Run {name: command} with `jobs` processes at a time -> {name: (returncode, seconds, last line)}.

    The output of each command goes to log_dir/<name><suffix>.txt.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = [pool.submit(run_worker, name, command, log_dir / f'{name}{suffix}.txt')
                   for name, command in commands.items()]
        for future in as_completed(futures):
            name, code, seconds, last = future.result()
            results[name] = (code, seconds, last)
            if on_done is not None:
                on_done(name, code, seconds, last, len(results), len(commands))
    return results


def _get(data, *keys):
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def _load(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def is_evaluated(folder):
    """The recording in <output>/<recording> has an extrinsic evaluation (older reports included)."""
    report = _load(Path(folder) / REPORT_NAME)
    return report is not None and report.get('extrinsic_verdict') in GRADES


def result_folders(output):
    """<output>/<recording> folders with a conversion result or a failure.json (not the _logs-like folders)."""
    output = Path(output)
    if not output.is_dir():
        return []
    return sorted(path for path in output.iterdir()
                  if path.is_dir() and not path.name.startswith(('_', '.'))
                  and ((path / REPORT_NAME).is_file() or (path / 'failure.json').is_file()))


def summarize(folder, run_error=None):
    """Summary row of <output>/<recording>; run_error: this run's conversion of it failed (last log line)."""
    folder = Path(folder)
    report = None if run_error is not None else _load(folder / REPORT_NAME)
    if report is None:
        failure = _load(folder / 'failure.json')
        error, seconds = run_error or 'no mano_fit_report.json', None
        if failure is not None:
            error = f"{failure.get('stage')}: {failure.get('error_type')}: {failure.get('error')}"
            seconds = failure.get('seconds_before_failure')
        return {'recording': folder.name, 'status': 'failed', 'sample_usable': False, 'seconds': seconds,
                'error': error}
    metrics = report.get('metrics') or {}
    hands = report.get('hands') or {}
    evaluation = _load(folder / EVAL_NAME) or {}
    errors = []
    if evaluation.get('status') == 'failed':
        errors.append(f"evaluation: {evaluation.get('error')}")
    video = report.get('comparison_video') or {}
    if video.get('status') == 'failed':
        errors.append(f"video: {video.get('error')}")
    return {'recording': folder.name, 'status': 'ok', 'sample_usable': report.get('sample_usable'),
            'extrinsic_status': _get(report, 'extrinsic', 'status'),
            'rotation_source': _get(report, 'extrinsic', 'rotation_source'),
            'verdict': report.get('extrinsic_verdict'),
            'frames': report.get('n_frames'), 'frames_any_hand_valid': _get(report, 'yield', 'frames_any_hand_valid'),
            'left_valid': _get(hands, 'left', 'frames_valid') or 0,
            'right_valid': _get(hands, 'right', 'frames_valid') or 0,
            'E1_median_px': _get(metrics, 'E1_heldout_reprojection', 'median_px'),
            'E1_p90_px': _get(metrics, 'E1_heldout_reprojection', 'p90_px'),
            'E2_bias_mm': _get(metrics, 'E2_depth', 'bias_mm'),
            'E3_rotation_max_deg': _get(metrics, 'E3_stability', 'rotation_max_deg'),
            'E3_translation_max_mm': _get(metrics, 'E3_stability', 'translation_max_mm'),
            'E4_mahalanobis2': _get(metrics, 'E4_rotation_consistency', 'mahalanobis2'),
            'sync_offset_ms': _get(metrics, 'sync', 'max_abs_best_offset_ms'),
            'left_mean_mm': _get(hands, 'left', 'valid_frames', 'all_points_mm_mean'),
            'right_mean_mm': _get(hands, 'right', 'valid_frames', 'all_points_mm_mean'),
            'video': (folder / 'comparison.mp4').is_file(),
            'seconds': _get(report, 'timings', 'total_seconds'), 'error': '; '.join(errors)}


def write_summary(output, run_errors=None, this_run=(), wall_seconds=None, jobs=None):
    """batch_summary.json / .csv of every recording in `output` (earlier runs included).

    run_errors: {recording: last log line} of conversions that failed in this run.
    """
    output = Path(output)
    run_errors = run_errors or {}
    names = sorted({folder.name for folder in result_folders(output)} | set(run_errors))
    rows = [summarize(output / name, run_errors.get(name)) for name in names]
    ok = [row for row in rows if row['status'] == 'ok']
    evaluated = [row for row in ok if row['verdict'] in GRADES]
    summary = {'recordings': len(rows), 'converted': len(ok), 'failed': len(rows) - len(ok),
               'with_valid_frames': sum((row['frames_any_hand_valid'] or 0) > 0 for row in ok),
               'frames': sum(row['frames'] or 0 for row in ok),
               'frames_any_hand_valid': sum(row['frames_any_hand_valid'] or 0 for row in ok),
               'evaluated': len(evaluated), 'usable': sum(row['sample_usable'] is True for row in evaluated),
               'verdicts': {grade: sum(row['verdict'] == grade for row in evaluated) for grade in GRADES},
               'with_video': sum(bool(row.get('video')) for row in ok),
               'this_run': sorted(this_run), 'wall_seconds': wall_seconds, 'jobs': jobs, 'rows': rows}
    (output / 'batch_summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8')
    with (output / 'batch_summary.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, SUMMARY_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return summary


if __name__ == '__main__':
    raise SystemExit('the entry is mano_convert.py:  python mano_convert.py --data <input> --output <output>')
