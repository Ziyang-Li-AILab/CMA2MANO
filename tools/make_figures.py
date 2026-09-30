# -*- coding: utf-8 -*-
"""修复前后的 GT 对比图。用带 matplotlib 的 Python 3.12 运行：

    python tools/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
BEFORE = ROOT / 'audit' / 'before' / 'summary.json'
AFTER = ROOT / 'audit' / 'after' / 'summary.json'
RESULTS = ROOT / 'results'
OUT = ROOT / 'figures'

JOINTS = ['食根', '食中', '食末', '中根', '中中', '中末', '小根', '小中', '小末',
          '名根', '名中', '名末', '拇根', '拇中', '拇末']
LENGTHS = [1, 2, 3, 4, 6, 8, 10, 15]
STATE_COLORS = ['#1F4E79', '#E09F3E', '#C44536', '#D9D9D9']  # fitted filled unknown absent
C_OLD, C_NEW, C_CMA, C_REF = '#C45C26', '#1F4E79', '#6B7280', '#2F6F4E'


def setup_font():
    path = Path(r'C:\Windows\Fonts\msyh.ttc')
    if path.is_file():
        font_manager.fontManager.addfont(str(path))
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False


def load_audit(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    rows = {}
    for rec in data['records']:
        for side in ('left', 'right'):
            hand = rec['hands'].get(side, {})
            if hand.get('present'):
                rows[(rec['name'], side)] = hand
    return data, rows


def keys_of(rows):
    return sorted(rows, key=lambda k: (k[0], k[1]))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(name, flush=True)


def scoreboard(before, after):
    b, a = before['total'], after['total']
    items = [
        ('扭转超标帧', b['twist_bad_frames'], a['twist_bad_frames'], False),
        ('beta 贴边界的手', b['hands_with_saturated_beta'], a['hands_with_saturated_beta'], False),
        ('左右 beta 距离', b['beta_left_right_l2_mean'], a['beta_left_right_l2_mean'], False),
        ('缺失帧里的 NaN', b['nan_in_absent_frames'], a['nan_in_absent_frames'], False),
        ('内部空洞帧', b['hole_frames'], a['hole_frames'], False),
        ('存在性闪烁窗口', b['windows_presence_flicker'], a['windows_presence_flicker'], False),
        ('含 NaN 的窗口', b['windows_contains_nan'], a['windows_contains_nan'], False),
        ('旋转跳变帧', b['rot_step_spike_frames'], a['rot_step_spike_frames'], False),
        ('残差接近 π 的帧', b['residual_risky_frames'], a['residual_risky_frames'], False),
        ('MediaPipe 中位 px', b['mp_err_px_p50_median'], a['mp_err_px_p50_median'], False),
        ('相对 CMA 的拟合 mm', b['fit_vs_cma_mm_mean'], a['fit_vs_cma_mm_mean'], True),
    ]
    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    y = np.arange(len(items))
    labels, old, new = [], [], []
    for name, vb, va, _ in items:
        labels.append(name)
        old.append(vb)
        new.append(va)
    # 各项量纲不同，画相对旧值的比例
    ratio = [va / vb if vb else 0 for vb, va in zip(old, new)]
    colors = [C_NEW if r <= 1 else C_OLD for r in ratio]
    ax.barh(y, ratio, color=colors, height=0.72)
    ax.axvline(1, color='#888888', lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('修复后 / 修复前')
    ax.set_title('GT 标签修复前后（小于 1 表示该项下降）')
    for i, (vb, va, r) in enumerate(zip(old, new, ratio)):
        text = f'{va:.2f}  ←  {vb:.2f}' if isinstance(va, float) and not float(va).is_integer() else f'{va:g}  ←  {vb:g}'
        ax.text(max(r, 0) + 0.02, i, text, va='center', fontsize=8, color='#333333')
    ax.set_xlim(0, max(ratio) * 1.38 + 0.05)
    ax.invert_yaxis()
    save(fig, 'scoreboard.png')


def timeline():
    reports = {}
    names = []
    for folder in sorted(p for p in RESULTS.iterdir() if (p / 'mano_fit_report.json').is_file()):
        report = json.loads((folder / 'mano_fit_report.json').read_text(encoding='utf-8'))
        frame = np.load(ROOT / 'audit' / 'after' / 'per_frame' / f'{folder.name}.npz')
        reports[folder.name] = (report, frame)
        names.append(folder.name)
    rows = len(names)
    fig, axes = plt.subplots(rows, 1, figsize=(13.5, 10.8), sharex=False)
    cmap = ListedColormap(STATE_COLORS)
    for ax, name in zip(axes, names):
        report, frame = reports[name]
        count = int(report['n_frames'])
        image = np.vstack([frame[f'{side}_gt_state'] for side in ('left', 'right')])
        ax.imshow(image, aspect='auto', cmap=cmap, vmin=0, vmax=3, interpolation='nearest')
        for gap in np.flatnonzero(frame['gap_after'][:-1]):
            ax.axvline(gap + 0.5, color='#111111', lw=0.6)
        for a, b in report['clip_ranges']:
            ax.plot([a, b - 1], [2.15, 2.15], color=C_REF, lw=2, solid_capstyle='butt')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['左', '右'], fontsize=8)
        ax.set_ylim(2.45, -0.5)
        ax.set_xlim(-0.5, count - 0.5)
        ax.set_ylabel(name[4:], fontsize=7, rotation=0, ha='right', va='center', labelpad=28)
        ax.tick_params(axis='x', labelsize=6, length=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1].set_xlabel('帧')
    axes[0].set_title('逐帧状态（蓝=拟合，橙=短空洞插值，红=无可信 GT，灰=不存在）  绿线=可训练片段  黑线=RGB 断点')
    fig.tight_layout()
    save(fig, 'timeline.png')


def beta_heatmap(before_rows, after_rows):
    keys = keys_of(after_rows)
    old = np.array([before_rows[k]['shape']['beta'] for k in keys])
    new = np.array([after_rows[k]['shape']['beta'] for k in keys])
    limit = max(3.0, np.nanmax(np.abs(np.concatenate([old, new]))))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 8.2), sharey=True)
    for ax, grid, title in ((axes[0], old, '修复前：每条录制、每只手各一组'),
                            (axes[1], new, '修复后：每个被试左右手共用一组')):
        im = ax.imshow(grid, aspect='auto', cmap='coolwarm', vmin=-limit, vmax=limit, interpolation='nearest')
        ax.set_xticks(range(10))
        ax.set_xticklabels([f'β{i}' for i in range(10)], fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([f'{n[4:]} {"左" if s == "left" else "右"}' for n, s in keys], fontsize=6)
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label='beta')
    fig.suptitle('MANO 形状系数（修复前大量贴在 ±3）', y=0.98)
    save(fig, 'beta.png')


def twist_plot(before_rows, after_rows):
    keys = keys_of(after_rows)
    old = np.array([before_rows[k]['pose']['twist_per_joint_p95_deg'] for k in keys])
    new = np.array([after_rows[k]['pose']['twist_per_joint_p95_deg'] for k in keys])
    ref = None
    for folder in RESULTS.iterdir():
        report_path = folder / 'mano_fit_report.json'
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding='utf-8'))
            ref = np.array(report['hands']['left']['twist_p95_deg']['mano_reference'], np.float64)
            break
    x = np.arange(15)
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.fill_between(x, np.percentile(old, 25, 0), np.percentile(old, 75, 0), color=C_OLD, alpha=0.18, lw=0)
    ax.fill_between(x, np.percentile(new, 25, 0), np.percentile(new, 75, 0), color=C_NEW, alpha=0.18, lw=0)
    ax.plot(x, np.median(old, 0), color=C_OLD, lw=1.8, marker='o', ms=4, label='修复前 p95 中位')
    ax.plot(x, np.median(new, 0), color=C_NEW, lw=1.8, marker='o', ms=4, label='修复后 p95 中位')
    ax.plot(x, ref, color=C_REF, lw=1.4, ls='--', marker='s', ms=3.5, label='MANO 训练姿态 p95')
    ax.set_xticks(x)
    ax.set_xticklabels(JOINTS, fontsize=8)
    ax.set_ylabel('绕骨轴扭转（度）')
    ax.set_title('各关节扭转的 95 分位（色带是 40 只手的四分位）')
    ax.legend(frameon=False, ncol=3, loc='upper right')
    ax.set_ylim(0, max(np.nanmax(old), np.nanmax(ref)) * 1.08)
    save(fig, 'twist.png')


def image_agreement(before_rows, after_rows):
    keys = keys_of(after_rows)
    cma = np.array([after_rows[k].get('image', {}).get('mp_err_cma_px_p50', np.nan) for k in keys])
    old = np.array([before_rows[k].get('image', {}).get('mp_err_px_p50', np.nan) for k in keys])
    new = np.array([after_rows[k].get('image', {}).get('mp_err_px_p50', np.nan) for k in keys])
    y = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8.6, 9.2))
    h = 0.25
    ax.barh(y - h, cma, height=h, color=C_CMA, label='CMA 手套')
    ax.barh(y, old, height=h, color=C_OLD, label='修复前 MANO')
    ax.barh(y + h, new, height=h, color=C_NEW, label='修复后 MANO')
    ax.set_yticks(y)
    ax.set_yticklabels([f'{n[4:]} {"左" if s == "left" else "右"}' for n, s in keys], fontsize=7)
    ax.set_xlabel('与 MediaPipe 的逐帧中位像素误差')
    ax.set_title('图像对齐（同一组 MediaPipe 关键点）')
    ax.legend(frameon=False, loc='lower right')
    ax.invert_yaxis()
    save(fig, 'image_agreement.png')


def temporal(before_rows, after_rows):
    keys = keys_of(after_rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6))
    pairs = [
        ('rot_step_deg_p99', '局部旋转逐帧跳变 p99（度）'),
        ('acc_rootrel_mm_p99', '根相对加速度 p99（mm/帧²）'),
    ]
    for ax, (field, title) in zip(axes, pairs):
        old = np.array([before_rows[k]['temporal'][field] for k in keys])
        new = np.array([after_rows[k]['temporal'][field] for k in keys])
        limit = max(np.nanmax(old), np.nanmax(new)) * 1.08
        ax.plot([0, limit], [0, limit], color='#BBBBBB', lw=0.8)
        ax.scatter(old, new, s=22, color=C_NEW, zorder=3)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('修复前')
        ax.set_ylabel('修复后')
        ax.set_title(title, fontsize=11)
    fig.suptitle('时间平滑：点在对角线下方表示修复后更稳', y=1.02)
    save(fig, 'temporal.png')


def holdout():
    bag = {L: {'mean': [], 'p95': [], 'rr': []} for L in LENGTHS}
    for path in sorted(RESULTS.glob('*/mano_fit_report.json')):
        report = json.loads(path.read_text(encoding='utf-8'))
        for hand in report['hands'].values():
            table = hand.get('fill_holdout_mm') or {}
            for L in LENGTHS:
                row = table.get(str(L))
                if not row:
                    continue
                bag[L]['mean'].append(row['mean_mm'])
                bag[L]['p95'].append(row['p95_mm'])
                bag[L]['rr'].append(row['rootrel_mean_mm'])
    xs = np.array(LENGTHS)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    mean = np.array([np.median(bag[L]['mean']) for L in LENGTHS])
    p95 = np.array([np.median(bag[L]['p95']) for L in LENGTHS])
    rr = np.array([np.median(bag[L]['rr']) for L in LENGTHS])
    ax.plot(xs, mean, color=C_NEW, marker='o', label='关节位置 中位均值')
    ax.plot(xs, p95, color=C_OLD, marker='o', label='关节位置 中位 p95')
    ax.plot(xs, rr, color=C_REF, marker='s', label='去掉手腕后 中位均值')
    ax.axvline(6, color='#888888', ls='--', lw=0.8)
    ax.text(6.15, p95.max() * 0.92, 'MAX_FILL = 6', color='#555555', fontsize=9)
    for mark, text, dy in ((3, '实际空洞 3 帧', 0.92), (4, '实际空洞 4 帧', 0.82)):
        ax.axvline(mark, color='#C44536', lw=0.6, alpha=0.7)
        ax.text(mark + 0.12, p95.max() * dy, text, color='#C44536', fontsize=8)
    ax.set_xticks(LENGTHS)
    ax.set_xlabel('人为挖掉的连续帧数')
    ax.set_ylabel('插值相对真实拟合的误差（mm）')
    ax.set_title('留出实验：短空洞插值能补到多准（40 只手的中位）')
    ax.legend(frameon=False)
    save(fig, 'holdout.png')
    summary = {str(L): {'median_mean_mm': float(np.median(bag[L]['mean'])),
                        'median_p95_mm': float(np.median(bag[L]['p95'])),
                        'median_rootrel_mm': float(np.median(bag[L]['rr']))} for L in LENGTHS}
    (OUT / 'holdout.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')


def main():
    setup_font()
    before, before_rows = load_audit(BEFORE)
    after, after_rows = load_audit(AFTER)
    scoreboard(before, after)
    timeline()
    beta_heatmap(before_rows, after_rows)
    twist_plot(before_rows, after_rows)
    image_agreement(before_rows, after_rows)
    temporal(before_rows, after_rows)
    holdout()
    print('wrote', OUT)


if __name__ == '__main__':
    main()
