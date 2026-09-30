# GT 标签审计：`outputs\unified_20`

记录 1 条，308 帧。阈值：`{"acc_spike_mm": 10.0, "rot_step_spike_deg": 30.0, "twist_pipdip_deg": 30.0, "twist_mcp_deg": 45.0, "residual_risky_rad": 2.8, "mp_mismatch_px": 40.0, "beta_saturated": 2.95, "wrist_teleport_mps": 3.0, "rig_jump_mps": 2.0, "rig_spin_dps": 400.0}`

## 总览

| 指标 | 值 |
| --- | ---: |
| recordings | 1 |
| frames | 308 |
| recordings_with_time_gaps | 1 |
| windows_total | 228 |
| windows_cross_time_gap | 2 |
| windows_presence_flicker | 12 |
| windows_contains_nan | 12 |
| hole_frames | 0 |
| edge_absent_frames | 12 |
| absent_on_screen_frames | 12 |
| absent_mediapipe_sees_frames | 12 |
| nan_in_absent_frames | 12 |
| hands_with_saturated_beta | 2 |
| saturated_beta_components | 11 |
| beta_left_right_l2_mean | 3.57 |
| fit_vs_cma_mm_mean | 10.10 |
| fit_jitter_spike_frames | 0 |
| rot_step_spike_frames | 2 |
| residual_risky_frames | 0 |
| twist_bad_frames | 604 |
| mp_err_px_p50_median | 10.59 |
| mp_err_cma_px_p50_median | 9.87 |
| mp_mismatch_frames | 0 |

## 逐手

| 记录 | 手 | exists% | holes | hole_frm | edge_abs_frm | abs_on_screen | abs_mp_sees | nan_absent | beta_sat | beta_max | fit_mm | fit_p95 | acc_rr_p99 | acc_rr_cma_p99 | jit_spk | rot_p99 | rot_spk | resid_risky | twist_pd_p95 | twist_bad | mp_px_p50 | mp_px_p90 | mp_cma_p50 | mp_mis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260820_112346 | left | 100.00 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 3.00 | 8.68 | 17.41 | 1.94 | 1.97 | 0 | 6.57 | 1 | 0 | 41.65 | 308 | 11.22 | 14.39 | 9.61 | 0 |
| 20260820_112346 | right | 96.10 | 0 | 0 | 12 | 12 | 12 | 12 | 6 | 3.00 | 11.53 | 22.65 | 4.40 | 4.39 | 0 | 14.81 | 1 | 0 | 36.17 | 296 | 9.96 | 21.42 | 10.13 | 0 |

## 时间断点（RGB 丢帧）

- 20260820_112346：帧0后 300 ms, 帧1后 67 ms；跨断点的 81 帧窗口 2/228

## exists=0 的帧

- 20260820_112346 right：12 帧（空洞 []，边缘 [[0, 12]]），画面内 12，MediaPipe 看到手 12；原始 CMA 原因 {'lost': 7, 'suspect': 2, 'settle': 6, 'teleport': 0, 'frozen': 0}
