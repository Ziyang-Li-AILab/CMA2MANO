# -*- coding: utf-8 -*-
"""convert_last_version：修复 CMA -> MANO GT 中会伤害 ACE-Ego-Hand 训练的问题。

  anatomy    MANO 解剖参考（屈曲轴、掌心法向、关节姿态分布）
  calibrate  无参考帧的 CMA->MANO 逐骨标定（修复手指绕骨轴扭转 ~90-115 度）
  shape      每个被试一组 beta（左右手共用，不贴边界）
  fit        固定 beta 的 MANO 重拟合（逐关节权重、扭转/非屈曲正则、时间平滑）
  labels     存在性状态、短空洞插值、时间分段、可训练片段、ACE 相机系标签导出
  convert    单条记录驱动
"""
