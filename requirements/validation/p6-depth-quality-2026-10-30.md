# P6 深度语义与相机质量诊断（2026-10-30）

本轮发现并修复 G1 observation 对缺失深度的编码错误，同时确认一个固定
reset 姿态下的相机方向和自遮挡范围。P6 尚未宣称动态深度质量合格。

> **2026-11-11 更新：**本文是旧后端固定阈值 `0.005` 下的历史验收记录。
> 当前锁定的 MuJoCo Warp fork 在 `create_render_context` 中新增
> `splat_depth_transmittance_threshold`，默认值为 `0.1`。下述数值尚未在新默认值
> 下复测，不能直接代表当前环境；现行接口和迁移说明见用户文档及
> `mjlab-3dgs-todo.md`。

## 1. 实际深度定义

依据安装的 `mujoco_warp/_src/render.py` 中 `ray_splat`、`shade_splats`
和最终 depth 写入路径，以及真实 GPU 解析测试：

- mesh 命中输出沿相机光轴的 Z-depth（米），不是沿归一化光线的欧氏距离。
- Gaussian 贡献按射线顺序合成；当剩余 transmittance 小于 0.005 时，记录
  当时 Gaussian 的 ray distance，再投影到相机光轴。深度不是高斯中心的
  最近距离、median depth，也不是 alpha 加权期望深度。
- 有颜色贡献不保证有 Gaussian depth。累计 opacity 未达到阈值时，仍可能
  有 RGB 而没有有效 splat depth；如果有后方 mesh，输出可退回 mesh 深度。
- 无 mesh 命中、且 Gaussian 未形成上述深度时，输出零。零可能表示射线
  无命中或累积不够不透明；单独 depth 通道无法再区分这两种原因。

解析测试覆盖：z=0 平面与 z=2 相机的全图深度为 2 米（含离轴像素）；
空场景为零；近乎不透明的中心 Gaussian 给出 2 米；opacity=0.1 的
Gaussian 无有效深度。测试依据当前安装实现，不声称所有 3DGS renderer
采用该语义。

## 2. 已修复的 observation 问题

原 `camera_depth` 将零值经 `clip(0.05,5)/5` 转成 0.01，等同于近处表面。
G1 现在显式设置 `zero_depth_as_far=True`，先将零替换为 cutoff，再裁剪和
归一化。普通任务默认值为 False，兼容原有行为；原始 sensor tensor 不变。
NaN 不被该选项吞掉，仍由 observation 的 `nan_policy="error"` 检测。

这修复的是缺失值编码，不是补全几何。归一化值 1 同时代表缺失或超量程，
不可当作已证实的自由空间；后续可评估独立 validity mask、coverage 阈值或
资产修复，但不能通过平滑图像掩盖无效深度。本修改不改变网络维度，旧视觉
checkpoint 仍可能在数值上加载，但输入语义发生变化，不能作为完全同分布恢复。

## 3. 真实 room1 固定姿态对照

运行：

```bash
uv run tests/run_g1_depth_diagnostics.py --output /tmp/p6-depth-diagnostics
```

使用 play 配置、seed=42、room1 manifest 和完全相同 qpos，qvel=0。
每种模式重复感知固定场景，再把 root 向 world +X 移动 0.5 米；不推进物理。
每次感知重新执行 forward/sense 并刷新 camera cache。

| 模式 | 缺失零值 | 正的近端深度 ≤0.05m | 有效深度 ≥5m | 固定姿态重复差异 |
|---|---:|---:|---:|---:|
| 机器人 + Gaussian | 19.14% | 0% | 25.52% | 0 |
| 仅机器人 | 100% | 0% | 0% | 0 |
| 仅 Gaussian | 19.14% | 0% | 25.52% | 0 |

组合图与 Gaussian-only raw depth 完全一致。该 reset 姿态下，相机视野未
击中机器人 mesh，故零值不是自遮挡；不推广到手臂抬起、转身或摔倒等姿态。
相机 forward world 约为 `(0.934,-0.103,-0.342)`，下俯角 19.99998 度。
移动后组合/仅 Gaussian 图像约 90.49% 像素改变，固定相机则重复一致。
有效正深度中位数约 3.15 米，最小约 1.14 米。

本次是固定 play/reset 姿态，不能与上一轮训练随机 reset 的 24.0%/28.2%
裁剪比例直接作因果对比。报告中的缺失和远端比例分别从 raw depth 统计。

原始 [JSON](p6-depth/diagnostics.json)，图像：
[组合](p6-depth/combined.png)、[仅机器人](p6-depth/robot-only.png)、
[仅 Gaussian](p6-depth/gaussian-only.png)。所有 PNG 使用修复后的编码：
白色既可能缺失，也可能远端。完整 raw NPY 保存在 `/tmp/p6-depth-diagnostics`。

## 4. 后端边界与验证结果

诊断中直接使用空 geom groups 的 splat-only 路径发生原生退出（139）。
在 group 5 放置 `(1000,1000,1000)` 的无碰撞小球、让 BVH 非空后诊断通过。
该几何离开相机视野，仅用于诊断；没有改变生产训练任务。这里记录的是
当前安装后端的空 mesh BVH 限制，尚未修复后端本身。

修复后的双环境 manifest 训练 smoke 通过两次 24-step PPO 更新，以及
checkpoint 重载后再一次更新；局部 reset 隔离通过。
见 [训练 smoke JSON](p6-depth/training-smoke.json)。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest \
  tests/test_g1_depth_semantics.py tests/test_g1_3dgs_vision_config.py \
  tests/test_camera_sensor.py -q
make check
```

结果：24 passed；Ruff、ty、Pyright 通过。
下一步应覆盖动态自遮挡/缺失值比例、不同资产组隔离和规模显存/吞吐；
本轮不改变 Gaussian 深度算法或降低其 opacity 阈值，也不证明实际视觉任务收益。
