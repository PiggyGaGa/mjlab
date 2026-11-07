# P6 时序、鲁棒性与性能验证（2026-11-07）

## 实现边界

新增独立任务
`Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth-Validity-History-Robust`，不改变
普通 G1、单帧 depth、单帧 depth+validity 或连续四帧任务。任务直接复用
主线 `ObservationManager` 的 history、delay、noise 和 partial-reset backfill：

- 保存 7 个连续控制步，CNN 以 stride 2 选择其中 4 帧，即
  `(t-6, t-4, t-2, t)`；
- depth 与 validity 使用相同的固定一控制步延迟；
- 训练时对归一化 depth 添加 `std=0.01` Gaussian range noise，并裁剪到
  `[0,1]`；play/evaluation 默认关闭随机 corruption；
- CNN 仅在项目专用 adapter 边界把四帧双通道合成 8 个输入通道。

这不是传感器物理噪声模型。当前 validity 保留 renderer 的 finite-positive
语义；Gaussian noise 只作用于 depth，不伪造 validity。

## 真实 renderer 与 PPO 验证

配置/模型测试验证 7 帧输入只生成 8 个 CNN 通道。真实 MJWarp renderer
测试验证 reset 回填、固定一帧延迟、范围裁剪及 actor 前向。以下命令完成
真实 rollout、PPO update、CNN 参数变化、actor/critic/optimizer checkpoint
重载和恢复后继续 update：

```bash
uv run tests/run_g1_3dgs_vision_smoke.py --robust-history \
  --num-envs 1 --steps 4 --iterations 1 --output /tmp/p6-robust-smoke
```

## 短性能对照

同一 RTX 3060 12 GiB、seed 42、12 steps、1 iteration；CUDA graph 均启用。
原始摘要见 `p6-temporal-benchmark-2026-11-07.json`。

| task | worlds | actor ms/batch | actor obs/s | rollout env-step/s | Torch peak MiB |
|---|---:|---:|---:|---:|---:|
| single depth | 1 | 0.369 | 2,712 | 31.1 | 34.5 |
| robust temporal | 1 | 0.397 | 2,519 | 15.7 | 39.5 |
| single depth | 4 | 0.345 | 11,582 | 41.8 | 37.8 |
| robust temporal | 4 | 0.395 | 10,138 | 37.0 | 61.6 |

样本很短，rollout 数值会受首次 kernel、reset 和调度抖动影响，不能根据
4-world robust 的单次较高 SPS 宣称时序任务更快。Torch peak 不包含 Warp、
OpenGL 或其他进程显存。该结果只证明 1/4 worlds 下两条链路可运行，并给出
当前机器的成本量级；更大 world 数应由目标训练机器按同一脚本复测。
