# P6 rollout 深度覆盖率与多资产隔离（2026-10-31）

## 真实仿真 rollout

```bash
uv run --locked tests/run_g1_3dgs_vision_smoke.py \
  --num-envs 4 --iterations 4 --depth-metrics \
  --scene-manifest src/mjlab/asset_zoo/3dgs/room1/manifest.json \
  --output /tmp/p6-rollout-depth-4
```

每个环境采样四次 24-step PPO rollout/update，并在 checkpoint 保存、
恢复后再运行一次 24-step rollout/update，共 120 帧/world、480 个环境帧。
使用随机初始化后的早期 PPO 策略，不是已训练的稳定行走策略。

统计从原始 sensor depth 计算，未从修复后的归一化图反推缺失值：

| world | 平均缺失 | 单帧最大缺失 | 平均正深度 ≥5m | 平均量程内深度 | done 次数 |
|---|---:|---:|---:|---:|---:|
| 0 | 30.02% | 62.01% | 25.30% | 44.68% | 1 |
| 1 | 20.11% | 47.82% | 26.47% | 53.42% | 2 |
| 2 | 21.93% | 37.50% | 31.56% | 46.51% | 2 |
| 3 | 15.76% | 55.92% | 20.84% | 63.40% | 2 |

全体平均缺失约 21.96%、远端约 26.04%、量程内约 52.00%。没有非有限
深度或正深度 ≤0.05m。done 只表示环境结束，未在此区分跌倒与其他原因；
图像包含自动 reset 后的帧，不能将所有帧归于连续运动轨迹。

四环境 CNN/PPO 更新、checkpoint 重载/继续更新、partial reset 隔离均通过。
原始 [运行报告](p6-rollout/report.json)、[逐帧统计](p6-rollout/frames.json)。
JSON 的计时包含 CPU 逐帧诊断开销，不作为正式吞吐/显存 benchmark。
没有使用 segmentation 将 rollout 深度归因到机器人自遮挡，不能把缺失率
称为自遮挡率；姿态分离证据仍以七姿态报告为准。

## 多资产、多相机隔离

`test_multi_asset_two_camera_world_isolation` 使用真实 MJWarp renderer：

- 两个合成高 opacity Gaussian，中心 Z 分别为 0m 和 -2m。
- 三个 world 的资产分组为 `(0,1,0)`，包含共享资产和不同资产。
- 两台相机初始高度分别为 2m 和 3m，光轴朝下。
- 中心深度分别为 `(2,4,2)` 和 `(3,5,3)` 米，与解析预期一致。
- 只将 world 0 的 rig 升高 0.5m，两台相机只改变该 world 的深度；
  其余 world 的完整图像逐像素不变，共享资产 world 2 也不受影响。
- 执行 world 0 的 scene/sensor reset 并恢复其初始 qpos 后，所有图像
  回到初始值，其他 world 的 qpos 不变。

这是合成场景的资产路由和 camera/state 隔离检查，不等于已完成多个真实
room 的碰撞模型原子切换。完整 G1 ManagerBasedRlEnv 的 partial reset
验证仍使用共享 room1 资产；不同真实资产组的任务级测试可后续加固。

## 结论与后续

缺失深度在真实早期 rollout 中占比明显，单通道 far 编码会混合 unknown 与
valid-but-distant。下一步应评估单独 validity 通道，并对比原一通道基线；
它改变网络输入，需记录模型版本和 checkpoint 兼容边界。不要将缺失值
直接解释为自由空间。当前数据尚不足以定义稳健的覆盖率验收阈值。

定向回归：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked pytest \
  tests/test_g1_depth_semantics.py tests/test_g1_3dgs_vision_config.py \
  tests/test_camera_sensor.py -q
```

结果：25 passed，`make check` 和 `make docs` 通过。
正式规模 benchmark、更多资产和训练稳定性仍未完成。
