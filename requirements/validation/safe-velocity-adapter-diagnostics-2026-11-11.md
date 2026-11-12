# Safe Velocity 规则 adapter 左右恢复诊断（2026-11-11）

本记录只验证固定 apartment 左右夹具上的规则状态链路，不是 P7 避障成功、
碰撞安全或训练效果验收。

## 1. 运行条件

- GPU：NVIDIA GeForce RTX 3060 12 GiB；
- environment step：0.02 s；
- seed：0；
- checkpoint：`logs/rsl_rl/g1_velocity/2026-10-24_16-15-47/model_29999.pt`；
- checkpoint SHA-256：
  `8ebe3be70a4078e24d07b13bde8be567e7096fd45335c263b8d269b82ac9a5ff`；
- nominal command：catalog 固定的 `(0.5, 0.0, 0.0)`；
- 每个场景 650 步，禁用 auto-reset，终止时立即停止记录。

命令：

```bash
uv run tests/run_g1_3dgs_safe_velocity_adapter.py \
  --checkpoint logs/rsl_rl/g1_velocity/2026-10-24_16-15-47/model_29999.pt \
  --scenario left_open --steps 650 --output /tmp/p7-left-open-12s.json

uv run tests/run_g1_3dgs_safe_velocity_adapter.py \
  --checkpoint logs/rsl_rl/g1_velocity/2026-10-24_16-15-47/model_29999.pt \
  --scenario right_open --steps 650 --output /tmp/p7-right-open-12s.json
```

## 2. 问题定位与标定

旧参数下，left/right 各 350 步均全部处于 `brake`。前方净空范围分别为
0.882--1.147 m 和 0.882--1.146 m；adapted forward command 最低约 0.235 m/s，
但 checkpoint 已基本停止，因此无法继续接近原 `0.6 m` STOP 门槛。

本轮增加 `search_trigger_distance_m=1.0`，让最低安全净空 `0.6 m` 与开始搜索
条件分离。6 秒预算可进入 `search`，但 left 实际只旋转约 0.376 rad（21.5 度），
尚未进入 catalog 的 30--80 度安全扇区便 blocked。根据实际 checkpoint 响应，
catalog 和六个固定 scenario 的搜索预算统一标定为 12 秒。

## 3. 标定后结果

| 场景 | 首次 STOP | 首次 SEARCH | 首次 RESUME | 首次 TRACK | blocked | terminated |
|---|---:|---:|---:|---:|---:|---:|
| left_open | 0.80 s | 0.82 s | 8.84 s，+0.511 rad | 8.86 s | 否 | 否 |
| right_open | 0.82 s | 0.84 s | 10.48 s，-0.499 rad | 10.50 s | 否 | 否 |

left mode 计数为 brake 47、stop 1、search 401、resume 1、track 201；right 为
brake 44、stop 2、search 490、resume 2、track 113。right 在继续前进后再次触发
一次短 search/resume，说明当前结果是局部规则行为，不是单次状态跳转伪造。

## 4. 仍未覆盖

- collision、非法 body contact、跌倒和最小净空；
- 九类 episode outcome 的真实轨迹归类；
- dead-end/false-alarm/both-open 真实 checkpoint 轨迹；
- 多 seed、命令范围、阈值敏感性和性能统计；
- learned policy、privileged、mesh 和 3DGS 基线公平比较。

因此 P7-2 仍为进行中。下一步应把真实轨迹接入 outcome/接触指标，再增加 P7
专用奖励；不能仅凭本记录关闭规则基线或 P7。

## 5. 轨迹 outcome 与接触指标复验

诊断脚本现已接入 task-local privileged trajectory evaluator。碰撞 baseline
检查 G1 根刚体 `pelvis` 与所有非 `ground` 场景 collision geom 的接触；每个
障碍 geom 有独立 contact sensor，force history 覆盖一个 policy step。足端和
其他肢体接触不纳入当前 collision outcome。evaluator 同时读取 `fell_over`、
允许区域、adapter mode 和机体世界轨迹。上述命令以相同 checkpoint、seed 和
650 步重新运行，结果为：

| 场景 | outcome | 路径长度 | 最小前向传感净空 | 最大障碍接触力 | 接触冲量 |
|---|---|---:|---:|---:|---:|
| left_open | `avoidance_resume` | 2.056 m | 0.962 m | 0 N | 0 N·s |
| right_open | `avoidance_resume` | 1.021 m | 0.933 m | 0 N | 0 N·s |

两条轨迹均未触发跌倒、越界、搜索超时或停滞，且恢复后位移超过 0.1 m。
这里的“最小前向传感净空”是 depth adapter 前向扇区的保守分位数，不是机器人
与 collision geometry 之间的精确最小距离。接触指标覆盖 apartment collision
MJCF 中除 `ground` 外的全部命名 geom。其他场景、多 seed、精确 geometry 净空、
非 base-link 非法接触和奖励训练仍未覆盖。

## 6. 六场景多 seed 正式规则验收（2026-11-12）

在首版左右诊断之后，catalog 增加每类场景的预期 outcome 和有限评估窗口。
多 seed 实测表明部分右侧/双侧开口轨迹的 checkpoint 偏航响应慢于 12 秒，
因此将搜索预算重新标定为 18 秒，障碍场景最长评估 22 秒。该预算仍然有限；
dead-end 必须在耗尽后进入稳定 `blocked`。开口与 dead-end 轨迹在达到其预期
outcome 时结束，避免把局部避障恢复验收混同为长距离导航；free/false-alarm
固定运行 10 秒。

```bash
uv run --no-sync tests/run_g1_3dgs_safe_velocity_adapter.py \
  --checkpoint logs/rsl_rl/g1_velocity/2026-10-24_16-15-47/model_29999.pt \
  --scenario free left_open right_open both_open dead_end false_alarm \
  --seed 0 1 2 --device cuda:0 --require-pass \
  --output /tmp/safe-velocity-acceptance-matrix-18s.json
```

| 场景 | 预期/实际 outcome | 通过 | base-link 碰撞 | 跌倒 | 三 seed 最小传感净空 |
|---|---|---:|---:|---:|---:|
| free | `continuous_tracking` | 3/3 | 0/3 | 0/3 | 1.300 m |
| left_open | `avoidance_resume` | 3/3 | 0/3 | 0/3 | 0.914 m |
| right_open | `avoidance_resume` | 3/3 | 0/3 | 0/3 | 0.923 m |
| both_open | `avoidance_resume` | 3/3 | 0/3 | 0/3 | 0.792 m |
| dead_end | `safe_blocked_stop` | 3/3 | 0/3 | 0/3 | 0.910 m |
| false_alarm | `continuous_tracking` | 3/3 | 0/3 | 0/3 | 1.301 m |

矩阵总计 `18/18` 通过，最大障碍接触力均为 0 N，全局最小前向传感净空为
0.7915 m，高于 0.6 m 安全停车净空。净空仍是 depth adapter 的前向扇区统计，
不是精确 collision geometry 距离。本节关闭 P7-2 规则基线验收；learned policy、
privileged/视觉对照、长期停滞与反复启停等问题归 P7-3/P7-4。
