# Safe Velocity blind/privileged 学习基线记录（2026-11-12）

本记录跟踪 P7-3 的工程链路与后续正式训练。当前只完成配置、真实环境 step 和
1-update PPO smoke，不把随机初始化策略的短运行解释为避障效果。

## 1. 基线边界

| 基线 | actor/critic 维度 | 可见信息 | renderer |
|---|---:|---|---|
| blind | 99/109 | G1 本体、历史 action、nominal command | 无 |
| privileged | 117/127 | blind + 9 射线距离和 9 有效位 | 无 |

两者使用相同 apartment collision、G1 joint-position action、PPO 网络宽度和
base-link 障碍碰撞代价。blind 使用现有 nominal velocity tracking；privileged
从 `pelvis` 机体系发出左 80° 至右 80° 的 9 条射线，只查询 collision geom
group 1/3，并以局部几何生成 stop-on-obstacle tracking target。privileged 输入不包含
世界位姿、scenario id、安全出口标签、目标位置或 evaluator outcome。

## 2. 1-update smoke

当前沙箱不可见 CUDA device，因此以下命令显式使用 CPU：

```bash
uv run --no-sync train Mjlab-Velocity-Safe-Blind-Unitree-G1-3DGS \
  --env.scene.num-envs 2 --agent.max-iterations 1 \
  --agent.logger tensorboard --gpu-ids None \
  --log-root /tmp/mjlab-safe-velocity-p7-3

uv run --no-sync train Mjlab-Velocity-Safe-Privileged-Unitree-G1-3DGS \
  --env.scene.num-envs 2 --agent.max-iterations 1 \
  --agent.logger tensorboard --gpu-ids None \
  --log-root /tmp/mjlab-safe-velocity-p7-3
```

| 基线 | steps | SPS | collection | learning | value loss | surrogate loss |
|---|---:|---:|---:|---:|---:|---:|
| blind | 48 | 33 | 1.342 s | 0.086 s | 0.6561 | -0.0677 |
| privileged | 48 | 31 | 1.459 s | 0.075 s | 0.6452 | -0.0709 |

两次运行均生成 `model_0.pt`，loss 有限。smoke 不满足 P7-3 退出条件；正式训练
前仍需冻结相同 environment-step 预算、训练 seed 数、warm-start 规则和评估
checkpoint。free tracking、障碍失败模式和 privileged 局部避障必须由独立
rollout 验收。

## 3. GPU 预算探针发现的 command 覆盖

RTX 3060 12 GiB 上的首轮 64-environment、3-update 探针达到约 1.6--2.1k SPS，
但日志显示通用 G1 `command_vel` curriculum 在首次 reset 后把 catalog 冻结的
`lin_vel_x=(0.0, 0.8)` 覆盖为 `(-1.0, 1.0)`。该探针因此只用于吞吐和问题定位，
不得作为 P7-3 效果数据。

safe-velocity 配置现已任务局部移除该 curriculum，保留 catalog 的 forward-only
linear range、零 lateral range 和 `ang_vel_z=(-0.6, 0.6)`。普通 G1 velocity
task 的 curriculum 不变。正式预算探针和训练必须在修正后重新运行。

修正后 privileged 以相同 64 environments、3 updates 重跑，curriculum manager
为空，吞吐从首个含编译开销的 1142 SPS 上升并稳定在约 1.68--1.70k SPS；loss
有限。按该结果估算，单条 1000-update 校准训练约需 15 分钟量级，但实际时间仍
取决于 checkpoint 保存与长时间 contact 分布。

## 4. locomotion actor warm-start

正式训练从现有标准 G1 locomotion checkpoint 的 actor 启动。转换工具为
`tests/prepare_g1_safe_velocity_warmstart.py`：blind 复制完整 99 维 actor；
privileged 复制前 99 维 actor 输入，并将新增 18 维局部几何输入对应的首层权重
置零。新增 normalizer 维度以 mean 0、variance/std 1 初始化。critic 和 optimizer
不从 locomotion 任务迁移，避免把不同 reward/critic observation 的状态错误带入。

两种输出 checkpoint 均已用目标 task 严格 resume，并在 CPU 上各完成 1 次 PPO
update；blind value loss 为 `0.0742`，privileged value loss 为 `0.0225`，loss
均有限。该验证只证明 checkpoint 结构和恢复链路正确，不计入策略效果。

首轮正式预算冻结为每种基线 64 environments、1000 updates、seed 42，使用同一
`model_29999.pt` actor 来源并依次训练。P7-3 在结果返回后检查 free tracking、
障碍失败模式和 privileged 局部避障；若趋势成立，再决定是否扩展多 seed。

## 5. Blind 首轮微调退化与校准

Blind seed 42 首轮按 64 environments、1000 updates 完成，但不满足验收条件。
`model_999.pt` 的平均回报从 `0.177` 降至 `-76.948`，后 50 updates 平均约
`-75.179`；线速度跟踪奖励约 `0.002`，策略无法站稳或向前行走，并出现不规则
后退。动作标准差从 `0.408` 增至 `0.458`。base-link 障碍碰撞为零不能解释为
安全，因为策略没有完成有效 locomotion。

逐参数检查确认转换后的 warm-start actor 与来源 `model_29999.pt` 完全一致，用户
也确认退化只出现在训练后的 `model_999.pt`，因此排除 actor 转换错误。后续校准
仅修改 safe-velocity learning runner：动作标准差固定为 `0.15`，学习率降至
`1e-4`，entropy coefficient 设为 `0`，每轮 PPO learning epochs 从 5 降至 2。
普通 G1 velocity runner 不变。先执行短校准并检查 deterministic rollout，不把
首轮 `model_999.pt` 用作 Blind 基线或 Privileged 对照。

修正后的 Blind checkpoint 已严格 resume 并在 CPU 上完成 5 updates；动作标准差
在所有 update 均保持 `0.15`，value/surrogate loss 有限。该 smoke 只验证配置、
固定噪声和更新链路，仍需 RTX 3060 上的 100-update 校准及 deterministic playback
确认步态没有再次退化。

100-update GPU 校准随后完成。`model_99.pt` 的 deterministic playback 可以站稳，
且动作标准差全程固定为 `0.15`；但它仍不满足 free tracking 验收。最后 20 updates
平均回报约 `-41.424`，线速度 tracking reward 约 `0.0025`，episode timeout 比例
约 `0.027`，训练采样仍以跌倒终止为主。该模型不得扩展到 1000 updates；需先
确认 deterministic playback 是否实际向前跟踪，避免把静止策略误判为稳定策略。

用户随后确认同一 `model_99.pt` 的 deterministic playback 不仅站稳，也能按正向
命令向前行走。因此将它冻结为候选 Blind checkpoint，不继续追加 PPO updates；
它通过的是定性 locomotion 门槛，不替代后续结构化 free tracking 指标。训练时
随机采样的大量跌倒仍作为失败证据保留。

Privileged 随后以相同保守设置完成 100-update GPU 校准并生成 `model_99.pt`。
动作标准差保持 `0.15`，最后 20 updates 平均回报约 `-46.762`，safe-velocity
tracking reward 约 `0.143`，episode timeout 比例约 `0.018`，训练采样同样主要
因跌倒终止。新增 18 维几何输入的首层权重已从零更新，但到 `model_99.pt` 的
L2 norm 仅约 `0.026`，尚不能仅凭训练曲线证明策略利用了局部几何。下一步必须
回放 both-open 场景，检查是否在障碍前减速、选向并通过开口，且无碰撞/跌倒。

回放确认 `model_99.pt` 能站稳并前进两步，但在障碍前停止超过 10 秒且完全不
转向。根因不是搜索时间不足，而是 privileged collision ray 的 miss 语义错误：
`RayCastSensor` 用 `-1` 表示最大距离内无命中，这是已知畅通；旧转换却将其标为
invalid。规则 adapter 因而把开阔扇区视为未知危险，无法进入正常 track/resume，
停车成为奖励下的局部最优。该 checkpoint 作废。

任务局部转换现将有限的负 ray distance 映射为 cutoff depth 且 validity 为 1；
只有 NaN/Inf 才标为 invalid。定向测试覆盖全 miss 时正常 tracking、前方命中时
朝无命中开阔侧 search，以及旋转到全 miss 方向后的 resume。相机 depth 的 invalid
保守语义和规则 3DGS adapter 均不改变。必须从 warm-start 重新开始校准，不能
恢复旧 `model_99.pt`；已有的 conservative
warm-start 可复用，因为其 18 维几何输入权重本来就是零初始化。

修正后的真实环境 reset/step 检查确认 9 个 ray validity 全部正确，严格 resume
后完成 2-env CPU 1-update PPO smoke：action std `0.15`、value loss `0.0541`、
surrogate loss `0.0030`，均有限。该结果验证接线与数值稳定性，不替代重新执行的
100-update GPU 行为校准。

ray-miss 修正后的 100-update GPU 校准已完成并生成新 `model_99.pt`。最后 20
updates 平均回报约 `-48.951`、safe-velocity tracking reward 约 `0.108`，timeout
比例约 `0.008`，随机训练采样仍主要跌倒；几何输入首层权重 L2 norm 约 `0.022`。
训练 telemetry 尚无改善证据，必须以修正语义后的 deterministic both-open 回放
确认是否出现制动、转向和恢复，不能直接扩展预算。

回放结果为前进、遇到障碍后停止，没有转向或恢复；因此仍不满足
`avoidance_resume`。ray-miss 语义修复解决了“开阔空间恒 invalid”，但当前
100-update 保守微调只让新增几何首层权重增长到约 `0.022` L2 norm，尚未学出
局部几何到转向动作的映射。该 checkpoint 也不作为合格 Privileged baseline，
不直接追加迭代；下一步需明确继续端到端学习，还是把 collision-truth adapter
输出的 safe twist 作为 oracle privileged 输入，两者实验含义不同。

## 6. Privileged 目标简化为遇障停车

用户确认 P7-3 的 privileged 学习基线不再要求转向或绕行。新的无状态目标只读取
9 条 privileged collision ray 中间的 3 条前向射线：任一前向射线无效，或最小
净空不大于 1.2 m 时，目标命令固定为 `[0, 0, 0]`；无风险时目标命令与 nominal
command 完全相同。因此即使 nominal command 给出很大的前进、侧移或偏航速度，
遇障后的 tracking target 仍是完全停车。侧面障碍不触发停车或转向，目标也没有
search、turn、resume 等内部状态。

纯逻辑测试覆盖大速度命令的全量清零、侧面障碍 passthrough 和前向无效射线的
保守停车。该修改只作用于 privileged 学习奖励；P7-2 deterministic adapter 的
制动、搜索、转向和恢复行为保持不变。新目标必须从 conservative warm-start
重新训练，不能恢复目标变更前的 `model_99.pt`。真实环境 reset/step 回归通过；
从 conservative warm-start 执行的 2-env CPU 1-update smoke 共采样 48 steps，
达到 31 SPS，value loss `0.0453`、surrogate loss `-0.0542`，action std 保持
`0.15`，数值有限。
