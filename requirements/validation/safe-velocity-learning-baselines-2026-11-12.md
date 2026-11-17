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

## 7. 长训练失败与 privileged stop-gate

用户随后以 4096 environments、5000 updates、seed 42 运行 stop-only
Privileged，实际采样约 4.9152 亿 environment steps。训练后 500 updates 的
平均回报约 `59.72`，episode length 固定为 `1000`，safe tracking reward 约
`1.176`，记录的 `fell_over` 和 base-obstacle collision 均为零；但用户确认
deterministic playback 中机器人无法站稳。相同任务和播放参数下，未微调的
`warmstart_conservative/model_0.pt` 仍能站稳，排除了环境和加载链路问题。

最终 actor MLP 相对 warm-start 的 L2 参数漂移为约 `21.8%`，几何输入首层权重
L2 增至约 `1.242`。日志的假稳定来自验收漏洞：`fell_over` 只检测超过 70° 的
倾倒，base-obstacle collision 只覆盖 pelvis 与障碍物，且零速度目标允许低矮、
非足端着地或其他静止姿态获得 tracking reward。该 run 的全部 checkpoint 不作为
合格基线，也不再通过增加 PPO updates 修复。

新增的 `Mjlab-Velocity-Safe-Privileged-Stop-Gate-Unitree-G1-3DGS` 不修改共享
runner。它保留 privileged 的 9 条 collision ray 和 117/127 observation 形状，
但在 actor/critic 的 command observation 边界执行无状态 stop/go gate：前向
射线危险或无效时传入 `[0, 0, 0]`，否则传入 nominal command。actor 继续加载
能站稳的 privileged warm-start，18 个几何输入的直接权重保持近零，因此低层
locomotion 不需要 PPO 微调。它是 collision-truth oracle baseline，不是视觉
策略。safe-velocity 配置另增加 pelvis 高度低于 0.5 m 和非足端接触地面终止；
真实 privileged warm-start 已由新 task 严格加载。seed 42 的 CPU deterministic
`both_open` rollout 在第 49 step（0.98 秒）触发 stop，完整运行 1100 steps
（22 秒）无终止，最低 pelvis 高度 `0.7530 m`、最大倾角 `1.834°`，最终平面位置
只比出生点前移约 `0.39 m`。`free` rollout 完整运行 500 steps（10 秒），stop
触发次数为零，无终止，最低 pelvis `0.7534 m`、最大倾角 `1.343°`，机器人持续
产生平面位移。该证据通过接线、稳定站立和 free/blocked 分离检查；用户 GUI
回放确认前仍不关闭 P7-3。

## 8. 连续特权安全教师（2026-11-14）

此前 Blind、两轮端到端 Privileged 校准、5000-update stop-only 长训练和硬
Stop-Gate 均作为论文实验资产保留；本节新增实现不删除、不重命名也不覆盖这些
task、run 或 checkpoint 记录。失败 checkpoint 仍不得恢复为新方案起点。

P7-4 主线改为 privileged safety supervision，而非蒸馏29维关节动作。新增
`Mjlab-Velocity-Safe-Privileged-Safety-Teacher-Unitree-G1-3DGS`，继续兼容
117/127 privileged warm-start，但只在 command observation 边界应用连续安全
缩放：当前第三版参数下，前向真值净空不小于 `1.4 m` 时 `alpha=1`，不大于
`0.4 m` 时
`alpha=0`，中间线性插值；任一前向射线无效时保守输出 `alpha=0`。三个速度
分量统一按 `safe_command = alpha * nominal_command` 缩放，因此 stop 状态不
残留侧移或偏航命令。

独立 `safety_teacher` observation group 输出四维监督标签：归一化前向净空、
真值有效位、`alpha` 和 stop label。runner 的 actor/critic observation groups
不包含该字段，避免特权标签泄漏到部署策略。新教师暂用于阈值标定和后续视觉
Student 数据接口。stop 阈值初版为 `0.8 m`，2026-11-14 在正式回放前调整为
`0.4 m`。play 配置现以 1 Hz 输出 nominal/safe/measured 三维速度、前向净空、
`alpha` 和 stop 状态，并固定使用场景协议的 nominal command；训练配置保持
静默且继续随机采样，用于 deterministic braking rollout 标定。

首次固定 `0.5 m/s` nominal command 的 GUI 回放显示：当日志为
`safe=(0.204, 0, 0)`、`clearance=0.972 m`、`alpha=0.409`、`stop=False` 时机器人
已经停止；后续 `safe=(0.192, 0, 0)`、`clearance=0.939 m`、`alpha=0.385` 的重复
日志同时给出 `measured_vx≈0`。这证明当前停车由冻结 locomotion actor 的低速
步态死区造成，而非 Safety Teacher 的 `0.4 m` stop 条件。第二版保持 stop
`0.4 m`，把 slow 从 `1.8 m` 收近到 `1.0 m`；在相同 nominal command 下，
`alpha≈0.4` 的位置由约 `0.96 m` 移至约 `0.64 m`。该参数仍需 GUI 制动验收，
也尚未实现数据持久化、监督训练或 DAgger。

第二版 GUI 回放不再在约 `0.94 m` 静止：`clearance=0.699--0.735 m`、
`safe_vx=0.249--0.279 m/s` 时 `measured_vx` 仍为 `0.208--0.232 m/s`。但下一条
周期日志反复跳回出生点的 `clearance=1.5--1.6 m`；用户确认机器人先碰撞障碍物
再重置，因此第二版不能记为安全停车，且 `1.0 m` slow 的制动距离不足。第三版
采用 slow `1.4 m`、stop `0.4 m`，预计 `alpha≈0.4` 的低速死区位置约为
`0.80 m`，在保守首版与碰撞第二版之间继续标定。play 诊断仍保留 reset 当步的
`episode_step` 和具体 `termination` 名称。用户确认第三版能够无碰撞稳定停车，
因此 P7-4A 以 stop `0.4 m`、slow `1.4 m` 完成标定；前两版证据继续保留。

## 9. 时序视觉数据接口（2026-11-14）

新增 `Mjlab-Velocity-Safe-Vision-Data-Unitree-G1-3DGS`，以 additive task-local
配置复用 P6 robust depth+validity observation，不修改共享相机或 observation
manager。视觉输入保持 `(B, 7, 2, 48, 64)`：7 个历史槽包含按 stride 2 选取的
4 帧、一步 observation lag、两步 sample-and-hold；训练配置启用既有深度噪声、
模糊、结构化 dropout 和整帧 dropout，play 配置关闭 corruption。

同一环境增加 9 条 collision-truth ray，只用于生成 normalized clearance、
validity、`alpha`、stop 四维标签和 teacher safe command。runner 继续使用 99/109
维 MLP locomotion actor/critic，`camera`、`safety_teacher` 和 ray geometry 均不
属于其 observation groups，因而该接口不训练29维关节策略，也不产生特权标签
泄漏。当前交付物建立在线配对与 chunked 持久化接口；尚未定义不对称加权损失
或训练视觉 Student，不能据此宣称视觉安全策略已经完成。

新增 chunked collector 后，使用 Blind `model_99.pt` 完成 1 environment × 2 step
CPU smoke：严格 actor load 成功，生成两个 chunk 和 manifest；回读字段为 camera
`(1,7,2,48,64)` float16、teacher label `(1,4)` float32，以及三维 nominal/safe/
measured velocity、int64 episode step 和 bool reset。manifest 记录 checkpoint
SHA-256、teacher 阈值、label 顺序、场景/seed 和 alpha/stop 统计。该 smoke 只
验证持久化接线，不构成数据规模或 Student 学习证据。

同时确认 SensorContext 会将 camera geom groups `[1,2]` 与 collision-truth ray
groups `[1,3]` 合并为 `[1,2,3]`。当前采集输入因此包含 collision mesh，可作为
mesh/collision-assisted + 3DGS baseline，但不得标注为纯 3DGS。纯 3DGS 训练前
必须隔离 camera 与 truth ray 的 geom-group 可见性，避免特权碰撞代理泄漏。

## 10. Studio 跨地图 Depth-Stop 探针（2026-11-15）

P7-5 结构化 runner 已扩展为在环境构建前接收自定义 manifest、spawn、nominal
command、allowed region 和明确 expected outcome。使用同一 Blind `model_99.pt`、
Studio `default` spawn、`0.5 m/s` 前进命令、seed 0 和最多 800 steps 运行：

```bash
uv run tests/run_g1_3dgs_safe_velocity_adapter.py \
  --checkpoint logs/rsl_rl/g1_3dgs_safe_velocity_blind/2026-11-12_22-38-51_blind_conservative_seed42_100/model_99.pt \
  --controller depth_stop \
  --scenario free \
  --scene-manifest src/mjlab/asset_zoo/3dgs/ArtanimMotionCaptureStudio/manifest.json \
  --scene-spawn-point default \
  --nominal-command 0.5 0.0 0.0 \
  --allowed-region -10 10 -10 10 \
  --expected-outcome safe_blocked_stop \
  --steps 800 --seed 0 --device cuda:0
```

评估器在 step 237（4.74 s）提前接受 `safe_blocked_stop`：84 steps 为 track、154
steps 为 brake，路径长 `1.171 m`，最小深度统计净空 `0.876 m`，base-link
collision force/impulse 均为零，无跌倒、越界、termination 或 reset。末帧深度
净空 `0.896 m`、alpha `0.496`、safe vx `0.248 m/s`，实测 body vx
`-0.031 m/s`，表明冻结 locomotion actor 再次在低速步态死区稳定停车。

随后使用同一配置运行 seed 0/1/2 且启用 `--require-pass`，结果为 3/3
`safe_blocked_stop`。三次分别在 237、229、467 steps 被提前接受，路径长为
`1.171/1.293/2.703 m`，最小深度净空为 `0.876/0.905/0.864 m`，最终实测 vx 为
`-0.031/-0.037/-0.040 m/s`；三次均无 collision、fall 或 termination。seed 2
路径明显更长，后续扩大 spawn 时应继续检查横向漂移和反复启停，不能只看最终 outcome。

为避免不同位置同时改变深度分布和动力学初态，Studio manifest 随后增加六个
`safe_velocity_pillar_00--05` 验收 spawn。六者 position 都是 `(0, 0, 0)`，只改变
机器人 yaw，使前向分别指向 collision MJCF 中的 `pillar_00--05`；原 `default`
保持不变。配置回归直接读取 MJCF 柱体中心，并验证六个 spawn 的目标 yaw 误差均不
超过 `1e-6 rad`。

同一 checkpoint、`0.5 m/s` nominal command、seed 0、最多 800 steps 的结果如下：

| Spawn | 接受时间 (s) | 路径长 (m) | 最小净空 (m) | 结果 |
|---|---:|---:|---:|---|
| `safe_velocity_pillar_00` | 4.82 | 1.302 | 0.856 | `safe_blocked_stop` |
| `safe_velocity_pillar_01` | 5.52 | 1.554 | 0.876 | `safe_blocked_stop` |
| `safe_velocity_pillar_02` | 5.88 | 1.788 | 0.869 | `safe_blocked_stop` |
| `safe_velocity_pillar_03` | 5.64 | 1.667 | 0.873 | `safe_blocked_stop` |
| `safe_velocity_pillar_04` | 7.08 | 2.290 | 0.880 | `safe_blocked_stop` |
| `safe_velocity_pillar_05` | 6.24 | 1.962 | 0.884 | `safe_blocked_stop` |

六次均提前接受，且 collision、fall、termination 全为零。这证明同一出生点下，
Depth-Stop 对六个障碍方向都能进入无碰撞停车；它仍不能替代 3DGS 视觉与 collision
overlay 的人工逐点对齐检查。原始本地 traces 为
`/tmp/safe-velocity-studio-pillar-00-seed0.json` 至
`/tmp/safe-velocity-studio-pillar-05-seed0.json`。

在保留上述同点组作为消融基线的同时，manifest 又增加六个位置互不相同的
`safe_velocity_generalization_pillar_00--05`。各 spawn 到目标柱体的初始距离为
`3.35--4.21 m`，并从各自位置重新计算目标 yaw。相同 checkpoint、命令、seed 和
评估窗口的结果如下：

| Spawn | Position (x, y) | 接受时间 (s) | 路径长 (m) | 最小净空 (m) | 结果 |
|---|---:|---:|---:|---:|---|
| `generalization_pillar_00` | `(-0.8, -0.6)` | 7.04 | 2.288 | 0.847 | `safe_blocked_stop` |
| `generalization_pillar_01` | `(0.7, -0.7)` | 7.60 | 2.512 | 0.886 | `safe_blocked_stop` |
| `generalization_pillar_02` | `(-0.6, 0.7)` | 8.08 | 2.744 | 0.849 | `safe_blocked_stop` |
| `generalization_pillar_03` | `(0.8, 0.5)` | 7.80 | 2.571 | 0.883 | `safe_blocked_stop` |
| `generalization_pillar_04` | `(-0.9, 0.1)` | 9.06 | 3.183 | 0.890 | `safe_blocked_stop` |
| `generalization_pillar_05` | `(0.5, 0.9)` | 8.06 | 2.660 | 0.886 | `safe_blocked_stop` |

六次均无 collision、fall 或 termination。该矩阵表明 Depth-Stop 在 Studio 内对
位置和朝向的联合变化保持了安全停车行为，但目前只有 seed 0，且障碍布局和地图
没有变化，因此只作为地图内初步泛化证据。原始本地 traces 为
`/tmp/safe-velocity-studio-generalization-pillar-00-seed0.json` 至
`/tmp/safe-velocity-studio-generalization-pillar-05-seed0.json`。

runner 随后增加 `--scene-spawn-prefix`，可由 manifest 自动选择整个 spawn 组，并
与多个 seed 组合成一个可失败的矩阵。正式三 seed 命令为：

```bash
uv run --no-sync tests/run_g1_3dgs_safe_velocity_adapter.py \
  --checkpoint logs/rsl_rl/g1_3dgs_safe_velocity_blind/2026-11-12_22-38-51_blind_conservative_seed42_100/model_99.pt \
  --controller depth_stop --scenario free \
  --scene-manifest src/mjlab/asset_zoo/3dgs/ArtanimMotionCaptureStudio/manifest.json \
  --scene-spawn-prefix safe_velocity_generalization_ \
  --nominal-command 0.5 0 0 --allowed-region -10 10 -10 10 \
  --expected-outcome safe_blocked_stop --steps 800 \
  --seed 0 1 2 --device cuda:0 --require-pass \
  --output /tmp/safe-velocity-studio-generalization-seed0-1-2.json
```

结果为 18/18 `safe_blocked_stop`，collision、fall、termination 计数均为 0。

| Spawn | 三 seed 最小净空 (m) | 路径范围 (m) | 接受时间范围 (s) |
|---|---:|---:|---:|
| `generalization_pillar_00` | 0.771 | 2.286--2.369 | 6.78--7.04 |
| `generalization_pillar_01` | 0.779 | 2.510--2.608 | 7.34--7.62 |
| `generalization_pillar_02` | 0.768 | 2.742--2.831 | 7.74--8.06 |
| `generalization_pillar_03` | 0.709 | 2.575--2.747 | 7.64--8.04 |
| `generalization_pillar_04` | 0.757 | 3.178--3.296 | 8.80--9.14 |
| `generalization_pillar_05` | 0.774 | 2.662--2.783 | 7.76--8.06 |

全矩阵净空范围 `0.709--0.892 m`，路径范围 `2.286--3.296 m`，接受时间范围
`6.78--9.14 s`。最小净空出现在 `generalization_pillar_03 / seed 2`，虽然仍高于
`0.4 m` stop threshold 且没有接触，但说明只看 seed 0 会高估最差停车余量。
该结果证明 Studio 地图内对位置、朝向和随机动力学变化的稳定性；跨地图、纯 3DGS
视觉依赖和视觉—碰撞对齐仍未验收。

## 11. Studio 视觉—碰撞对齐验收（2026-11-17）

解释更正：下述历史命令曾默认将 group-1 合成可视柱体与 Gaussian 重建比较。
柱体是额外添加的可视且可碰撞物体，不要求出现在原始 Gaussian 中，所以 0/6
不能判定为视觉管线失败。18/18 停车属于“3DGS 背景 + 可视 mesh 障碍”证据。
当前 runner 默认改为 group 3 隐藏物理代理的离线投影；复现以下旧报告时应增加
`--reference-geom-groups 1`。参考深度始终不得输入策略。

新增 `tests/run_g1_3dgs_safe_velocity_alignment.py`，在完全相同的 manifest spawn、
seed、机器人根位姿和 MJWarp 相机位姿下分别捕获：

- Gaussian-only depth：保留 3DGS，仅用远处 group-5 占位几何避免空 BVH；
- collision-only depth：关闭 Gaussian，仅渲染 collision group 1。

验收只评价 collision 投影像素。默认要求投影不少于 20 像素、Gaussian 有效覆盖率
不少于 90%，且至少 80% 像素深度误差不超过 `0.15 m`。overlay 图例为：绿色表示
误差内对齐，红色表示两者都有深度但误差超限，紫色表示 collision 像素没有有效
Gaussian 深度，灰度背景为反色 Gaussian depth。

```bash
uv run --no-sync tests/run_g1_3dgs_safe_velocity_alignment.py \
  --scene-manifest src/mjlab/asset_zoo/3dgs/ArtanimMotionCaptureStudio/manifest.json \
  --scene-spawn-prefix safe_velocity_generalization_ \
  --seed 0 --tolerance 0.15 \
  --minimum-collision-pixels 20 \
  --minimum-gaussian-coverage 0.9 \
  --minimum-aligned-fraction 0.8 \
  --overlay-scale 8 \
  --output requirements/validation/safe-velocity-studio-alignment-2026-11-17
```

| Spawn | Collision pixels | Gaussian coverage | Aligned | Median abs error (m) | P90 abs error (m) |
|---|---:|---:|---:|---:|---:|
| `generalization_pillar_00` | 183 | 0.841 | 0.000 | 1.015 | 1.494 |
| `generalization_pillar_01` | 219 | 0.557 | 0.000 | 1.142 | 1.558 |
| `generalization_pillar_02` | 159 | 0.629 | 0.000 | 1.142 | 1.354 |
| `generalization_pillar_03` | 138 | 0.696 | 0.000 | 1.244 | 1.423 |
| `generalization_pillar_04` | 289 | 0.657 | 0.000 | 1.341 | 1.883 |
| `generalization_pillar_05` | 431 | 0.812 | 0.000 | 1.561 | 1.860 |

结果为 0/6，通过人工检查六张放大 overlay 也没有发现绿色投影。Studio 文件
注释说明柱体是固定伪随机测试布局；本次测量确认它们是额外添加的可视物体，
无需在 Gaussian 重建中对应。此前 Depth-Stop 18/18 使用合法的可视柱体深度，
可作为可视 mesh 障碍停车证据，仍不能写成纯 3DGS 障碍感知成功。测量报告完整
保留；撤销此前将其解释为资产对齐失败的结论。真实重建障碍的隐藏物理代理仍需
对齐验收，额外可视柱体按其自身可视/碰撞几何一致性验收。

该矩阵证明相同深度阈值和冻结 actor 能在第二张 3DGS manifest、三组随机动力学
参数下产生无碰撞停车，但不单独证明 Studio 的视觉表面与随机测试 collision 柱体
逐点对齐。正式跨地图结论仍需 collision overlay 人工检查、多个 spawn，以及
视觉/碰撞严格对齐的新资产。原始本地 traces 写入
`/tmp/safe-velocity-studio-default*.json`，不作为长期版本化产物。
