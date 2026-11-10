# P7：3DGS 安全感知速度跟踪需求

最后更新：2026-11-10

状态：P7-0 场景协议和 P7-1 独立任务骨架已完成；P7-2 的九类 outcome
判定、规则 depth adapter 核心及真实环境 observation 接入已实现，P7 专用奖励、
轨迹效果验收和训练尚未完成。

## 1. 目的

本文档定义 P7 的任务边界、MDP、场景协议、基线、指标、测试和阶段验收。
项目级进度以 `mjlab-3dgs-todo.md` 为准；P6 视觉输入和通用训练要求见
`mjlab-3dgs-training.md`。相关项目对比、能力边界和选型依据见
`mjlab-3dgs-research-report.md` 第 6.4 节。

P7 的目标是：G1 在具有独立碰撞几何的 3DGS 场景中，根据期望速度命令、
本体状态和时序深度安全行走。前方无障碍时跟踪速度；发现碰撞风险时减速
或停止，选择安全方向；必要时原地旋转搜索无碰撞方向，然后恢复速度跟踪。

P7 不是导航任务，不要求机器人到达世界坐标目标，也不假设机器人具备
LiDAR、GNSS、VIO、SLAM、外部动捕或其他全局/局部位置估计。

## 2. 核心任务定义

### 2.1 输入、输出和闭环

首版闭环为：

```text
nominal velocity command + proprioception + temporal depth/validity
                              ↓
                  safe perceptive locomotion policy
                              ↓
                    target joint positions
```

- nominal command 表示用户或上层系统希望机器人跟踪的机体速度。
- 策略在安全空间内尽量跟踪 nominal command。
- 碰撞风险高时，安全性优先于瞬时速度跟踪误差。
- 策略可停止平移并原地旋转，通过连续深度观测搜索安全方向。
- 找到满足净空条件的方向后，策略朝该方向恢复期望速度大小。
- 机器人不需要知道自身在 apartment 世界坐标中的位置。

### 2.2 行为状态语义

评估器应能从行为和传感状态区分以下阶段；实现不要求显式建立有限状态机：

1. `track`：前方安全，跟踪 nominal velocity command；
2. `brake`：风险上升，降低平移速度并避免继续逼近；
3. `stop`：在安全净空外稳定停止平移；
4. `search`：原地或近似原地旋转，扫描候选方向；
5. `resume`：选定安全方向并重新建立速度跟踪；
6. `blocked`：在搜索预算内没有安全方向，保持安全站立而非强行通过。

状态标签只用于奖励归因、诊断和评估，不直接输入 actor。

### 2.3 安全优先级

行为优先级为：

```text
avoid collision/fall
  > brake and maintain clearance
  > find a safe direction
  > resume velocity tracking
  > minimize command-tracking error
```

该优先级避免策略为了追踪前进命令持续撞向墙体，也避免策略通过永久停车或
持续旋转获得虚假安全成功。

## 3. 实施原则与边界

### 3.1 必须复用的现有能力

- 复用 mjlab 的 `ManagerBasedRlEnv`、command/observation/reward/event/
  termination manager，不建立第二套训练框架。
- 复用现有 G1 velocity task 的 `UniformVelocityCommand`、关节目标 action、
  reset、接触传感器、速度跟踪和 locomotion 正则项。
- 复用 P5 scene manifest 和已对齐的 PLY/collision/spawn metadata。
- 复用 P6 的单帧 depth、depth+validity、时序 history、robust corruption 和
  asymmetric critic 配置。
- 复用 RSL-RL runner、checkpoint、恢复训练和日志链路。

### 3.2 新增代码的放置原则

- P7 使用独立 task id 和任务专用配置，不改变已有 G1 velocity task 行为。
- 避障奖励、诊断状态、任务指标和 safety shield 优先放在 P7 命名空间。
- 现有 `UniformVelocityCommand` 能表达任务输入，不增加 position command。
- 只有被多个非 P7 任务验证需要的能力，才考虑提取到共享模块。
- 不修改上游默认参数，不复制已有 manager、历史缓存或策略 runner。
- 每次实现变更必须同步更新本文档和项目 TODO，并检查相对 `main` 的差异。
- 安全框架优先参考 ABS，G1 深度关节控制优先参考 Hiking in the Wild，
  3DGS 仿真对照参考 GaussGym；这些是分层参考，不表示整体复现任何项目。

### 3.3 首版范围内

- apartment 平坦支撑面中的安全速度跟踪、制动、停车、转向搜索和恢复。
- 静态墙体、柱体和静态碰撞几何。
- actor 使用速度命令、本体状态、历史 action 和 P6 视觉输入。
- free、left-open、right-open、both-open、dead-end 等局部障碍布局。
- blind、privileged、mesh vision、3DGS vision 和视觉破坏消融。
- 可独立开关、结果单独报告的 safety shield。

### 3.4 首版范围外

- 世界坐标 PointGoal、waypoint 到达率、全局路径规划和地图构建。
- 目标相对位置、GPS、定位结果或最短路径作为 actor observation。
- 单级台阶、楼梯、间隙、稀疏落脚点和复杂野外地形。
- 动态行人、动态障碍和移动目标。
- 语言命令、语义目标搜索和开放词汇导航。
- 重新设计 G1 底层 action 接口。
- P8 的跨场景泛化或语义碰撞生成算法。

## 4. 场景与数据协议

### P7-FR-1：冻结 apartment 基线资产

首个任务使用当前 apartment scene manifest。它包含共坐标系的 3DGS PLY、
collision MJCF、spawn metadata、地面、墙体代理和两个同时可见且可碰撞的
不透明柱体。

- 实验记录必须保存 manifest 路径、仓库 commit、任务配置和生成参数。
- 一次训练运行中不得静默替换 PLY、collision 或 transform。
- P8 新资产必须使用新版本标识，并在相同协议下做替换消融。
- 视觉和碰撞必须作为整体变换，不得只移动 collision 障碍。

### P7-FR-2：局部障碍 scenario catalog

每个 scenario 至少记录：

- 唯一 id 和 train/validation/test 划分；
- 初始机器人位姿及允许的随机范围；
- nominal velocity command 分布；
- 障碍布局类别和碰撞 geom 集合；
- 初始可见方向及允许的安全出口集合；
- 允许区域、越界边界和最小几何净空；
- 搜索角度/时间预算及人工检查结果。

首轮至少覆盖：

- `free`：前方无遮挡，用于验证正常速度跟踪；
- `left_open`：正前方受阻，左侧存在安全方向；
- `right_open`：正前方受阻，右侧存在安全方向；
- `both_open`：左右均可通行，不强制唯一转向；
- `dead_end`：当前搜索范围内无安全出口，应保持安全站立；
- `false_alarm`：远端或非碰撞视觉结构，不应过早停车。

scenario catalog 必须进入版本控制。初版先建立固定列表和确定性回归，再在
列表限定范围内随机采样。

实现记录（2026-11-10）：新增版本化 `p7_scenarios.json`，记录六类 scenario、
split、manifest spawn、nominal command、碰撞 geom、安全出口扇区、允许区域、
最小净空、搜索预算和预期行为。apartment manifest 新增六个 `p7_*` 命名出生点。
2026-11-10 用户已完成六类场景的交互检查，所有 `review_status` 更新为
`approved`。这表示出生方向、视觉/碰撞和安全出口协议可以作为 P7 首版固定
夹具；当前净空和搜索阈值仍需在实际制动实验中继续标定。

### P7-FR-3：数据划分

- 同一布局实例不得同时出现在训练和测试划分。
- 测试至少保留未见出生偏航、障碍距离和左右开口组合。
- apartment 内留出布局只表示组合泛化，不宣称跨场景泛化。
- 跨场景结果必须使用另一组视觉/碰撞资产并单独报告。
- 各策略使用相同 scenario、seed、episode 上限和评估预算。

## 5. Command、observation 与 action

### P7-FR-4：nominal velocity command

首版直接复用现有 `UniformVelocityCommand`。初始训练协议应从受限命令开始：

- 使用机体坐标系的直接 `(lin_vel_x, lin_vel_y, ang_vel_z)` 命令；首版关闭
  `heading_command`，不引入需要世界 yaw 目标的航向跟踪；
- 主要使用正向 `lin_vel_x`，使障碍交互可归因；
- 初期可固定或限制 `lin_vel_y`；
- 包含小范围 `ang_vel_z` 和独立原地旋转样本；
- 随课程逐步扩大速度和转向范围；
- command 在正常行走区间保持足够久，避免频繁重采样掩盖避障行为。

nominal command 表示期望运动，不是必须无条件跟踪的硬约束。进入危险区域时，
策略允许暂时偏离命令；恢复阶段应重新跟踪期望速度大小，但不要求恢复原世界
航向，因为任务没有定位或全局目标。

### P7-FR-5：actor observation

完整 actor observation 包含：

- 现有 G1 proprioception；
- nominal velocity command；
- previous action；
- P6 的时序 depth+validity；
- P6 已定义的延迟、噪声、模糊和 dropout。

actor 不得读取：

- base 世界位置或世界 yaw；
- 目标坐标、目标方向或目标距离；
- 障碍世界坐标、collision geom id 或占据地图；
- scenario 类别、安全出口标签或参考转向；
- 最短路径、未来传感帧或 evaluator 状态。

必须保留以下 observation 变体：

1. proprioception + velocity command 的 blind 基线；
2. 单帧 depth；
3. 单帧 depth+validity；
4. 时序 depth+validity；
5. 时序 robust depth+validity。

训练默认继续使用 P6 的 `64×48` 输入。更高分辨率只能作为独立消融。

### P7-FR-6：critic 和 evaluator 信息

- symmetric critic 与 actor 使用相同信息。
- asymmetric critic 可读取局部碰撞距离、接触风险或无噪声状态。
- privileged 字段必须逐项列出，不得泄漏到部署 actor。
- evaluator 可读取世界位姿和碰撞几何计算轨迹与净空，但这些不是传感输入。
- 不使用到达世界目标的 reward 或 outcome。

### P7-FR-7：action 和控制频率

首版沿用现有 G1 velocity task 的关节目标 action、控制频率和 action scale。
策略直接输出关节位置目标；不在 P7 同时引入新的高层导航策略或定位模块。

## 6. MDP 行为要求

### P7-FR-8：无障碍速度跟踪

在 `free` scenario 中：

- 机器人应稳定跟踪 nominal linear/angular velocity；
- 不应无故停车、绕行或持续旋转；
- 相比现有 G1 velocity task 的跟踪误差和稳定性退化必须受限；
- 视觉输入不得给普通 velocity task 带来默认渲染开销。

### P7-FR-9：减速与安全停车

前方碰撞风险上升时：

- 机器人应在接触前降低朝障碍方向的速度；
- 停车过程不得跌倒、明显滑移或用身体撞击完成制动；
- 停止后保持可配置最小净空；
- 无效/缺失深度应触发保守行为，但不能导致永久停止策略。

### P7-FR-10：方向搜索与恢复

- 前方受阻时允许偏离 nominal yaw/forward tracking。
- 机器人可以原地或近似原地旋转，使相机观察新的候选方向。
- 搜索期间应限制向未知或受阻方向的平移。
- `left_open`/`right_open` 应选择对应安全侧。
- `both_open` 可选择任一满足净空的方向，不惩罚非唯一正确解。
- 发现安全方向后应结束持续旋转并恢复平移速度跟踪。
- `dead_end` 在搜索预算耗尽后应保持稳定站立，不强制继续前进。
- 禁止以左右高频摆动、无限旋转或反复启停替代有效方向选择。

### P7-FR-11：奖励

奖励由现有 locomotion 项和 P7 安全感知项组成：

- 安全区域的 linear/angular velocity tracking；
- 姿态、步态、能耗和 action smoothness；
- 碰撞、跌倒、越界和非法接触代价；
- 风险上升时的提前减速和安全净空；
- 安全停止、有效搜索和发现安全方向；
- 选定安全方向后的恢复跟踪；
- 无障碍场景的不必要停车/旋转代价；
- 搜索振荡、无限旋转、长期停滞和反复启停代价。

若奖励使用 collision geometry 计算局部净空，它属于训练特权信号，必须记录
并通过视觉消融证明 actor 没有获得该真值。奖励不得使用世界目标位置或路径。

### P7-FR-12：termination 与 outcome

episode 至少区分：

- `continuous_tracking`：在时间窗内安全持续跟踪；
- `avoidance_resume`：完成制动/搜索并恢复行走；
- `safe_blocked_stop`：无安全出口时保持安全站立；
- `collision`；
- `fall`；
- `out_of_bounds`；
- `search_timeout`；
- `stagnation`；
- `episode_timeout`。

每个 episode 只有一个主要 outcome，诊断事件可单独计数。不得使用
`goal_reached`、SPL 或世界目标距离作为 P7 完成依据。

## 7. Safety shield

### P7-FR-13：可独立开关的安全层

safety shield 位于学习策略之外，默认可以完全关闭。可部署 shield 只能使用
本体状态、nominal command 和相机观测；使用碰撞真值的版本必须标记为
oracle shield。

动态停止距离至少考虑：

```text
d_stop = v_forward * t_delay + v_forward² / (2 * a_brake) + margin
```

也可增加 TTC 判定。延迟、可靠减速度和 margin 必须通过 G1 制动标定获得。
shield 至少支持限制前进和强制稳定站立。若 shield 负责选择转向，它必须作为
独立 deterministic baseline，不得把转向成功归因于学习策略。

shield off/on 必须分开报告介入率、误触发、漏触发和最终 outcome。

## 8. 基线、消融与指标

### P7-FR-14：最低基线集合

在相同协议和预算下至少比较：

1. 现有 blind velocity tracking；
2. blind + deterministic depth safety adapter；
3. privileged local-geometry policy；
4. mesh single-depth；
5. 3DGS single-depth；
6. 3DGS depth+validity；
7. 3DGS temporal robust depth+validity；
8. symmetric/asymmetric critic；
9. 最佳策略的 shield off/on。

deterministic adapter 至少实现“制动—原地旋转—选取最大安全扇区—恢复命令”，
作为学习方法能否超过简单规则的基线。

### P7-FR-15：视觉依赖测试

评估至少包含正常、全零、冻结首帧、时间打乱、depth/validity 错配、跨 world
错配和留出障碍布局。若视觉策略没有稳定优于 blind/规则基线，或视觉破坏不
改变行为，不得声称策略学会了视觉避障。

### P7-FR-16：核心指标

所有指标按 scenario 类别和总集合分别报告：

- free-space linear/angular velocity tracking error；
- 碰撞率、非足端碰撞率、接触冲量和跌倒率；
- 首次风险发现时的速度和距离；
- 制动距离、停车时间和停止净空；
- 正确安全侧选择率；
- 搜索成功率、搜索角度和搜索时间；
- avoidance resume rate 和恢复跟踪时间；
- dead-end 安全停车率；
- 不必要停车率、不必要旋转率和 false-alarm rate；
- 振荡、无限旋转、停滞和超时率；
- 最小障碍净空；
- shield 介入率、误触发率和漏触发率；
- 训练 SPS、渲染 SPS、推理延迟和 real-time factor。

不报告目标到达率、路径效率或 SPL。轨迹只用于分析局部避障行为。

### P7-FR-17：统计协议

- 超参数选择只使用训练/验证划分。
- 比较记录训练 seed、evaluation seed、资产版本和 checkpoint。
- 至少报告多 seed 均值、离散程度和 episode 数。
- 失败 episode 必须保存 outcome 和关键安全指标。
- 视频用于质检，不替代结构化指标。

## 9. 测试要求

### P7-TEST-1：配置和上游回归

- 导入并实例化已有 G1 velocity task，确认默认 observation/action 不变。
- 纯本体任务不得创建相机渲染工作负载。
- P7 task 必须通过独立 task id 显式启用。

### P7-TEST-2：command 与 observation

- 验证 P7 复用 `UniformVelocityCommand` 的 shape、范围和 resampling。
- 验证首版关闭 `heading_command`，actor 不依赖世界 yaw 目标。
- 验证 actor observation 不包含世界位置、目标位置或 privileged 字段。
- 验证单帧/历史视觉 shape、有限值、history 顺序和 corruption 对齐。
- 验证 reset 后 command/history 只影响对应 environment。

### P7-TEST-3：scenario 和 outcome

- 校验 scenario id、障碍类别、安全出口和数据划分。
- 用确定性状态/脚本轨迹触发九类主要 outcome。
- 验证 `both_open` 接受任一安全方向。
- 验证 `dead_end` 停车成功不要求世界目标。
- 验证永久停车、永久旋转和高频摆动不计 avoidance success。

### P7-TEST-4：奖励与接触

- 检查 free-space tracking reward 与现有 velocity task 方向一致。
- 验证危险状态允许安全偏离 nominal command。
- 验证允许足端接触和非法 body 接触分类。
- 检查 reset 首步、episode 边界和无效深度。

### P7-TEST-5：方向搜索和 shield

- 测试左开口、右开口、双开口和 dead-end 的临界条件。
- 测试停止距离/TTC 的零速、反向速度和无效深度边界。
- 验证 shield off 不改变 action。
- 验证 deterministic adapter/shield 的每次介入被记录。

### P7-TEST-6：训练链路

- 单环境和多环境 reset/step 通过。
- 完成短 PPO update、checkpoint 保存、重载和恢复 update。
- blind 路径无 renderer，视觉路径兼容当前 CUDA graph 约束。
- 正式关闭阶段前执行 `make check` 和相关完整测试。

## 10. 分阶段交付

### P7-0：协议与场景夹具

- 冻结 apartment 资产、scenario catalog、命令范围和数据划分。
- 预览 free/left/right/both/dead-end/false-alarm 场景。
- 标定碰撞风险、停车净空、搜索预算和 outcome 语义。

退出条件：固定 scenario 可重复加载，视觉/碰撞和安全出口人工检查通过。

完成（2026-11-10）：catalog、六个 manifest spawn、命令初值、split 和结构化
校验已实现；六类场景可由配置工厂重复加载，并已完成人工视觉/碰撞/安全出口
确认。风险距离、停车净空和搜索预算保留为 P7-2/P7-3 的可复现实验参数，后续
标定必须版本化修改，不能静默改变已记录实验。

人工预览可先使用零动作策略和现有 Play 入口，例如：

```sh
uv run play Mjlab-Velocity-Safe-Unitree-G1-3DGS \
  --agent zero \
  --scene-manifest src/mjlab/asset_zoo/3dgs/apartment/manifest.json \
  --scene-spawn-point p7_left_open \
  --viewer native
```

将最后一个参数依次替换为六个 catalog `spawn_point`。该命令只用于初始视觉、
碰撞和方向检查，零动作机器人会跌倒，不能用作行为验收。

### P7-1：安全速度跟踪任务骨架

- 增加独立 P7 task id，复用现有 velocity command 和 joint action。
- 接入 P6 observation，不增加 position/goal command。
- 完成配置、reset 隔离、无定位泄漏和非 P7 回归测试。

退出条件：单/多 environment 可确定性 reset/step，不要求策略收敛。

完成（2026-11-10）：已注册独立任务
`Mjlab-Velocity-Safe-Unitree-G1-3DGS`。它复用 P6 robust temporal
depth+validity、asymmetric critic、`UniformVelocityCommand` 和 G1 joint
position action；关闭 heading/world command，移除 push event，使用 12 m
world spacing，并只在该任务加载 apartment manifest。1/2 environment 的
reset/step 定向测试通过，已有 G1 velocity 默认配置回归通过。P7-1 只证明
任务骨架和隔离边界，不表示奖励或避障效果已经完成。

### P7-2：奖励、outcome 与规则基线

- 实现安全优先的奖励、九类 outcome 和结构化指标。
- 实现 deterministic depth safety adapter。
- 用脚本轨迹验证制动、搜索、恢复和 dead-end 停车语义。

退出条件：全部 outcome 可确定性触发，规则基线可完成基础左右避障。

2026-11-10 进度：新增任务局部 `P7DeterministicDepthAdapter`，消费归一化
depth+validity，将视野分为左/前/右扇区，输出 track、brake、stop、search、
resume、blocked 模式和安全 twist。它支持单帧/时序输入、无效深度保守停车、
固定搜索方向、搜索超时、恢复限速、介入计数和完全关闭时的精确 passthrough。
新增安全优先级固定的九类 `P7Outcome` 判定，并用合成深度/状态测试覆盖左右
选向、dead-end timeout、恢复和全部 outcome。

新增 `Mjlab-Velocity-Safe-Adapter-Unitree-G1-3DGS` 注册任务，将 Warp camera 的
原始深度在任务局部 observation term 中转换为 depth+validity，再由 adapter
生成 3 维安全速度命令。相机仍在 scene 中更新，但 camera group 不进入 blind
actor；actor 保持标准 G1 MLP observation/action 接口，可加载维度兼容的既有
velocity checkpoint。真实 1/2 environment reset/step 验证了安全命令确实进入
actor observation，且未修改通用 manager、command、sensor 或 velocity MDP。
P7 专用奖励、真实轨迹 outcome、checkpoint 场景 rollout 和基础左右避障效果
验收仍未完成，因此 P7-2 保持进行中。

定向验证覆盖 adapter、catalog/config、apartment manifest 和现有 velocity task，
结果为 `46 passed`；新增文件的 Ruff、ty 和 Pyright 检查通过。

2026-11-10 人工 rollout：标准 G1 velocity checkpoint
`logs/rsl_rl/g1_velocity/2026-10-24_16-15-47/model_29999.pt` 可由 adapter 任务
正常加载和运行，证明 99 维 blind actor 接口与既有 locomotion policy 兼容。
人工运行多个 apartment 出生点时未观察到碰撞，但 `p7_left_open` 和
`p7_right_open` 基本不恢复平移，因此不能判定左右避障通过。当前待验证原因
包括出生点已接近风险阈值、向下相机视野中的近地深度影响保守分位数，以及
搜索后前方净空未达到恢复阈值。下一步先记录 adapter mode、三扇区净空、
valid ratio、nominal/adapted command 和搜索时间，再据此标定 ROI、出生距离及
制动/恢复阈值；不能仅因没有碰撞就关闭 P7-2。

### P7-3：blind 与 privileged 基线

- 训练 blind velocity tracking 和 privileged local-geometry policy。
- 校验正常跟踪、奖励量级和障碍失败模式。

退出条件：free 场景保持稳定跟踪，privileged 策略能完成协议内局部避障。

### P7-4：视觉训练

- 接入 mesh vision 和 P6 各级 3DGS 视觉配置。
- 完成短 PPO、恢复训练和多 seed 训练。
- 执行视觉关闭、冻结和错配消融。

退出条件：视觉策略稳定优于 blind，并与规则基线公平比较。

### P7-5：安全层与正式评估

- 标定延迟、制动能力和稳定站立行为。
- 分别评估 policy、规则 adapter 和 shield 组合。
- 在冻结测试集报告安全、恢复、跟踪和性能指标。

退出条件：核心指标、失败轨迹、介入统计和性能数据完整可复现。

### P7-6：资产替换与扩展

- 用通过 P8 检查的新 collision 资产替换基线。
- 执行视觉—碰撞偏差消融。
- 评估动态障碍、复杂地形或独立定位/导航模块的后续里程碑。

世界坐标导航如果未来需要，应作为独立工作包，明确定位传感器和接口，不能
重新把仿真 world pose 静默加入 P7 actor。

## 11. P7 完成标准

P7 只有同时满足以下条件才能标记完成：

1. 独立 task id、scenario、MDP、指标和 checkpoint 可复现；
2. actor 不读取位置、目标、地图或碰撞真值；
3. 现有 G1 velocity task 默认行为和测试不回归；
4. free 场景速度跟踪退化处于冻结阈值内；
5. 左/右开口完成制动、搜索、选向和恢复；
6. dead-end 能安全停车，且不存在永久停车/旋转奖励漏洞；
7. blind、规则、privileged、mesh 和 3DGS 基线采用相同预算；
8. 视觉破坏实验支持或否定视觉依赖结论；
9. 多 seed、性能和失败案例数据齐全；
10. `make check`、相关测试和正式验证通过；
11. requirements、TODO、用户文档和 changelog 与代码同步。

数值阈值必须在 P7-0/P7-3 标定后冻结。短 PPO update、单次 demo 或依赖
oracle pose 的结果都不能作为 P7 效果完成证据。

## 12. 实现前仍需冻结的参数

- nominal linear/angular velocity command 的首版范围；
- 障碍风险距离、TTC、停车速度、停车 dwell 和安全净空；
- 安全方向的深度统计、扇区宽度和最小连续可见时间；
- 原地旋转速度、搜索角度、搜索时间和恢复条件；
- 允许接触 body、碰撞冲量、跌倒姿态和越界范围；
- episode 时长、停滞/振荡窗口和恢复跟踪窗口；
- reward 权重、curriculum、训练预算和 seed 数量；
- deterministic adapter 与 learned policy 的接口边界；
- 是否需要增加与柱体严格视觉/碰撞匹配的障碍布局。

这些参数应通过 P7-0 场景检查、P7-1 确定性测试和 P7-3 基线逐步冻结，
不得在最终测试结果出来后反向调整。
