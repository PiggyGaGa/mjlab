# 安全感知速度跟踪需求（P7）

最后更新：2026-11-17

状态：P7-0 场景协议、P7-1 独立任务骨架和 P7-2 奖励/outcome/规则基线
均已完成；P7-3 blind 候选已冻结，privileged 基线已收敛为“前方遇障停车”的
简单目标；端到端低层 PPO 失败后，当前主线转为“深度稳健净空 → 连续速度门控
→ 冻结 locomotion actor”的分层路线。特权教师和视觉 Student 实验原样保留，
作为学习对照和后续可选扩展。P7-2 的完成不表示真实障碍 GUI 矩阵、动态 safety
shield 或长期效果已经验收。

当前收口决定（2026-11-17）：按用户要求，本轮规则深度安全层及 36 组场景验证
标记为完成，不再追加场景测试、复测或为全通过继续调整规则。实测结果保留为
33/36（障碍停车 30/30、零介入跟踪 3/6），三个轻微减速项作为已知限制归档。
该决定不将未通过项改为通过，也不表示后续学习式感知策略已经完成。下文早期
场景扩展与待测计划保留为历史记录，不再作为本轮待办。

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

调整后的首版部署闭环为：

```text
temporal depth/validity ──> robust clearance threshold ──> speed scale alpha
nominal velocity command ────────────────────────────────> safe velocity command
                                                                  ↓
                                                    frozen locomotion actor
                                                                  ↓
                                                      target joint positions
```

- nominal command 表示用户或上层系统希望机器人跟踪的机体速度。
- 首版安全层不识别障碍类别，只根据深度净空缩放速度。
- 可选 safety student 不直接输出关节动作；它只估计净空/有效性和安全速度缩放。
- 已验证的 locomotion actor 保持冻结并负责站立、行走和关节协调。
- 系统在安全空间内尽量跟踪 nominal command。
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
- 失败校准、被替代基线及其 task/config/checkpoint 元数据必须作为论文实验
  证据保留。后续方案使用新增 task id 和 run name，不覆盖或静默删除旧实验；
  删除大型本地日志或 checkpoint 前必须取得用户明确确认。

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

实现记录（2026-11-10，2026-11-11 重构）：版本化
`safe_velocity_scenarios.py` 以冻结 dataclass 和常量记录六类 scenario、split、
manifest spawn、nominal command、碰撞 geom、安全出口扇区、允许区域、最小净空、
搜索预算和预期行为。它替代运行时读取的 `p7_scenarios.json`，避免外部 JSON
缺失或解析失败影响 G1 任务注册。apartment manifest 保留六个 `p7_*` 出生点；
这些是资产协议中的数据标识，不作为 Python 类名或文件名。
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
冻结的 locomotion actor 继续直接输出关节位置目标；视觉安全学生只生成结构化
安全标签或 safe velocity command，不新增高层导航策略或定位模块。

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

2026-11-10 进度：新增任务局部 `SafeVelocityDepthAdapter`，消费归一化
depth+validity，将视野分为左/前/右扇区，输出 track、brake、stop、search、
resume、blocked 模式和安全 twist。它支持单帧/时序输入、无效深度保守停车、
固定搜索方向、搜索超时、恢复限速、介入计数和完全关闭时的精确 passthrough。
新增安全优先级固定的九类 `SafeVelocityOutcome` 判定，并用合成深度/状态测试覆盖左右
选向、dead-end timeout、恢复和全部 outcome。

新增 `Mjlab-Velocity-Safe-Adapter-Unitree-G1-3DGS` 注册任务，将 Warp camera 的
原始深度在任务局部 observation term 中转换为 depth+validity，再由 adapter
生成 3 维安全速度命令。相机仍在 scene 中更新，但 camera group 不进入 blind
actor；actor 保持标准 G1 MLP observation/action 接口，可加载维度兼容的既有
velocity checkpoint。真实 1/2 environment reset/step 验证了安全命令确实进入
actor observation，且未修改通用 manager、command、sensor 或 velocity MDP。
首版 P7 专用奖励此前尚未接入；完整场景/多 seed rollout 和学习策略训练仍未
完成，因此 P7-2 保持进行中。

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

2026-11-11 诊断与标定：新增
`tests/run_g1_3dgs_safe_velocity_adapter.py`，固定 catalog nominal command，逐步
记录 adapter mode、三扇区净空/valid ratio、nominal/adapted command、搜索时间、
机体位姿和终止状态。旧参数下 left/right 各 350 步均全部停在 `brake`；前方
净空约为 0.88--1.15 m，始终不会触及原 `0.6 m` STOP 条件，而低速 command 已
使旧 locomotion checkpoint 基本停止。为区分最低安全净空与开始搜索条件，新增
`search_trigger_distance_m=1.0`，保留 `stop_clearance_m=0.6`；首轮 6 秒搜索只
产生约 21.5 度实际偏航，未进入 30 度安全扇区，因此将 catalog 搜索预算标定为
12 秒。

使用同一 `model_29999.pt`、seed 0 和 650 步诊断后，`left_open` 在 8.84 秒、
约 +0.511 rad 恢复，`right_open` 在 10.48 秒、约 -0.499 rad 首次恢复；两者均
出现 `brake -> stop -> search -> resume -> track`，未 blocked、未终止。该结果只
证明规则基线在这两个固定夹具上的基本状态链路，尚未检查碰撞、跌倒、最小净空、
outcome 和多 seed，不能作为 P7-2 正式退出证据。详细记录见
`validation/safe-velocity-adapter-diagnostics-2026-11-11.md`。

2026-11-11 轨迹 evaluator：P7 场景增加仅用于评估的 base-link 障碍接触传感器。
G1 base link 固定为根刚体 `pelvis`，允许它接触的场景 geom 只有 `ground`；与
任意其他 `pillar_*`/`surface_*` 碰撞均记为障碍碰撞。该逻辑只检查 `pelvis`
body，不把足端或其他肢体接触混入当前 baseline collision outcome。新增纯逻辑
`SafeVelocityTrajectoryEvaluator`，以世界位置、适配器模式、跌倒状态和接触力为
privileged evaluator 输入，输出唯一 outcome，并累计路径长度、深度传感净空、
最大接触力和接触冲量。确定性脚本轨迹可触发全部九类 outcome；dead-end 的
`safe_blocked_stop` 要求 blocked 后持续稳定站立，刚耗尽搜索预算但尚未满足
保持时间则归为 `search_timeout`。

同一 checkpoint/seed 的 left/right 650 步按上述 base-link 逻辑重跑，均归为
`avoidance_resume`；`pelvis` 障碍接触力和冲量均为 0，未跌倒、未越界、未停滞；
最小前向深度传感净空分别为 0.962 m 和 0.933 m。该结果补齐固定左右夹具的真实
轨迹 outcome，但它不是 collision geometry 的最小几何距离，也未覆盖其他场景、
多 seed、非 base-link 非法接触或奖励训练，因此仍不能关闭 P7-2。详细记录继续
保存在同一 validation 文档。

2026-11-11 首版任务奖励：保留现有 G1 姿态、步态、能耗和 action smoothness
项，但在 safe-velocity 任务中关闭无条件 nominal linear/angular tracking，改为
奖励跟踪 deterministic depth adapter 生成的安全 body-frame twist。空旷时安全
twist 等于 nominal command，奖励方向和总权重与原两项一致；风险区则允许减速、
停车、固定方向搜索和 blocked 站立。另聚合现有 `pelvis`—非地面障碍接触传感器
作为独立负奖励，不向 actor 暴露碰撞真值。普通 G1 velocity task 的奖励配置不变。
纯奖励、碰撞聚合、配置、真实 reset/step 和已有 adapter/outcome 定向测试共
`48 passed`。随后以 2 个环境、TensorBoard 本地 logger 完成 1 次 PPO update：
48 个采样步，collection 1.738 s，learning 0.330 s，value loss `0.6771`、surrogate
loss `-0.0762`，训练链路无 NaN/Inf。首次使用默认 W&B logger 的运行因本机没有
API key 在 rollout 前退出，不计为训练失败。当前权重仍是待正式短训练、恢复训练、
全场景和多 seed 实验标定的初值，尚未完成长期停滞、反复启停等学习策略漏洞
验收，因此 P7-2 继续进行。

2026-11-12 正式验收：为六类场景写入预期 outcome 和有限评估窗口，并将
规则 adapter 的搜索预算从 12 秒重新标定为 18 秒。原因是旧 locomotion
checkpoint 在部分 seed 中的实际原地偏航响应较慢；18 秒仍是有限预算，
`dead_end` 在预算耗尽后必须进入并保持 `blocked`，不会无限搜索。障碍场景
最多评估 22 秒；达到预期的 `avoidance_resume` 或 `safe_blocked_stop` 后立即
结束该条局部验收轨迹，free/false-alarm 则固定评估 10 秒。

同一 `model_29999.pt` 在 free、left-open、right-open、both-open、dead-end 和
false-alarm 上使用 seed 0/1/2，共 18 条真实环境轨迹，catalog 预期 outcome
`18/18` 通过：free/false-alarm 为 `continuous_tracking`，三类开口场景为
`avoidance_resume`，dead-end 为 `safe_blocked_stop`。全部轨迹均无 `pelvis`
障碍接触和跌倒；全矩阵最小前向深度传感净空为 0.792 m，高于协议的 0.6 m
安全停车净空。该值仍是 depth adapter 的扇区统计量，不是精确 geometry 距离。

训练链路除首次 1-update smoke 外，又从其 `model_0.pt` 成功恢复并完成下一次
update：2 个环境、48 个采样步，collection 1.692 s、learning 0.329 s，value
loss `0.7272`、surrogate loss `-0.1005`，数值有限。至此 P7-2 的安全优先奖励、
九类 outcome、deterministic depth adapter、基础左右避障、六场景多 seed 规则
验收和 checkpoint 恢复链路均满足退出条件，P7-2 关闭。学习策略奖励漏洞、
blind/privileged 对照和视觉训练分别进入 P7-3/P7-4，不作为 P7-2 阻塞项。

### P7-3：blind 与 privileged 基线

- 训练 blind velocity tracking 和 privileged local-geometry stop policy。
- 校验正常跟踪、奖励量级、遇障停车和障碍失败模式。

退出条件：free 场景保持稳定跟踪；privileged 策略在前方净空不大于 1.2 m
或前向射线无效时停止，并在 nominal command 持续为任意线速度/角速度时保持
稳定停车且不发生 base-link 障碍接触或跌倒。本基线不要求转向、绕行或自动恢复。

2026-11-12 工程进度：新增 renderer-free blind 与 privileged local-geometry
独立 task。blind actor/critic 保持标准 G1 MLP 的 99/109 维 observation，只读取
本体状态、历史 action 和 nominal command；场景仍使用冻结的 apartment collision，
但不加载 Gaussian asset、相机或 camera observation，奖励恢复现有 nominal
linear/angular tracking，并保留 base-link 障碍接触代价。

privileged task 同样不创建相机或 Gaussian renderer，在 `pelvis` 机体系增加
从左至右 80° 到 -80° 的 9 条 collision-truth 射线，只查询 apartment 的 pillar
和 surface geom group。actor/critic 各增加 9 个归一化距离和 9 个有效位，形成
117/127 维 MLP observation。该输入明确标记为
non-deployable privileged local geometry，不泄漏 world pose、scenario id、出口
标签或未来状态。

两个 task 均以 2 environments、24 steps 完成 CPU 1-update smoke：blind 为
33 SPS、value loss `0.6561`、surrogate loss `-0.0677`；privileged 为 31 SPS、
value loss `0.6452`、surrogate loss `-0.0709`，均无 NaN/Inf。该 smoke 只证明
模型维度、reward 和 PPO 链路可运行，不能证明 free tracking 或局部避障效果。
当前执行沙箱不可见 GPU；正式同预算训练、checkpoint warm-start 方案、多 seed
评估和 P7-3 退出条件仍未完成。

首轮 64-environment GPU 吞吐探针发现通用 G1 `command_vel` curriculum 会在
首次 reset 把 catalog 的 forward-only `0.0--0.8 m/s` 覆盖成 `-1.0--1.0 m/s`。
safe-velocity 任务现已任务局部关闭该 curriculum，确保 blind、privileged、规则
和后续视觉基线都遵守同一冻结 command protocol；普通 G1 velocity curriculum
保持不变。修正后的 privileged 64-environment 探针稳定在约 1.68--1.70k SPS；
修正前探针不得作为正式训练数据使用。

为保证 blind/privileged 与既有 G1 locomotion 的比较不被随机初始化主导，新增
`tests/prepare_g1_safe_velocity_warmstart.py`。它只迁移标准 locomotion actor：
blind 直接复制 99 维 actor，privileged 复制相同 99 维输入并将新增 18 维局部
几何输入列置零；critic、optimizer 和迭代计数保持目标任务的新初始化。两种转换
checkpoint 均已通过严格 resume，并各完成 1 次 CPU PPO update，loss 有限。
正式 1000-update、seed 42 训练由用户在 RTX 3060 上执行，结果返回后再按冻结的
free tracking 与局部避障协议验收。

Blind 首轮 1000-update 训练发生步态退化：最终策略无法站稳或向前行走，并出现
不规则后退。转换后的 actor 已确认与来源 checkpoint 逐参数一致，且用户确认
warm-start 正常、退化只存在于训练后的 `model_999.pt`。因此该 checkpoint 不计入
基线结果。safe-velocity learning runner 改用任务局部的保守微调参数：固定
`0.15` 动作标准差、`1e-4` 学习率、零 entropy coefficient 和每轮 2 个 PPO
learning epochs；普通 G1 runner 不变。短校准 rollout 通过前不启动 Privileged
正式训练。

修正配置已用 Blind warm-start 在 CPU 上完成 5 次 PPO update，动作标准差稳定为
`0.15` 且 loss 有限。下一步先运行 100-update GPU 校准并回放最终 checkpoint；
只有站立和前向 locomotion 保持正常，才扩展训练预算和启动 Privileged。

100-update GPU 校准的 `model_99.pt` 在 deterministic playback 中可以站稳，但
训练曲线仍显示平均回报为负、线速度跟踪接近零且大多数采样 episode 跌倒。
因此“可以站稳”尚未通过 free tracking 条件，1000-update 扩展保持暂停；必须先
确认模型是否实际按正向命令行走，并重新决定 Blind 是冻结 locomotion control
还是继续进行不可观测障碍下的 PPO 微调。

用户已确认 `model_99.pt` 在 deterministic playback 中能够站稳并向前行走。
该 checkpoint 因此冻结为候选 Blind baseline，不再继续 PPO 微调；后续用结构化
free rollout 量化速度误差、跌倒和碰撞。随机训练 telemetry 的退化不被该定性
观察覆盖，也不把 Blind 描述为已具备避障能力。

Privileged 保守配置也已完成 100-update GPU 校准。固定噪声生效且几何输入首层
权重已从零更新，但平均回报仍为负、训练 episode 主要跌倒，几何权重幅度很小。
因此不能依据 checkpoint 存在或碰撞 reward 为零宣称学会避障；必须先用
deterministic both-open rollout 检查减速、选向、通过开口和恢复跟踪。

实际回放显示该 checkpoint 在障碍前长期停止且不转向。定位到 privileged
collision-ray 转换错误地把 `-1` no-hit 当成 camera-style invalid；但对有限长度
真值射线而言，no-hit 表示 cutoff 范围内已知畅通。修正后 no-hit 映射为最大
clearance 和有效位 1，只有非有限传感值无效。该修改只影响 privileged local
geometry，不改变相机 depth、规则 adapter 或普通任务。旧 Privileged checkpoint
作废，需从零权重几何输入的 conservative warm-start 重新训练。

修正语义后的第二次 100-update 回放仍只表现为遇障停止，不转向、不恢复，未达到
`avoidance_resume`。保守更新保护了 locomotion，但新增几何权重学习过慢。
继续工作前必须保持实验定义清晰：端到端 privileged policy 需要可学习且无隐藏
状态歧义的几何目标；若直接把规则 adapter 的 safe twist 输入 actor，则属于
oracle-controller upper bound，不能再描述为学习到的局部避障策略。

用户随后将 P7-3 privileged 目标明确简化为“遇障停车”：前向中间 3 条射线只要
有一条无效，或最小净空不大于 1.2 m，就把目标的前进、侧移和偏航速度全部置零，
不受 nominal command 大小影响；无风险时原样跟踪 nominal command。该目标无状态，
不执行搜索、转向、选路或自动恢复，侧面障碍也不会触发转向。P7-2 deterministic
adapter 的制动—搜索—转向—恢复逻辑保持独立且不变。此前两个 privileged
checkpoint 只作为目标变更前的失败校准记录，不能恢复用于新目标训练。

2026-11-13 的 4096-environment、5000-update 长训练进一步确认端到端 PPO 不适合
当前简单停车目标。训练 telemetry 后期达到平均回报约 59.7、episode length 1000、
记录跌倒为零，但 deterministic playback 无法站稳；未经微调的 privileged
warm-start 在同一 play 配置下仍能站稳。最终 actor MLP 相对 warm-start 的参数
漂移约 21.8%，说明失败来自微调而不是环境或 checkpoint 加载。旧 `fell_over`
只检测超过 70° 的倾倒，零速度 tracking、仅 pelvis-障碍碰撞和缺少低高度/非足端
着地终止共同允许了静止但不稳定的奖励捷径；该长训练 checkpoint 全部作废。

为避免修改共享 RSL-RL 模型，新增独立 privileged stop-gate task。它保留 117 维
checkpoint 接口和零初始化的 18 维局部几何权重，但使用 9 条真值射线的中间 3 条
只在 observation 边界把 nominal command 切换为原命令或 `[0, 0, 0]`，直接复用
已验证能站稳的 warm-start locomotion actor，不再端到端微调 29 关节策略。这个
baseline 是 collision-truth oracle，不得描述为学习到的视觉策略；P7-4 再用
mesh/3DGS depth 学习 stop/go 感知。safe-velocity task 同时增加 pelvis 低于
0.5 m 和非足端接触地面的终止条件，普通 G1 velocity task 保持不变。stop-gate
已严格加载原 privileged warm-start 完成 CPU deterministic rollout：`both_open`
在第 49 step 触发停车，22 秒内无终止，最低 pelvis 0.753 m、最大倾角 1.83°；
`free` 在 10 秒内零次误停车、无终止，最低 pelvis 0.753 m、最大倾角 1.34°。
仍需用户 GUI 回放确认可观察行为，因此 P7-3 暂不关闭。

### P7-4：视觉训练

- P7-4A：以 collision ray 真值建立确定性连续安全教师，输出 normalized
  clearance、validity、速度缩放 `alpha`、stop label 和 safe command；这些标签
  只能进入独立 teacher observation group，不得进入部署 actor。
- P7-4B：接入 mesh vision 和 P6 时序 3DGS depth+validity，先用不对称安全损失
  监督学习视觉安全学生，不训练29维关节策略。
- P7-4C：如离线监督出现状态分布偏移，再执行 DAgger；只有监督和 DAgger
  验收后才允许只针对安全学生进行小规模 RL 微调。
- 执行视觉关闭、冻结、错配、单帧和无 validity 消融。

退出条件：视觉学生在 free 场景不过早停车，在 approach/blocked 场景输出保守
减速/停车；冻结 locomotion actor 保持站立和行走；视觉结果稳定优于 blind，
并与 collision-truth teacher 和规则基线公平比较。

2026-11-14 启动 P7-4A：新增独立
`Mjlab-Velocity-Safe-Privileged-Safety-Teacher-Unitree-G1-3DGS`。旧硬
Stop-Gate 和全部端到端 Privileged 实验原样保留。新教师以中间前向真值射线
计算连续 `alpha`，按 `safe_command = alpha * nominal_command` 平滑制动，并在
独立 `safety_teacher` observation group 输出四维标签：归一化前向净空、真值
有效位、`alpha` 和 stop label。actor/critic 不读取该标签，117/127 checkpoint
接口保持兼容。首版阈值为 stop `0.8 m`、slow `1.8 m`；2026-11-14 将 stop
调整为 `0.4 m`。固定命令回放确认长减速区间会在 `0.94 m` 左右进入 actor 的
低速步态死区，因此第二版将 slow 收近为 `1.0 m`。Safety Teacher 的 play
配置每秒打印一次
nominal/safe/measured 三维速度、前向净空、`alpha` 和 stop 状态；训练配置保持
静默。自动 reset 时不受 1 Hz 限制，立即附加打印 episode step 和具体 termination
名称。
play 同时固定使用所选场景的 nominal command（当前六个场景均为
`[0.5, 0.0, 0.0]`），不再随机采样训练速度。视觉 Student、数据采集和 DAgger
尚未实现。

首次固定命令 GUI 回放观察到：`clearance=0.972 m`、`alpha=0.409`、
`safe_vx=0.204 m/s` 且 `stop=False` 时机器人已经停止。该结果表明当前冻结
locomotion actor 存在低速步态死区，而不是 Safety Teacher 触发停车。第二版
`0.4--1.0 m` 区间预计把 `alpha≈0.4` 对应位置移至约 `0.64 m`；在 GUI 确认
制动安全前仍不把该参数标记为已通过标定。

第二版 GUI 日志显示机器人在 `clearance=0.699--0.735 m`、
`safe_vx=0.249--0.279 m/s` 时仍保持 `measured_vx=0.208--0.232 m/s`，随后净空
跳回出生点的 `1.5--1.6 m`，证明 episode 自动 reset，而非成功停车。具体终止
原因为机器人碰撞障碍物后重置，证明 `1.0 m` 的制动距离不足。第三版取两次
失败参数的中间值 slow `1.4 m`，预计 `alpha≈0.4` 对应约 `0.80 m`；stop 仍为
`0.4 m`。用户确认第三版能够无碰撞稳定停车，P7-4A 标定通过；前两版仍作为
低速死区和制动不足的失败证据保留。

2026-11-14 启动 P7-4B：新增
`Mjlab-Velocity-Safe-Vision-Data-Unitree-G1-3DGS`。该任务直接复用 P6 robust
时序 3DGS depth+validity，Student 输入形状为 `(B, 7, 2, 48, 64)`；同时用 9 条
collision ray 生成四维 `safety_teacher` 标签。camera、teacher label 和真值射线
均不进入 99/109 维 frozen locomotion actor/critic，actor 只接收 teacher 缩放后的
三维 command。训练配置保留视觉 corruption，play 配置关闭 corruption 并固定
场景 nominal command。在线接口、chunked 数据持久化和 smoke 回读已经完成；
加权监督损失与 Student checkpoint 尚未实现。

同日新增 `scripts/tools/collect_safe_velocity_vision_data.py`，严格加载 frozen
locomotion actor，按 chunk 原子写入 float16 camera history、四维 label、
nominal/safe/measured velocity、episode step 和 reset mask，并用 manifest 记录
checkpoint SHA-256、场景、seed、阈值、label 顺序和统计量。采集使用冻结 actor
已经验收的 deterministic play observation，但单独重新开启 camera corruption；
chunk 同时按 termination manager 注册顺序保存终止标志，manifest 汇总各终止原因。
1 environment × 2 step CPU smoke 已生成 2 个 chunk 并以 `weights_only=True` 回读，
所有 shape/dtype 正确。因此数据持久化接口已完成；加权监督损失和 Student
checkpoint 仍未实现。

首次 4 environments × 100 step GPU 采集保留为失败证据：400 个样本的
`mean_alpha=1.0`、`stop_fraction=0.0`，clearance 始终为 `3.392--5.000 m`，
measured forward velocity 均值仅 `0.067 m/s`、最大 `0.158 m/s`；reset mask 共 32
个，其中 4 个是初始 episode boundary，最长 episode step 仅 12。该运行使用训练态
locomotion observation，机器人尚未接近障碍物便反复终止，数据不得进入 Student
训练。termination 字段加入后必须用新目录重跑短探针，并确认终止计数为零，再扩大
采集预算。

修正后的 deterministic locomotion / corrupted camera 配置已用相同 Blind
`model_99.pt` 完成 1 environment × 30 step CPU 探针：四个 termination term
（time out、fell over、pelvis too low、non-foot ground contact）计数均为零，
`mean_alpha=0.9932`。短探针回读同时发现 CPU 上 `.cpu()` 不产生副本会使 buffer
持有可变张量别名，现已对所有入 chunk 张量显式 clone；该修复不改变 GPU 数值，
但保证 CPU/GPU 数据格式一致。仍需在 GPU 上用新输出目录完成多环境短探针，
确认终止为零且 episode step 连续，再进行正式采集。

第二次 4 environments × 200 step GPU 探针同样作为失败证据保留：四个环境各自
触发 15 次 `pelvis_too_low`，最长 episode 均仅 12 step，总计 60 次；没有
`fell_over` 或 non-foot contact。所有环境 `alpha=1.0`，measured forward velocity
均值仅约 `0.068--0.083 m/s`。这证明问题不是单个异常环境，也不是视觉教师提前
制动。进一步检查发现 play 配置虽然关闭 actor observation corruption，却仍执行
startup friction、encoder bias 和 base-COM randomization。首版离线数据现默认关闭
这三项 dynamics randomization，只保留 camera corruption；脚本提供显式
`--enable-dynamics-randomization` 供后续鲁棒性探针使用，并在 manifest 记录选择。
chunk 新增 `env_index` 和 `base_height`，用于逐环境审计。必须先通过 nominal
单环境 GPU 探针，再单独评估 dynamics-randomized 数据，不能混合解释。

随后关闭全部 startup dynamics randomization 的 4-environment CPU 探针仍精确
复现同步下沉：四个环境的 pelvis 高度均从 `0.780 m` 按相同轨迹降至
`0.494 m`，每 13 个采样点触发 `pelvis_too_low`。因此 dynamics randomization
不是这次自由落体的根因。最初曾将其暂时归因为 reconstructed scene 未复制；
进一步检查 MuJoCo Warp 数据模型后纠正该解释：`nworld` 已提供物理隔离 world，
同一 MJCF model 会在每个 world 中复用；单一 Gaussian asset 在未指定
`world_asset_ids` 时也通过 group 0 分配给全部 world。无需复制 collision 文件或
3DGS 点云。

真正错误是 safe-velocity 任务额外设置了 `env_spacing=12 m`，把每个隔离 world
中的机器人从 scene-local 原点移出有限 apartment，而 collision 和 splat 仍使用
各 world 的 scene-local 坐标。任务现改为 `env_spacing=0`，使所有 world 内的
机器人、collision、3DGS camera 和 truth ray 保持同一局部坐标；world 之间由
MuJoCo Warp 隔离，不会因坐标相同而互相碰撞。采集器同步解除单环境限制。旧
4-environment 自由落体数据继续作为错误 origin 配置的回归证据，而不是资产未复制
的证据。

修复后使用相同 Blind `model_99.pt` 完成 4 world × 30 step CPU collector runtime：
四个 world 的 termination 全为零，pelvis 高度范围均为 `0.754--0.780 m`，episode
step 均连续覆盖 `0--29`，collision-ray clearance 均为 `1.355--1.527 m`、alpha
均为 `0.955--1.0`。初始 camera frame 在启用独立 corruption 后跨 world 不完全
相同，证明视觉增强仍逐 world 生效。该结果验收物理站立、truth ray 和 camera
的多环境局部坐标一致性；仍需用户 GPU 长探针确认吞吐与完整制动轨迹。

用户随后完成 4 world × 200 step GPU teacher-rollout：800 samples、4 chunks，
`mean_alpha=0.6253`，所有 termination 均为零。逐 world 回读均得到 pelvis
`0.753--0.780 m`、episode step `0--199`、minimum/final clearance
`0.826/0.864 m`、minimum/mean alpha `0.426/0.625` 和 final measured forward
velocity `-0.004 m/s`；每个 world 的唯一 reset mask 是初始 episode boundary。
结果精确复现单环境 teacher rollout，GPU 多环境 reconstructed-scene origin 修复
验收通过。

首个有效单环境 GPU teacher-rollout 共 200 samples，所有 termination 为零，
`mean_alpha=0.6253`。回读显示 minimum/final clearance 分别为 `0.826/0.864 m`，
minimum alpha 为 `0.426`，最后 measured forward velocity 为 `-0.004 m/s`，pelvis
最低仍为 `0.753 m`。机器人稳定停在约 `0.86 m` 的 locomotion 低速步态死区，
因此 `stop_fraction=0` 不是采集故障，但也证明只用 teacher command rollout 永远
无法覆盖 `clearance <= 0.4 m` 的 stop 正样本。

采集器现把 behavior command 与 supervision 解耦：默认 `teacher` rollout 保留稳定
自由空间/减速轨迹；显式 `--rollout-command nominal` 让冻结 locomotion actor 执行
nominal command，同时继续保存 collision-truth teacher labels 和 safe command，
用于补齐近障碍/stop 标签。schema version 2 chunk 新增 `executed_command`，manifest
记录 rollout mode；nominal rollout 只能解释为仿真数据覆盖手段，不能作为安全策略
效果。两种 mode 必须使用独立目录并在训练采样时保留来源标识。

1 environment × 30 step CPU nominal-mode smoke 已验证 schema version 2：
`executed_command` 全程严格等于 nominal `[0.5, 0, 0]`，末帧 teacher safe target
已经独立降至 `[0.4775, 0, 0]`，所有 termination 为零，chunk 字段完整回读。
该 smoke 只验收命令/标签解耦接口，不代替 GPU stop-positive 覆盖验证。

首个 1 environment × 250 step GPU nominal rollout 到达近障碍区域，但在获得旧
stop 正样本前触发一次 non-foot ground contact。终止前最小 clearance 为
`0.466 m`、最小 alpha 为 `0.066`，随后 ray clearance 跳变并 reset；这证明
`0.4 m` command-zero 距离小于当前机器人/场景可达的无碰撞 pelvis-ray 净空。
该失败数据继续保留，不能通过延长 rollout 解释为缺少采样。

为保持用户已验收的连续制动曲线，`stop_distance_m=0.4` 和 safe command 计算均
不变；仅将二值 stop 监督定义为 `alpha <= 0.1`（未知几何仍为 stop）。本次轨迹在
碰撞前提供 11 个这样的正样本，对应 clearance `0.466--0.496 m`。该阈值表达
locomotion 已进入不可有效行走的尾段，而不是声称 safe command 已严格为零；
Student 仍以连续 alpha 为主要安全目标。

修正 origin 后的 4-world nominal rollout 已在 149 step 内生成 596 samples，
`mean_alpha=0.5737`、`stop_fraction=0.0738`，四类 termination 均为零。逐 world
均提供 11 个 stop positive，clearance `0.466--0.496 m`、alpha `0.066--0.096`，
episode step 连续 `0--148` 且 minimum pelvis `0.754 m`。因此 nominal/teacher
解耦、可达 stop label 和多环境采集链路均通过。

为避免不同 scenario 继续手工猜测碰撞前 step，collector 新增可选
`--stop-after-stop-samples N`：累计达到 N 个 stop positive 后，在执行下一次 actor
action 前结束并正常 flush，manifest 同时记录 requested/actual steps 和目标数。
该机制只控制数据覆盖终点，不改变 teacher label 或 safe command。

`dead_end` 首次 4-world nominal stop-target 采集在 127/500 steps 自动结束，生成
508 samples、44 个 stop positives，`mean_alpha=0.5583`、`stop_fraction=0.0866`，
所有 termination 为零。每个 world 各提供 11 个 positive，stop clearance
`0.399--0.492 m`、alpha `0.0--0.092`、minimum pelvis `0.754 m`，证明自动终点
在不同 spawn/朝向下能覆盖 command-zero 区域并在碰撞前结束。

`left_open` 的对称场景采集也在 108/500 steps 自动结束，生成 432 samples、
44 个 stop positives，`mean_alpha=0.4678`、`stop_fraction=0.1019`，所有
termination 为零。每个 world 各提供 11 个 positive，stop clearance
`0.409--0.496 m`、alpha `0.009--0.096`、minimum pelvis `0.754 m`。该结果确认
左侧开放布局不会破坏制动标签或并行 rollout。

`right_open` 随后使用完全相同的配置得到相同汇总和逐 world 结果：108/500
steps、432 samples、44 个 stop positives、`mean_alpha=0.4678`、
`stop_fraction=0.1019`，所有 termination 为零；每个 world 各有 11 个 positive，
stop clearance `0.409--0.496 m`、alpha `0.009--0.096`、minimum pelvis
`0.754 m`。左右布局的数据量、标签范围和稳定性对称性验收通过。下一步补充
`free` 和 `false_alarm` teacher rollout 的纯负样本，约束视觉 Student 的误停车率。

`free` teacher rollout 已完成 4 world × 200 steps，生成 800 个纯负样本，
`mean_alpha=1.0`、`stop_fraction=0`，所有 termination 为零。每个 world 都有
200 个连续样本、episode step `0--199`、alpha 恒为 `1.0`、minimum pelvis
`0.754 m`，且除初始 episode boundary 外没有 reset。该数据属于 catalog 的
`train` split，可用于约束自由空间误减速。`false_alarm` 属于 `test` split，后续
采集仅用于留出误停车评估，不得混入 Student 训练。

`false_alarm` teacher rollout 已完成 4 world × 200 steps，生成 800 个留出
纯负样本，`mean_alpha=1.0`、`stop_fraction=0`，所有 termination 为零。每个
world 都有 200 个连续样本、episode step `0--199`、alpha 恒为 `1.0`、minimum
pelvis `0.754 m`，且除初始 episode boundary 外没有 reset。该数据保持在 `test`
split，只用于衡量视觉结构引发的误停车。

同日完成首版 task-local 离线 Student 与训练入口。Student 将七帧
depth+validity 按有序时间通道输入小型 CNN，并在视觉特征后拼接 nominal command，
输出 normalized clearance、validity、alpha 和 stop 四个 logit。监督损失组合
clearance Smooth-L1、validity BCE、非对称 alpha Smooth-L1 和带正类权重的 stop
BCE；其中 alpha 高估默认权重为普通误差的 4 倍，stop 正类权重默认由训练集
负正样本比计算。数据加载器强制执行 catalog split：`free`、`left_open`、
`both_open` 用于训练，`right_open` 用于 validation，`dead_end`、`false_alarm`
只在 validation 选出 `best.pt` 后评估一次，避免 test leakage。训练器拒绝覆盖
非空输出目录，并保存 config、逐 epoch metrics、`last.pt`、`best.pt` 和最终 test
metrics。

真实数据 1-epoch CPU smoke 已完成：1828 train、432 validation、1308 test
samples 均成功加载，反向传播、checkpoint 保存/回读和一次性 test evaluation
通过；train/validation loss 分别为 `1.47391/2.13157`。单轮 validation
`false_stop_rate=0.9485`，仅证明未收敛初始化和训练链路可运行，不作为模型效果。
正式 GPU 训练与收敛验收仍待执行。

首个 50-epoch GPU run `offline_seed0_v1` 原样保留为失败证据。validation 在
`right_open` 上达到 alpha MAE `0.0610`、零误停和零漏停，但 held-out test 的
alpha MAE 为 `0.20688`、误停率 `0.1970`、漏停率 `0.9091`，不能接入在线闭环。
逐场景回读证明 `false_alarm` 实际 alpha MAE `0.0084` 且误停为零；全部错误来自
`dead_end`，其 alpha MAE 为 `0.5194`，44 个 stop 中漏掉 40 个，并在早期负样本
上过早停车。stop threshold 从 `0.1` 扫描至 `0.9` 均无法同时修复，排除单纯阈值
校准问题。

进一步逐 step 诊断发现 v1 在 `dead_end` 第 0 步就把 teacher alpha `1.0` 预测为
约 `0.043`。原因是首版 CNN 的 global average pooling 丢失空间位置，把两侧近墙
当作正前方障碍；独立 stop head 又与低 alpha 输出相互矛盾。v2 改为复用 mjlab
已有 `SpatialSoftmax`，将特征位置与全局强度共同输入 head；head 只回归
clearance、validity 和 alpha，stop logit 由 `(0.1 - alpha) / 0.02` 连续导出，保证
stop 与 alpha 定义一致。checkpoint/config 增加 `model_version=2`，最终测试报告
同时保存 combined 和逐 scenario 指标。真实 split 的 v2 1-epoch CPU smoke 已通过
反向传播、checkpoint 回读及逐场景报告；该单轮未收敛指标不作为效果。下一步使用
新 run name 重新执行 50-epoch GPU 训练，v1 不覆盖、不删除。

50-epoch GPU run `offline_spatial_seed0_v2` 同样保留为失败证据。best epoch 31
的 combined test alpha MAE 为 `0.22528`、误停率 `0.3663`、漏停率为零；
`false_alarm` 仍为零误停且 alpha MAE `0.02714`，但 `dead_end` alpha MAE
`0.53730`、误停率 `0.9978`，即几乎全程过早停车。SpatialSoftmax 修复了输出
一致性，但固定柱体训练轨迹仍不足以泛化到整面墙布局。

`dead_end` 几何审计排除出生点穿墙：spawn `(0.0, -4.4)` 朝向 `-Y`，墙体中心
`y=-5.9`、半厚 `0.1 m`，近表面 `y=-5.8`，初始几何净空正好 `1.4 m`。首帧四个
world 的 truth clearance 均为 `1.4 m`、alpha 约 `1.0`、pelvis `0.780 m`，无接触
或 termination；camera 前向平均归一化深度约 `0.275`，对应 `1.375 m`，与 truth
一致。因而 v2 失败不是 spawn、碰撞或深度异常，而是训练几何多样性不足。

2026-11-15 将下一阶段主目标明确为“根据深度几何阈值修改速度”，而不是识别墙、
柱子或场景类别。新增独立
`Mjlab-Velocity-Safe-Depth-Stop-Unitree-G1-3DGS` 任务，复用现有 blind G1
locomotion actor 和 task-local `SafeVelocityDepthAdapter`。该任务在完整前向图像
ROI 中聚合有效深度，以 q10 分位数作为稳健净空：净空不大于 `0.4 m` 时 alpha
为零，不小于 `1.4 m` 时 alpha 为一，中间线性缩放整个 nominal twist。无效深度
保守停车，安全层不产生新的横向速度或角速度，也不进入 search、resume 或 blocked
状态机。因此墙状和柱状深度只要具有相同的深度分布，就产生相同安全命令。

原 `Safe-Adapter` 的中间扇区搜索/转向行为保持默认不变；v1 global-pooling
Student、v2 SpatialSoftmax Student、对应数据和 checkpoint 也全部保留，继续作为
“有限柱体训练不能可靠泛化到墙面”的失败实验依据，不因路线调整而删除。合成墙/柱
形状等价、连续停车不转向、无效深度停车和任务隔离测试已纳入定向回归，当前结果为
`70 passed`。下一步需用同一个冻结 locomotion checkpoint 在 GUI 中分别回放墙、
柱和其它障碍，记录 q10 净空、alpha、实际速度、最小碰撞净空与误停车率，再决定
是否加入时序滤波。

同日完成首轮 GUI 形状与误报验证。默认柱体场景中，q10 净空从 `1.261 m`
下降后稳定在约 `0.923 m`，safe vx 稳定为约 `0.262 m/s`，实际速度进入冻结
locomotion actor 的低速步态死区并归零；到 episode step 301 无 termination。
`dead_end` 墙面场景同样在约 `0.819 m` 净空、`0.209 m/s` safe vx 处稳定停止，
到 step 251 无碰撞、终止或 reset。因此柱体与墙面均通过 `0.5 m/s` nominal
指令的无碰撞停车回放。

`false_alarm` 的冻结评估窗口为 10 秒。step 1--451 内 q10 净空保持
`1.432--1.461 m`、alpha 恒为 `1.0`、safe vx 恒为 `0.5 m/s`，满足窗口内零误
减速/误停车。step 501 后机器人已离开协议窗口并接近另一个真实障碍；安全层随后
按深度减速并在约 `0.82 m` 停止，不能记为 false-alarm 失败。

上述回放还确认单纯 `safe = alpha * nominal` 无法对任意大的 nominal 指令给出
统一制动上界。Depth-Stop 任务因此新增绝对前向速度上限 `0.5 m/s`：实际下发
前向速度绝对值不超过 `0.5 * alpha`，所以 nominal 大于 `0.5 m/s` 时不会放大
已验收的制动曲线，并在 alpha 为零时必然停车。该字段默认是 `None`，旧 Adapter
及其它任务保持原行为；合成测试覆盖 `2.0 m/s` nominal 在 free、brake 和 stop
三种净空下分别输出 `0.5`、`0.25` 和 `0.0 m/s`。

下一步针对整图 q10 的小目标盲区完成离线筛选。候选统计使用 7 帧中位数、宽度为
图像 `1/4` 的重叠水平滑窗、窗内 q20，以及垂直 ROI `0.2--0.85`。在现有六场景
chunk 上，`free` 和 `false_alarm` 的误停车率均为零，`both_open`、
`left_open`、`right_open`、`dead_end` 的漏停车率均为零；代价是有障碍场景比
整图 q10 更早减速。对应 alpha MAE 分别为 `0.098`、`0.025`、`0.026`、
`0.095`、`0.000`、`0.000`。这些指标只用于统计器筛选，不替代真实 GUI 验收。

Depth-Stop 任务已切换到上述局部时序统计，旧 Adapter 默认仍使用单帧、无滑窗的
原始扇区统计。实现只缓存每个环境最近 7 个 clearance/valid-ratio 标量，不缓存
额外图像，因此不会按环境数增加七帧图像显存。首次观测和 reset 后用当前统计填满
历史，避免未初始化帧触发停车；之后以时间中位数抑制少数瞬时坏帧。合成 `48×64`
测试中的窄障碍只占全图约 3.4%，整图 q10 会忽略它，而 `1/4` 滑窗 q20 能把
`0.9 m` 障碍转换为 `alpha=0.5`。单帧瞬态不会触发制动，连续 4/7 帧出现后触发。
下一验收项是重新 GUI 回放 `false_alarm`、柱体和墙面，确认更保守的局部统计没有
引入实际误停车，并新增真实细障碍资产/夹具验证小目标检测。

局部时序版本的三场景 GUI 回归已完成。`false_alarm` 在冻结的前 500 step 内
clearance 为 `1.746--2.142 m`、alpha 恒为 1、safe vx 恒为 `0.5 m/s`，无误减速
和 termination；窗口外接近真实障碍后正常减速，并在约 `0.86--0.87 m` 停止。
默认双柱场景在约 `0.696 m` 净空、`0.148 m/s` safe vx 处稳定停止，到 step 201
无 termination。`dead_end` 墙面在约 `0.732 m`、`0.166 m/s` safe vx 处稳定停止，
到 step 251 无碰撞、termination 或 reset。因此局部 q20 + 7 帧标量中位数没有
破坏旧三场景行为，且已覆盖半径 `0.18 m` 圆柱、整墙和窗口内视觉结构。

P7-5 自动评估入口同步扩展为 `--controller depth_stop`。该模式复用原结构化 trace、
碰撞力、路径长度、最小净空、termination 和多 scenario/seed 汇总，但对有
`blocking_geoms` 的场景将期望结果设为 `safe_blocked_stop`，对 `free` 和
`false_alarm` 保持 `continuous_tracking`。轨迹 evaluator 新增默认保持
`(BLOCKED,)` 的可配置停车模式；Depth-Stop 显式把连续低速 `BRAKE/STOP/BLOCKED`
都视为停车候选，从而正确表示冻结 locomotion actor 在 alpha 尚未为零时进入步态
死区的实际行为。正式多 seed GPU 矩阵仍待运行。

runner 已用相同 Blind `model_99.pt` 完成单环境 CPU 真实环境 smoke：
`depth_stop + false_alarm + seed 0 + 5 steps` 成功构建 99/109 维 actor/critic、加载
checkpoint、执行 camera depth safety layer 并写出 JSON trace；结果为
`continuous_tracking`，零 intervention、零碰撞、零 termination，最小统计净空
`1.807 m`。该 smoke 只验收 runner 接线，不代替完整时间窗口和多 seed GPU 结果。

为后续资产替换，Depth-Stop 配置工厂现可在环境构建前接收任意 scene manifest、
命名 spawn point 和固定 nominal command。base-link 障碍接触传感器不再固定引用
apartment geom 名称，而是从所选 manifest 的 collision MJCF 中枚举除 `ground` 外
的全部 geom；未传参数时仍使用原 apartment scenario，已有 task id 和实验不变。
使用仓库内 `ArtanimMotionCaptureStudio` 作为第二地图的配置回归已通过：PLY、MJCF、
default spawn、六个 `pillar_00--05` 接触目标和 `0.4 m/s` 固定命令均来自第二地图，
且没有 apartment `surface_*` 名称泄漏；单环境真实构建、reset 和 step 也已通过，
actor 为 99 维有限值且 reward 有限。该测试证明任务装配已与地图解耦，但 Studio
现有随机碰撞柱体没有视觉—碰撞匹配验收，因此不能据此宣称跨地图安全行为通过；
下一步仍需一张视觉/碰撞对齐的新 manifest 做真实 GPU/GUI 停车矩阵。

新增任务专用播放入口 `scripts/tools/play_safe_velocity_map.py`，用于执行上述
构建顺序。它复用标准 `run_play`、viewer、checkpoint loader 和 Depth-Stop task，
但把已由 manifest 构建完成的 env config 注册为仅当前进程可见的临时任务，再按
原有接口调用 `run_play`，不再使用通用 CLI 的构建后 scene override。示例：

```bash
uv run scripts/tools/play_safe_velocity_map.py \
  --checkpoint-file logs/rsl_rl/g1_3dgs_safe_velocity_blind/2026-11-12_22-38-51_blind_conservative_seed42_100/model_99.pt \
  --scene-manifest /path/to/map/manifest.json \
  --scene-spawn-point default \
  --nominal-command 0.5 0.0 0.0 \
  --device cuda:0 \
  --collision-visible
```

脚本保持单环境回放，并沿用 Depth-Stop 的限速和每秒诊断输出。共享的 `run_play`
接口和实现均不需要修改，所有已有 `play` 命令仍从 registry 加载配置且行为不变。
播放/安全配置联合回归为 `42 passed`；脚本 `--help`、ruff、ty 和 pyright 均通过。
正式跨地图结果必须记录 manifest、spawn、checkpoint 和 nominal command，且仍需
人工确认深度中障碍与 collision overlay 对齐。

P7-5 runner 也支持相同的构建前 manifest 参数，并要求自定义地图显式指定
`expected_outcome`，避免把 apartment catalog 语义静默套用到新地图。Studio
default spawn、seed 0 的首个 GPU 结构化探针在 step 237/4.74 s 提前达到
`safe_blocked_stop`：路径长 `1.171 m`，最小深度统计净空 `0.876 m`，零
base-link collision、零跌倒、零 termination。末帧 alpha `0.496`、safe vx
`0.248 m/s`、实测 vx `-0.031 m/s`，说明第二地图同样通过冻结 actor 的低速步态
死区停车。后续 seed 0/1/2 矩阵达到 3/3 `safe_blocked_stop`，最小净空范围
`0.864--0.905 m`，均无 collision、fall 或 termination；seed 2 路径较长
（`2.703 m`），仍需关注漂移。该结果是单 spawn/三 seed 行为证据，不替代
collision overlay 对齐检查和视觉/碰撞严格匹配的新地图验收。

Studio manifest 进一步增加六个同点异向验收 spawn：
`safe_velocity_pillar_00--05` 的 position 均为 `(0, 0, 0)`，orientation 分别按
collision MJCF 中 `pillar_00--05` 的中心方向计算，原 `default` 不变。配置测试从
MJCF 反算目标方位并验证每个 yaw，误差上限为 `1e-6 rad`。使用同一 checkpoint、
`0.5 m/s` 命令、seed 0 和最多 800 steps 的 GPU 矩阵达到 6/6
`safe_blocked_stop`；路径长 `1.302--2.290 m`，最小深度净空
`0.856--0.884 m`，全部零 collision、fall 和 termination。该结果关闭 Studio
同点多朝向行为验收，但仍不替代 collision overlay 人工对齐检查，也不证明随机
collision 柱体与 3DGS 视觉表面严格对应。

为把“朝向变化”和“位置变化”分开评估，manifest 保留上述同点组，并新增
`safe_velocity_generalization_pillar_00--05`。六个泛化 spawn 使用六个互不相同、
非原点的位置，距对应柱体中心 `3.35--4.21 m`，朝向仍由目标柱体与出生位置之差
计算；原 `default` 与同点组均不修改。几何回归同时验证位置、唯一性、目标距离和
yaw。使用同一 checkpoint、`0.5 m/s` 命令、seed 0 的 GPU 矩阵达到 6/6
`safe_blocked_stop`；路径长 `2.288--3.183 m`，最小深度净空
`0.847--0.890 m`，全部零 collision、fall 和 termination。这提供了位置与朝向
联合变化的初步泛化证据，但单 seed、单地图仍不能代表完整泛化能力。

验收 runner 新增可选 `--scene-spawn-prefix`，按 manifest 顺序展开所有名称前缀
匹配的 spawn，并与已有 `--seed` 做笛卡尔积；原 `--scene-spawn-point` 单点入口
保持不变且两者互斥。矩阵 summary 现在记录 accepted、collision、fall、
termination 数量，以及净空、路径和时长范围。使用前缀
`safe_velocity_generalization_` 和 seed 0/1/2 的单命令 GPU 验收完成 18/18
`safe_blocked_stop`，零 collision、fall 和 termination。全矩阵最小净空为
`0.709 m`，路径范围 `2.286--3.296 m`，停车接受时间 `6.78--9.14 s`；最小净空
来自 `generalization_pillar_03 / seed 2`。该结果关闭 Studio 地图内多位置三 seed
验收；这是 3DGS 背景加显式可视柱体的测试，跨地图和纯 3DGS 障碍测试仍待完成。

2026-11-17 新增独立视觉—碰撞对齐 runner：对同一 manifest spawn、seed 和机器人
相机位姿分别渲染 Gaussian-only（仅 group 5 占位 BVH + 3DGS）与
collision-only（仅 group 1、关闭 Gaussian），只在 collision 投影像素上比较深度。
默认通过条件为 collision 像素不少于 20、Gaussian 有效覆盖率不少于 90%，且
至少 80% 像素深度误差不超过 `0.15 m`。工具同时保存放大八倍的 overlay：绿色为
误差内对齐，红色为有 Gaussian 深度但误差超限，紫色为没有有效 Gaussian 深度。

历史诊断中，Studio 六个 `safe_velocity_generalization_*` spawn 的结果为 0/6：Gaussian 覆盖率
`0.557--0.841`，对齐比例全部为 0，中位绝对深度误差 `1.015--1.561 m`，P90
绝对误差 `1.354--1.883 m`。这些 group-1 柱体是额外添加的可视且可碰撞几何，
不要求在原始 3DGS 中有对应表面。2026-11-17 修正验收解释：该 0/6 只表示柱体
不在 Gaussian 重建中，不构成视觉输入错误或停车失败。此前 18/18 是有效的
“3DGS 背景 + 可视 mesh 柱体”停车证据，不等于纯 3DGS 障碍感知成功。旧报告和
overlay 原样保留，按此说明解释。

相机只渲染 3DGS 及可视 mesh（group 1/2）；group 3 隐藏碰撞只能参与物理、
接触检测和教师监督。viewer 显示隐藏碰撞的调试开关不决定相机的渲染范围。
当前 Depth-Stop 主线使用相机深度生成安全速度，用 pelvis 与非地面障碍物的
ContactSensor 判定物理接触；不挂载 collision-truth ray，也不读取 teacher 标签。
`found > 0` 表示接触存在；奖励的 `10 N` 力阈值用于筛选惩罚，不表示更小力的
接触不存在。正式 runner 更保守，将当前 `found>0` 或历史任意正接触力均记为
碰撞，记录目标 geom、当前 found 和历史峰值；接触历史覆盖一个控制步。
Vision-Data/Privileged 的真值射线仅属于可选监督或历史基线。
alignment runner 现在默认投影 group 3，仅用于离线对比重建表面与物理代理，
参考深度不进入策略；`--reference-geom-groups 1` 可复现历史柱体诊断。
Studio 当前 group 3 只有支撑地面，不包含重建障碍的物理代理，因此不能用它
声称已完成真实场景障碍碰撞验收。

旧 Vision-Data 任务把 camera `[1,2]` 和教师 ray `[1,3]` 交给共享 RenderContext，
导致 `[1,2,3]` 合并，旧数据保留为 collision-assisted 基线。当前 Vision-Data
默认改用任务侧 `CollisionTruthRaySensor`，复用上游射线 frame、buffer 和后处理，
调用原生 `mujoco_warp.rays(rc=None)` 直接查询碰撞模型。教师射线不加入相机的
SensorContext，因此碰撞标签和可见深度可来自同一状态，不需要双环境锁步，也不
需要修改共享 sensor/simulation/play 代码。代价是教师直接射线没有 BVH 加速，
当前只用于离线监督，吞吐需另行评估。

采集器默认使用隔离路径，manifest 新增 `visual_input`，记录视觉来源、相机组、
教师组、查询方式和 `collision_only_in_camera=false`。schema 仍为 2，tensor
格式不变。显式 `--legacy-shared-render-context` 可复现旧图像，不覆盖旧数据。
历史 manifest 没有该字段，不能自动视为隔离数据；训练时应显式选择新采集目录。

2026-11-17 CPU 验证：隐藏盒体高度改变及两个 world 独立位移时，教师距离按
几何关系变化，可视平面深度保持 2 m；奖励读取、`forward()`、观测读取之间
没有 `scene.update()` 时，也能取得新位姿。真实 G1 frozen `model_99.pt` 的
2 world × 2 step 采集生成 4 samples、2 chunks，零 termination；新旧模式教师
标签最大差为 0，视觉张量有 79,520 个值不同。schema-v2 Student loader 成功
回读新格式。启动检查验证相机 BVH 实际 geom group 只包含 1/2。当前沙箱未能
访问 CUDA；后续经沙箱外检查确认 GPU 可用并启动正式停车矩阵，见下述记录。
GPU 采集吞吐仍待单独验证。详细证据见
`validation/safe-velocity-visual-isolation-2026-11-17.md`。

2026-11-17 正式停车验收 runner 补齐 pelvis 接触事件 trace（含 obstacle geom、
found、历史峰值力）、精确终止原因和实际 camera BVH geom groups。Depth-Stop
启动检查要求无共享 raycast、相机 BVH 仅 group 1/2，发生任何 termination 时
均不接受该 run；原实验与 shared play/sensor/simulation 代码保留。
新增 telemetry 回归覆盖 found 存在但历史力为零、接触已结束但历史力仍为正、
完全无接触三种情况。正式 GPU 矩阵结果记录于
`validation/safe-velocity-depth-contact-2026-11-17/README.md`。

首个 GPU 探针暴露旧 evaluator 在 BRAKE/BLOCKED 内速度反复超阈值时，不会清除
此前累积的静止时间；因此出现宣称静止 1 s、末尾实际仅连续静止 0.9 s 的情况。
现已修正为速度或模式不满足条件就清零，需连续平面速度不超过 0.1 m/s 满 1 s
才接受停车；trace 汇总另独立反算 `final_stationary_hold_s`。修正只影响任务侧
验收判据，不改变深度处理、速度门控、locomotion checkpoint 或物理参数。
首轮旧判据矩阵完整保留，复验用新文件名记录，不能将首轮分数当成最终验收。

修正后的 RTX 3060 正式复验共 36 组：apartment 六场景 × seed 0/1/2 得到
15/18，Studio 六个泛化 spawn × seed 0/1/2 得到 18/18；障碍停车合计 30/30，
free/false_alarm 连续零介入跟踪为 3/6。全部零 pelvis 接触、跌倒、termination，
且实际 camera BVH 仅 `[1,2]`、无共享 raycast。停车末尾连续静止均为 1.00--1.02 s。
最小深度统计净空为 apartment 0.6786 m、Studio 0.7131 m。
free/seed 1、2 与 false_alarm/seed 1 存在轻微减速，最低 alpha 为
0.9373、0.9055、0.9384。free/seed 1 首次介入的 108 个近像素均匹配侧方可视
柱体 `pillar_01`；相机输入正常，问题在整片 ROI 的窗口净空未区分运动通道。
其余失败项尚未逐像素归因。按用户收口决定，保留全部结果与已知限制，不再开展
速度方向/通行区域敏感性场景测试，也不通过隐藏可视物体、改变阈值或缩短窗口
宣称全矩阵通过。本轮验证工作完成，学习式感知安全层另行开展。

### P7-5：安全层与正式评估

- 标定延迟、制动能力和稳定站立行为。
- 分别评估 policy、规则 adapter 和 shield 组合。
- 在冻结测试集报告安全、恢复、跟踪和性能指标。

退出条件：核心指标、失败轨迹、介入统计和性能数据完整可复现。

### P7-6：资产替换与扩展

- 使用 manifest-driven Depth-Stop 配置在环境构建前同时替换 3DGS、collision、
  spawn 和 base-link 障碍接触目标；不要只在环境配置完成后覆盖资产路径。
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
