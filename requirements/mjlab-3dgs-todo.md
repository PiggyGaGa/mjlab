# mjlab 3DGS 项目 TODO 与进度

最后更新：2026-11-25

## 0. 项目总览

### 0.1 阶段状态表

| 阶段 | 工作包 | 状态 | 下一交付物 |
|---|---|---|---|
| P0 | 需求与调研 | 第一轮完成 | 持续补充论文材料 |
| P1 | 3DGS 资产导入 | 基础完成 | 资源上限与导入优化 |
| P2 | MJWarp 原生渲染 | 基础完成 | 高分辨率吞吐、孔洞与遮挡边界验证 |
| P3 | Play 前端（Viser/外部 viewer） | **完成（按当前范围收口）** | 新版 wheel 后续发布；增强验证转为后续加固 |
| P4 | 外部碰撞输入 | **MVP 完成，阶段收口** | 接收 P8 后续泛化资产作替换对照 |
| P5 | 统一场景资产 | **完成** | 为 P7/P8 提供稳定场景契约 |
| P6 | YAM Vision → G1 视觉训练适配 | **完成（工程接口与 P7-ready 验收范围）** | 视觉收益、多种子消融和更大规模训练转入 P7/P10/P11 |
| P7 | 3DGS 安全感知速度跟踪 | 规则深度安全层及 36 组场景验证完成，按当前范围收口 | 学习式感知安全层（另行开展） |
| P8 | 自动碰撞生成 | **完成，按用户决定收口** | 地板拼接、独立语义准确率与安全保证作为已知限制，不再追加修补 |
| P10 | 性能与发布 | **进行中：固定资产对照完成，缓存验证中** | 缓存回归验收、传感/训练吞吐矩阵及完整测试 |
| P11 | 论文实验与写作 | 持续进行 | 基线、消融和论文草稿 |

“基础完成”表示核心链路已经实现，但扩展能力和最终验收仍存在
TODO。“MVP 完成，阶段收口”表示当前约定范围已经验收，未完成的增强项
保留到相应后续阶段，不再阻塞当前阶段。

### 0.2 项目依赖树

```text
mjlab 3DGS
├── P0 需求与调研                                  [第一轮完成]
├── 视觉基础
│   ├── P1 PLY 导入、规范化、多 world              [基础完成]
│   ├── P2 MuJoCo Warp RGB/depth                   [基础完成]
│   └── P3 Play 前端：Viser、视频、外部 viewer     [完成，按当前范围收口]
├── 物理场景
│   ├── P4 外部 collision-only MJCF                [MVP 完成，阶段收口]
│   ├── P5 PLY/MJCF manifest 与统一 transform      [完成]
│   └── P8 3DGS 自动碰撞代理生成                   [完成；按当前范围收口]
├── 视觉闭环训练
│   ├── P6 YAM Vision → G1 depth observation       [完成]
│   ├── P7 3DGS 安全感知速度跟踪               [规则基线场景验证完成；学习式后续]
│   └── 单台阶、楼梯、复杂地形与间隙               [后续扩展]
└── 质量与论文
    ├── P10 测试、文档、性能和发布                 [进行中；固定资产对照完成]
    └── P11 基线、消融、论文材料                   [持续进行]
```

关键执行依赖：

```text
P4 外部碰撞输入
  └── P5 统一视觉/碰撞资产
      ├── P6 训练视觉 observation
      │   └── P7 安全速度跟踪（冻结 apartment 基线资产）
      └── P8 自动碰撞代理生成（与 P7 并行）

P7 安全速度跟踪基线 + P8 生成资产
  └── 手工/自动碰撞替换对照与视觉—碰撞偏差实验
```

P7 与 P8 互不作为启动前置条件。P6 视觉链路已完成，P7 当前可直接冻结
局部障碍协议并建立安全速度跟踪 MDP 与 blind/privileged 基线。P8 依赖
已建立的 P4/P5
输入契约，不等待 P7 完成。两条线成熟后再联合验收，不能因为并行计划已
确定就标记代码、训练或算法已开始/完成。

共享边界：P7 固定已校验的 apartment 资产、配置与测试集作为基线；P8
输出独立版本的 collision MJCF/mesh、manifest、内容指纹及 provenance，
不修改训练接口。
自动资产通过几何/接触检查后，再在相同视觉资产和评估协议下替换对照，
不在一次训练中途静默覆盖基线资产。GPU 密集训练/重建按资源预算安排，
任务并行不要求重负载同时占满同一张卡。

### 0.3 每次开发后的强制更新

每次完成代码、测试、文档、实验或重要技术决策后，必须在
同一次
工作中更新：

1. 本节阶段状态表中的状态和下一交付物；
2. 对应章节的详细复选框；
3. “当前验证证据”中的测试命令、通过数量和人工验证结果；
4. 文末“更新记录”的日期和变更摘要；
5. 如果是用户可见功能，同时更新 README、详细 RST 文档和 changelog。

不能只修改详细 TODO 而不更新顶部总览，也不能仅因代码存在
就标记完成；必须有相应验证证据。后续智能体开始工作前应
先阅读本表，
结束工作前再次核对本表。

## 1. 文档用途

本文档是 mjlab 3DGS 项目的项目级进度入口，用于同步已经完成、
正在验证、下一步和长期研究工作。详细需求和技术论证仍以以下
文档为准：

- `mjlab-3dgs-training.md`：训练、碰撞和视觉闭环需求；
- `mjlab-3dgs-safe-velocity.md`：P7 安全感知速度跟踪的任务与验收契约；
- `mjlab-3dgs-play-viser.md`：独立 Play 与 Viser 需求；
- `mjlab-3dgs-research.md`：调研范围与论文证据要求；
- `mjlab-3dgs-research-report.md`：已有工作和第一轮调研结论。

状态约定：

- `[x]`：代码或文档已经完成，并有测试或人工验证证据；
- `[ ]`：尚未完成；
- **部分完成**：已经具备部分能力，但尚未满足完整验收标准；
- **延后**：已有明确边界，但不属于当前里程碑。

完成一项用户可见功能时，必须同时更新本 TODO、详细文档、README
最小示例和 changelog。代码任务只有在目标测试通过后才能标记
完成；
里程碑只有在 `make check` 和相应测试通过后才能关闭。

## 2. 当前状态摘要

### 已达到的能力

- 已能读取 Graphdeco 风格 3DGS PLY，并转换为 MuJoCo Warp 和 Viser
  所需数据。
- 已能通过 MuJoCo Warp 原生 renderer 生成包含 3DGS 的 RGB/depth。
- 已能在 Viser 中使用原生 Gaussian renderer 实时显示 3DGS 和机器人，
  不使用相机截图作为背景。
- 已提供 G1 velocity 和 tracking 的 3DGS Play 配置、预训练 tracking demo、
  PLY 命令行覆盖、多机器人显示和按需视频录制。
- 已通过自定义 MuJoCo 原生 viewer 的 room1 冒烟测试：加载 799,083 个
  Gaussian，预训练 G1 tracking 策略连续运行 200 步，关节位置数值有限，
  测试完成后窗口正常关闭。用户已确认 passive UI 修复，并决定 P3 按当前
  范围收口；新版 wheel 后续发布，增强验证不再阻塞 P3。
- 已支持通过 `TerrainEntityCfg(terrain_type="mjcf")` 加载静态
  collision-only MJCF；训练、Play 和 demo 均可用外部碰撞替换 plane。
- 已支持 primitive、多个凸 mesh、heightfield、相对资产路径、静态性
  校验，以及 group 3 碰撞代理调试显示。
- 已提供 room1 地面与六柱碰撞资产及基于标准 demo policy 的人工
  Play 冒烟脚本。
- 已提供 apartment 场景专用碰撞生成脚本：将 20,468 个占据 voxel 合并为
  89 个静态表面 box，再删除体积小于 0.1 m³ 的 53 个碎片，保留 36 个；
  地面及两个相机可见且具碰撞的柱体不参与体积过滤，导出
  collision-only MJCF。该脚本包含 apartment 专用范围、地面和障碍设定，
  不代表已实现跨场景或语义碰撞生成。过滤后不再保证覆盖所有原占据 voxel，
  小型真实表面也可能被移除，接触敏感用途仍需检查碰撞叠加效果。
- 已提供场景无关的 center-only voxel/box 命令行基线：从任意兼容 PLY 导出
  collision-only MJCF、独立哈希报告和 box 到原始 Gaussian ID 的确定性映射。
  apartment 真实资产复跑得到与专用原型相同的 20,468 个占据 voxel、89 个
  合并候选及 36 个 surface box；这不代表跨场景几何质量已验收。
- 已提供 schema v1 scene manifest，将 PLY、collision MJCF、坐标约定、
  统一 transform 和 spawn-point metadata 绑定为一个入口；
  train/play/demo 均可加载。
- 独立 G1 3DGS Depth 训练任务已读取相机 observation，单/双环境 CNN/PPO
  短更新与 checkpoint 恢复通过；普通本体 G1 任务仍不创建相机工作负载。

### 尚未达到的能力

- 独立落体和定量接触稳定性已通过；G1 在 room1 外部碰撞场景中的
  连续运行已由用户人工确认。velocity 闭环策略已在普通 plane 上完成
  32 环境、20 秒、零跌倒的静态站立验证；room1 上的策略级定量接触
  指标留作后续加固，不阻塞 P4 MVP 收口。
- P6 已完成深度语义、动态覆盖率、1/4-world 短性能和 PPO 链路验收；尚未
  证明视觉任务增益，长训练与任务级多种子对照属于 P7/P11。
- P7 规则深度安全层已完成减速/停车与 36 组场景验收；学习式安全层、主动
  转向搜索、安全选向和恢复跟踪仍未完成。P7 不包含世界坐标导航；单台阶、
  楼梯和复杂地形后移。
- P8 场景无关的 center-only voxel/primitive MVP、内部 mesh/TSDF/heightfield
  表示对照、固定预算选择、SplatTransform 外部输出及实际语义/含视觉任务对照
  已完成；未测独立语义准确率、不保证碰撞安全，也未证明语义策略收益。
- 尚未完成可选预处理缓存和正式性能基准；二者属于后期性能优化，不阻塞
  当前功能里程碑。

### 当前首要目标

P3、P4、P7 已按当前范围收口，P5、P6 已完成。2026-11-24 用户明确将 P8
标记完成，地板拼接等已知限制归档，不再作为当前修补主线。当前推进 P10：
固定 apartment 视觉资产与既有 G1 checkpoint/motion 的原有/通用自动碰撞
CPU/CUDA/native 对照已完成并于 2026-11-25 核验；当前实现并验证默认关闭的
版本化预处理缓存，随后为传感/训练吞吐矩阵及完整测试。原有模型的额外可见
柱体在两组实验副本中保持一致，
不得悄悄改变场景来宣称碰撞生成收益；原始 P7/P8 资产保持不变。P6 复用
YAM Vision 的
CameraSensor、标准化 depth、二维 observation group、Spatial-Softmax CNN
和 RSL-RL；单帧、validity、时序、robust 与 asymmetric 小规模 PPO 更新及
恢复训练链路已通过。项目保留参考 GaussGym 的真实重建场景视觉运动路线，
继续复用 mjlab、G1 与 MuJoCo Warp 原生渲染，RGB 作为后续对照。P8 输出必须
作为独立版本与 P7 已冻结的 apartment 视觉/碰撞基线替换对照，不能在训练
中途覆盖。NPZ 渲染缓存仍属于 P10 后期优化；本轮 source-ID sidecar 只服务
碰撞生成 provenance，不是渲染缓存。

## 3. 已完成工作

### A. 需求、范围与研究定位

- [x] 将训练逻辑与 Play 逻辑拆分为两个独立需求模块。
- [x] 确定本仓库只实现 Viser Play；MuJoCo viewer/simulate 前端由外部仓库
  负责。
- [x] 建立训练需求、Play 需求、调研需求和第一轮调研报告。
- [x] 调研 GaussGym、MuGS、DISCOVERSE、SplatGym、SplatTransform 等相关工作。
- [x] 记录 MuGS 使用独立 Gaussian renderer、依赖较复杂，与本项目复用
  MuJoCo Warp 原生 renderer 的路线不同。
- [x] 将项目意义、机器人应用方向和最小使用示例写入 README。
- [x] 确定首个机器人为 Unitree G1。
- [x] 2026-10-31 调整研究主线：参考 GaussGym 的真实重建场景视觉运动/
  导航训练；首个闭环改为目标到达，增加 RGB 与 depth 对照，台阶任务后移。
- [x] 2026-11-07 明确 P7 为 3DGS 目标条件视觉避障：不可通行障碍前
  减速、停车或转向，并以无碰撞、不跌倒的目标到达为验收结果；该定义已于
  2026-11-09 因无定位传感器约束被下条决策取代。
- [x] 2026-11-09 建立 P7 独立需求基线，定义场景协议、速度命令、MDP、
  safety shield、基线、指标、测试和分阶段验收；未将需求完成记为代码完成。
- [x] 2026-11-09 按真实机器人无定位传感器的约束重构 P7：取消 PointGoal/
  世界坐标导航和目标位置 observation，改为速度命令条件下的制动、停车、
  原地旋转搜索、安全方向选择与恢复跟踪。

### B. 3DGS 资产导入

- [x] 实现 `GaussianSplatAssetCfg`、`GaussianSplatSceneCfg` 和
  `GaussianSplatData`。
- [x] 支持 ASCII 和 binary little-endian Graphdeco 风格 PLY。
- [x] 支持 position、rotation、scale、opacity、SH DC 或 RGB 字段。
- [x] 支持 log/linear scale、logit/linear opacity 和四元数归一化。
- [x] 支持平移、旋转和统一缩放。
- [x] 支持 opacity threshold、world bounds 和 maximum splat count 过滤。
- [x] 支持多个 splat asset 和 `world_asset_ids` 映射。
- [x] 支持在没有 CameraSensor 时由 Scene 持有规范化 splat 数据，供 Viser
  使用。
- [x] 增加合成 PLY 的字段、过滤、变换、分组和异常输入单元
  测试。
- [x] 增加真实 PLY 的加载与 MuJoCo Warp 渲染慢速测试。

当前限制：

- [ ] 支持高阶球谐的视角相关颜色；当前只使用 SH DC 静态颜色。
- [ ] **后期优化：**建立版本化 NPZ/cache；当前环境初始化直接解析一次
  PLY，不影响后续逐步训练或渲染吞吐。
- [ ] 建立安全的文件大小、元素数和内存上限。
- [ ] 支持资产 manifest、来源、许可证和内容指纹。

### C. MuJoCo Warp 原生 3DGS 渲染

- [x] 将 splat position、rotation、scale、RGBA、address 和 world group 传给
  MuJoCo Warp render context。
- [x] 支持 CameraSensor RGB 和 depth 输出。
- [x] 复用 MuJoCo Warp 原生 renderer，没有引入第二套训练相机
  rasterizer。
- [x] 保持未配置 3DGS 时的原有 sensor 路径。
- [x] 为真实 PLY 提供原生 renderer 集成测试。

仍需正式验证：

- [x] 记录 splat depth 为相机轴向 Z-depth；它不是期望/最近点深度，未命中
  mesh 且累计 opacity 未达到阈值时返回零。2026-10-30 的验收使用旧后端固定的
  ``transmittance < 0.005`` 语义，作为历史证据保留。
- [x] 2026-11-11 锁定 MuJoCo Warp fork ``8ffab7a5``。其
  ``create_render_context`` 已通过 ``splat_depth_transmittance_threshold`` 将
  depth 表面判定与 RGB 合成终止条件解耦，默认值为 ``0.1``，并校验严格位于
  ``(0, 1)``。mjlab 当前沿用上游默认值，没有增加本地兼容层。
- [ ] 在当前默认阈值 ``0.1`` 下重新比较孔洞率、漂浮点误检、RGB/depth
  对齐、渲染吞吐及已有 checkpoint 的观测分布；旧 ``0.005`` 验收结果不得
  直接作为新默认值的质量证据。
- [ ] 验证不同相机数量、分辨率和更新频率下的显存及吞吐。
- [x] 当前 1/2/4-world P6 视觉 rollout 与 PPO 更新已在仿真 CUDA graph
  启用时通过；更多相机、分辨率和更新频率组合仍归入性能矩阵。
- [ ] 建立 segmentation 能力边界和测试。

### D. P3 Play 前端（Viser 与外部 viewer）

**完成（2026-10-30，按用户确认的当前范围收口）。** 已有 Viser/native
核心链路、tracking/velocity 联调、接口契约、自动控制测试及性能初测证据，
用户确认 passive UI 已修复。最终 `make check` 和 13 项定向回归已通过。
用户决定后续再发布更新版本；新版 wheel 发布/安装，以及下列尚未勾选的
人工画面对照、物理键鼠、完整 MJWarp 接触重放、多场景、长期资源和正式
性能验证均保留为后续发布或加固任务，不再作为 P3 完成的阻塞条件。
未勾选项不代表已执行或通过，已有验证边界继续保留。

#### D1. Viser 原生 3DGS Play

- [x] 使用 Viser `add_gaussian_splats` 原生接口显示，不再传输低分辨率
  截图。
- [x] 在同一交互式视口中显示静态 splat 和动态机器人 mesh。
- [x] splat 只在初始化时上传，不在每个 physics step 重传。
- [x] 同步 Viser world offset，使机器人和 splat 使用一致的环境偏移。
- [x] 普通 Viser Play 不创建 MuJoCo Warp camera，避免重复渲染。
- [x] 保留现有浏览器 orbit、zoom、跟随、暂停、单步和环境选择
  能力。
- [x] 支持一个 splat 场景中显示多个仿真机器人。
- [x] 增加 Viser splat 创建和 world offset 单元测试。
- [x] 增加碰撞代理实时显示开关。

**部分完成：**

- [ ] 为多个不同场景实现并验证机器人 world、splat group 和 collision
  scene 的原子切换。
- [ ] 对浏览器断连、重连和长期运行资源占用进行专项测试。
- [ ] 增加 PLY/collision 包围盒、坐标轴和 spawn 对齐调试模式。
- [ ] 对 splat 与机器人互相遮挡进行正式图像回归测试。

#### D2. 外部 MuJoCo viewer/simulate 联调

本仓库不重复实现外部前端，只维护稳定场景/状态接口并记录跨仓库测试
步骤和结果。

- [x] 定义供外部 MuJoCo viewer/simulate 仓库使用的版本 1 场景与状态接口，
  见 `mjlab-3dgs-native-interface.md`；覆盖规范化数组、PLY 编码、状态同步、
  资产选择和生命周期。跨版本兼容性及不同碰撞模型切换不在承诺范围内。
- [x] 接入 `mujoco.gaussian.GaussianAssets` 与 passive viewer 的
  `gaussian_ply`/`load_gaussian_ply` 接口，复用现有机器人状态同步；
  无 3DGS、资产组切换和缺失定制 wheel 的错误提示已有单元测试。
- [x] 记录外部仓库地址、测试分支/commit、依赖版本和启动命令，见下文。
- [x] 使用 room1 `manifest.json` 同时加载 PLY、collision MJCF 和
  `default` spawn point。
- [x] 加载标准 demo 的 G1 tracking checkpoint 和 motion，原生 renderer
  就绪后完成 200 步策略推理与仿真，检查同步后的关节位置为有限值。
- [x] 加载本地 G1 velocity `model_10000.pt`，通过显式旧版 phase observation
  适配完成真实窗口策略运行；显示帧上的 qpos/qvel 与 MJWarp 状态逐项一致。
  直接使用当前标准配置加载失败（旧 actor 98 维，当前 99 维），未修改
  checkpoint 或生产任务默认配置，不据此声称完整训练环境兼容或零跌倒。
- [ ] 确认 3DGS、机器人 mesh 和 collision proxy 的坐标、尺度、朝向及
  遮挡关系与 Viser 结果一致。
- [x] 使用非对称位置、各向异性 scale 和非单位旋转验证 native PLY 往返的
  position、covariance、DC RGBA 与 Viser 的规范化输入一致；不替代画面对照。
- [ ] 验证 collision proxy 的显示/隐藏不会改变真实物理接触。
- [x] 使用相同 CPU 初始状态/控制，比较 group 3 隐藏/显示下的 100 步
  接触重放：qpos/qvel、接触对和接触距离完全一致；两轮分别包含
  680 和 533 个接触样本。完整 MJWarp 策略轨迹对照仍待补。
- [ ] 验证交互相机、暂停/继续、reset、退出和资源释放。
- [x] 四个真实窗口通过 callback/action queue 的暂停、单步、reset、继续、
  相机参数往返、关闭和临时 PLY 清理；物理键盘/鼠标和长期资源测试待补。
- [x] 用户人工确认原生交互窗口中 F8 可切换 3DGS 显示，但左侧
  `3DGS > Show / Hide` 按钮置灰，不能点击；该结果不代表 UI 完整验收通过。
- [x] 定位外部 MuJoCo `simulate/simulate.cc` 的 `Make3DgsSection`：
  radio 控件使用类别 `5`，`UiPredicate` 将其限制为非 passive 模式；
  mjlab 使用 `launch_passive()`。F8 直接调用 `mjgs::ToggleVisible()`，
  不受该条件限制。
- [x] 用户于 2026-10-30 确认外部 MuJoCo 的 passive UI 问题已修复。
  本轮未独立核对具体实现、修复 commit 或 wheel 内容。
- [ ] 重新构建并发布修复后的 wheel，更新 mjlab 的 Release URL 与
  `uv.lock`，记录版本及 hash；当前未确认已安装 wheel 含此修复。
- [ ] 用 `uv run tests/run_g1_3dgs_native_demo.py --interactive` 人工验证
  Show/Hide 可点击、与 F8 状态一致，且仅改变 3DGS 可见性。
- [ ] 对比无 3DGS 与启用 3DGS 的 Actual RT、帧率、显存和加载时间，
  保存测试日期、硬件、命令和结果。
- [x] 完成 viewer off/on/off/on 四轮各 10 秒初测，记录 Actual RT、提交
  频率、设备总显存与窗口加载时间。正式显示 FPS、进程显存和完整场景
  初始化耗时尚未测量，不关闭上一项。
2026-10-30 六项验收的详细结果见
`validation/p3-acceptance-2026-10-30.md`，原始 JSON 和直接加载失败记录
一并保存在 `validation/`。自动工具为 `tests/run_g1_3dgs_acceptance.py`。

当前联调记录（2026-10-29）：

- 外部仓库：`git@github.com:PiggyGaGa/mujoco.git`；本地路径
  `/home/yzt/Documents/3DGS/mujoco`，分支 `feature/3dgs`，当前 HEAD
  `0a2f9118f4b9ad695a6c55328d3549446a3e6e7a`，记录时工作树干净。
  此为当前源码版本记录，尚未独立核对 wheel 构建来源与该 commit 的对应关系。
- 已安装包：`mujoco==3.16.0+3dgs`，CPython 3.13；GPU 为 NVIDIA
  GeForce RTX 4070 Laptop GPU（8 GiB），OpenGL 4.6，驱动 580.159.03。
- 自动冒烟命令：
  `uv run --no-sync tests/run_g1_3dgs_native_demo.py --steps 200`。
  实测 renderer 状态为 `ready`，加载 799,083 个 Gaussian，完成 200 步，
  进程退出码为 0。此测试不证明遮挡正确、交互控制完整或接触指标合格。
- 当前交互入口：`uv run tests/run_g1_3dgs_native_demo.py --interactive`；
  添加 `--collision-visible` 可显示碰撞代理，交互模式尚未专项验收。
- 初次验证时使用 `--no-sync` 保留手动安装的定制 wheel。2026-10-30
  已将 Linux x86_64 Python 3.12/3.13 的 uv source 改为远程
  `https://github.com/PiggyGaGa/mujoco/releases/tag/v1.0` 中的对应 wheel，
  更新 `uv.lock` 并记录 SHA-256；当前无需 `--no-sync` 或相邻源码目录。
  这些 wheel 要求 glibc >= 2.35；其他平台/Python 版本仍使用官方 MuJoCo，
  不提供此定制原生 3DGS viewer。
- [x] 通过 GitHub Release wheel 分发定制 MuJoCo，并接入 uv 按平台和
  Python 版本选择依赖；README、脚本帮助、RST 指南和 changelog 已同步。
- P3 已按用户确认的当前范围完成；D1 增强项和 D2 未勾选项作为后续
  发布/加固任务保留，不阻塞当前阶段收口。

### E. G1 任务、CLI 与 checkpoint Play

- [x] 注册 `Mjlab-Velocity-Flat-Unitree-G1-3DGS`。
- [x] 注册 `Mjlab-Tracking-Flat-Unitree-G1-3DGS`。
- [x] 让 tracking 3DGS demo 复用现有预训练 checkpoint 和 motion。
- [x] 增加 `demo --use-3dgs True`。
- [x] 增加 `--gaussian-splat-file` Play/demo 路径覆盖。
- [x] `--use-3dgs False` 时忽略 PLY 参数，不校验或解析该文件。
- [x] 3DGS demo 默认一个环境，并允许通过 `--num-envs` 显示多个
  机器人。
- [x] 为 G1 3DGS task 增加较高的 solver iteration/line-search iteration 预算。
- [x] 增加 velocity、tracking、Play override 和 demo 参数测试。
- [ ] Play 自动校验 checkpoint、visual asset 和 collision asset 的兼容性。

重要边界：

- 当前 `unitree_g1_flat_3dgs_env_cfg(play=False)` 直接返回普通 flat 配置。
- 当前的 train 命令不会加载 PLY、不会创建 3DGS CameraSensor，
  也不会使用视觉 observation。
- 未提供外部碰撞时，Play 继续使用 flat plane；提供 scene manifest 或
  collision MJCF 后，机器人改为与外部碰撞场景接触。

### F. 视频录制

- [x] VideoRecorder 支持从指定 RGB CameraSensor 录制。
- [x] `g1_3dgs_video` sensor 只在同时请求视频和该 sensor 时按需启用。
- [x] 录制路径使用 MuJoCo Warp 原生 3DGS 合成，而 Viser 保持
  浏览器原生渲染。
- [x] 修正布尔参数用法，`--video` 与 `--video-sensor` 分别传递。
- [x] 增加按需 sensor、重复启用和视频 wrapper 测试。
- [ ] 对视频帧数、分辨率、时长、资源释放和异常退出进行端到端测试。

### G. 文档

- [x] README 包含项目意义、当前边界和最小 demo。
- [x] `gaussian_splats.rst` 包含格式、配置、训练成本、Play、demo、
  多机器人、录制和性能说明。
- [x] API、sensor index、viewer 文档和 changelog 已接入 3DGS 内容。
- [x] API 与使用指南已记录 scene manifest 的类型、加载函数、CLI 和
  分开指定的兼容接口。
- [x] 文档明确 3DGS 当前只负责视觉，不参与 physics、contact 或
  raycast。
- [x] 2026-10-25 完整 Sphinx 文档构建成功。

### H. 当前验证证据

- [x] 2026-11-21 P8 MVP 验证（非完整 P8 验收）：``make check`` 全绿，碰撞生成与 apartment
  定向回归 ``24 passed``，``make docs`` 成功；共享 ``play.py``、``demo.py`` 及其
  配置测试保持零差异。完整 ``make test`` 首次被系统 ROS pytest plugin 缺失
  ``lark`` 阻断；禁用外部 plugin 并把 Warp cache 指向 ``/tmp`` 后，P8 新增测试
  全部通过，仓库仍有 4 个缺失未纳入仓库的 ``scene_transform.ply`` slow-test
  失败和 1 个既有 Studio pillar alpha 0.8/测试期望 1.0 失败，末尾 ``wandb``
  测试在受限网络环境中不退出。因此不把完整仓库测试写成全绿。
- [x] 2026-11-21 P8 独立 CPU collision benchmark：新增不经过 viewer、3DGS
  renderer 或 policy 的统一落球探针工具，记录输入 hash、编译时间、physics SPS、
  ``ncon/nefc`` 与 solver iteration。Artanim 同为 3×3 probe、1,000 warmup、
  10,000 measured steps 时，center-only 306 primitive 为 20,382 steps/s，
  0.2 m scale 上限的 covariance-aware 276 primitive 为 20,563 steps/s；接触均值
  分别为 12.0846/12.0825，约束均值为 48.3384/48.3300。这里只是一轮 CPU
  小探针结果，用于证明测量链路与量级，不声明 0.9% 差异可复现或代表含视觉吞吐。
- [x] 2026-11-21 P8 独立质量 probe：新增 ``--evaluation-points`` NPZ，与生成用
  known-free sidecar 分离；输出 reference surface 最近代理距离 mean/p95/max 和
  free-space false-occupied count/fraction。三阶楼梯回归 surface max distance 为
  0.075 m、free probe 误占为 0；桌底、3×3 坑洞、双层间隙在无约束时全部 probe
  被误占，约束后均为 0。report 同时记录 grid shape、连通分量、丢弃 box、映射
  覆盖率、primitive/voxel 比及 primitive-only 的 0 mesh vertex/face 语义。
- [x] 2026-11-21 P8 semantic sidecar 契约：CLI/API 接受与原 PLY vertex 一一
  对齐的整数 ``(N,)`` 标签；voxel 采用确定性多数票，跨标签 occupied voxel
  作为 merge 禁区，auto fit 扩张到其他标签 source point 时回退 AABB。unknown
  默认 ``-1``，保留为 geometry-only residual。双层 synthetic 在无标签时被填实，
  标为 ``3/-1`` 后分为两个边界且 source-ID 全覆盖；输出 NPZ 记录每 primitive
  标签，report 记录 sidecar hash、unique label 与逐标签计数。尚未声称 SceneSplat
  解码质量通过。
- [x] 2026-11-21 P8 covariance-aware 基线：新增 Graphdeco scale/rotation 解码、
  quaternion 归一化与 ``R diag(scale²) Rᵀ``，可选加入 candidate center covariance
  后再做 PCA；``max_gaussian_scale`` 在 voxelization 前过滤任一轴过大的 splat。
  Artanim 最大 scale 分布为 median 0.0228 m、p99 0.1309 m、max 124.8851 m；以
  0.2 m 上限和 covariance-aware auto 生成 554 merged candidate、172 primitive
  （163 box、9 cylinder），映射 183,409 Gaussian，MuJoCo 编译为 173 geom（含
  ground）成功。该证据建立可运行对照，不证明其质量优于 center-only 205
  primitive 结果。
- [x] 2026-11-21 P8 known-free 约束阶段：生成 API 与 CLI 接受可选 ``(N, 3)``
  world-space ``.npy`` 点，输入文件哈希、点数及网格内唯一 voxel 数进入 report；
  occupied/free 冲突直接报错。merge 切分优先减少仍包围 free voxel 的子候选，
  auto sphere/cylinder/capsule/OBB 若侵入约束点则回退候选 AABB。synthetic 桌底、
  3×3 坑洞和双层间隙均从“无约束时被填实”变为零 probe 误占，box 预算分别
  不超过 12/6/2；碰撞生成与 apartment 回归 ``17 passed``。未提供 sidecar 时
  apartment 仍生成 20,468 occupied voxel/36 box，MJCF SHA-256 为
  `df1fe9cc451ae4a980f2605f0f5a0fc76a3886df6252ef229baacec08fd806f8`，NPZ 为
  `3ba252d650ce25c5cc295ad360a580a195e5573269cb96c66cc1f19fd48034aa`。
- [x] 2026-11-20 P8 平面 merge 残差阶段：新增默认关闭的绝对距离约束，仅对
  PCA 判定为近似平面的 merged candidate 生效。synthetic 门洞在默认与 4 cm
  约束下均保持可通行；三阶楼梯不再被单个 ramp-like box 跨越，两个自由空间
  probe 未被占用，支撑高度为 0.1/0.3/0.5 m，使用 4 个 box。Artanim 以 4 cm
  阈值生成 554 个 merged candidate 和 205 个 primitive（196 box、9 cylinder），
  MuJoCo 编译为 206 个 geom（含 ground）成功；相比无约束的 190 个 primitive
  增加 15 个。两次 MJCF SHA-256 均为
  `004dcacbaf05bd6037af01661ec74030c8b34ecb1a279e8e7ccf42cd2566bf15`，NPZ 均为
  `8f98317f910113949cc72cb2a18b29020125c74031da79bef8b39d487d1abfef`；相关
  play/demo/collision 聚焦回归 `30 passed`，尚未完成人工 overlay 质量验收。
- [x] 2026-11-20 P8 center-only 多 primitive 原型：synthetic sphere、cylinder、
  capsule、带门洞平面及 source mapping/XML 回归通过。Studio 默认 auto 运行从
  16,400 个 occupied voxel、466 个 merged candidate 生成 190 个 primitive：
  183 box、7 cylinder、0 sphere、0 capsule，映射 196,189 个 Gaussian；7 个
  cylinder 相对 oriented-box 误差改善 33%--49%，MuJoCo 编译为 191 个 geom
  （含 ground）成功。两次 auto 输出的 MJCF SHA-256 均为
  `ebde35bf8f50a8a93f4d0b34072aaeba64662aab8e523826a4153230d9a786a0`，NPZ
  均为 `28e223819dfd113c6d076dc16fa6cf73c1976f8b3a46f2da0ecb1620e8e529f2`。
  结果证明类型选择、导出和字节确定性可运行，不代表 7 个 cylinder 的语义或
  碰撞质量已人工验收。
- [x] 2026-11-20 Studio 类型着色诊断资产生成成功：正常与 debug MJCF 均编译为
  191 个 geom，逐项比较 `geom_type/pos/quat/size/friction/contype/conaffinity/
  condim` 完全一致；source-ID NPZ SHA-256 均为
  `28e223819dfd113c6d076dc16fa6cf73c1976f8b3a46f2da0ecb1620e8e529f2`。该证据
  证明着色不改变物理，不替代 Native Viewer 人工语义验收。
- [x] 2026-11-20 P8 scene-independent center-only collision baseline 定向回归：
  `tests/test_3dgs_collision_generation.py` 与既有 apartment 合并回归共
  `13 passed`；Ruff check、定向 ty 和 Pyright 通过。真实 apartment PLY 经通用
  CLI 两次独立生成均得到 20,468 个 occupied voxel、89 个 merged candidate 和
  36 个 retained surface box；两次 MJCF SHA-256 均为
  `df1fe9cc451ae4a980f2605f0f5a0fc76a3886df6252ef229baacec08fd806f8`，两次
  source-ID NPZ SHA-256 均为
  `3ba252d650ce25c5cc295ad360a580a195e5573269cb96c66cc1f19fd48034aa`。
  该证据验证接线、既有资产复现和字节确定性，不代表第二场景或碰撞质量通过。
- [x] 2026-11-09 完成 P7 无定位安全速度跟踪 requirements 一致性检查；
  `git diff --check` 通过，`src/` 和 `tests/` 无本轮变更。离线运行：

  ```sh
  UV_CACHE_DIR=/tmp/mjlab-uv-cache \
    uv run --no-sync sphinx-build -j auto docs docs/_build
  ```

  Sphinx 构建成功。该证据只验证 requirements/changelog 格式与交叉口径，
  不表示 P7 MDP、策略或训练已经实现。
- [x] 2026-11-10 apartment 体积阈值进一步提高到 0.1 m³（替代下述初版）：
  从初版 68 个表面 box 中再删除 32 个，剩余 36 个；脚本同步调整
  `MIN_SURFACE_BOX_VOLUME`，恰好等于阈值的 box 保留。地板、两个柱体、
  保留 geom 的尺寸/位置/属性及默认透明度不变。MuJoCo CPU 编译
  39 个 geom 成功。专项测试 `4 passed`，Ruff 格式/检查及定向 ty 通过。
  体积过滤可能删除真实小表面，仍需在 viewer 中检查局部碰撞缺口。
- [x] 2026-11-10 apartment 小 box 过滤初版验证（后续阈值已提高）：
  `uv run --no-sync pytest -q tests/test_apartment_collision.py` 为 `4 passed`；
  覆盖碎片删除、阈值边界保留及原有合并回归。Ruff 格式/检查、定向 ty 和
  `git diff --check` 通过。当前 XML 删除 21 个小于 0.01 m³ 的表面 box，
  剩余 68 个；所有保留 geom 属性及默认透明度保持原样。MuJoCo CPU 编译
  71 个 geom 成功，出生点上方 (0, 0, 1) 六方向射线均命中场景边界。
  联合 manifest 测试为 `14 passed, 1 failed`，失败来自未修改的 Studio
  柱体 alpha=0.8 与旧测试要求 1.0 不一致；本次未调整该场景。
- [x] 2026-11-09 运行：

  ```sh
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=/tmp/mjlab-uv-cache \
    uv run --no-sync pytest -q \
    tests/test_apartment_collision.py tests/test_scene_manifest.py
  ```

  `14 passed`。覆盖 apartment box 合并保持原占据 voxel、保留一米门洞、
  空输入、manifest 无 SHA 契约、MJCF dependency 一致性，以及两个 group 1
  柱体同时相机可见并启用碰撞。
- [x] 2026-11-09 用户人工检查高分辨率 vision smoke 导出的 RGB/raw depth/
  policy depth，确认柱体与 3DGS 背景的基本关系正确；同时观察到地面和周边
  仍有 3DGS 孔洞。该结果支持诊断链路，不代表深度孔洞或 RGB/depth 阈值
  问题已经解决。
- [x] 2026-11-09 文档核对 MuJoCo Warp 当前 splat 深度语义并保留
  depth+validity 方案；RGB 合成与 depth 判定阈值解耦记为不阻塞 P6/P7 的
  后续优化，当前未修改 MuJoCo Warp。
- [x] 2026-10-31 核对 GaussGym 作者项目页及官方训练说明，更新任务定位、
  RGB/时序建模候选与分阶段计划；本项是规划调整，没有复现 GaussGym，
  未新增训练结果或关闭 P6/P7。来源见主调研报告第 6.4 节。
- [x] 2026-10-31 完成 P8 语义引导碰撞简化的一手资料调研，核对 room1
  PLY 字段与现有 manifest 边界。见
  [专题报告](mjlab-3dgs-p8-semantic-collision-research.md)及
  [BibTeX](mjlab-3dgs-p8-references.bib)。本项仅为文献/接口证据，
  未安装或复现外部方法，没有碰撞质量或加速实验结论。
- [x] 2026-10-31 独立 Depth-Validity 任务与单通道同 seed/预算短训练均通过，
  初始 depth 逐像素一致，mask 能区分缺失与有效远端；checkpoint 严格
  跨编码加载被拒绝。相关回归 `29 passed`，见
  `validation/p6-validity-2026-10-31.md`；不声称任务收益。
  最终 `make check` 与 `make docs` 均通过。
- [x] 2026-10-31 四环境共 480 个环境帧的 raw depth 统计及 PPO/恢复更新
  通过；合成双资产/三 world/双相机的深度解析值和状态/reset 隔离通过。
  定向相机回归 `25 passed`，见 `validation/p6-rollout-isolation-2026-10-31.md`。
  本轮 `make check` 与 `make docs` 均通过。
- [x] 2026-10-31 `run_g1_depth_diagnostics.py --pose-sweep` 完成七姿态三路
  分离渲染及重复一致性检查；原始 JSON/样例图已归档。`make check` 和
  `make docs` 通过，保留动态 rollout 与多资产组待办。
- [x] 2026-10-30 深度质量诊断及修复后双环境 PPO smoke 通过，见
  `validation/p6-depth-quality-2026-10-30.md`；深度语义、G1 配置和 camera
  回归 `24 passed`，最终 `make check` 通过。
- [x] 2026-10-30 P6 单环境默认 PLY 和双环境 room1 manifest smoke 均通过，
  命令、完整 agent 配置、loss、图像和 JSON 见
  `validation/p6-runtime-2026-10-30.md`。两组均启用仿真 CUDA graph；
  仅检查小规模链路，不声称训练收敛或视觉收益。
- [x] 2026-10-30 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_g1_3dgs_vision_config.py -q`：
  `4 passed`（含真实 depth runtime）。
- [x] 2026-10-30 P6 最终 `make check` 通过：Ruff、ty、Pyright 无错误。
- [x] 2026-10-30 `make docs` 构建成功，包含 P6 方法出处与 smoke 用法。
- [x] 2026-10-30 用户明确要求 P3 标记完成，并决定后续发布更新版本。
  依据已有最终 `make check`、13 项定向测试、真实窗口联调及用户确认的
  passive UI 修复，按当前范围收口；未执行的增强验证仍保留原状态。
- [x] 2026-10-30 用户反馈 native demo 交互模式下 F8 可用、左侧 3DGS
  Show/Hide 按钮置灰。源码检查确认是类别 `5` 禁止 passive 模式；
  随后用户确认 passive UI 已修复。修复 wheel 的发布、安装和按钮/F8
  联动验证尚未确认。
- [x] 2026-10-30 使用用户指定 run 的 `model_10000.pt`，在显式
  `--legacy-phase-observations` 下完成四轮真实原生窗口验收；命令、硬件、
  性能和兼容性限制见 `validation/p3-acceptance-2026-10-30.md`。
- [x] 2026-10-30 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_native_gaussian_contract.py tests/test_native_viewer_gaussians.py tests/test_native_viewer_actions.py tests/test_viser_update_policy.py -q`：
  `13 passed`；最终 `make check` 通过，Ruff、ty、Pyright 均无错误。
- [x] 2026-10-30 使用 Release wheel 和正常依赖同步运行
  `uv run --locked tests/run_g1_3dgs_native_demo.py --steps 200`，
  安装版本为 `3.16.0+3dgs`，renderer 就绪，200 步通过，退出码为 0。
  `uv lock --offline` 和 `uv run --offline ruff check tests/run_g1_3dgs_native_demo.py`
  均通过；`--locked` 仅要求锁文件保持最新，未跳过依赖同步。
- [x] 2026-10-29 原生 MuJoCo viewer 的 room1 G1 tracking 冒烟通过：
  `uv run --no-sync tests/run_g1_3dgs_native_demo.py --steps 200`。
  实际打开原生窗口，renderer 就绪并完成 200 步，正常退出。
- [x] 2026-10-29 定向回归：
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync pytest tests/test_native_viewer_gaussians.py tests/test_g1_3dgs_vision_config.py -q`，
  `7 passed`。禁用自动加载是为了避免系统 ROS pytest 插件缺少 `lark`
  导致收集失败。
- [x] 2026-10-29 新增脚本的 Ruff format/check、ty 和 Pyright 检查通过；
  均使用 `uv run --no-sync`。本轮未运行全仓 `make check` 或 `make test`，
  不作为 P3 整体关闭的证据。
- [x] 用户已人工确认真实 PLY 能在 Viser 中原生显示。
- [x] 用户已人工确认移除普通 Play 的 MJWarp camera 后卡顿改善。
- [x] 用户测得同一机器上稳态 Actual RT 约为：无 3DGS `0.6`，Viser 3DGS
  `0.53`。这是初步观察，不是正式 benchmark。
- [x] 2026-10-24 运行 7 个相关测试文件：`43 passed`。
- [x] 测试覆盖 importer、render context、Play/demo 配置、video、Viser、
  velocity 和 tracking 注册。
- [x] 2026-10-25 运行外部 MJCF terrain 与 Play 配置定向测试：
  `26 passed`；`uv run ty check` 和 `uv run pyright` 均通过。
- [x] 外部碰撞测试覆盖 primitive、多个相对路径凸 mesh、heightfield、
  plane fallback/显式保留、多 world、缺失资产、路径逃逸和动态模型拒绝。
- [x] 2026-10-25 运行 terrain/velocity/Tyro 回归测试：`32 passed`；
  `make docs` 构建成功。
- [x] 2026-10-25 `make check` 通过：Ruff、ty、Pyright 均无错误。
- [x] 2026-10-25 用户使用 `run_g1_3dgs_collision_demo.py` 完成 G1 在
  room1 3DGS 与外部碰撞场景中的连续运行人工验证。
- [x] 2026-10-25 运行 Viser collision、Play 配置和 MJCF terrain 联合测试：
  `33 passed`；Collision GUI 实时切换及 G3 状态同步已有自动测试。
- [x] 2026-10-25 运行 manifest、MJCF、Play/demo 和 Tyro 定向回归：
  `37 passed`；当时的 room1 真实 PLY/MJCF hash 校验、`make check` 与
  `make docs` 均通过。manifest 内容 hash 契约已于 2026-11-09 按项目决策
  移除，该历史结果不代表当前加载器仍执行 SHA-256 校验。
- [x] 2026-10-25 运行完整 P4 MJCF 定向回归：`23 passed`。覆盖定量
  落体、G1 contact sensor、terrain-height group 重定向及 velocity 环境
  初始化/单步；另行诊断确认无闭环策略的 G1 默认姿态无法静态站立。
- [x] 2026-10-25 使用 `model_29999.pt` 对 velocity 闭环策略执行 32 环境、
  20 秒零命令站立验证：零跌倒，平均水平速度约 `0.0024 m/s`，平均最终
  漂移约 `1.59 cm`；该实验使用普通 plane，room1 策略级接触指标延后。
- [x] 2026-10-25 运行 P5 manifest、Play、reset event 和 MJCF terrain
  定向回归：`75 passed`；`make check`、`make docs` 和 `git diff --check`
  均通过。
- [x] 2026-10-25 新增 P6 G1 depth 配置测试：3 个静态配置测试通过；
  单环境真实 room1 PLY runtime slow test 通过，观测 shape 为
  `(1, 1, 48, 64)`，dtype/range/有限性和非恒定 depth 均满足约束。
- [x] 2026-10-25 运行 P6、CameraSensor、RSL-RL runner 与 Play 配置联合
  回归：`46 passed, 1 skipped`；`make type` 和 `make docs` 均通过，
  `git diff --check` 无错误。
- [ ] 本轮尚未运行完整 `make test`。

本地受限环境运行测试需要将缓存放到可写目录，并关闭系统 ROS
pytest 插件自动加载：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
UV_CACHE_DIR=/tmp/mjlab-uv-cache \
WARP_CACHE_PATH=/tmp/mjlab-warp-cache \
uv run pytest <tests>
```

## 4. P4：外部碰撞输入 MVP（阶段性完成）

目标：不考虑如何简化 3DGS，先让外部准备好的碰撞几何成为
稳定的软件输入。

### C0. 输入契约

- [x] 明确 PLY 是视觉表示、MJCF 是物理入口，二者属于同一逻辑
  场景。
- [x] 明确一个 MJCF 可以引用多个 collision mesh、heightfield 和 primitive。
- [x] 定义 `visual/`、`collision/meshes/`、`collision/heightfields/` 目录布局。
- [x] 要求 MJCF 资产使用相对路径，禁止绝对路径和 `..` 路径逃逸。
- [x] 要求一个凸分解部件对应一个独立 mesh asset/geom。
- [x] 确定外部 MJCF 通过 `TerrainEntityCfg` 接入，不使用普通
  mocap entity。
- [x] 确定后续绑定文件使用项目自定义、版本化的 `manifest.json`；
  同时保留直接接收 PLY 和 MJCF 两个路径的临时实验接口。
- [x] 定义 collision-only MJCF 配置类型及公共 API。
- [x] 规范输入支持静态 primitive、多个凸 mesh 和 heightfield。
- [x] 明确 SDF 不在 MVP 支持列表内，不作为首版阻塞项。
- [x] 定义碰撞 friction、contact dimension、priority、`contype` 和
  `conaffinity` 的覆盖规则。
- [x] 定义视觉 PLY 与碰撞 MJCF 的单位、坐标轴和 transform 契约。

### C1. 场景加载

- [x] 在 `Scene.compile()` 前把 collision-only MJCF 作为静态 terrain
  加入场景。
- [x] 保证碰撞资产不创建 joint、mocap body、actuator、sensor 或额外机器人。
- [x] 实现未提供外部碰撞时的 plane fallback。
- [x] 实现提供外部碰撞时默认关闭隐藏 plane，避免双重接触。
- [x] 允许显式保留 plane，仅用于调试或过渡。
- [x] 所有并行 world 首版共享同一静态碰撞场景。

### C2. CLI 与错误处理

- [x] 增加 `--collision-mjcf-file` 和训练 terrain 配置入口。
- [x] 增加显式 keep-plane 参数；不传外部碰撞时沿用 plane fallback。
- [x] 校验文件存在、引用资产存在和 MJCF 可以编译。
- [x] 校验至少存在一个启用碰撞的 geom。
- [x] 拒绝 joint、mocap body、actuator 和 sensor。
- [ ] 对单个任意凹 mesh 给出凸碰撞语义告警或拒绝。
- [ ] 对 unsupported geom、错误单位和明显包围盒不匹配给出上下文
  错误。

以上两项属于输入诊断加固，保留为后续质量任务，不阻塞 P4 MVP。

### C3. 调试与测试

- [x] 外部碰撞统一进入 group 3；Play 可通过
  `--collision-visible True` 设置初始状态，并可在 Viser 的
  `Visualization > Collision` 中实时显示/隐藏。
- [ ] 增加 PLY 与碰撞包围盒、坐标轴和出生点叠加检查。
- [x] 增加 primitive 碰撞加载测试。
- [x] 增加多个凸 mesh 碰撞加载测试。
- [x] 增加 heightfield 碰撞加载测试。
- [x] 增加 plane fallback、plane 互斥和非法 MJCF 测试。
- [x] 增加多 world 共享碰撞测试。
- [x] 增加 room1 固定伪随机柱子碰撞场景和标准 G1 demo policy
  人工 Play 脚本。
- [x] 使用标准 demo policy 完成 G1 在 room1 外部碰撞上的连续运行冒烟。
- [x] 增加独立落箱测试：2 秒后中心高度位于 95--105 mm，末段高度
  标准差小于 0.5 mm，最大允许穿透 10 mm，末速度小于 0.01 m/s，
  overflow 为 0。
- [x] 验证 G1 velocity 完整环境可在 room1 外部碰撞上初始化并单步，
  脚部 contact sensor 能识别标准 `terrain` 根，terrain-height sensor
  自动读取 collision group 3。
- [ ] 使用闭环站立策略增加 G1 静态站立冒烟和定量测试。开环默认
  关节位置目标实验在约 3 秒内跌倒，骨盆最低约 0.133 m，不能作为
  站立验收。
- [x] 对独立落体记录穿透、接触抖动、末速度和 solver overflow。
- [ ] 对带策略的 G1 记录穿透深度、接触抖动、solver overflow 和
  持续运行时长。
- [x] 更新 README、3DGS 指南和 changelog。

PLY/碰撞叠加诊断和 room1 策略级定量指标均保留为后续加固。P4 当前
验收边界是“能够稳定接收预先准备的 collision-only MJCF 并用于训练、
Play 和 demo”；从 PLY 自动生成该 MJCF/mesh 明确属于 P8。

## 5. 统一场景资产与缓存

- [x] 定义版本化 scene manifest schema。
- [x] manifest 绑定 visual PLY、collision MJCF、统一 transform 和
  出生点。
- [x] manifest 记录 visual/collision 路径、资产类型、单位约定和 schema
  版本；2026-11-09 按项目决策移除主资产及 dependency SHA-256 字段和
  运行时内容校验，允许本地重建资产后直接验证。
- [x] 自动核对 MJCF 全部外部依赖都已出现在 manifest dependencies 中，
  并拒绝未引用的多余声明。
- [x] 自动碰撞生成的转换参数和工具版本归入 P8 provenance，不作为
  P5 阻塞项。
- [x] NPZ 原子写入和缓存失效转入 P10 后期性能优化，不作为 P5
  阻塞项。
- [x] 定义标准资产包；`splats.npz` 和 `collision_report.json` 为后期
  可选产物：

```text
scene/
  manifest.json
  visual/
    splats.ply
    splats.npz             # 后续缓存
  collision/
    collision.xml
    meshes/
    heightfields/
  reports/
    collision_report.json
```

- [x] 支持统一 asset-to-world 刚体变换。
- [x] 增加 manifest schema、路径边界、dependency、transform 和不匹配测试；
  内容指纹不再属于 manifest v1 加载契约。
- [x] train/play/demo 增加 `--scene-manifest`，同时保留分开的 PLY/MJCF
  接口；两种输入模式互斥。
- [x] 消除 G1 3DGS 资产的开发机绝对默认路径，改为仓库资产、显式参数、
  环境变量或 manifest。
- [x] 将 manifest spawn point 经统一 transform 后应用到机器人
  reset/origin；默认选择 `default`，并支持 `--scene-spawn-point`。

## 6. P6：YAM Vision → G1 3DGS 视觉训练适配

P6 不从零实现视觉训练框架。仓库中的 YAM Vision 已经提供以下成熟链路：

```text
CameraSensor
  → (B, C, H, W) RGB/depth observation
  → 独立 camera observation group
  → SpatialSoftmaxCNNModel
  → RSL-RL actor/critic
```

P6 把该链路迁移到 G1 velocity task，并保持普通 G1 任务完全不创建
相机或渲染开销。它还需为 P7 提供能区分未知/远端、表达障碍接近速度
并可注入传感延迟和噪声的视觉输入。首个注册任务为
`Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth`。

算法来源与论文写作依据见 `mjlab-3dgs-p6-method.md`，BibTeX 见
`mjlab-3dgs-p6-references.bib`：PPO（Schulman et al., 2017）、GAE
（Schulman et al., 2015）、Spatial Softmax 编码结构（Levine et al.,
JMLR 2016）及 3DGS 表示（Kerbl et al., 2023）。本项目复用仓库 YAM
Vision/RSL-RL 工程链路，不使用 guided policy search，不将这些算法
作为原创贡献。运行证据见 `validation/p6-runtime-2026-10-30.md`。

### V0. YAM Vision 复用与静态配置

- [x] 保留单通道 `3DGS-Depth` 基线，新增独立 `3DGS-Depth-Validity`
  task/experiment，camera 按 channel 拼接为 `(B,2,48,64)`；共享同一
  原始 camera 缓存，无额外渲染，PPO、奖励和本体 observation 保持一致。
- [x] 新建真正加载 3DGS 的 training env 配置，不再在 `play=False` 时
  直接退回普通 flat 任务。
- [x] 增加安装在 G1 torso/head 区域、向前下方俯视 `20°` 的训练
  CameraSensor。
- [x] 首版使用 `64×48` depth；RGB-D 和纯 RGB 后续扩展。
- [x] 复用 YAM Vision 的 `(B, 1, H, W)` 形式，将标准化 depth 接入
  独立 `camera` ObservationManager group。
- [x] 复用 YAM Vision 的 Spatial-Softmax CNN，并让 actor/critic 同时读取
  本体 observation 和 `camera` group。
- [x] 保持 action space、command 和 velocity locomotion 控制接口不变；
  保留其余奖励，仅移除依赖共享 BVH `foot_height_scan` 的
  `foot_clearance` 与 `foot_swing_height`。
- [x] 首版相机每个控制 step 更新、无历史帧，将 `0.05–5.0 m` 裁剪并
  归一化至 `[0, 1]`。
- [x] 普通 proprioception G1 配置不创建 CameraSensor、3DGS 或 renderer
  workload，并增加配置回归测试。
- [x] 移除视觉任务中与相机共享 MJWarp BVH、会把隐藏 plane 合入渲染
  上下文的 `foot_height_scan` 及其两个奖励依赖；critic 继续使用 contact
  特权观测。

### V1. 运行时正确性

- [x] 编译单环境场景，确认相机安装位置、约 20° 下俯视线与 robot
  self-occlusion；固定姿态、七姿态扫描和真实 rollout 证据见下列报告。
- [x] 完成固定 reset 姿态诊断：实际下俯角约 20 度，robot-only 无命中，
  combined 与 Gaussian-only raw depth 一致；固定姿态重复结果相同，
  平移 0.5m 后约 90.5% 像素改变。动态姿态自遮挡仍待验收。
- [x] 2026-10-31 七种可控姿态扫描通过，含 yaw、pitch 和单臂抬起。
  抬臂时机器人遮挡约 19.14% 像素，最近深度约 0.133m；不能作为
  缺失噪声过滤。缺失率约 14.52%–28.48%，仍需真实 rollout 覆盖率统计。
  见 `validation/p6-pose-sweep-2026-10-31.md`；不是动力学轨迹验收。
- [x] 解析 GPU 测试确认光轴 Z-depth、零值缺失、Gaussian opacity 阈值
  深度语义。修复 G1 将缺失零值映射为近障碍的问题，显式启用
  `zero_depth_as_far=True`；其他任务默认行为不变。
- [x] 评估动态缺失深度覆盖率与 validity mask；四环境 480 个早期 rollout
  帧平均缺失约 21.96%，单帧最高 62.01%。映射到 far 不代表自由空间，
  因此保留独立 validity 通道和对齐 dropout。
- [x] validity 接入与语义验证完成：raw finite-positive 为 1，零值为 0；
  有效超量程深度仍为 1。单通道与双通道各完成四次 24-step PPO 更新、
  重载后一次更新；仅翻转 mask 会改变动作输出，不代表已学会有效利用。
  实际任务收益留待多种子实验，不关闭上方综合评估项。
- [x] 真实四环境早期 PPO rollout 采样 480 个环境帧，平均缺失约 21.96%、
  远端约 26.04%、量程内约 52.00%；单帧缺失最高 62.01%。包含 auto-reset
  后图像，不声称稳定步态分布或动态自遮挡归因。validity 的接入验证已完成，
  有效性收益仍待任务实验。
- [x] 验证默认 room1 PLY 能进入单环境 depth observation。
- [x] 验证通过 `--scene-manifest` 加载的 room1 PLY 能进入 depth observation；
  双环境运行、形状与有限数值检查通过。不同替代资产另行验证。
- [x] 验证单环境 depth 的 shape 为 `(1, 1, 48, 64)`、dtype 为
  `float32`、范围为 `[0, 1]`，且图像不是常数。
- [x] 单/双环境 rollout 中记录连续 depth 帧与 PNG/NPY，逐帧差异非零；
  仅证明帧在更新，不据此证明深度语义、遮挡或几何精度正确。
- [x] 运行 RSL-RL CNN 前向、两次 24-step rollout/PPO 更新，loss/gradient
  有限且 actor CNN 权重改变；actor/critic/optimizer 保存重载通过，并
  继续一次 24-step rollout/PPO 更新。相机置零会改变输出，仅验证连通性。
- [x] 新增 `tests/run_g1_3dgs_vision_smoke.py` 的逐帧差异检查，冻结图像时报错。
- [x] 视觉 smoke 支持可配置宽高（诊断默认 `320×240`）和固定基座姿态，
  保存对齐的 RGB、raw depth、policy depth 及 validity PNG/NumPy；训练任务
  默认分辨率仍为 `64×48`，未由诊断默认值改变。

### V2. 性能与并行训练

- [x] 验证多 world splat group、camera tensor 和 reset 隔离。
- [x] 两环境共享 room1 资产的 camera tensor 与局部 reset 隔离通过；
  reset env 0 后 env 1 的 qpos 和 depth 完全不变；四环境 smoke 也通过。
- [x] 合成双资产/三 world/双相机验证分组 `(0,1,0)` 的解析深度、单 world
  移动与 scene/sensor reset 隔离；两个 world 共享同一资产也不串扰。
  不同真实资产的完整任务级测试仍待补，不关闭上方整体验收项。
- [x] 验证视觉 observation 与 CUDA graph 的兼容性。
- [x] 当前机器 1/2/4 环境的实际仿真 CUDA graph 启用，视觉 rollout 和 PPO
  更新通过；尚未关闭多配置兼容性验收项。
- [x] 完成 P6 当前 `64×48`、1/4-world、单帧/robust 时序的短吞吐、actor
  推理和 Torch peak 对照；更新频率已覆盖每步与两步 sample-and-hold。
  不同 splat 数、更高分辨率和总 GPU 显存基准转入 P10，见
  `validation/p6-temporal-robustness-2026-11-07.md`。
- [x] 首版采用 YAM Vision 的独立 Spatial-Softmax CNN encoder，不把 depth
  展平后直接送入 MLP。
- [x] P6 输入错误由现有 PLY/schema/manifest 校验和 camera observation 的
  明确类型、缺失 depth、NaN/Inf 错误处理覆盖；显存不足继续保留后端原始
  异常，不吞掉或伪装为有效观测。
- [ ] 安装后端在空 mesh BVH / splat-only 配置下仍可能原生退出（139）；
  诊断工具使用离屏无碰撞占位几何。该外部后端问题转入 P10，不阻塞带
  robot/collision BVH 的 P6 训练任务收口。

### V3. P7-ready 时序感知与鲁棒性

- [x] 新增独立四帧 depth+validity 基线任务，直接复用主线
  `ObservationManager` history 与 partial-reset backfill；camera 保持
  `(B,4,2,48,64)`，仅在现有 Spatial-Softmax CNN 边界合并为 8 通道。
  配置、模型前向和真实 renderer reset/step 测试通过；单帧任务未改动。
- [x] 增加可配置的 depth+validity 帧数与 stride；注册 robust 基线保存 7 帧、
  以 stride 2 选择 4 帧 `(t-6,t-4,t-2,t)`，同时保留所有单帧/连续帧基线。
- [x] 增加固定一控制步延迟、两控制步 sample-and-hold、量程噪声、模糊、
  对齐 depth+validity 的结构化缺失/局部遮挡与整帧 dropout；参数可独立置零
  做消融，真实深度诊断仍单独保留。
- [ ] 在 P7 固定任务/预算下多种子比较单帧 depth、depth+validity 与
  时序 depth+validity，报告速度跟踪、制动、搜索、恢复、碰撞和稳定性并做
  mask-only 消融。
- [x] 注册 actor-only camera / asymmetric MLP critic 对照，真实 PPO update、
  checkpoint 重载与恢复后 update 通过；效果比较留给 P7 多种子实验。
- [ ] 当前 main/RSL-RL 没有可复用的 privileged teacher → depth student
  distillation 接口；避免在 P6 自建并行训练框架，转入 P11 独立研究项。
- [x] 测量单帧/时序 encoder 在 1/4 worlds、每步/两步更新下的 SPS、Torch
  显存和 actor 推理延迟；更高分辨率/规模转入 P10，RGB 作为后续对照。

P6 的完成边界是可训练、可恢复、可诊断且可消融的 P7-ready 视觉输入链路，
不是安全速度跟踪效果结论。上面两项未勾选内容分别依赖 P7 MDP 和新的蒸馏研究，
不属于 P6 工程接口的完成条件。

## 7. P7：3DGS 安全感知速度跟踪

P7 的首要目标是：G1 根据 nominal velocity command、本体状态和时序深度
在 3DGS 场景中安全行走。前方安全时跟踪速度；发现障碍时减速或停车，
必要时原地旋转搜索无碰撞方向，然后朝安全方向恢复速度跟踪。任务不定义为
导航，不要求世界坐标目标，也不假设 LiDAR、VIO、SLAM 或其他定位传感器。
方法参考 Hiking in the Wild 的 G1 深度感知关节控制和位置到速度命令设计，
以及 PHP 的 privileged teacher → depth student 路线，但不声称复现其算法。
当前主线不再让视觉或真值几何直接重训29维低层 Actor，而是训练安全速度学生
并冻结已验证的 locomotion Actor。

P7 的详细任务、接口、指标和完成条件以
`mjlab-3dgs-safe-velocity.md` 为准。本节只跟踪项目进度，不能替代该
需求契约。

### N0. 首个安全速度跟踪任务

- [x] 形成独立 P7 requirements，明确上游复用边界、scenario catalog、
  observation/action、奖励/终止、safety shield、基线和验收阶段。
- [x] 选择 apartment 作为首个平坦支撑面、墙体和少量障碍的重建场景；
  PLY/collision 共用坐标系，两个柱体同时具备不透明视觉和碰撞几何。
- [x] 明确 actor 不读取世界位置、目标位置、地图或碰撞真值；世界位姿仅可
  用于 evaluator，不能作为可部署传感输入。
- [x] 2026-11-09 完成相关项目对比：ABS 作为任务/安全机制首要参考，Hiking
  in the Wild 作为 G1 深度关节控制参考，Omni-Perception 作为安全速度跟踪
  参考，GaussGym 作为 3DGS 仿真参考；没有项目可直接覆盖完整 P7。
- [x] 冻结 free、left-open、right-open、both-open、dead-end 和 false-alarm
  catalog、manifest 出生位姿、命令初值和数据划分；六类场景已于
  2026-11-10 完成人工视觉/碰撞/安全出口确认。
- [x] 复用现有 `UniformVelocityCommand` 的直接 body-frame twist，首版关闭
  `heading_command`；前方安全时跟踪命令，危险时允许暂时偏离并优先制动、
  搜索和选取安全方向。
- [ ] 在前方受阻时学习减速/停车、近似原地旋转扫描、选择安全侧并恢复
  平移速度；没有安全出口时保持稳定站立。
- [ ] 使用 P6 的时序 depth+validity 和本体历史，显式评估传感延迟、深度缺失
  及机器人制动距离；单帧 depth 保留为对照，不作为安全上界。
- [x] 注册 `Mjlab-Velocity-Safe-Unitree-G1-3DGS` 独立任务，沿用现有关节
  目标动作和 P6 时序 depth+validity，不新增定位、PointGoal 或高层导航策略；
  1/2 environment reset/step 已通过。
- [ ] 增加安全区速度跟踪、提前减速、安全停车、方向搜索、恢复跟踪及
  碰撞代价，防止永久停车、无限旋转、左右振荡和反复启停等奖励漏洞。
- [ ] 在学习策略外设计可单独开关的 safety shield：按速度、系统延迟、制动
  能力和余量计算动态安全距离或 TTC，越界时限制前进或强制稳定站立。
  分别报告 shield 开/关结果，不把硬安全层的成功归因于策略学习。
- [ ] 在同预算下比较 blind、deterministic depth adapter、privileged、
  mesh vision 与 3DGS vision；规则 adapter 使用“制动—旋转—最大安全扇区—
  恢复”的可解释基线。
- [x] 实现规则 adapter 的任务局部纯逻辑与九类 outcome 判定：支持制动、
  停车、固定方向搜索、恢复、dead-end timeout、无效深度保守处理、关闭时
  passthrough 和介入计数。任务局部轨迹 evaluator 已接入世界边界、跌倒和
  `pelvis` base-link 对所有非地面场景 geom 的障碍接触，并以脚本轨迹确定性
  触发全部九类 outcome。
- [x] 注册 `Mjlab-Velocity-Safe-Adapter-Unitree-G1-3DGS`，在任务局部
  observation term 中读取 Warp camera depth 并向标准 blind G1 MLP 注入安全
  twist；camera 不进入 actor observation，真实 reset/step 接入测试通过。规则
  基线已用既有 checkpoint 完成六场景 × seed 0/1/2 的正式规则验收：18/18
  outcome 符合 catalog，均无 base-link 障碍接触和跌倒，最小前向传感净空
  0.792 m。该指标不是精确 geometry 净空，学习策略效果仍归 P7-3/P7-4。
- [x] 保留全部 Blind、端到端 Privileged、stop-only、Stop-Gate 配置与实验
  记录作为论文证据；失败方案只标记为 failed/superseded/diagnostic，不覆盖
  task id、run name 或 checkpoint 元数据。删除本地大型产物前需用户确认。
- [x] 完成 P7-4A：新增 task-local 连续 privileged safety teacher，以前向
  collision truth 输出 clearance/validity/alpha/stop 标签和 safe command；标签
  位于独立 observation group，不进入 actor/critic，旧硬 Stop-Gate 保持不变。
  第三版 stop `0.4 m`、slow `1.4 m` 已由用户 GUI 确认无碰撞稳定停车；首版低速
  死区和第二版碰撞重置结果保留为失败证据。
- [ ] 接入 P6 时序 depth+validity Student，先进行加权监督学习；若 rollout
  发生状态分布偏移再增加 DAgger，不在该阶段微调 locomotion Actor。已新增
  `Mjlab-Velocity-Safe-Vision-Data-Unitree-G1-3DGS`，把 `(B, 7, 2, 48, 64)` robust
  3DGS observation 与四维 collision-truth label 配对，且二者均不进入 frozen
  locomotion actor/critic；chunked collector 已通过 2-sample CPU smoke，保存
  camera/label/command/measured/reset 并生成带 checkpoint hash 的 manifest。
  多环境重建场景已修正为在 MuJoCo Warp 隔离 world 中使用 scene-local 零 origin，
  不再把机器人平移出固定 collision/3DGS；4 world × 30 step CPU collector runtime
  已验证零 termination、pelvis 稳定、ray clearance 一致且 camera corruption 独立。
  4 world × 200 step GPU teacher rollout 也已生成 800 samples，零 termination，
  并逐 world 精确复现单环境高度、clearance、alpha 和最终停车速度；并行 origin
  修复验收通过。4 world nominal rollout 进一步生成 596 samples 和 44 个可达
  stop positives，零 termination；collector 可按 stop-positive 数量在下一动作前
  自动结束。`dead_end` 也在 127/500 steps 自动生成 508 samples/44 positives，
  零 termination，并覆盖 alpha 0 的 command-zero 区域。`left_open` 在 108/500
  steps 自动生成 432 samples/44 positives，零 termination，且四个 world 各得到
  11 个 positive；`right_open` 使用相同配置得到完全相同的汇总和逐 world 标签
  范围，左右对称性验收通过。`free` teacher rollout 已生成 800 个 alpha 恒为
  1.0 的纯负样本，四个 world 各 200 个连续样本且零 termination，可用于训练时
  约束自由空间误减速。属于 test split 的 `false_alarm` 也已生成 800 个 alpha
  恒为 1.0 的纯负样本，零 termination，只用于误停车评估。task-local Student、
  split-enforcing chunk loader、非对称 alpha/加权 stop 损失和独立训练入口已实现；
  真实 split 的 1-epoch CPU smoke 已完成反向传播、best/last checkpoint 和一次性
  test evaluation。首个 50-epoch `offline_seed0_v1` 在 held-out test 上得到 alpha
  MAE 0.20688、误停率 0.1970、漏停率 0.9091，已标记为失败并保留。逐场景诊断
  确认 `false_alarm` 零误停，错误集中在未见墙面布局 `dead_end`；global average
  pooling 丢失左右/前方位置是主要结构问题。v2 已复用现有 SpatialSoftmax 保留
  空间位置，并由 alpha 连续导出 stop，消除两个输出不一致；真实 split 的
  1-epoch smoke 及逐场景报告通过。v2 的 50-epoch GPU run 在 `false_alarm` 保持
  零误停，但 `dead_end` alpha MAE 0.53730、误停率 0.9978，仍未通过。几何审计
  确认 dead-end spawn 距前墙 1.4 m，truth ray、camera depth、pelvis 高度和接触
  状态均正常；失败来自固定柱体训练轨迹无法泛化至整墙，而非出生点穿墙。该结论
  促使主线改用显式前向深度几何阈值；墙面训练布局不再是首版部署的必要条件，
  Student 只保留为可选学习对照。旧共享 RenderContext 把
  camera `[1,2]`
  与 ray `[1,3]` 合并为 `[1,2,3]`，所以首版数据只标记为 mesh/collision-assisted
  + 3DGS 对照；2026-11-17 新采集路径在任务侧隔离教师射线，见下文。
- [ ] 通过关闭、冻结或错配视觉，以及改变障碍布局的测试，验证视觉依赖；
  不因机器人能在平地跟踪速度就宣称策略学会了视觉避障。
- [ ] 报告 free-space 跟踪误差、碰撞/跌倒率、制动距离、停止净空、左右
  安全侧选择率、搜索/恢复成功率、dead-end 停车率、不必要停车/旋转率及
  端到端 SPS；不报告目标到达率、路径效率或 SPL。
- [ ] P8 资产通过几何/接触检查后，固定视觉和评估协议进行碰撞替换消融。
- [x] Depth-Stop 配置支持在环境创建前传入 scene manifest、spawn point 和固定
  nominal command，并从所选 collision MJCF 动态生成除地面外的 base-link
  障碍接触传感器。第二地图 `ArtanimMotionCaptureStudio` 配置回归确认加载其
  PLY/MJCF/default spawn 和六根柱体，且不残留 apartment geom 名称；单环境真实
  reset/step 也通过。该资产的视觉—碰撞未对齐，真实跨地图安全行为仍待新资产验证。
- [x] 新增 `scripts/tools/play_safe_velocity_map.py`，在环境创建前用 manifest、spawn
  和 nominal command 构建 Depth-Stop 配置，再复用标准 `run_play` 和 viewer；
  避免通用 `play --scene-manifest` 的构建后覆盖留下旧地图接触传感器。配置通过
  进程级临时 task 注册，`run_play` 及其它共享播放代码不变。播放/配置回归
  `42 passed`，脚本帮助、ruff、ty、pyright 通过。
- [x] P7-5 runner 支持自定义 manifest、spawn、nominal command、allowed region
  和显式 expected outcome。Studio default 的 seed 0/1/2 GPU 矩阵 3/3 达到
  `safe_blocked_stop`，最小深度净空 `0.864--0.905 m`，均零碰撞、跌倒和
  termination；seed 2 路径达到 `2.703 m`，仍需检查漂移。该结果不关闭
  视觉—碰撞对齐验收。
- [x] Studio manifest 增加 `safe_velocity_pillar_00--05` 六个同点异向验收 spawn；
  它们均从 `(0, 0, 0)` 出生并分别正对 collision MJCF 中的六根柱体，原
  `default` 不变。几何回归验证 yaw 误差不超过 `1e-6 rad`。同一 checkpoint、
  seed 0 的 GPU 矩阵 6/6 达到 `safe_blocked_stop`，最小深度净空
  `0.856--0.884 m`，均零碰撞、跌倒和 termination。该结果关闭多朝向行为验收，
  但不关闭 collision overlay 和视觉—碰撞严格对齐检查。
- [x] 在不删除同点组的前提下，增加
  `safe_velocity_generalization_pillar_00--05` 六个异位置、异朝向 spawn，距各自
  目标柱体 `3.35--4.21 m`。几何回归验证位置唯一、非原点、目标距离和朝向。
  同一 checkpoint、seed 0 的 GPU 矩阵 6/6 达到 `safe_blocked_stop`，路径长
  `2.288--3.183 m`、最小深度净空 `0.847--0.890 m`，均零碰撞、跌倒和
  termination。后续仍需多 seed 和新地图验证，不能把本结果扩大解释为完整泛化。
- [x] runner 新增 `--scene-spawn-prefix`，按 manifest 顺序批量选择 spawn 并与 seed
  形成验收矩阵；原单点参数不变。矩阵 summary 汇总碰撞、跌倒、termination、
  净空、路径和时长范围。六个泛化 spawn × seed 0/1/2 的 GPU 矩阵 18/18 达到
  `safe_blocked_stop`，均零碰撞、跌倒和 termination；最小净空 `0.709 m`，路径
  `2.286--3.296 m`，接受时间 `6.78--9.14 s`。后续工作转向跨地图和
  视觉—碰撞对齐验证。
- [x] 新增 Gaussian-only 与 collision-only 同相机位姿深度对齐 runner，并对
  Studio 六个泛化 spawn 生成定量报告和放大 overlay。默认 `0.15 m` 容差下结果
  0/6：Gaussian 覆盖率 `0.557--0.841`、对齐比例全部为 0、中位绝对误差
  `1.015--1.561 m`。这些柱体是额外可视 mesh，不要求在原始 3DGS 内对应；
  0/6 仅保留为历史诊断，撤销用它否定 18/18 可视柱体停车结果的解释。
- [x] 相机输入契约明确为 3DGS + 可视 mesh；group 3 只用于物理与教师。
  Vision-Data 默认采用任务侧直接碰撞射线，排除共享渲染组污染；旧采集路径可用
  `--legacy-shared-render-context` 显式复现。manifest 记录视觉来源及隔离状态。
  alignment runner 默认 group 3，仅作为离线参考；group 1 可选用于历史诊断。
  CPU 双 world G1 采集 4 samples、零 termination，新旧教师标签完全一致，
  相机张量 79,520 个值不同；合成动态隐藏障碍验证标签刷新与视觉隔离。
  GPU 吞吐和正式采集待验证，证据见 visual-isolation 验证记录。
- [x] Depth-Stop 主线验收“可见深度调速、pelvis 接触事件判碰撞”。apartment
  和 Studio 配置均不挂载教师射线、相机仅 group 1/2、障碍接触传感器覆盖
  一个控制步历史。双 world 隐藏墙体回归覆盖接触/无接触隔离、分离后接触清除
  与短暂撞击历史保留；整个过程相机深度不受隐藏墙体影响。证据归入
  `validation/safe-velocity-visual-isolation-2026-11-17.md`。
- [x] 完成真实 G1 checkpoint 的 GPU 可见深度/接触矩阵，实际 camera BVH 仅
  group 1/2、无共享教师 ray。修正停车计时为连续静止 1 s，保留旧判据首轮和
  新判据复验。严格复验 apartment 15/18、Studio 18/18，障碍停车 30/30，
  全部零 pelvis 接触/跌倒/termination。净空最小为 0.6786/0.7131 m，停车末尾
  连续静止 1.00--1.02 s。完整 trace 和说明位于
  `validation/safe-velocity-depth-contact-2026-11-17/`。
- [x] 2026-11-17 按用户决定，将本轮规则基线及 36 组场景验证标记完成并收口，
  不再追加场景、复测矩阵或为追求全通过继续调参。结果固定为 33/36，障碍
  停车 30/30、零介入跟踪 3/6，全部零 pelvis 接触/跌倒/termination。
  free/seed 1、2 和 false_alarm/seed 1 的轻微减速作为已知限制归档，未视为
  已修复，也不将其重新标记通过。全部 trace、失败记录和旧实验继续保留。

历史扩展设想（本轮不执行）：侧方障碍/通行区域敏感性、其它失败项逐像素归因、
Studio 重建障碍 collision 对齐与新增地图验收。它们不再是本项完成的前置条件；
后续如需重新开展场景测试，需用户明确提出。

### 7.1 后续扩展：单级台阶、楼梯与地形行走

以下保留原单台阶方案作为后续课程，不再作为首个闭环验收前置条件。
先使用手工对齐资产；视觉训练使用 P6 链路，自动碰撞生成仍可独立推进。

### S0. 手工资产基线

- [ ] 选择或制作一个带单级台阶的最小 3DGS 视觉资产。
- [ ] 为该台阶制作严格对齐的 box 或 collision-only MJCF。
- [ ] MVP 固定台阶高度，只随机机器人出生位置和朝向。
- [ ] 验证视觉表面、碰撞表面、机器人出生点和相机坐标对齐。

### S1. 环境与 MDP

- [ ] 基于 G1 velocity task 创建独立的 perceptive single-step task id。
- [ ] 随机出生距离，初始建议范围为 `0.6–1.5 m`。
- [ ] 随机偏航，初始建议范围为 `±10°`。
- [ ] 随机前进速度指令，避免固定时间抬脚策略。
- [ ] 不向非特权策略提供台阶距离、高度或真值 collision geometry。
- [ ] 复用 velocity tracking、姿态、能耗和平滑奖励。
- [ ] 增加平台到达、稳定接触、足端净空和成功奖励。
- [ ] 增加脚尖撞击、非足端碰撞、冲量、滑动和穿透惩罚。
- [ ] 增加摔倒、停滞、超时和越界 termination。
- [ ] 定义 base 越过边缘、双脚稳定接触并保持站立的成功条件。

### S2. 训练与对照

- [ ] 训练 proprioception-only blind policy。
- [ ] 训练读取真值台阶参数/高度图的 privileged upper bound。
- [ ] 训练普通 mesh depth 视觉策略。
- [ ] 训练 3DGS depth 视觉策略。
- [ ] 在相同随机种子、训练预算和测试集下比较四组策略。
- [ ] 执行冻结相机、关闭相机和遮挡相机实验，证明策略真实使用
  视觉。
- [ ] 导出 checkpoint，并由独立 Play 模块复现评估 episode。

### S3. 指标与验收

- [ ] 记录成功率、摔倒率和任务回报。
- [ ] 记录脚尖/非足端碰撞次数和最大穿透。
- [ ] 记录接触冲量、足端净空和速度跟踪误差。
- [ ] 记录训练吞吐、显存和推理 real-time factor。
- [ ] 对碰撞台阶施加 `±1 cm`、`±2 cm`、`±5 cm` 高度/位置偏差。
- [ ] 绘制视觉—碰撞误差对落脚、碰撞、穿透和成功率的影响
  曲线。
- [ ] 只有 3DGS 视觉策略相对 blind policy 有可复现增益时，
  才标记视觉闭环完成。

### S4. 课程扩展

- [ ] 为平地和不同台阶高度分别制作严格配对的视觉/碰撞资产。
- [ ] 扩展到 `8 cm`、`12 cm`、`16 cm` 单台阶课程。
- [ ] 扩展到上台阶、平地和下台阶混合。
- [ ] 扩展到连续两到三级楼梯。
- [ ] 扩展到窄平台、间隙或 sparse footholds。
- [ ] 最终使用真实重建场景和自动生成碰撞代理复现实验。

## 8. 自动碰撞代理生成

该阶段在外部碰撞输入接口稳定后开展。它只负责输出符合既定
契约的资产包，不应修改训练、Play 或物理场景加载接口。
P4/P5 契约已经建立，P8 可与 P7 并行，不依赖视觉训练或台阶策略收敛。
首个交付物是可复现的碰撞资产生成与导出基线，后续接入 P7 作替换对照。
当前已完成 apartment 专用原型：脚本对 Gaussian 中心进行 opacity/bounds
过滤、体素化、连通分量清理和递归 box 合并，并导出 collision-only MJCF。
其地面范围、阈值、可见柱体和门洞回归均针对当前 apartment 调整，尚不能
视为任意 3DGS 场景的通用生成器，也不具备语义部件识别能力。
专题结论见
[语义简化调研](mjlab-3dgs-p8-semantic-collision-research.md)：先做有限平面/
box 几何基线，再接可替换语义前端，残差使用有限数量的凸体或局部 mesh。

P8 MVP 于 2026-11-21 完成以下范围：场景无关且确定性的
PLY→voxel→box/简单 primitive 生成、source-ID/provenance、显式 known-free 与
语义 sidecar 契约、有限平面/free-side 约束、synthetic 五类结构真值、第二真实
场景生成与 MuJoCo 编译、表面距离统计及独立 CPU collision benchmark。未包含
模型推理的“语义 sidecar”不等于自动语义识别。后续已完成内部
TSDF/marching-tetrahedra、mesh/heightfield 混合和固定预算选择/CPU 消融；
CUDA policy 与 native-3DGS 任务联合评价后续已完成。
2026-11-24 用户明确不做手工类别标注、算法不能判断时保留 unknown；按此最新
口径，实际特征/语义解码、unknown/几何回退及固定预算 CPU/CUDA/native 对照
已完成工程验收。独立真值准确率改为可选研究验证，未测即未测，不用 synthetic
人工标签替代，也不以收口声明暗示语义有效、接触安全或跨场景质量已获保证。

### G0. 算法基线

- [x] 完成语义前端、平面/primitive 重建和碰撞生成方法的首轮专题调研，
  区分 PLY-only、多视角图像与原始 mesh 输入条件，记录论文与公开代码边界。
- [x] 建立 apartment 场景专用 center/voxel/box 原型：过滤 Gaussian 中心、
  清理小连通分量、在受限表面扩张下合并 box，并生成 collision-only MJCF；
  明确其场景常量、人工障碍和无语义能力边界。
- [x] 建立 center-only 有限平面/box MVP 基线，保留门洞、地面边界和台阶
  边缘。auto 模式已有局部 PCA oriented box、投影支持率和大孔洞拒绝，opt-in
  平面残差约束通过 synthetic 门洞和三阶楼梯自由空间/支撑高度回归；显式
  known-free 证据可锚定 slab 自由侧与厚度。跨 candidate 共面合并和无负证据的
  PLY-only 实体侧推断保留为研究扩展。
- [x] 小区域评估实际 SceneSplat 特征解码、空间分布与 unknown 行为；在相同
  geom 预算下比较 geometry-only 与自动语义方案。按 2026-11-24 用户确认，
  不要求手工类别标注；独立准确率未验收，人工标注接口保留为可选诊断。
  **2026-11-24 用户澄清：**不要求用户手工分类，自动算法不确定的类别应输出
  unknown。人工标注与独立准确率不再作为自动生成流程的运行前置条件；保留
  评分接口作为可选诊断，未测指标仍明确记为未验证，不能伪造真值或语义收益。
  2026-11-24 已恢复并核验实际 `(659010,768)` 特征，完成官方 InteriorGS
  72 类解码；0.01 margin 实验门限下 79.48% 为 unknown，不能视为标签准确。
  详见 [实际特征诊断](validation/p8-scenesplat-artanim-2026-11-24.md)；
  另完成实际标签的 32-geom 同预算分区/CPU 落体诊断，两份 MJCF 均编译通过。
  该轮没有独立 surface/free probes，人工标签质量及语义收益验收仍未完成。
  已增加独立三视图/图例及盲标样本导出工具；中心区域 23,933 行中，
  原始 floor 19,901 行、过滤后 floor 0 行，说明 0.01 margin 不可直接作为
  地面剔除规则。48 个样本的人工标注留空，待独立核验后才计算质量。
  独立评分工具已实现 source-ID/XYZ 校验、abstention-aware precision/recall/IoU
  与阈值诊断；实际 0/48 标注报告为 awaiting_annotations，不产生虚假准确率。
  已增加 opt-in 自动回退包装器：unknown 超过 50% 时直接采用纯几何基线；
  否则语义候选必须不降低 source mapping coverage，且 mean/p95/max source
  距离不超过基线 1.05 倍。Artanim 79.48% unknown 实测触发回退，输出
  257 geoms（254 box、3 cylinder）；语义标签保持 unknown，不反向伪造类别。
- [x] 建立场景无关的 center-only 点云/voxel/axis-aligned box 基线；输出
  collision-only MJCF、独立 report 和 box→原始 PLY vertex ID 映射。apartment
  真实资产复跑与现有专用原型得到相同的 20,468/89/36 统计。
- [x] 增加显式 `--primitive-mode auto`：按局部协方差特征和归一化表面误差
  选择 oriented box、sphere、cylinder 或 capsule，拟合失败时回退原有 bounded
  axis-aligned box；round primitive 还必须相对 oriented-box 基线降低至少 30%
  误差，避免仅因 PCA 形状比率误判；默认 `boxes` 模式及既有输出保持不变。
- [x] 建立利用 Gaussian scale、rotation、opacity 的 opt-in covariance-aware
  基线：opacity 继续参与前置过滤，scale/rotation 构造每点 covariance 并加入
  candidate PCA；默认 center-only 不分配 covariance。
- [x] 比较 voxel occupancy、TSDF 和 marching-tetrahedra 等值面。固定
  Artanim crop 的 exact occupancy、0.1 m TSDF 和 0.2 m coarse TSDF 均导出
  确定性 OBJ；report 记录生成时间、顶点/面、拓扑与 source-surface 距离。
  exact occupancy 有 245 条 non-manifold edge，TSDF 与 coarse TSDF 为闭合、
  winding-consistent，故不把所有 voxel surface 泛称为可用 collision mesh。
- [x] 比较 heightfield、mesh/primitive 混合表示。新增 lowest/highest surface、
  bounded hole fill 与多层歧义列诊断；Artanim crop 的 42×42 heightfield 有
  205 个歧义列，混合 MJCF 为 1 heightfield + 24 residual primitives，并与
  43-primitive 基线用相同 falling probes 完成 CPU contact benchmark。
- [x] 复现 SplatTransform 3.12.0（commit ``5d5be2b``）公开工程方案。相同
  Artanim crop/0.1 m voxel/0.1 opacity 在 llvmpipe WebGPU 上生成官方 voxel
  octree 与 ``faces``/``smooth`` collision GLB；smooth 为 2,227 vertices、
  4,406 faces、12 components、watertight，10k crop center 距离 mean/p95/max
  0.1024/0.2174/0.6447 m。软件 adapter 时间不代表真实 GPU；凹 GLB 在 MuJoCo
  单 mesh geom 中会退化为整体凸包。后续已完成 component-hull 与确定性 V-HACD
  转换：12/64 geoms 的体积膨胀为 1.442×/1.172×，21 个 manifest spawn free
  probe 均未误占，并完成相同 3×3 falling-probe CPU contact 对照。
- [x] 完成 MuJoCo/MuJoCo Warp 当前 SDF 路径的接口与本地可用性评估：SDF 是
  需实现和注册的 engine plugin，并以迭代梯度查询参与碰撞；当前 venv 没有可
  直接复用的 SDF plugin 动态库/示例。故不把它伪装成已运行基线，实际实现与
  primitive/mesh 的接触性能比较转为需要新后端工作的研究扩展。

### G1. 生成管线

2026-11-24 后续地板质量加固：新增默认关闭的水平面合并/顶面对齐路径，重用
原始 candidate footprint 并集，不新增无限 plane，不改变 play/demo。它不是整房
地板语义识别；只合并近水平、同标签且已有顶面高度接近的块，保留明显层高差、
斜坡、显式 known-free 和 footprint 孔洞。原始已验收碰撞不覆盖，候选另存并
测量源点覆盖和 MuJoCo 向下射线，详见
[地板拼接加固](validation/p8-horizontal-surfaces-2026-11-24.md)。

apartment 原型已局部覆盖 opacity/bounds 过滤、体素化、连通分量清理、
box 合并和 collision-only MJCF 导出。下列项目按“可配置、跨场景、带报告的
通用管线”口径验收，因此在原型泛化前保持未完成。

- [x] 在独立 collision report/sidecar 中记录生成参数、确定性 seed 语义、工具
  版本、输入/输出哈希、坐标假设和 Gaussian ID 映射；保持 manifest v1 不变。
- [x] 实现模型无关的整数语义 sidecar 输入与部件分组；voxel 多数票、跨标签
  merge 禁止、auto fit 跨界回退均确定性执行。unknown 标签保留为独立几何残差，
  不自动当自由空间或丢弃；output→source CSR 同步记录 primitive 标签。另提供
  SceneSplat-style aligned feature + class embedding 的 blocked cosine 解码器，按
  similarity/margin 阈值输出 unknown 与置信度报告；不内置 checkpoint/text encoder。
- [x] 实现有限 footprint、厚度和自由侧约束的 MVP，按误差与占据边界合并/拆分
  box；避免填实门洞、桌底或未观测区域。当前 opt-in 平面残差只阻止近似
  平面 candidate 跨越弯折/台阶过度合并；另已支持显式 world-space known-free
  点作为负占据约束，merge 优先隔离约束 void，auto primitive 侵入时回退安全
  AABB。配置 ``planar_slab_thickness`` 后，footprint 内邻近 known-free 点以符号
  一致率判断自由侧，拟合平面作为自由面，厚度只向相反实体侧延伸；证据缺失或
  两侧冲突时保留对称拟合。PLY-only 自动实体侧推断、跨 candidate 合并及全局
  固定 geom 预算优化保留为研究扩展。
- [x] 实现 opacity、scale、quantile bounds 和离群点过滤：支持最大 Gaussian
  axis scale、密度阈值、小连通分量及 quantile 空间边界；参数及统计进入 report。
- [x] 实现 center-only 体素降采样、连通分量清理和有界 scan-gap 填充；更一般
  的表面孔洞/自由空间推断仍归后续有限平面与 mesh 路线。
- [x] 实现表面/占据重建：exact exposed occupancy surface、binary TSDF 与
  无 scikit-image 依赖的 marching tetrahedra zero isosurface 均有 synthetic
  watertight 回归和 Artanim 实际输出。
- [x] 实现网格修复、法线统一和简化：合并顶点、删除重复/退化面、清理未引用
  顶点、修复 winding/normals，并报告 boundary/non-manifold/watertight；提供
  vertex clustering，并以 coarse occupancy remeshing 生成保持流形的简化对照。
- [x] 导出含 box、sphere、cylinder、capsule 的 collision-only MJCF；自动模式
  在 report 中逐 primitive 记录类型、位姿、尺寸、误差、oriented-box 对照误差、
  相对改善、candidate 和 source Gaussian 数。另支持确定性 OBJ、inline
  heightfield 以及外部 GLB 的 component/V-HACD 多凸 mesh MJCF；file-backed
  benchmark 只在显式 opt-in 时接受 bundle-local mesh 并哈希全部依赖。
- [x] 增加显式 `--debug-color-by-type` 人工 overlay 诊断：box 低透明绿色、
  sphere 洋红、cylinder 橙色、capsule 蓝色、ground 低透明灰色；仅写 MJCF
  `rgba`，不改变碰撞类型、位姿、尺寸、接触参数和 source mapping。
- [x] 保证相同输入和参数输出字节稳定的 MJCF 与 source-ID NPZ；算法无随机
  采样，report 将 seed 记录为 null。

### G2. 质量与任务感知简化

- [x] 建立含门洞、桌腿、台阶、坑洞和多层结构的 synthetic ground truth；
  单独统计误占自由空间、支撑高度与边缘误差。五类结构均已有定向回归：门洞
  与桌底/桌腿检查 free-space，三阶楼梯验证 0.1/0.3/0.5 m 支撑高度，3×3 坑洞
  与双层间隙验证负占据约束及 box 预算；独立 evaluation probe 输出表面距离及
  false-occupied 数量/比例。
- [x] 比较固定 geom 预算下的语义消融，并分别测量离线生成时间、独立
  physics SPS、接触/求解开销与含视觉的端到端吞吐。固定预算选择、synthetic
  人工真值的 geometry-only/semantic-priority 2-geom 对照及 CPU generation/SPS/
  ncon/nefc/solver 报告已完成。相同 manifest 的 CUDA policy 与 native 3DGS
  视觉实测也已完成：无渲染为 76.10--80.20 control SPS，native actual realtime
  为 0.498--0.571、24.90--28.54 submissions/s。2026-11-24 又完成真实
  SceneSplat raw/unknown 方案与 geometry-only 的 32-geom 对照：三种子
  CUDA 几何为 3/3 跑满、两种语义均 2/3，未证明语义收益；四种 native
  均 ready、500 submissions、overflow=0，actual RT 0.485--0.526。
  策略运行部分 seed 有 LS_ITERATIONS 警告，逐项记录而不伪装成全部稳定。
- [x] 输出 input/filtered/mapped splat、occupied voxel、连通分量、grid/bounds、
  merged/retained/discarded box、primitive 类型/标签及简化比率；primitive-only
  表示明确记录 mesh vertex/face 为 0，避免与 mesh 路线混淆。
- [x] primitive-only 输出没有三角面、重复面或非流形拓扑；导出前显式拒绝
  非有限中心/四元数/尺寸、非单位四元数、非正尺寸及与 geom 类型不匹配的尺寸
  元数。mesh/heightfield 路线仍须单独执行拓扑检查。
- [x] 计算碰撞表面与高置信度 Gaussian 的距离指标。``--evaluate-source-surface``
  在 Artanim center-only 306 primitive 上得到 mean/p95/max =
  0.0714/0.1948/1.4517 m；该输入覆盖指标不是独立 ground truth。
- [x] 完成 Gaussian scale、rotation、opacity 的首个可执行消融接口与真实资产
  对照：opacity 始终参与过滤，scale 上限可清理极端 splat，scale+rotation
  covariance 可选参与 PCA。Artanim 0.2 m scale/covariance 对照为 276 primitive，
  source distance mean/p95/max = 0.0739/0.1998/2.9931 m，未优于 center-only；因此
  不将 covariance-aware 设为默认，也不从 primitive 数减少推断质量改善。
- [x] 提供显式任务优先级选择接口：auto 模式可固定 ``--max-primitives``，
  ``--semantic-priorities`` 对地面/台阶/扶手/落脚区/可操作表面标签加权，
  ``--point-importance`` 对具体 source Gaussian 加权；排序稳定且重建 CSR。
  该项只声明接口与 synthetic 效果，不声称 SceneSplat 已正确识别这些类别。
- [x] 建立几何误差、接触稳定性、策略性能和仿真吞吐的联合评价。新增独立
  manifest-driven G1 evaluator，在不修改 Play/demo 的前提下固定 checkpoint、
  motion、seed、起始帧与有限时域，将既有几何证据、MPKPE/R-MPKPE/末端与关节
  速度误差、终止项、ncon/nefc/solver、overflow bit flag 和 policy+env SPS 写入
  同一报告。Artanim 三种子 CPU 对照中平面控制 3/3 跑满，其余四种自动代理均
  仅 1/3 跑满，主要由 ``ee_body_pos`` 终止，证明几何距离较小不等于任务可用。
  companion native-3DGS evaluator 已按相同 manifest 完成 CUDA 实测；五个表示
  均完成 500 次提交、``gaussian_status=ready`` 且 overflow bitmask 为 0。CUDA
  policy/native 报告哈希和逐表示结果见验证记录；共享 Play/demo 未修改。

## 10. 性能、质量与发布

已完成的首个子任务见 [P10 固定资产 benchmark](mjlab-3dgs-p10-benchmarks.md)。
该轮仅为碰撞替换的性能/接触/跟踪对照，不重开 P7 场景验收，不做新训练或 PR。

- [x] 冻结 apartment 原有碰撞与通用自动碰撞实验副本，显式给两组保留相同
  人工柱体，校验自动产物对应的 PLY/report 哈希，原始资产不覆盖。
- [x] 完成相同落体 probe 与三种子 CPU policy 对照并归档原始结果。两组均
  3/3 跑满 500 步、overflow=0；reference/automatic 的 control SPS 为
  22.67/19.63、MPKPE 为 0.11799/0.13235 m，不宣称自动碰撞收益。
- [x] 用户完成相同两变体的 CUDA policy/native-3DGS 对照，2026-11-25 已核验
  碰撞、PLY、checkpoint/motion 哈希与协议。CUDA 两组均 3/3 跑满，SPS 为
  80.51/80.59，MPKPE 为 0.09757/0.15803 m；seed 42 均有 LS_ITERATIONS。
  native 两组均 ready、500 submissions、overflow=0，actual RT 0.770/0.760。
  不宣称自动碰撞收益，不覆盖原有资产；P10 其它项目保持未完成。

- [ ] **后期优化：**实现版本化 `splats.npz` 预处理缓存、原子写入，
  并在源 PLY、导入参数、transform 或 schema 变化时自动失效。
  2026-11-25 已实现可选 cache_dir、内容/参数 key、NPZ 数据校验和原子替换，
  正在完成回归、make check 与文档验收；未配置缓存时不做哈希或缓存 I/O。
  真实 apartment/Artanim 已测未缓存、首次建缓存与热缓存三阶段，输出数组
  完全相同。证据见 [缓存验证](validation/p10-splat-cache-2026-11-25.md)。
- [ ] 为无 3DGS、Viser-only、MJWarp camera 和视觉训练分别建立正式
  benchmark。
- [ ] 记录初始化、PLY 解析、BVH、浏览器上传、physics、policy 和
  rendering 时间。
- [ ] 建立目标 GPU、splat 数、world 数、分辨率和目标吞吐矩阵。
- [x] 2026-10-25 运行 `make check` 通过。
- [ ] 运行完整 `make test`，包括经明确配置的 slow/real-asset 测试。
- [x] 运行 `make docs`。
- [ ] 检查当前所有 3DGS 修改，避免覆盖工作树中的用户改动。
- [x] 2026-11-09 将 apartment 碰撞生成、可见碰撞柱体、manifest 无 SHA
  契约和视觉诊断整理为提交 ``e735d0b``。
- [ ] 按仓库规范准备 PR 描述；未经用户明确命令不创建 commit 或
  PR。

## 11. 论文与研究材料

- [x] 建立 prior-art 第一轮报告和本项目必要性初步判断。
- [x] 记录从 Gaussian 到碰撞、视觉—物理偏差和任务感知简化的
  研究方向。
- [x] 确定单级台阶为最小视觉闭环验证任务。
- [ ] 持续更新论文、代码、数据集、许可证和复现状态。
- [ ] 建立方法对比表和正式 bibliography/BibTeX 数据库。
- [ ] 将工程 baseline 与可声明的算法贡献严格区分。
- [ ] 固定论文研究问题、假设、基线、消融和统计检验方案。
- [ ] 保存所有训练配置、随机种子、资产 hash、checkpoint 和原始
  指标。
- [ ] 形成论文背景、相关工作、方法、实验和限制章节草稿。

## 12. 当前未决事项

- [ ] 首个真实 3DGS 资产由哪个工具和版本生成，字段语义是否完全
  固定？
- [ ] 当前真实场景是否包含适合作为单级台阶实验的区域？
- [x] 训练首版使用 `64×48` depth 和 YAM Vision 的独立
  Spatial-Softmax CNN encoder。
- [ ] 目标 GPU、首版 splat 数、训练 world 数和最低吞吐是多少？
- [ ] 首版碰撞输入是否只接受 collision-only MJCF，还是同时提供受限
  raw mesh convenience API？

## 13. 推荐执行顺序

1. P8 按用户决定完成并收口，保留现有产物和地板拼接等限制，不追加修补。
2. P10 固定资产的原有/自动碰撞 CPU/CUDA/native 对照已完成，按顺序继续
   版本化缓存与性能矩阵，不重开 P7 已收口场景验收或 P8 地板修补。
3. 固定视觉资产和评估协议，对比手工/自动碰撞、视觉依赖及未见布局/场景。
4. 基础视觉运动闭环成立后扩展到单台阶、楼梯、复杂地形与接触敏感实验。

## 14. 更新记录

- 2026-11-25：启动 P10 可选 3DGS 预处理缓存，独立模块加默认关闭的资产配置
  字段，不改 play/demo、manifest schema、训练或物理加载接口。基于 source
  内容/导入参数/schema 校验失效，损坏重建、原子写入和失败回退均已有定向
  测试；真实两场景读入计时与 exact-array 对照已完成，整体检查仍在验证。

- 2026-11-25：核验用户完成的 P10 apartment CUDA/native 报告，输入及协议
  一致。两组 CUDA 均跑满三个种子，自动方案吞吐近似但平均跟踪误差较大；
  两组 seed 42 的 LS_ITERATIONS 警告原样保留。native 两组均 ready、500
  submissions、overflow=0。固定资产对照子任务完成，P10 整体仍进行中。

- 2026-11-24：用户明确将 P8 标记完成并转下一任务。更新当前主线为 P10
  固定资产替换 benchmark。新增通用实验准备工具，显式规范化共同障碍的位姿、
  形状、接触与外观参数，冻结副本并记录输入/输出及生成报告哈希；不覆盖
  apartment 原始碰撞、manifest 或 P8 通用产物，不修改 play/demo。
  两变体三种子 CPU 与独立落体已完成，9 项定向测试、make check、make docs
  通过；CUDA/native 尚待桌面运行，不提前标记 P10 完成。

- 2026-11-24：按用户不手工分类、不能判断则 unknown 的要求收口 P8 工程链路。
  实际 SceneSplat、unknown 保留、257-geom 几何回退、原始/过滤语义同预算
  CPU/CUDA/native 对照均完成。42 项定向测试、make check、make docs 通过，
  play/demo 零差异。负结论与 LS_ITERATIONS 警告保留，独立准确率未测。
  证据见 [实际语义验收记录](validation/p8-scenesplat-artanim-2026-11-24.md)。

- 2026-11-21：核验用户完成的 P8 CUDA/native 输出。CUDA policy 报告沿用相同
  checkpoint、motion、manifest 和 seeds，五表示法为 76.10--80.20 control SPS，
  是 CPU 的 2.88--3.49 倍；SplatTransform 两表示的成功率由 CPU 1/3 变为 CUDA
  0/3，明确记录而不假设跨设备轨迹等价。native 3DGS 五表示均 ready 并完成
  500 次提交，actual realtime 0.498--0.571，24.90--28.54 submissions/s，无
  overflow bit。GPU 路径 SplatTransform GLB 与 llvmpipe 输出字节一致；补录的
  Vulkan GPU 0 日志显示 voxelization 0.036 s、总耗时 0.714 s、峰值 GPU
  13.6 MB。``nvidia-smi`` 已归档主机唯一 NVIDIA GPU 为 RTX 3060 12 GiB、
  driver 580.82.07；``--list-gpus`` 已归档 adapter 0/1 为 RTX 3060、adapter 2
  为 llvmpipe。运行显式选择 ``--gpu 0``，NVIDIA/SplatTransform 性能证据闭环。
- 2026-11-21：完成 P8 CPU 策略级联合评价协议。新增本地 G1 collision-policy
  benchmark、三种子 Artanim 五表示法报告及 native 3DGS companion benchmark；
  报告关闭 auto-reset 以保留真实终止步，并将 Warp overflow 解码为 bit flag。
  平面控制成功率 1.0，43 primitives、heightfield+residual、SplatTransform
  component hull/V-HACD 均为 0.333；后四者多在 401--428 步因末端/anchor 偏差
  终止。其 CUDA/native 补充结果见上一条更新。
- 2026-11-21：完成 P8 内部表示法与固定预算阶段。新增 exact occupancy、TSDF
  marching tetrahedra、拓扑修复/法线统一、coarse manifold remeshing、
  heightfield+residual primitive，以及 Artanim crop 的距离/拓扑/生成时间和
  统一 CPU contact 对照。新增 ``--max-primitives``、semantic/point importance
  稳定排序及 matched-budget 消融工具；synthetic 2-geom 人工真值显示层间误占
  由 1 降为 0，同时保留被舍弃表面的误差代价。SceneSplat 官方预处理已验证
  659,010 行对齐，但 checkpoint 受 Hugging Face 条款/登录限制且本环境无 CUDA；
  SplatTransform 固定官方 commit、安装 3.12.0 并通过 llvmpipe 实际生成
  faces/smooth collision GLB；smooth 为 2,227 vertices/4,406 faces 且闭合，
  但凹 GLB 不可直接作为 MuJoCo 单 mesh geom。SceneSplat 实际推理和视觉/策略
  联合评价不作完成声明；共享 ``play.py``/``demo.py`` 保持零 diff。
- 2026-11-21：补齐语义特征解码边界。新增 aligned ``.npy``/tensor-only ``.pt``
  feature 与 class embedding 的确定性 cosine decoder，分块避免一次分配完整
  ``N×C``，以 best similarity/top-two margin 把低置信度保留为 unknown，并输出
  labels、confidence NPZ 与输入/输出 hash。synthetic 顺序、阈值、unknown 与非法
  shape/ID 回归通过；这仍不替代 gated SceneSplat checkpoint 的真实推理和标签评估。
- 2026-11-21：补齐 SplatTransform→MuJoCo 可执行外部对照。新增 GLB/OBJ 的
  component hull 与 V-HACD 多凸 MJCF 转换、volume/source-distance/free-probe
  报告，以及默认拒绝、显式 opt-in 的 bundle-local mesh benchmark。Artanim
  smooth GLB 的 12-component hull 与 64-piece V-HACD 体积膨胀为 1.442×/1.172×，
  21 个 spawn-column free probe 均为 0 false occupied；同一 3×3 probe 单轮为
  27,456/20,689 SPS。关闭 async 后两次 V-HACD sampled hashes 一致。该阶段没有
  修改 ``play.py``/``demo.py``，也不把稀疏 free probe 当全场景真值。
- 2026-11-21：完成 P8 自动碰撞生成 MVP，但按用户要求不作完整 P8 收口。MVP
  核心范围包括确定性通用
  primitive 生成、provenance/source mapping、known-free 与 semantic sidecar、
  有限平面/free-side 约束、synthetic 真值、Artanim 真实资产距离/编译证据及
  rendering-independent CPU collision benchmark。该记录描述当时状态；后续
  已补内部 TSDF/mesh/heightfield，但外部语义模型、SplatTransform/SDF 和任务
  联合评价继续作为完整 P8 未完成项；未修改共享
  ``play.py``/``demo.py``。
- 2026-11-21：增加独立 evaluation NPZ 和统一几何统计。reference surface 点
  输出最近 proxy surface mean/p95/max 距离，free 点输出误占数量/比例；输入不
  参与生成，避免约束点自证。同步补齐 grid、连通分量、映射覆盖和简化统计，
  synthetic 五类结构的首版 ground-truth 项按当前 primitive 基线范围完成。
- 2026-11-21：增加可替换 semantic frontend 的 sidecar 契约。每个原始 PLY
  vertex 对齐一个整数标签；voxel 多数票后按标签分组 merge，其他标签 occupied
  voxel 作为边界，unknown 保留为 geometry residual。NPZ/report 保存输出标签与
  输入哈希。当前只完成接线和 synthetic 边界回归，SceneSplat 小区域标签质量
  评估仍未完成。
- 2026-11-21：新增 opt-in covariance-aware PCA 与最大 Gaussian scale 过滤。
  复用 Graphdeco scale/rotation 定义构造 world-space covariance，只在显式启用时
  参与 primitive 分类；默认 center-only 路径无额外内存和输出变化。Artanim
  0.2 m scale 上限对照生成 163 box + 9 cylinder 并通过 MuJoCo 编译，质量优劣
  保持待后续统一指标与接触实验判断。
- 2026-11-21：在显式 known-free 负证据上增加 opt-in planar slab 单侧锚定。
  footprint 内、指定距离内的 free 点对局部法向两侧投票；达到一致率后将拟合
  平面固定为自由面，并把配置厚度仅向相反侧延伸。上下自由侧及双侧歧义
  synthetic 回归通过；缺少证据时不猜测，默认输出不变。
- 2026-11-21：新增显式 known-free 点 sidecar 约束，不从 PLY 未占据区域猜测
  自由空间。merge box 和 auto primitive 均不得覆盖网格内约束点，冲突输入失败；
  free-aware guillotine score 在保留 void 的同时控制 box 数。新增桌底/桌腿、坑洞
  和双层结构 synthetic 回归，默认路径保持字节兼容。该接口为相机 ray carving、
  人工标注或 synthetic truth 提供通用接入点；实体侧推断和 slab 单侧厚度锚定
  仍未完成。
- 2026-11-20：为 center-only box merge 增加 opt-in 平面残差约束。它只在候选
  占据 voxel 呈近似平面时限制 voxel center 到最佳拟合平面的最大绝对距离，
  防止楼梯或弯折表面被粗 fill 条件合并为 ramp-like 大 box；默认关闭以保持
  既有字节输出。synthetic 门洞和三阶楼梯回归通过；完整自由侧厚度锚定以及
  桌腿、坑洞、多层结构真值仍是下一步。
- 2026-11-20：在保持默认 box-only 输出不变的前提下新增 opt-in auto primitive
  模式。每个 merged voxel candidate 使用 PCA 特征分类，再以表面误差验证
  oriented box、sphere、cylinder 或 capsule；round fit 必须比 oriented-box
  基线至少降低 30% 误差。平面支持稀疏或含大孔洞、误差超阈值和点数不足均
  回退 bounded axis-aligned box。新增逐 primitive 对照误差 report 与 CSR
  source-ID mapping；Studio 保守 auto 得到 183 box + 7 cylinder，这 7 个候选
  的相对改善为 33%--49%，尚待 native overlay 人工质量检查和参数标定。
- 2026-11-20：修复提高 round-vs-box 相对改善阈值时小候选可能被二次体积过滤
  的问题；绝对 round 拟合通过但相对收益不足时保留同一 candidate 的 oriented
  box，并在 report 中记录 rejected round 类型、误差和相对改善。Studio 的
  30%/40%/50% sweep 现在均保持 190 个 primitive，分别为 7/4/0 个 cylinder。
- 2026-11-20：为 auto primitive 增加 opt-in 按类型着色的 MJCF 诊断输出，便于
  在 Native Viewer 中定位 7 个 cylinder 并对照 3DGS 判断是否误判；该选项只改
  `rgba`，不作为几何或接触质量已验收的证据。
- 2026-11-20：启动 P8 通用几何基线。新增场景无关 center-only PLY→voxel/
  axis-aligned box→collision-only MJCF 工具，输出独立 SHA-256 report 与确定性
  box→source Gaussian ID NPZ，不修改 manifest v1、训练、渲染或物理加载接口。
  synthetic 门洞/source-ID/floor/grid-limit 回归与既有 apartment 合并回归共
  8 项通过；
  apartment 真实资产复跑得到 20,468 个占据 voxel、89 个合并候选和 36 个
  surface box。有限平面、第二场景质量、语义与接触性能保持未完成。

- 2026-11-17：明确深度来源为 3DGS 和可视 mesh，隐藏 group 3 仅用于物理和
  教师。Vision-Data 使用任务侧原生直接射线，避免共享 RenderContext 合并污染，
  保留显式旧路径与所有历史产物。修正 Studio group-1 柱体的验收解释，0/6
  不再作为可视柱体停车失败证据。共享 play/sensor/simulation 无新增修改。

- 2026-11-14：结合 Hiking in the Wild 与 PHP 调整 P7-4 主线。端到端
  Privileged 低层 PPO 作为失败诊断完整保留，不再用增加迭代修复；新增独立
  continuous safety teacher task，以真值前向射线输出 normalized clearance、
  validity、速度缩放 `alpha`、stop label，并用 `alpha` 平滑缩放 nominal
  command。teacher label 不进入 actor/critic，稳定 locomotion actor 保持冻结。
  首版阈值为 stop `0.8 m`、slow `1.8 m`；同日将 stop 调整为 `0.4 m`。play
  模式现以 1 Hz 打印 nominal/safe/measured velocity、前向
  净空、`alpha` 和 stop，并固定使用场景协议的 nominal command，不再随机采样
  训练速度；训练模式不打印且仍保留原采样范围。固定命令 GUI 回放在
  `safe_vx=0.204 m/s`、`clearance=0.972 m`、`alpha=0.409`、`stop=False` 时已停止，
  后续日志又在 `safe_vx=0.192 m/s`、`clearance=0.939 m`、`alpha=0.385` 时确认
  `measured_vx≈0`。这说明冻结 actor 存在低速步态死区；第二版将 slow 收近为
  `1.0 m`，预计实际停止位置约 `0.64 m`。视觉 Student、数据采集和 DAgger
  仍属于下一交付物。
  第二版 GUI 回放在 `0.699--0.735 m` 仍有正向实测速度，随后重复跳回出生净空，
  用户确认该 reset 发生在机器人碰撞障碍物后，说明第二版制动距离不足。第三版
  slow 改为 `1.4 m`，预计在约 `0.80 m` 进入低速死区；stop 仍为 `0.4 m`，等待
  GUI 无碰撞停车验收。用户随后确认第三版验证通过。play 诊断仍会在 reset 当步
  立即输出 episode step 和具体 termination 名称。
- 2026-11-12：启动 P7-3。新增 renderer-free blind 与 privileged local-geometry
  task，均复用冻结 apartment collision、标准 G1 joint action 和 MLP PPO。blind
  保持 99/109 维 observation，不创建 Gaussian asset/camera；privileged 用
  `pelvis` 机体系 9 条 collision-truth 射线向 actor/critic 各增加 18 维距离/
  有效位，并用同一局部几何构造安全 tracking target。两个 task 的 2-env CPU
  1-update smoke 均通过，分别为 33/31 SPS，loss 有限。正式训练预算、标准 G1
  checkpoint warm-start、free tracking 和多场景局部避障效果尚未验收，P7-3
  保持进行中。
- 2026-11-15：将后续主目标收敛为基于深度几何阈值的通用速度安全层，而不是
  识别墙、柱或场景类别。新增独立 `Safe-Depth-Stop` 任务，对完整前向 ROI 使用
  q10 稳健净空，并以 `0.4 m` stop、`1.4 m` full-speed 连续缩放 nominal
  command；该路径只减速/停车，不主动转向。原 Adapter 搜索路径和 Student v1/v2
  失败实验均保留。合成墙/柱等价、无效深度停车及配置隔离定向回归为
  `70 passed`；真实 GUI 墙/柱/其它障碍回放与时序滤波决策待完成。
  随后 GUI 已确认柱体在约 `0.923 m`、墙面在约 `0.819 m` 净空无碰撞停车；
  `false_alarm` 在冻结 10 秒窗口内 alpha 恒为 1 且保持全速，窗口外遇到真实障碍
  后才减速停车。新任务进一步增加默认关闭的 `0.5 m/s` 绝对前向速度上限，使
  更大的 nominal 指令不能放大已验收的制动曲线；合成 `2.0 m/s` 指令测试覆盖
  free/brake/stop 输出。
  针对整图 q10 可能忽略细障碍的问题，离线六场景筛选后新增 7 帧 clearance
  中位数、`1/4` 宽重叠滑窗和窗内 q20，仅 Depth-Stop 任务启用。离线结果在
  `free/false_alarm` 零误停、四个障碍布局零漏停；合成约占全图 3.4% 的窄障碍
  已从全速改为 alpha 0.5，单帧异常被时间中位数抑制。仍需重新 GUI 验证既有三
  场景，并增加真实细障碍夹具后才能关闭该项。
  三场景 GUI 回归随后通过：`false_alarm` 前 500 step 保持全速且零 termination，
  双柱和 dead-end 墙面分别在约 `0.696 m`、`0.732 m` 净空稳定停车且无碰撞。
  现有半径 `0.18 m` 圆柱已是冻结资产中的窄障碍，不修改已验收资产。P7-5 runner
  已新增 `--controller depth_stop` 多场景/多 seed 模式，并允许 evaluator 把
  BRAKE/STOP/BLOCKED 的持续静止识别为安全停车；正式 GPU 矩阵待运行。
  相同 Blind `model_99.pt` 的 `false_alarm` 5-step CPU smoke 已成功加载并生成
  JSON trace，结果 continuous tracking、零介入/碰撞/termination；该结果仅验收
  runner 接线。
- 2026-11-15：启动跨地图解耦。Depth-Stop 工厂新增可选 scene manifest、spawn
  和 nominal command 输入，并在环境构建前从所选 collision MJCF 生成 base-link
  障碍接触传感器；默认 apartment 注册任务和旧实验保持不变。仓库内 Studio
  manifest 的第二地图配置回归和真实 reset/step 验证六根柱体且无 apartment geom
  泄漏。由于 Studio 随机碰撞柱体尚无视觉匹配，该结果只关闭配置装配，不关闭
  跨地图 GUI 行为验收。
- 2026-11-15：新增 manifest-aware 安全速度 GUI 播放工具。该工具先构建所选地图
  的完整 Depth-Stop env config，再注册为仅当前播放进程可见的任务并调用标准
  `run_play`，使 viewer、checkpoint 加载和 alpha 诊断可直接复用，同时避免构建后
  替换地图导致旧 contact sensor 残留。通用 play 代码与默认路径不变；下一步等待
  视觉/碰撞对齐的新 manifest 做 GUI 停车矩阵。
- 2026-11-15：结构化 runner 同步支持 manifest-aware 自定义地图。首个 Studio
  default/seed 0 探针在 4.74 s 无碰撞进入安全静止，证明同一深度门控和冻结 actor
  可以跨 manifest 复用；随后 seed 0/1/2 矩阵 3/3 安全停车且零碰撞/跌倒/
  termination。由于仍只有单 spawn 且 collision overlay 尚未逐点验收，不宣称
  完整跨地图泛化。
- 2026-11-12：P7-3 首轮 64-env GPU 吞吐探针发现通用 G1 `command_vel`
  curriculum 会在首次 reset 覆盖 safe-velocity catalog 的 `0.0--0.8 m/s`
  forward-only 命令。现已仅在 safe-velocity 任务中关闭该 curriculum，保证各
  P7 基线遵守冻结 command protocol；普通 G1 velocity task 不变。修正前探针
  仅用于发现配置问题，不计入正式训练结果。
- 2026-11-12：为 P7-3 新增 locomotion actor warm-start 转换工具。blind 复制
  标准 99 维 actor；privileged 复制前 99 维并将新增 18 维局部几何输入权重
  置零；critic 和 optimizer 保持新初始化。两种 checkpoint 均已严格 resume
  并完成一次有限 loss 的 CPU PPO update。首轮正式预算冻结为 64 environments、
  1000 updates、seed 42，由用户依次训练后返回结果验收。
- 2026-11-12：Blind 首轮 1000-update 微调失败：最终平均回报约 `-76.95`，线速
  跟踪奖励接近零，策略无法站稳、向前行走且不规则后退。warm-start actor 与
  来源 checkpoint 逐参数一致，退化仅发生在 PPO 微调后。现将 Blind/Privileged
  的任务局部微调设置改为固定 `0.15` 动作标准差、`1e-4` 学习率、零 entropy
  coefficient 和每轮 2 learning epochs；普通 G1 runner 不变。首轮 checkpoint
  作废。修正配置已完成 5-update CPU resume smoke，动作标准差稳定在 `0.15`，
  loss 有限；下一步先运行 100-update GPU 校准和 deterministic playback，再继续
  正式对照。
- 2026-11-12：Blind 保守配置的 100-update GPU 校准完成，`model_99.pt` 的
  deterministic playback 可以站稳，动作标准差保持 `0.15`；但最后 20 updates
  平均回报约 `-41.42`、线速度 tracking reward 约 `0.0025`，训练采样仍主要因
  跌倒结束。该模型尚未通过 free tracking，不扩展到 1000 updates；先确认是否
  实际向前跟踪，再决定冻结原 locomotion actor 或继续微调。
- 2026-11-12：用户确认 Blind `model_99.pt` 的 deterministic playback 能站稳且
  向前行走。该模型冻结为候选 Blind baseline，不再追加 PPO updates；后续仍需
  结构化 free rollout 量化跟踪误差和终止状态，且不把它解释为避障策略。
- 2026-11-12：Privileged 保守配置完成 100-update GPU 校准。动作标准差固定为
  `0.15`，但最后 20 updates 平均回报约 `-46.76`、timeout 比例约 `0.018`，训练
  采样仍主要跌倒；新增几何输入首层权重虽从零更新，L2 norm 仅约 `0.026`。
  需先回放 both-open 检查真实避障行为，不扩展正式预算。
- 2026-11-12：Privileged `model_99.pt` 回放在障碍前停止超过 10 秒且不转向。
  根因是 local-geometry 转换把 collision ray 的 `-1` no-hit 错当成无效深度，
  导致开阔方向也被 adapter 判为未知危险、无法 track/resume。现将有限负距离
  映射为 cutoff clearance 且 validity=1，仅 NaN/Inf 无效；定向测试覆盖 clear
  tracking、朝开阔侧 search 和 resume。真实 reset/step 与 1-update CPU smoke
  通过，loss 有限；旧 checkpoint 作废，需重新执行 100-update GPU 校准。
- 2026-11-12：ray-miss 修正后的 Privileged 100-update GPU 校准完成。最后 20
  updates 平均回报约 `-48.95`、safe tracking reward 约 `0.108`，随机采样仍主要
  跌倒，训练曲线未显示明确改善。下一步只做 deterministic both-open 回放；
  未确认制动、转向、恢复前不扩展训练预算。
- 2026-11-12：修正后 `model_99.pt` 回放为前进、遇障停止，但不转向、不恢复，
  仍未达到 `avoidance_resume`。该 checkpoint 记为失败校准，不直接追加迭代。
  后续需区分端到端 privileged 几何策略与直接输入规则 safe twist 的 oracle
  controller upper bound，避免混淆实验结论。
- 2026-11-12：用户将 P7-3 privileged 目标简化为“遇障停车”，不再要求转向、
  绕行或自动恢复。新的无状态 tracking target 在中间 3 条前向射线任一无效，
  或最小净空不大于 1.2 m 时，把前进、侧移和偏航命令全部置零；该结果不受
  nominal command 大小影响，侧面障碍不会触发转向。P7-2 规则 adapter 保持不变；
  目标变更前的 privileged checkpoint 不恢复，下一步从 conservative warm-start
  重新执行 100-update 校准并验证无碰撞稳定停车。纯逻辑、真实 reset/step 和
  2-env CPU 1-update smoke 已通过；smoke 的 value/surrogate loss 分别为
  `0.0453`/`-0.0542`，action std 保持 `0.15`。
- 2026-11-13：4096-env、5000-update Privileged 长训练的后期 telemetry 虽显示
  平均回报约 59.7、episode length 1000 且记录跌倒为零，deterministic playback
  中机器人仍无法站稳；同配置下 warm-start 可站稳。最终 actor MLP 相对
  warm-start 漂移约 21.8%，该 run 全部作废。新增独立 privileged stop-gate，
  只在 observation 边界用 collision ray 将 nominal command 切换为原命令或零，
  复用稳定 warm-start 而不微调关节策略；同时增加 pelvis 低于 0.5 m 和非足端
  着地终止。该任务属于 collision-truth oracle。CPU deterministic 验收中，
  `both_open` 在 0.98 秒触发停车并稳定保持至 22 秒，`free` 10 秒内无误停车；
  两者均无终止且 pelvis 最低高度约 0.753 m。下一步用户 GUI 回放确认行为。
- 2026-11-12：P7-2 完成。六类 catalog 场景增加预期 outcome 与有限评估窗口；
  根据旧 locomotion checkpoint 的多 seed 偏航响应，将规则 adapter 搜索预算从
  12 秒重新标定为 18 秒，障碍场景最长评估 22 秒。`model_29999.pt` 在六场景 ×
  seed 0/1/2 的真实环境矩阵中 `18/18` 通过：free/false-alarm 持续跟踪，三类
  开口场景完成避障恢复，dead-end 安全 blocked；全部无 `pelvis` 障碍接触和
  跌倒，最小前向传感净空 0.792 m。首次 1-update 产物也成功恢复并完成下一次
  PPO update，loss 有限。P7-3 下一步训练 blind/privileged 学习基线；长期奖励
  漏洞和视觉策略效果不属于本次规则基线结论。
- 2026-11-11：接入首版 safe-velocity 任务奖励。保留通用 G1 locomotion 项，
  仅在显式 safe-velocity 任务中用 depth adapter 的安全 twist 替换无条件 nominal
  velocity tracking，并增加 `pelvis` 对所有非地面障碍接触的独立负奖励；普通
  G1 velocity 默认奖励不变。纯逻辑、碰撞聚合、配置和真实 reset/step 等定向
  回归共 `48 passed`。2 个环境、48 个采样步的 1-update PPO smoke 也已用本地
  TensorBoard logger 跑通，loss 有限；默认 W&B logger 因未配置 API key 在
  rollout 前退出。权重、奖励量级、完整六场景、多 seed、正式/恢复训练和学习
  策略漏洞仍待验收，P7-2 不关闭。
- 2026-11-11：将安全速度运行时代码从阶段编号命名重构为语义化的
  `SafeVelocity*` API；冻结的 `safe_velocity_scenarios.py` 替代
  `p7_scenarios.json` 和运行时 JSON 解析。现有 task ID 字符串、scenario id、
  manifest spawn id 与行为保持兼容，P7 仅作为需求阶段编号保留。
- 2026-11-11：新增 P7 privileged 轨迹 evaluator；碰撞 baseline 定义为 G1
  `pelvis` base link 与任意非 `ground` 场景 geom 接触，足端和其他肢体接触暂不
  纳入该 outcome。evaluator 汇总唯一 outcome、路径长度、深度传感净空、最大
  接触力和接触冲量。九类 outcome 的脚本轨迹测试通过；同一 checkpoint/seed
  的 left/right 650 步重跑均为 `avoidance_resume`，无 base-link 障碍接触、跌倒、
  越界或停滞，最小前向深度传感净空分别为 0.962 m 和 0.933 m。仍需其他场景、
  多 seed、geometry 净空和 P7 奖励，P7-2 不关闭。
- 2026-11-11：新增 P7 adapter 结构化诊断脚本，记录 mode、三扇区净空/
  valid ratio、输入/输出 command、搜索时间和机体轨迹。实测定位旧参数的恢复
  停滞来自 `0.6 m` STOP 门槛在低速策略停住后不可达；新增独立的 `1.0 m`
  搜索触发距离。旧 checkpoint 的实际原地旋转响应约需 8--10 秒进入安全扇区，
  因此将 catalog 搜索预算从 6 秒标定为 12 秒。seed 0 的 left/right 650 步均
  完成 `brake-stop-search-resume-track` 且未 blocked/终止；这不是碰撞、outcome
  或多 seed 正式验收，P7-2 继续进行。
- 2026-11-10：使用现有 G1 velocity `model_29999.pt` 人工运行 adapter 任务的
  多个 apartment 出生点，checkpoint 和 99 维 actor 接口正常，未观察到碰撞；
  但 `p7_left_open`/`p7_right_open` 基本不恢复平移，左右避障尚未通过。下一步
  增加 mode、扇区净空、valid ratio、输入/输出 command 和搜索时间诊断，确认
  近地深度、出生距离及恢复阈值的影响后再标定，不以保守停车替代避障成功。
- 2026-11-10：将 deterministic depth adapter 接入独立可运行任务
  `Mjlab-Velocity-Safe-Adapter-Unitree-G1-3DGS`。相机只供任务局部 adapter
  消费，blind actor 保持标准 G1 MLP 接口并接收安全 twist；未改通用 manager、
  command、sensor 或 velocity MDP。真实环境 reset/step 与相邻回归共
  `46 passed`；P7 专用奖励、轨迹 outcome 和 checkpoint 避障效果仍待完成。
- 2026-11-10：用户确认六类 P7 场景的出生方向、视觉/碰撞和安全出口，catalog
  `review_status` 全部更新为 `approved`，P7-0 与已有 reset/step、无定位泄漏、
  非 P7 回归证据共同关闭 P7-1。开始 P7-2：新增任务局部 deterministic depth
  adapter 和九类互斥 outcome 判定，合成测试覆盖左右搜索、阻塞超时、恢复、
  无效深度、关闭 passthrough 与安全优先级；相关定向回归 `42 passed`，实际
  rollout 和奖励仍待实现。
- 2026-11-10：启动 P7 实现。新增版本化 scenario catalog 和 apartment 六个
  P7 命名出生点，覆盖 free/left/right/both/dead-end/false-alarm、数据划分、
  命令初值、安全出口和搜索/净空初值。新增独立 P7 task id，复用现有 G1
  joint action、`UniformVelocityCommand`、P6 robust temporal depth+validity
  和 asymmetric critic；未修改 manager/command/scene 核心或其他任务默认值。
  catalog 仍待人工视觉/碰撞审批，奖励、outcome、adapter、训练均未实现。
- 2026-11-09：补充 P7 相关项目调研。确认 ABS 最接近任务与安全恢复机制，
  Hiking in the Wild 最接近 G1 深度关节控制，Omni-Perception 最接近
  速度跟踪与反应式避碰的联合训练，GaussGym 最接近 3DGS 仿真条件；没有
  单一公开项目覆盖完整 P7，详细依据见主调研报告第 6.4 节。
- 2026-11-09：按真实机器人没有位置传感器或定位模块的约束，将 P7 从
  PointGoal/目标条件导航重构为 ``3DGS 安全感知速度跟踪``。新增
  ``mjlab-3dgs-safe-velocity.md``，明确 actor 只读取 velocity command、
  本体状态和时序 depth+validity；复用 ``UniformVelocityCommand`` 和现有
  joint action，不读取世界位置、目标位置、地图或碰撞真值。任务状态改为
  正常跟踪、制动、停车、原地旋转搜索、安全选向、恢复跟踪和无出口站立；
  取消目标到达率、路径效率和 SPL，运行时代码与训练仍未开始。
- 2026-11-09：同步提交 ``e735d0b`` 后的项目状态。manifest v1 已移除
  visual/collision/dependency SHA-256 字段和运行时内容校验，仍保留路径边界、
  dependency 一致性、schema 和 transform 校验；视觉 smoke 支持 `320×240`
  诊断、固定姿态及 RGB/raw/policy depth 导出。apartment 当前产物为 89 个
  表面 box、地面和两个视觉/碰撞一致的柱体，相关定向回归 `14 passed`。
  P7 进入资产/协议准备阶段，但安全速度跟踪 MDP、奖励和训练均未开始。
- 2026-11-09：确认 apartment 的 ``generate_collision.py`` 是 P8 已实现的
  场景专用几何原型，而非纯手工资产或通用算法。它已覆盖 Gaussian 中心
  过滤、体素化、连通分量清理、box 合并和 MJCF 导出；跨场景参数推断、
  泛化质量、语义碰撞生成及系统消融留待后续详细设计。主线调整为 P7，
  P8 作为并行算法线继续推进。
- 2026-11-07：P6 按 P7-ready 工程边界完成。新增 stride-2 四帧 robust
  depth+validity、固定延迟、低频 sample-and-hold、量程噪声、模糊、对齐
  结构化/整帧 dropout、asymmetric critic、ONNX 前向兼容、1/4-world 性能
  证据及统一 smoke 入口。单帧与普通 G1 默认路径不变；视觉收益、多种子
  任务实验、蒸馏、更大规模和外部空 BVH 后端问题分别转入 P7/P10/P11。
- 2026-11-07：P6 新增独立四帧 depth+validity 任务，复用主线 observation
  history 与 reset 回填，不修改共享 manager 或单帧默认行为；新增时序 CNN
  薄适配层及真实 renderer 测试。stride、延迟/噪声、训练收益与规模性能
  仍待完成，P6 保持进行中。
- 2026-11-07：明确 P6 以时序 depth+validity、传感延迟/噪声和规模验收为
  P7-ready 收尾目标。P7 改为 3DGS 目标条件视觉避障：持续行走时对
  不可通行障碍减速、停车或转向，并在无碰撞、不跌倒的前提下到达
  目标。增加制动/TTC safety shield、停车距离、最小间距和不必要停车率
  验收。参考 Hiking in the Wild 的感知运动设计，不声称复现。
- 2026-10-31：按用户方向将 GaussGym 式真实重建场景视觉运动/导航作为
  主线，P7 首个闭环改为随机目标到达与障碍绕行，单台阶、楼梯与复杂地形
  转为后续课程。P6 保留 depth/validity 并规划 RGB；P8 优先工作顺序不变。
  本次仅调整研究与验收计划，不表示算法迁移或新增训练已完成。
- 2026-10-31：按用户要求先开展 P8，完成语义引导简化碰撞的首轮专题
  调研，补充主报告、独立方法报告和 BibTeX。建议有限平面/box 几何基线、
  SceneSplat 语义前端与残差凸分解；记录门洞/边界约束、输入差异及公平
  消融和性能协议。实现与外部复现尚未开始，不宣称性能收益。
- 2026-10-31：按用户确认将 P7 与 P8 调整为并行工作线。P7 先使用
  手工对齐资产建立视觉任务基线；P8 基于已稳定的 P4/P5 契约独立生成
  碰撞资产，不再置于 P7 完成之后。记录资产版本冻结、联合评估和 GPU
  资源边界；本次仅更新执行规划，不改变尚未开展工作的实际完成状态。
- 2026-10-31：新增独立 depth+validity 对照任务，保留单通道基线与独立
  experiment。完成 raw mask 语义、channel 拼接、真实 reset/step 对齐、
  CNN/PPO 短训练及 checkpoint 编码边界测试；两组初始 depth 一致。
  回归 `29 passed`，保存方法、对照报告和 mask 图像。mask 是工程观测
  设计，不宣称算法创新或任务收益。
- 2026-10-31：新增逐帧 raw depth 覆盖率统计，四环境早期 PPO/恢复训练
  共 480 环境帧通过，记录缺失/远端/量程内比例与 done 次数。新增双资产、
  三 world、双相机的解析深度及隔离测试，定向回归 `25 passed`。
  保存原始报告，区分合成路由检查与真实多资产任务验收；下一步评估
  validity 通道及规模性能，P6 仍保持进行中。
- 2026-10-31：继续 P6 深度诊断，完成七姿态机器人/Gaussian/组合对照，
  确认抬臂产生真实近处自遮挡；记录各姿态缺失/远端比例及验证边界。
  补充 `--pose-sweep` 入口、归档报告与图像，P6 仍保持进行中。
- 2026-10-30：继续 P6 深度质量验证，新增机器人/Gaussian 分离诊断及
  解析深度测试，确认固定相机朝向与当前姿态无自遮挡；修复 G1 缺失
  零深度被编码为近障碍的问题。修复后双环境 PPO/恢复训练 smoke 通过，
  相关回归 `24 passed`、`make check` 通过。记录空 BVH 原生退出及诊断
  占位几何方案，动态覆盖率、validity mask 与后端修复仍待后续完成。
- 2026-10-30：启动 P6 运行时验证，新增视觉 smoke 脚本；单环境默认 PLY
  与双环境 room1 manifest 均通过深度更新、CNN/PPO 更新、checkpoint
  重载和继续训练，双环境局部 reset 隔离通过。新增方法来源文档与
  BibTeX，区分 PPO/GAE、Spatial Softmax、3DGS 和本项目工程适配。
  深度质量、人工自遮挡、多资产组和规模性能仍待验证，P6 不标记完成。
- 2026-10-30：补全 README 中原生验收脚本的用法，增加旧 velocity 策略
  示例、采样时长与 JSON 输出、交互前端对照命令，以及自动验证边界。
- 2026-10-30：按用户明确决定将 P3 标记为完成，当前范围阶段收口。
  新版 wheel 由用户后续发布；人工对照、物理键鼠、完整 MJWarp 接触
  重放、长期资源、多场景和正式性能验证转为后续发布/加固任务。
  保留历史验收记录及未完成复选框，不将范围调整记为新增测试通过。
- 2026-10-30：用户确认外部 MuJoCo passive UI 问题已修复，将修复项
  标记完成；保留修复 wheel 发布/安装及 Show/Hide 与 F8 联动验收待办，
  尚未记录具体修复 commit 和包版本。
- 2026-10-30：记录 P3 人工交互发现的原生 3DGS Show/Hide 按钮置灰问题，
  确认其 UI 类别限制与 passive viewer 不兼容，F8 路径不受限制。
  新增外部控件修复、wheel 发布与依赖更新、按钮/F8 联动验收待办；
  不将已有程序回调测试当作完整 UI 验收。
- 2026-10-30：推进 P3 六项验收，新增真实窗口验收/性能脚本、native PLY
  几何往返测试和版本 1 接口契约。发现用户旧 velocity checkpoint 的
  98/99 维 observation 不兼容，记录直接加载失败并提供显式 phase 适配。
  四轮窗口自动控制、状态同步、清理及 CPU 碰撞显示接触重放通过；性能
  初测和原始 JSON 已保存。定向测试 `13 passed`、`make check` 通过。
  人工遮挡/交互、完整 MJWarp 重放和正式性能指标仍未完成，P3 不整体关闭。
- 2026-10-30：将 MuJoCo 本地 wheel 路径替换为远程 v1.0 Release URL，
  支持 Linux x86_64 Python 3.12/3.13，更新 uv 锁文件及文档，移除当前
  启动命令对 `--no-sync` 和相邻 MuJoCo 源码目录的依赖。
- 2026-10-29：新增原生 MuJoCo 3DGS G1 room1 有限步数冒烟脚本和交互
  入口，实际完成 renderer 加载与 200 步 tracking 策略仿真；定向回归
  `7 passed`，新增脚本格式、lint 和类型检查通过。更新 P3 为核心功能
  完成、原生 viewer 冒烟通过，补录外部仓库版本、硬件和命令；保留
  velocity checkpoint、Viser 对照、交互控制、接触不变性和性能验收。
- 2026-10-25：启动 P6，并明确复用 YAM Vision 而不是重新建设视觉训练
  框架。新增 G1 3DGS depth 训练任务、前下视 `64×48` CameraSensor、
  标准化 camera observation group、Spatial-Softmax CNN runner、静态配置
  回归和真实 room1 PLY 单环境 runtime slow test；普通 G1 任务保持无相机
  开销。由于相机与 terrain-height ray 共用 MJWarp BVH，首版移除
  `foot_height_scan` 及其两个奖励依赖，P7 再补任务专用足端指标。
- 2026-10-25：将原 P9 外部前端工作合并到 P3 Play。P3 现在统一管理
  Viser 主实现、视频路径以及外部 MuJoCo viewer/simulate 联调验收；本仓库
  仍不重复实现外部前端。
- 2026-10-25：在 P3 增加外部 MuJoCo viewer/simulate 3DGS 前端的联调
  验收 TODO，要求记录外部仓库版本，并验证 room1 manifest、velocity
  checkpoint、视觉/碰撞对齐、运行控制、资源释放和性能，防止后续遗漏。
- 2026-10-25：在 README 和 3DGS 指南中增加本地 G1 velocity checkpoint
  配合 room1 manifest、spawn point、Viser 和 collision 可视化的完整 Play
  示例，并说明零命令站立与 `--agent zero` 的区别。
- 2026-10-25：确认 3DGS NPZ 缓存不是训练或 Play 的功能前置条件；PLY
  只在场景初始化时解析一次，之后使用内存/GPU 数据。将原子缓存和失效
  策略转入 P10 后期性能优化，将碰撞生成 provenance 归入 P8，并正式
  完成 P5。
- 2026-10-25：开始 P5 第二阶段实现。manifest 现在会解析 collision MJCF
  的 mesh、heightfield、texture 等外部文件引用，并要求 dependencies 与之
  完全一致；选中的 spawn point 会与 `world_from_scene` 合成并应用到机器人
  reset 位置和朝向，Play/demo/train 支持 `--scene-spawn-point`。
- 2026-10-25：按当前项目范围将 P4 外部碰撞输入 MVP 阶段性收口。
  已完成预制 collision-only MJCF 的加载、验证、调试显示及基础物理测试；
  凹 mesh/单位诊断、PLY 对齐叠加和 room1 策略级定量指标留作后续加固，
  从 PLY 自动生成碰撞代理继续归入 P8，不再阻塞 P4。
- 2026-10-25：完成 P4 其余自动验证：新增落箱稳定性、穿透、速度和
  overflow 指标；验证 G1 velocity 外部碰撞环境。修复外部 terrain 根名
  与脚部 contact sensor 不兼容，以及 terrain-height sensor 仍读取旧
  plane group 导致空 BVH 的问题。开环 G1 静态站立未通过，保留为需要
  闭环站立策略的明确待办。
- 2026-10-25：用户确认 `run_g1_3dgs_collision_demo.py` 已完成 G1 在
  room1 场景中的连续运行；将该项计入 P4 人工验收，剩余落体、静态
  站立和定量接触稳定性验证。
- 2026-10-25：实现 scene manifest schema v1 和严格加载器；绑定 PLY、
  collision MJCF、dependency hash、统一 world transform 与 spawn metadata，
  接入 train/play/demo 并保留互斥的分开指定接口；room1 冒烟脚本改用
  manifest，消除 G1 默认资产的开发机绝对路径。
- 2026-10-25：在 Viser `Visualization` 面板增加 Collision 开关，运行中
  可实时切换 group 3，并与 `Groups > Geoms > G3` 的状态同步。
- 2026-10-25：根据首次 room1 Play 日志修复 worldbody 直挂 geom 的
  初始化崩溃；加载器会自动创建静态 `terrain` 根，并增加完整
  `Scene.initialize()` 回归测试。
- 2026-10-25：在 `asset_zoo/3dgs/room1/collision/` 增加可复现的地面与
  六柱碰撞 MJCF；增加 `tests/run_g1_3dgs_collision_demo.py`，复用标准
  demo checkpoint、motion 和 policy，同时加载 room1 PLY 与碰撞代理。
- 2026-10-25：实现 collision-only MJCF terrain MVP；增加训练配置以及
  Play/demo CLI，支持静态 primitive、多个凸 mesh、heightfield、plane
  互斥/显式保留、group 3 调试显示和 fail-fast 校验，并补充定向测试与文档。
- 2026-10-25：澄清场景 manifest 不是行业标准；统一采用
  `manifest.json`，MVP 阶段不依赖 manifest。
- 2026-10-25：确定 PLY、collision MJCF 和 mesh 的关系与标准目录布局；
  确定通过 `TerrainEntityCfg` 的新 MJCF 类型加载静态碰撞场景。
- 2026-10-24：在文档顶部增加阶段状态表、项目依赖树和
  每次开发后的
  强制更新规则。
- 2026-10-24：创建项目级 TODO；核对当前代码、测试、文档和需求；
  将“视觉 Play 已完成”与“视觉训练/碰撞尚未完成”明确分离；
  确定外部碰撞输入为下一里程碑。
