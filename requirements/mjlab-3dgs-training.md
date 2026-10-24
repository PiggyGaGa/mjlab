# mjlab 3DGS 训练模块需求

项目总体进度、已完成证据和后续执行顺序见 `mjlab-3dgs-todo.md`。

## 1. 文档目的

本文档定义 mjlab 在机器人训练阶段加载 3D Gaussian Splatting（3DGS）
环境、生成简化碰撞代理并执行强化学习训练的需求和任务拆解。

训练模块与策略回放模块是两个独立模块：

- 训练模块负责资产导入、碰撞生成、物理场景、传感器观测和训练；
- Play 模块负责加载训练产物并通过 Viser 展示策略运行效果；
- 训练命令不得隐式启动 Play，Play 命令也不得包含训练逻辑；
- 两个模块只通过稳定的场景资产契约、环境配置和 checkpoint 交接。

本阶段只维护需求和实施计划。收到明确执行命令后才开始实现。

## 2. 背景与技术基础

mjlab 已经具备 MuJoCo 场景组合、MuJoCo Warp 多 world GPU 仿真、
manager-based RL、RGB/深度相机和接触传感器。

当前 MuJoCo Warp 3.16 的 `create_render_context` 原生支持以下 splat 输入：

- position；
- rotation；
- scale；
- RGBA；
- splat group offset；
- 每个 world 的 splat group 选择。

训练模块必须复用这套原生能力，不实现自定义 Gaussian rasterizer。

3DGS 只提供视觉表达，不提供 MuJoCo 可用的碰撞表面。因此训练环境由
两个空间对齐的部分组成：

1. 用于 RGB/深度观测的 3DGS 视觉资产；
2. 用于动力学、接触和奖励计算的简化碰撞代理。

## 3. 目标

1. 支持把标准 3DGS PLY 转换为版本化的 mjlab 场景资产。
2. 使用 MuJoCo Warp 原生 splat API 渲染训练相机观测。
3. 从 3DGS 几何信息生成稳定、可简化、可复现的碰撞代理。
4. 保证视觉资产、碰撞代理、机器人出生点使用统一坐标变换。
5. 支持单环境验证和多环境并行训练。
6. 支持纯本体感觉训练和视觉训练，两者不产生不必要的相互开销。
7. 训练产物能够由独立 Play 模块加载，但训练过程不依赖任何 viewer。

## 4. 范围

### 4.1 范围内

- 3DGS PLY 读取、校验、规范化和缓存；
- splat 字段到 MuJoCo Warp 原生参数的适配；
- 3DGS 到简化碰撞代理的离线生成；
- 视觉与碰撞坐标对齐和质量验证；
- 3DGS 静态环境实体与训练场景配置；
- 共享场景和按 world 选择场景；
- RGB/深度视觉观测；
- 强化学习训练、checkpoint 和复现信息记录；
- 测试、性能基准、示例和训练文档。

### 4.2 范围外

- 3DGS 模型训练或从照片/视频重建 3DGS；
- 动态或可变形 3DGS；
- Gaussian ellipsoid 直接参与 MuJoCo 碰撞；
- 自定义 Gaussian CUDA/Warp rasterizer；
- Viser、MuJoCo viewer 或其他前端回放界面；
- checkpoint 的交互式播放；
- 外部仓库中 MuJoCo viewer/simulate 3DGS 功能的实现与维护。

## 5. 模块边界与交接契约

训练模块建议输出以下内容：

```text
scene_name/
  manifest.json
  splats.npz
  collision/
    collision.xml
    meshes/                  # 可选；一个或多个凸分解网格
    heightfields/            # 可选；2.5D 地形高度场
  collision_report.json
  preview.png              # 可选

training_run/
  checkpoint.pt
  resolved_env_config.*
  scene_manifest_ref.*
```

Play 模块只能依赖公开的环境构建 API、场景 manifest 和 checkpoint，
不得导入训练循环的内部实现。

`manifest.json` 至少记录：

- schema 版本；
- 源文件指纹；
- 字段解释和数组 dtype；
- splat 数量和包围盒；
- 坐标变换；
- 碰撞生成参数和输出指纹；
- 明确配置的机器人出生点；
- 资产来源和许可证信息（若提供）。

## 6. 功能需求

### TR-FR-1：3DGS 导入与规范化

- 首版支持一种严格定义的标准 3DGS PLY。
- 至少读取位置、旋转、尺度、不透明度和颜色所需字段。
- 输出 MuJoCo Warp 所需的以下数组：
  - `position: (N, 3)`；
  - `rotation: (N, 4)`；
  - `scale: (N, 3)`；
  - `rgba: (N, 4)`。
- 必须明确四元数顺序、log-scale 激活、opacity 激活和颜色范围。
- 仅有球谐颜色时，首版至少支持从 DC 分量生成静态 RGB。
- 高阶视角相关球谐若不支持，必须显式告警或拒绝，不能静默误解释。
- 必须拒绝缺失字段、NaN/Inf、非法尺度、非法四元数和空资产。
- 支持 opacity、尺度、包围盒和最大 splat 数量过滤。
- 规范化缓存包含 schema、源指纹和全部转换参数。
- 缓存写入必须是原子的；源数据或参数变化时必须失效。

### TR-FR-2：坐标与单位

- 目标单位为米，世界上方向与 MuJoCo 一致。
- 支持平移、旋转和统一缩放。
- 同一变换必须同时作用于 splat、碰撞代理和出生点。
- 非统一缩放首版不支持；使用时必须显式报错。
- 转换报告记录完整变换、源/目标包围盒和尺度。
- 提供坐标轴、包围盒、出生点和碰撞代理的叠加检查方式。

### TR-FR-3：MuJoCo Warp 原生渲染

- 通过 `mujoco_warp.create_render_context` 的原生 splat 参数接入。
- 支持零个、一个或多个规范化 3DGS 资产。
- 支持 `splat_adr` 和 `splat_group_id`。
- 未配置 3DGS 时，现有相机、分割和 raycast 行为保持不变。
- RGB 必须正确合成 3DGS 和 MuJoCo 几何体的遮挡关系。
- 深度必须遵循并记录当前 MuJoCo Warp 的 splat 深度语义。
- 首版可不为 splat 分配 segmentation id，但不得误标为 MuJoCo geom。
- 静态场景只在初始化时构建 splat BVH，不做无意义的逐 step refit。

### TR-FR-4：碰撞代理生成

- 提供独立离线命令生成碰撞代理。
- 管线至少包含：
  1. opacity、尺度和空间范围过滤；
  2. 根据 Gaussian 中心、尺度和方向构造带权空间样本；
  3. 离群点去除和体素降采样；
  4. 表面或占据重建；
  5. 小型悬浮连通分量清理和可配置孔洞修复；
  6. 网格修复、法线统一和简化；
  7. MuJoCo Warp 碰撞兼容性验证；
  8. 质量与性能报告。
- 正式选择算法前，至少比较以下方案：
  - SplatTransform voxel/SVO 与 collision mesh，作为公开工程基线；
  - center-only voxel 与利用 scale、rotation、opacity 的 covariance-aware
    occupancy；
  - voxel occupancy/TSDF 与 marching cubes；
  - heightfield 与 mesh/primitive 的混合表示；
  - 原始图像和相机可用时的表面重建方法，与仅 PLY 输入的方法分组比较；
  - 当前 MuJoCo Warp 支持条件下的 SDF/mesh 表示。
- 选择依据包括薄结构、孔洞、漂浮点、脚底接触稳定性、面数、
  生成耗时、仿真吞吐和依赖成本。
- 允许配置裁剪范围、地面方向、高度和保留区域。
- 相同输入、参数和种子必须产生相同结果。
- 允许用户提供碰撞 mesh 覆盖自动生成结果。
- 普通 voxelization、marching cubes 或调用现有工具只能作为工程能力，不能在
  缺少方法增量和定量证据时宣称为碰撞算法创新。

### TR-FR-4A：外部碰撞资产输入

自动简化算法实现前，系统必须先支持把已经准备好的碰撞资产独立接入
3DGS 场景，以便分别验证视觉、物理和坐标对齐链路。

- MVP 的规范输入格式为 collision-only MJCF（`collision.xml`），而不是单个
  任意凹 OBJ。MJCF 作为容器可以引用多个凸 OBJ/STL、MuJoCo primitive 和
  heightfield，并能保存 friction、contact dimension、priority、`contype` 和
  `conaffinity` 等物理属性。
- 碰撞 MJCF 只允许包含静态环境资产和固定 body；不得包含 actuator、自由关节、
  机器人或训练逻辑。
- 完整 3D 室内环境应使用多个凸碰撞 geom 的并集。单个 mesh 只有在已知为凸体时
  才能直接作为碰撞输入；不得把任意凹场景 mesh 的凸包误认为真实碰撞表面。
- 2.5D 地面优先允许使用 heightfield；primitive、凸 mesh、heightfield 可以在
  同一个 collision-only MJCF 中混合。
- SDF 是后续可选输入形式，首版不作为 MVP 的强制依赖。
- 碰撞资产必须在 `Scene.compile()` 前加入 MuJoCo 场景；它是物理模型输入，
  不是策略 observation。若策略需要接触信息，应继续通过 contact sensor 或
  现有 observation manager 显式暴露。
- 配置和命令行至少提供等价于 `collision_mjcf_file` /
  `--collision-mjcf-file` 的入口。未配置外部碰撞时，现有 plane 行为保持不变。
- 启用外部碰撞后，是否保留 plane 必须显式配置；默认不得同时保留一个与外部
  地面重叠的隐藏 plane，以免产生双重接触。
- 3DGS PLY 与碰撞 MJCF 必须共享单位、坐标轴和 world transform。MVP 要求输入
  资产已经对齐且单位为米；后续由统一 scene manifest 对两者应用同一刚体变换。
- 初始化时必须校验文件存在、MJCF 可编译、至少存在一个启用碰撞的 geom、环境
  body 固定、引用资产完整以及碰撞类型受当前 MuJoCo Warp 支持。
- 碰撞 geom 的视觉显示与物理启用相互独立：默认不在最终画面显示，但必须可以
  通过调试开关显示半透明碰撞表面。
- MVP 中所有并行 world 共享同一套静态碰撞资产。不同 world 使用不同碰撞场景
  属于后续的多场景扩展。

### TR-FR-5：碰撞质量

- 输出输入/保留 splat 数、顶点/面数、连通分量、包围盒和简化比例。
- 检查退化面、重复面、非有限顶点和明显非流形问题。
- 提供 3DGS 与半透明碰撞代理的叠加预览。
- 输出碰撞表面到高置信度 Gaussian 样本的距离统计。
- 提供落体、静态接触、机器人站立或行走的物理冒烟测试。
- 不得出现主要地面穿透、不可接受抖动或数值爆炸。

### TR-FR-6：训练场景配置

- 提供类型明确的 3DGS 场景资产配置。
- 配置包含视觉缓存、碰撞资产、统一变换、group、加载选项和渲染开关。
- 3DGS 环境作为静态场景实体，与机器人、灯光和传感器组合。
- 环境原点和机器人出生点必须显式配置。
- 碰撞摩擦、接触维度、优先级和 mask 可配置。
- MVP 支持所有 world 共享同一场景。
- 扩展模式支持每个 world 选择不同但视觉/物理严格匹配的场景。

### TR-FR-7：训练行为

- 继续使用现有 manager-based RL 框架，不建立第二套训练框架。
- 纯本体感觉任务默认不创建相机渲染工作负载。
- 视觉任务按配置的分辨率和更新频率执行渲染。
- 相机输出继续使用当前 GPU tensor 路径。
- reset、出生点随机化、命令、reward、termination 和 contact sensor 正常工作。
- 多 world 机器人状态相互隔离。
- 提供至少一个最小训练示例。
- 训练日志记录 3DGS 资产、碰撞资产和转换配置指纹。
- checkpoint 必须能由独立 Play 模块加载。

### TR-FR-8：命令行与错误处理

- 导入、碰撞生成和校验提供独立命令行入口及 `--help`。
- 长任务显示阶段和进度。
- 失败时不留下可被误认为有效的完整缓存。
- 缺少可选依赖时给出明确安装说明。
- 视觉/碰撞 manifest 不匹配时 fail fast。
- splat 超限、显存不足和格式不兼容时给出上下文信息。

### TR-FR-9：最小视觉闭环验证任务

首个端到端训练任务定义为 G1 的“随机起点单级台阶视觉跨越”
（3DGS-based Perceptive Single-Step Traversal）。该任务是验证平台能力的 MVP，
不以完整楼梯、跑酷、导航或操作为首个目标。

#### TR-FR-9.1：任务场景

- 视觉场景由 3DGS PLY 表示，物理场景由与其严格对齐的静态台阶碰撞表示。
- 自动碰撞生成完成前，允许使用手工编写的 box 或 collision-only MJCF 台阶。
- 机器人使用 G1，动作空间沿用现有 velocity locomotion 任务，避免同时更换底层
  控制接口。
- 相机安装在头部或胸部并朝前下方观察台阶和近场落脚区域。
- 同一个视觉/碰撞资产对中不得只移动碰撞台阶。MVP 通过随机机器人出生位置和
  朝向改变相对关系，以保持 3DGS 与碰撞严格对齐。
- 不同台阶高度必须分别提供严格对齐的视觉/碰撞资产对，并通过多场景选择进行
  随机化；不得在固定 PLY 下单独缩放或替换碰撞台阶。

#### TR-FR-9.2：视觉闭环成立条件

- 随机化机器人到台阶的初始距离、初始朝向和前进速度指令。
- 训练场景至少混合平地和多种台阶高度，使固定时间抬脚策略不能解决任务。
- 不向非特权策略直接提供台阶距离、高度、碰撞 mesh 或真值高度图。
- 成功策略应在撞击台阶立面前，根据逐帧视觉观测提前调整步态和足端净空。
- 相机观测必须在控制过程中持续更新；只在 episode 开始读取一次图像不算视觉
  闭环。

MVP 先使用一个固定高度的台阶，建议随机出生距离 `0.6–1.5 m`、偏航 `±10°`。
多场景阶段再加入平地、`8 cm`、`12 cm`、`16 cm` 台阶资产对。具体数值必须根据
G1 的安全可达范围通过冒烟测试确认后再固定。

#### TR-FR-9.3：Observation 与训练顺序

- 基础 observation 包含现有 proprioception、velocity command 和 previous
  action。
- 第一阶段增加 3DGS CameraSensor 的 depth 或其紧凑编码，优先验证几何闭环。
- 第二阶段增加 RGB-D，对颜色、曝光和背景进行必要随机化。
- 第三阶段才评估纯 RGB 策略；不得把纯 RGB 训练作为碰撞输入 MVP 的阻塞项。
- privileged teacher 可以读取真值台阶参数或高度图，但 student 和最终部署策略
  不得读取这些特权信息。

#### TR-FR-9.4：奖励、终止与成功定义

- 复用现有 velocity tracking、姿态稳定、能耗和平滑性奖励。
- 增加平台到达、稳定足端接触和合理足端净空奖励。
- 惩罚脚尖撞击立面、非足端碰撞、过大接触冲量、滑动和明显穿透。
- 摔倒、超时、长期停滞或越出允许区域时终止 episode。
- MVP 成功条件为 base 越过台阶边缘、双脚在上平台形成稳定接触，并保持站立
  至少一个可配置时间窗口。

#### TR-FR-9.5：对照实验与指标

必须在相同训练预算和测试场景上比较：

1. 仅 proprioception 的 blind policy；
2. proprioception 加真值台阶参数或高度图的 privileged upper bound；
3. 普通 mesh 渲染 depth 的视觉策略；
4. 3DGS 渲染 depth/RGB 的视觉策略。

至少记录成功率、摔倒率、脚尖/非足端碰撞次数、最大穿透、接触冲量、足端净空、
速度跟踪误差、训练吞吐和推理 real-time factor。测试集必须包含训练范围内场景和
未见过的出生位置/台阶参数。

为定量评价视觉—碰撞偏差，应对碰撞台阶施加受控的高度或位置偏移，例如
`±1 cm`、`±2 cm`、`±5 cm`，报告各指标随偏差的变化。该实验只用于评估，正常
训练资产仍必须严格对齐。

#### TR-FR-9.6：递进路线

1. 固定台阶资产，随机机器人出生位置；
2. 平地与多种单级台阶混合；
3. 上台阶、平地和下台阶混合；
4. 连续两到三级楼梯；
5. 窄平台、间隙或 sparse footholds；
6. 使用真实重建 3DGS 和自动生成的碰撞代理复现实验。

只有前一阶段通过既定验收标准后才进入下一阶段，避免动作技能、视觉表示和碰撞
质量问题同时出现而无法归因。

## 7. 非功能需求

### TR-NFR-1：性能

- 非视觉训练不得产生逐 step 3DGS 渲染开销。
- 基准覆盖 splat 数、相机分辨率、相机数量、更新频率、碰撞面数和 world 数。
- 多 world 尽量共享只读视觉资产。
- 性能阈值根据目标 GPU 和真实场景基准确定。

### TR-NFR-2：兼容性与维护

- 无 3DGS 的现有任务、相机和 raycast 无回归。
- 不修改 MuJoCo Warp 安装包源码。
- 资产 I/O、碰撞生成、场景配置和传感器接入分层实现。
- 公共 API 有类型标注并通过 `make check`。
- 测试使用小型、可再分发的合成资产。
- 大型真实资产不进入普通 Git 历史。

### TR-NFR-3：安全

- 不执行资产中的代码或命令。
- 解析前检查文件大小、元素数量和内存上限。
- 示例资产记录来源和许可证。

## 8. 验收标准

### TR-AC-1：导入与缓存

小型标准 PLY 能生成缓存和 manifest；重复运行命中缓存；修改源或参数后
缓存失效。非法输入产生明确错误。

### TR-AC-2：渲染观测

固定相机中能看到 splat，机器人在 splat 前后移动时遮挡正确，RGB/深度
能够进入 observation pipeline。

### TR-AC-3：碰撞与对齐

刚体和机器人不能穿过主要地面；静态接触稳定；叠加预览和误差指标证明
视觉与碰撞使用相同坐标。

外部 collision-only MJCF 能够在不启用自动简化算法的情况下加载。仅配置视觉
PLY 时继续使用显式选择的 plane fallback；配置外部碰撞时机器人接触来自该资产，
且不会因隐藏 plane 重叠产生双重接触。凹场景作为单 mesh 输入时必须被拒绝或给出
明确的凸包语义告警。

### TR-AC-4：训练

示例任务可以短时训练并保存 checkpoint；纯本体感觉配置不调用相机渲染；
视觉配置能够读取含 3DGS 的 GPU 图像。

### TR-AC-5：多环境

多个 world 可独立 reset 和 step。启用多场景后，视觉 group 与物理场景
严格对应。

### TR-AC-6：回归与质量

无 3DGS 配置时现有关键测试通过；`make check` 和最终 `make test` 通过；
训练模块不依赖 Viser 或其他 viewer 的运行状态。

### TR-AC-7：单级台阶视觉闭环

- G1 能使用持续更新的 3DGS 相机观测完成随机起点的单级台阶跨越。
- 3DGS 视觉策略相对 blind policy 在随机台阶测试集上表现出可复现的性能增益；
  如果没有增益，不能宣称视觉闭环已经验证。
- 相机冻结、遮挡或关闭时性能应出现可解释的下降，以证明策略实际使用视觉，而
  不是依赖固定时序或记忆场景。
- mesh depth、3DGS depth/RGB 和 privileged upper bound 的差异有完整报告。
- 视觉—碰撞偏移实验能量化对落脚、碰撞、穿透和任务成功率的影响。

## 9. 需求拆解

### Training Phase 0：技术验证

- TR-T0.1：用最小数组验证 MuJoCo Warp RGB、深度和遮挡。
- TR-T0.2：验证 CUDA graph 和多 world splat group。
- TR-T0.3：固定标准 PLY 字段、四元数、scale、opacity 和 SH DC 约定。
- TR-T0.4：比较 voxel/TSDF、heightfield 混合和 SDF/mesh 碰撞路径。
- TR-T0.5：测试碰撞复杂度对稳定性和吞吐的影响。
- TR-T0.6：形成 MVP 格式、缓存和碰撞算法决策记录。

交付物：最小实验、基准结果和 API 草案。

### Training Phase 1：资产导入

- TR-T1.1：定义配置、manifest 和 schema 版本。
- TR-T1.2：实现安全 PLY 读取和严格校验。
- TR-T1.3：实现 Gaussian 字段规范化。
- TR-T1.4：实现过滤、坐标变换和统计。
- TR-T1.5：实现指纹、缓存和原子写入。
- TR-T1.6：增加 CLI、fixture 和单元测试。

依赖：Training Phase 0。

### Training Phase 2：原生渲染接入

- TR-T2.1：场景/传感器配置声明 3DGS 资产。
- TR-T2.2：传递 splat 数组、offset 和 group id。
- TR-T2.3：保持无 3DGS、raycast、RGB、深度和分割路径兼容。
- TR-T2.4：实现静态 BVH 生命周期。
- TR-T2.5：增加遮挡、深度和多 world 测试。
- TR-T2.6：记录显存、初始化和渲染吞吐。

依赖：Training Phase 1。

### Training Phase 3：碰撞代理

- TR-T3.1：实现选定的过滤和表面重建算法。
- TR-T3.2：实现网格修复和简化。
- TR-T3.3：实现 MuJoCo 资产导出与加载校验。
- TR-T3.4：实现用户 mesh 覆盖和接触参数配置。
- TR-T3.5：实现误差、统计和复现报告。
- TR-T3.6：增加确定性、网格质量和物理测试。

依赖：Training Phase 0、1。

### Training Phase 3A：外部碰撞输入 MVP

- TR-T3A.1：定义 collision-only MJCF 约束和场景资产配置。
- TR-T3A.2：在场景编译前加载静态碰撞 entity。
- TR-T3A.3：实现外部碰撞与 plane fallback 的互斥/显式共存语义。
- TR-T3A.4：实现文件、固定 body、collision geom、资产引用和类型校验。
- TR-T3A.5：实现共享 transform、包围盒和调试可视化检查。
- TR-T3A.6：增加 primitive、凸 mesh、heightfield、无效凹 mesh 和多 world 测试。
- TR-T3A.7：增加 CLI 使用说明、README 最小示例和 changelog 条目。

该阶段不生成或简化碰撞几何，只建立稳定的碰撞输入接口。它应先于自动碰撞代理
生成完成，使后续算法只需输出符合契约的 MJCF 资产包。

依赖：Training Phase 0、1；可与 Training Phase 2 并行。

### Training Phase 4：统一场景

- TR-T4.1：绑定视觉资产与碰撞代理。
- TR-T4.2：统一应用坐标变换。
- TR-T4.3：实现原点、出生点和 group 映射。
- TR-T4.4：实现对齐预览。
- TR-T4.5：增加指纹、group 和变换不匹配测试。
- TR-T4.6：提供场景 demo。

依赖：Training Phase 1、2、3A。Training Phase 3 生成的自动碰撞代理在接口稳定后
替换手工碰撞资产，不阻塞首个视觉闭环 MVP。

### Training Phase 5：训练闭环

- TR-T5.1：基于 G1 velocity task 建立单级台阶任务和手工碰撞基线。
- TR-T5.2：增加前向下视 CameraSensor 及 depth observation 编码。
- TR-T5.3：实现出生位置、朝向、命令和台阶课程随机化。
- TR-T5.4：实现台阶成功、足端净空、非法接触、穿透和终止指标。
- TR-T5.5：分别训练 blind、privileged、mesh vision 和 3DGS vision 基线。
- TR-T5.6：执行相机冻结/关闭实验，确认策略真实依赖视觉。
- TR-T5.7：执行视觉—碰撞位置和高度偏移敏感性实验。
- TR-T5.8：验证纯本体感觉路径不渲染，视觉路径兼容 CUDA graph。
- TR-T5.9：记录资产、配置、随机种子、checkpoint 和全部评估指标。
- TR-T5.10：执行单环境、多环境和目标规模性能基准。
- TR-T5.11：导出 checkpoint，并由独立 Play 模块复现实验 episode。

依赖：Training Phase 4。

### Training Phase 6：质量收尾

- TR-T6.1：完成多场景视觉/物理映射。
- TR-T6.2：增加真实资产端到端测试。
- TR-T6.3：完成异常中断、缓存和资源上限测试。
- TR-T6.4：补充训练教程和故障排查。
- TR-T6.5：更新 changelog。
- TR-T6.6：运行 `make check` 和 `make test`。

依赖：前述训练阶段。

## 10. 待确认事项

1. 首个真实资产由哪个 3DGS 工具导出？
2. 首个真实场景中是否存在适合构建单级台阶实验的区域？
3. 视觉策略首版使用原始 depth、降采样 depth，还是图像 encoder 输出？
4. 场景是 2.5D 室外环境还是完整 3D 室内环境？
5. 目标 GPU、splat 数量、world 数和期望吞吐是多少？

已经确认首个机器人为 Unitree G1，首个端到端任务为随机起点单级台阶视觉跨越，
并采用 depth-first、RGB-D/RGB 后续扩展的实施顺序。

未知时默认采用标准 3DGS PLY、静态完整 3D 环境、共享场景、可选视觉
观测，并从 Training Phase 0 开始。

