# mjlab 3DGS 项目 TODO 与进度

最后更新：2026-10-25

## 0. 项目总览

### 0.1 阶段状态表

| 阶段 | 工作包 | 状态 | 下一交付物 |
|---|---|---|---|
| P0 | 需求与调研 | 第一轮完成 | 持续补充论文材料 |
| P1 | 3DGS 资产导入 | 基础完成 | 资源上限与导入优化 |
| P2 | MJWarp 原生渲染 | 基础完成 | 训练性能与深度验证 |
| P3 | Play 前端（Viser/外部 viewer） | 基础完成 | 外部 MuJoCo viewer 联调验收 |
| P4 | 外部碰撞输入 | **MVP 完成，阶段收口** | 后续由 P8 生成碰撞资产 |
| P5 | 统一场景资产 | **完成** | 进入 P6 视觉训练 |
| P6 | YAM Vision → G1 视觉训练适配 | **进行中** | depth runtime smoke test |
| P7 | 单级台阶任务 | 未开始 | 对齐资产和四组基线 |
| P8 | 自动碰撞生成 | 研究规划 | collision generator |
| P10 | 性能与发布 | 部分完成 | check/test/docs 与 benchmark |
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
│   └── P3 Play 前端：Viser、视频、外部 viewer     [基础完成]
├── 物理场景
│   ├── P4 外部 collision-only MJCF                [MVP 完成，阶段收口]
│   ├── P5 PLY/MJCF manifest 与统一 transform      [完成]
│   └── P8 3DGS 自动碰撞代理生成                   [研究规划]
├── 视觉闭环训练
│   ├── P6 YAM Vision → G1 depth observation       [进行中]
│   ├── P7 单级台阶视觉跨越                        [未开始]
│   └── 多台阶、间隙、真实重建场景                 [后续]
└── 质量与论文
    ├── P10 测试、文档、性能和发布                 [部分完成]
    └── P11 基线、消融、论文材料                   [持续进行]
```

关键执行依赖：

```text
P4 外部碰撞输入
  └── P5 统一视觉/碰撞资产
      └── P6 训练视觉 observation
          └── P7 单级台阶闭环
              ├── 视觉—碰撞偏差实验
              └── P8 自动碰撞代理替换手工碰撞
```

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
- 已支持通过 `TerrainEntityCfg(terrain_type="mjcf")` 加载静态
  collision-only MJCF；训练、Play 和 demo 均可用外部碰撞替换 plane。
- 已支持 primitive、多个凸 mesh、heightfield、相对资产路径、静态性
  校验，以及 group 3 碰撞代理调试显示。
- 已提供 room1 地面与六柱碰撞资产及基于标准 demo policy 的人工
  Play 冒烟脚本。
- 已提供 schema v1 scene manifest，将 PLY、collision MJCF、SHA-256、
  坐标约定、统一 transform 和 spawn-point metadata 绑定为一个入口；
  train/play/demo 均可加载。
- 当前训练任务仍不加载 PLY 或 3DGS CameraSensor，策略尚未使用视觉
  observation。

### 尚未达到的能力

- 独立落体和定量接触稳定性已通过；G1 在 room1 外部碰撞场景中的
  连续运行已由用户人工确认。velocity 闭环策略已在普通 plane 上完成
  32 环境、20 秒、零跌倒的静态站立验证；room1 上的策略级定量接触
  指标留作后续加固，不阻塞 P4 MVP 收口。
- 尚未让 G1 训练策略读取 3DGS CameraSensor observation。
- 尚未建立单级台阶视觉闭环训练任务。
- 尚未自动从 3DGS 生成碰撞代理。
- 尚未完成可选预处理缓存和正式性能基准；二者属于后期性能优化，不阻塞
  当前功能里程碑。

### 当前首要目标

P4 已暂时收口，P5 已完成。当前进入 P6：不重新建设视觉强化学习框架，
而是复用仓库已有的 YAM Vision 路径，将 CameraSensor、标准化 depth、
二维 observation group、Spatial-Softmax CNN 和 RSL-RL 接线适配到 G1；
之后准备严格对齐的单台阶 PLY/MJCF 资产对并进入 P7。从 PLY 自动生成
碰撞代理属于 P8。NPZ 预处理缓存仅优化初始化时间，不阻塞视觉训练，
转入 P10 后期性能优化。

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
- [x] 确定首个视觉闭环任务为随机起点单级台阶跨越，采用
  depth-first 路线。

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

- [ ] 记录 splat depth 的精确定义、遮挡边界和已知误差。
- [ ] 验证不同相机数量、分辨率和更新频率下的显存及吞吐。
- [ ] 验证视觉训练路径的 CUDA graph 行为。
- [ ] 建立 segmentation 能力边界和测试。

### D. P3 Play 前端（Viser 与外部 viewer）

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

- [ ] 定义供外部 MuJoCo viewer/simulate 仓库使用的稳定场景与状态接口。
- [ ] 记录外部仓库地址、测试分支/commit、依赖版本和启动命令。
- [ ] 使用 room1 `manifest.json` 同时加载 PLY、collision MJCF 和
  `default` spawn point。
- [ ] 加载本地 G1 velocity checkpoint，确认机器人状态和策略运动连续
  同步到 viewer/simulate。
- [ ] 确认 3DGS、机器人 mesh 和 collision proxy 的坐标、尺度、朝向及
  遮挡关系与 Viser 结果一致。
- [ ] 验证 collision proxy 的显示/隐藏不会改变真实物理接触。
- [ ] 验证交互相机、暂停/继续、reset、退出和资源释放。
- [ ] 对比无 3DGS 与启用 3DGS 的 Actual RT、帧率、显存和加载时间，
  保存测试日期、硬件、命令和结果。

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
  `37 passed`；room1 真实 PLY/MJCF hash 校验通过，`make check` 与
  `make docs` 均通过。
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
- [x] 记录 visual/collision 主资产 hash、单位和 schema
  版本。
- [x] 对主资产和 manifest 已声明 dependency 实现 hash 不匹配 fail fast。
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
- [x] 增加 manifest schema、hash、路径边界、transform 和不匹配测试。
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

P6 只负责把该链路迁移到 G1 velocity task，并保持普通 G1 任务完全不创建
相机或渲染开销。首个注册任务为
`Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth`。

### V0. YAM Vision 复用与静态配置

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

- [ ] 编译单环境场景，确认相机安装位置、视线方向与 robot self-occlusion。
- [x] 验证默认 room1 PLY 能进入单环境 depth observation。
- [ ] 验证通过 `--scene-manifest` 覆盖的 PLY 能进入 depth observation。
- [x] 验证单环境 depth 的 shape 为 `(1, 1, 48, 64)`、dtype 为
  `float32`、范围为 `[0, 1]`，且图像不是常数。
- [ ] 验证 depth 随机器人运动持续更新。
- [ ] 运行 RSL-RL CNN 前向与短 rollout，确认 actor/critic observation
  group 和 checkpoint 保存/加载正常。
- [ ] 增加冻结帧或逐帧差异检查，避免策略读取到不更新的图像。

### V2. 性能与并行训练

- [ ] 验证多 world splat group、camera tensor 和 reset 隔离。
- [ ] 验证视觉 observation 与 CUDA graph 的兼容性。
- [ ] 测量不同 splat 数、分辨率、更新频率和 world 数下的吞吐及
  显存。
- [x] 首版采用 YAM Vision 的独立 Spatial-Softmax CNN encoder，不把 depth
  展平后直接送入 MLP。
- [ ] 为渲染失败、显存不足和无效相机配置增加错误处理。

### V3. 后续训练增强（不阻塞 P6 MVP）

- [ ] 对比 actor/critic 都读取相机与 asymmetric critic（只读特权状态）。
- [ ] 评估 privileged teacher → depth student 的 RSL-RL distillation。
- [ ] 增加深度噪声、延迟、低频更新、遮挡和 dropout。
- [ ] 评估 frame history 或 recurrent policy，避免单帧深度缺乏运动信息。

## 7. 首个端到端任务：G1 单级台阶视觉跨越

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

### G0. 算法基线

- [ ] 建立 center-only 点云/voxel 基线。
- [ ] 建立利用 Gaussian scale、rotation、opacity 的 covariance-aware 基线。
- [ ] 比较 voxel occupancy、TSDF 和 marching cubes。
- [ ] 比较 heightfield、mesh/primitive 混合表示。
- [ ] 复现或对照 SplatTransform 等公开工程方案。
- [ ] 评估 MuJoCo Warp 当前版本下的 SDF 路径。

### G1. 生成管线

- [ ] 在 collision report/manifest provenance 中记录生成参数、随机种子、
  工具名称和版本。
- [ ] 实现 opacity、scale、bounds 和离群点过滤。
- [ ] 实现体素降采样、连通分量清理和孔洞处理。
- [ ] 实现表面/占据重建。
- [ ] 实现网格修复、法线统一和简化。
- [ ] 导出 collision-only MJCF 和引用资产。
- [ ] 保证相同输入、参数和种子输出可复现。

### G2. 质量与任务感知简化

- [ ] 输出 splat、顶点、面、连通分量、包围盒和简化统计。
- [ ] 检查退化面、重复面、非有限数据和明显非流形问题。
- [ ] 计算碰撞表面与高置信度 Gaussian 的距离指标。
- [ ] 研究 Gaussian scale、rotation、opacity 对碰撞表面的贡献。
- [ ] 优先保留地面、台阶、扶手、落脚区和可操作表面。
- [ ] 建立几何误差、接触稳定性、策略性能和仿真吞吐的联合
  评价。

## 10. 性能、质量与发布

- [ ] **后期优化：**实现版本化 `splats.npz` 预处理缓存、原子写入，
  并在源 PLY、导入参数、transform 或 schema 变化时自动失效。
- [ ] 为无 3DGS、Viser-only、MJWarp camera 和视觉训练分别建立正式
  benchmark。
- [ ] 记录初始化、PLY 解析、BVH、浏览器上传、physics、policy 和
  rendering 时间。
- [ ] 建立目标 GPU、splat 数、world 数、分辨率和目标吞吐矩阵。
- [x] 2026-10-25 运行 `make check` 通过。
- [ ] 运行完整 `make test`，包括经明确配置的 slow/real-asset 测试。
- [x] 运行 `make docs`。
- [ ] 检查当前所有 3DGS 修改，避免覆盖工作树中的用户改动。
- [ ] 将当前未提交的 3DGS 工作整理成范围清晰的 commit。
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

1. 完成 P6 CNN runner 构建、短 rollout、多环境隔离与性能验证。
2. 建立一个严格对齐的单台阶 PLY/MJCF 资产对。
3. 建立随机起点单级台阶 MDP，并先跑 blind/privileged 冒烟训练。
4. 训练 mesh vision 与 3DGS vision，对视觉闭环进行消融验证。
5. 完成视觉—碰撞偏差实验和正式性能基准。
6. 在稳定接口之上实现自动碰撞代理生成。
7. 扩展到多台阶、间隙、真实重建环境和论文实验。

## 14. 更新记录

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
