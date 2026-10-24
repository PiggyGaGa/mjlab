# mjlab 3DGS 项目 TODO 与进度

最后更新：2026-10-24

## 0. 项目总览

### 0.1 阶段状态表

| 阶段 | 工作包 | 状态 | 下一交付物 |
|---|---|---|---|
| P0 | 需求与调研 | 第一轮完成 | 持续补充论文材料 |
| P1 | 3DGS 资产导入 | 基础完成 | manifest 与缓存 |
| P2 | MJWarp 原生渲染 | 基础完成 | 训练性能与深度验证 |
| P3 | Viser Play | 基础完成 | 多场景和碰撞调试 |
| P4 | 外部碰撞输入 | **下一步** | collision-only MJCF MVP |
| P5 | 统一场景资产 | 未开始 | manifest、hash、transform |
| P6 | 视觉训练管线 | 未开始 | G1 depth observation |
| P7 | 单级台阶任务 | 未开始 | 对齐资产和四组基线 |
| P8 | 自动碰撞生成 | 研究规划 | collision generator |
| P9 | Play/外部前端 | 部分完成 | manifest 与稳定接口 |
| P10 | 性能与发布 | 部分完成 | check/test/docs 与 benchmark |
| P11 | 论文实验与写作 | 持续进行 | 基线、消融和论文草稿 |

“基础完成”表示核心链路已经实现，但扩展能力和最终验收
仍存在
TODO。只有详细任务、测试、文档和验收全部完成后，阶段才能标记
为“完成”。

### 0.2 项目依赖树

```text
mjlab 3DGS
├── P0 需求与调研                                  [第一轮完成]
├── 视觉基础
│   ├── P1 PLY 导入、规范化、多 world              [基础完成]
│   ├── P2 MuJoCo Warp RGB/depth                   [基础完成]
│   └── P3 Viser 原生 Play 与视频                  [基础完成]
├── 物理场景
│   ├── P4 外部 collision-only MJCF                [下一步]
│   ├── P5 PLY/MJCF manifest 与统一 transform      [未开始]
│   └── P8 3DGS 自动碰撞代理生成                   [研究规划]
├── 视觉闭环训练
│   ├── P6 G1 depth observation                    [未开始]
│   ├── P7 单级台阶视觉跨越                        [未开始]
│   └── 多台阶、间隙、真实重建场景                 [后续]
├── 展示与对接
│   └── P9 Viser 完善、外部 viewer 稳定接口        [部分完成]
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
- 当前训练配置仍是原来的 flat 任务；3DGS 只在 Play 配置加载，
  物理地面仍是隐藏 plane。

### 尚未达到的能力

- 尚不能从命令行或场景配置加载独立的 collision-only MJCF 环境。
- 尚未让 G1 训练策略读取 3DGS CameraSensor observation。
- 尚未建立单级台阶视觉闭环训练任务。
- 尚未自动从 3DGS 生成碰撞代理。
- 尚未完成版本化 scene manifest、预处理缓存和正式性能基准。

### 当前首要目标

先实现“外部碰撞输入 MVP”，即让已经准备好的 collision-only MJCF 在
`Scene.compile()` 前进入 MuJoCo/MuJoCo Warp 物理模型。自动碰撞简化暂不
阻塞该目标。完成后使用手工台阶碰撞建立 G1 单级台阶视觉闭环
任务。

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
- [ ] 建立版本化 NPZ/cache，当前环境初始化仍直接解析 PLY。
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

### D. Viser 原生 3DGS Play

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

**部分完成：**

- [ ] 为多个不同 splat 场景实现并验证 Viser world/group 原子切换。
- [ ] 对浏览器断连、重连和长期运行资源占用进行专项测试。
- [ ] 增加碰撞代理半透明显示开关。
- [ ] 对 splat 与机器人互相遮挡进行正式图像回归测试。

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

重要边界：

- 当前 `unitree_g1_flat_3dgs_env_cfg(play=False)` 直接返回普通 flat 配置。
- 当前的 train 命令不会加载 PLY、不会创建 3DGS CameraSensor，
  也不会使用视觉 observation。
- 当前 Play 中机器人仍与隐藏 plane 接触，而不是与重建场景接触。

### F. 视频录制

- [x] VideoRecorder 支持从指定 RGB CameraSensor 录制。
- [x] `g1_3dgs_video` sensor 只在同时请求视频和该 sensor 时按需启用。
- [x] 录制路径使用 MuJoCo Warp 原生 3DGS 合成，而 Viser 保持
  浏览器原生渲染。
- [x] 修正布尔参数用法，`--video` 与 `--video-sensor` 分别传递。
- [x] 增加按需 sensor、重复启用和视频 wrapper 测试。

### G. 文档

- [x] README 包含项目意义、当前边界和最小 demo。
- [x] `gaussian_splats.rst` 包含格式、配置、训练成本、Play、demo、
  多机器人、录制和性能说明。
- [x] API、sensor index、viewer 文档和 changelog 已接入 3DGS 内容。
- [x] 文档明确 3DGS 当前只负责视觉，不参与 physics、contact 或
  raycast。
- [ ] 构建完整 Sphinx 文档并检查链接与渲染结果。

### H. 当前验证证据

- [x] 用户已人工确认真实 PLY 能在 Viser 中原生显示。
- [x] 用户已人工确认移除普通 Play 的 MJWarp camera 后卡顿改善。
- [x] 用户测得同一机器上稳态 Actual RT 约为：无 3DGS `0.6`，Viser 3DGS
  `0.53`。这是初步观察，不是正式 benchmark。
- [x] 2026-10-24 运行 7 个相关测试文件：`43 passed`。
- [x] 测试覆盖 importer、render context、Play/demo 配置、video、Viser、
  velocity 和 tracking 注册。
- [ ] 本轮尚未运行真实大 PLY slow test。
- [ ] 本轮尚未运行 `make check`、完整 `make test` 和 `make docs`。

本地受限环境运行测试需要将缓存放到可写目录，并关闭系统 ROS
pytest 插件自动加载：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
UV_CACHE_DIR=/tmp/mjlab-uv-cache \
WARP_CACHE_PATH=/tmp/mjlab-warp-cache \
uv run pytest <tests>
```

## 4. 下一里程碑：外部碰撞输入 MVP

目标：不考虑如何简化 3DGS，先让外部准备好的碰撞几何成为
稳定的软件输入。

### C0. 输入契约

- [ ] 定义 collision-only MJCF 配置类型及公共 API。
- [ ] 规范输入支持静态 primitive、多个凸 mesh 和 heightfield。
- [ ] 明确 SDF 是否在首版暴露；默认不作为 MVP 阻塞项。
- [ ] 定义碰撞 friction、contact dimension、priority、`contype` 和
  `conaffinity` 的覆盖规则。
- [ ] 定义视觉 PLY 与碰撞 MJCF 的单位、坐标轴和 transform 契约。

### C1. 场景加载

- [ ] 在 `Scene.compile()` 前把 collision-only MJCF 作为静态 entity
  加入场景。
- [ ] 保证碰撞资产不创建自由关节、actuator 或额外机器人。
- [ ] 实现未提供外部碰撞时的 plane fallback。
- [ ] 实现提供外部碰撞时默认关闭隐藏 plane，避免双重接触。
- [ ] 允许显式保留 plane，仅用于调试或过渡。
- [ ] 所有并行 world 首版共享同一静态碰撞场景。

### C2. CLI 与错误处理

- [ ] 增加等价于 `--collision-mjcf-file` 的配置/命令行入口。
- [ ] 增加显式 plane fallback/keep-plane 参数。
- [ ] 校验文件存在、引用资产存在和 MJCF 可以编译。
- [ ] 校验至少存在一个启用碰撞的 geom。
- [ ] 拒绝动态 body、free joint 和 actuator。
- [ ] 对单个任意凹 mesh 给出凸碰撞语义告警或拒绝。
- [ ] 对 unsupported geom、错误单位和明显包围盒不匹配给出上下文
  错误。

### C3. 调试与测试

- [ ] 支持在 Viser 中显示/隐藏半透明碰撞代理。
- [ ] 增加 PLY 与碰撞包围盒、坐标轴和出生点叠加检查。
- [ ] 增加 primitive 碰撞加载测试。
- [ ] 增加多个凸 mesh 碰撞加载测试。
- [ ] 增加 heightfield 碰撞加载测试。
- [ ] 增加 plane fallback、plane 互斥和非法 MJCF 测试。
- [ ] 增加多 world 共享碰撞测试。
- [ ] 增加落体、机器人站立和接触稳定性冒烟测试。
- [ ] 更新 README、3DGS 指南和 changelog。

## 5. 统一场景资产与缓存

- [ ] 定义版本化 scene manifest schema。
- [ ] manifest 绑定 visual PLY/cache、collision MJCF、统一 transform 和
  出生点。
- [ ] 记录源资产 hash、碰撞资产 hash、转换参数、单位和 schema
  版本。
- [ ] 实现视觉/碰撞指纹不匹配时 fail fast。
- [ ] 实现原子写入和源文件/参数变化后的缓存失效。
- [ ] 定义标准资产包：

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

- [ ] 支持统一 asset-to-world 刚体变换。
- [ ] 增加 manifest schema、hash、transform 和不匹配测试。
- [ ] 消除 G1 3DGS 资产的开发机绝对默认路径，改为显式参数、
  环境变量或 manifest。

## 6. 训练用 3DGS 视觉 observation

### V0. 最小相机管线

- [ ] 新建真正加载 3DGS 的 training env 配置，不再在 `play=False` 时
  直接退回普通 flat 任务。
- [ ] 增加安装在 G1 头部或胸部、朝向前下方的训练 CameraSensor。
- [ ] 首版使用 depth；RGB-D 和纯 RGB 后续扩展。
- [ ] 把 depth 或其紧凑编码接入 ObservationManager。
- [ ] 保持 action space 和现有 velocity locomotion 控制接口不变。
- [ ] 配置图像分辨率、更新频率、历史帧和 normalization。
- [ ] 纯 proprioception 配置不得创建 camera/render workload。

### V1. 性能与并行训练

- [ ] 验证单环境 camera observation 的 shape、dtype、范围和更新时序。
- [ ] 验证多 world splat group、camera tensor 和 reset 隔离。
- [ ] 验证视觉 observation 与 CUDA graph 的兼容性。
- [ ] 测量不同 splat 数、分辨率、更新频率和 world 数下的吞吐及
  显存。
- [ ] 决定首版直接降采样 depth，还是使用独立 image encoder。
- [ ] 为渲染失败、显存不足和无效相机配置增加错误处理。

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

## 9. Play 后续与外部前端对接

- [ ] Play 加载统一 scene manifest，而不是分别接收松散路径。
- [ ] Play 校验 checkpoint、visual asset 和 collision asset 兼容性。
- [ ] Viser 增加碰撞代理显示开关和对齐调试模式。
- [ ] 多场景时同步切换机器人 world、splat group 和 collision scene。
- [ ] 对视频帧数、分辨率、时长、资源释放和异常退出进行端到端
  测试。
- [ ] 建立长期运行和浏览器断连/重连测试。
- [ ] 定义供外部 MuJoCo viewer/simulate 仓库使用的稳定场景与
  状态接口。
- [ ] **延后：**本仓库不重复实现外部 MuJoCo viewer/simulate 的 3DGS
  前端。

## 10. 性能、质量与发布

- [ ] 为无 3DGS、Viser-only、MJWarp camera 和视觉训练分别建立正式
  benchmark。
- [ ] 记录初始化、PLY 解析、BVH、浏览器上传、physics、policy 和
  rendering 时间。
- [ ] 建立目标 GPU、splat 数、world 数、分辨率和目标吞吐矩阵。
- [ ] 运行 `make check`。
- [ ] 运行完整 `make test`，包括经明确配置的 slow/real-asset 测试。
- [ ] 运行 `make docs`。
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
- [ ] 训练首版使用原始降采样 depth，还是单独的 image encoder？
- [ ] 目标 GPU、首版 splat 数、训练 world 数和最低吞吐是多少？
- [ ] 首版碰撞输入是否只接受 collision-only MJCF，还是同时提供受限
  raw mesh convenience API？
- [ ] scene manifest 使用 JSON、TOML 还是现有配置序列化格式？

## 13. 推荐执行顺序

1. 完成外部 collision-only MJCF 输入、plane fallback 和验证测试。
2. 建立一个严格对齐的单台阶 PLY/MJCF 资产对。
3. 实现 G1 前向下视 depth observation。
4. 建立随机起点单级台阶 MDP，并先跑 blind/privileged 冒烟训练。
5. 训练 mesh vision 与 3DGS vision，对视觉闭环进行消融验证。
6. 完成视觉—碰撞偏差实验和正式性能基准。
7. 在稳定接口之上实现自动碰撞代理生成。
8. 扩展到多台阶、间隙、真实重建环境和论文实验。

## 14. 更新记录

- 2026-10-24：在文档顶部增加阶段状态表、项目依赖树和
  每次开发后的
  强制更新规则。
- 2026-10-24：创建项目级 TODO；核对当前代码、测试、文档和需求；
  将“视觉 Play 已完成”与“视觉训练/碰撞尚未完成”明确分离；
  确定外部碰撞输入为下一里程碑。
