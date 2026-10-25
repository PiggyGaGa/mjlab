# mjlab 3DGS Play（Viser）模块需求

项目总体进度、已完成证据和后续执行顺序见 `mjlab-3dgs-todo.md`。

## 1. 文档目的

本文档定义独立的 3DGS 策略回放模块。该模块加载已经训练完成的策略，
在 mjlab 仿真中执行推理，并通过 Viser 前端展示机器人在 3DGS 环境中的
运动效果。

Play 与训练是两个不同模块和执行流程：

- Play 只消费环境配置、场景 manifest 和 checkpoint；
- Play 不负责训练策略、生成碰撞代理或转换原始 3DGS PLY；
- `play` 命令独立启动，不由训练命令自动触发；
- Play 的生命周期、错误和资源释放不能影响训练模块设计。

## 2. 前端范围

本仓库只实现基于 Viser 的 3DGS Play 界面。

MuJoCo viewer/simulate 的 3DGS 界面已经在外部仓库实现，不属于本仓库
本阶段的开发范围。后续对接时，本仓库只提供稳定的场景资产、仿真状态和
策略输出接口，不在这里重复实现该前端。

### 2.1 范围内

- 通过现有 `play` 流程加载环境与 checkpoint；
- 通过 Viser 原生 Gaussian Splat renderer 在浏览器 3D 视口实时渲染；
- 通过 MuJoCo Warp CameraSensor 生成视频和传感器画面；
- Viser 控制面板中的播放控制、相机控制、环境选择和状态信息；
- 视频或帧录制；
- 回放性能、错误处理、测试和文档。

### 2.2 范围外

- 训练逻辑和训练 UI；
- 原始 PLY 转换；
- 自动碰撞生成；
- MuJoCo viewer/simulate 前端的 3DGS 实现；
- 将每个 Gaussian 转换成 Viser mesh/point primitive；
- 在本仓库复制外部仓库的 viewer 实现；
- 首版提供浏览器内自由编辑 3DGS 资产。

## 3. 目标

1. 用户可用独立 `play` 命令加载训练好的 checkpoint。
2. Viser 能显示机器人与 3DGS 正确合成后的实时画面。
3. 机器人和 3DGS 之间具有正确深度和遮挡关系。
4. 用户可以暂停、继续、单步、重置、调速和切换环境。
5. 用户可以选择固定、自由或跟随机器人相机。
6. 实时查看和视频录制共享规范化 splat 数据与空间变换。
7. 没有 3DGS 的现有 Viser Play 行为保持兼容。

## 4. 架构边界

现有 `ViserMujocoScene` 负责在浏览器 3D 视口中显示 MuJoCo mesh。Play 将
规范化后的 Gaussian 中心、协方差、颜色和透明度传给 Viser 原生
`add_gaussian_splats` 接口，由浏览器实时渲染，而不是传输相机截图。

```text
scene manifest + checkpoint
          |
          v
独立 play 运行循环
          |
          +----> MuJoCo Warp physics
          |
          +----> Viser native 3DGS + MuJoCo mesh
          |                    |
          |                    v
          |             浏览器交互式 3D 视口
          |
          +----> Warp CameraSensor ----> Camera Feeds / MP4
```

Viser 的同一 3D 视口显示 splat、机器人结构和调试图元。隐藏物理 plane
继续参与碰撞，但通过 geom group 配置不参与可视化。

## 5. 输入契约

Play 模块接收：

- 可由现有环境注册系统构建的环境配置；
- 训练模块生成的版本化场景 manifest；
- 与环境配置兼容的策略 checkpoint；
- 可选的相机、分辨率、帧率、环境索引和录制配置。

Play 启动时必须验证：

- manifest schema 受支持；
- splat 缓存和碰撞资产指纹一致；
- checkpoint 与环境 observation/action 规格兼容；
- 选择的环境索引和 splat group 合法；
- 至少存在一台可用于 3DGS 合成画面的相机，或能够创建专用 viewer 相机。

## 6. 功能需求

### PL-FR-1：独立执行

- `play` 独立创建环境、策略、viewer 和推理循环。
- 不导入或调用训练 loop。
- 支持本地 checkpoint 和现有远程 checkpoint 获取方式。
- 支持 zero/random policy 以便无 checkpoint 调试场景。
- 提供 Tracking 3DGS 任务，并允许 ``demo --use-3dgs True`` 复用内置预训练
  checkpoint 和 motion；视觉扩展不得改变 observation/action 规格。
- 环境关闭时释放 Viser、Warp render context 和视频资源。

### PL-FR-2：Viser 3DGS 画面

- 使用 Viser 原生 Gaussian Splat renderer 实时渲染，不使用截图背景。
- Viser 主 3D 视口同时显示 3DGS 和 MuJoCo mesh。
- 画面同时包含 3DGS、机器人和允许显示的其他 MuJoCo geom。
- splat 与机器人必须正确执行深度测试和颜色合成。
- 用户可通过浏览器相机自由旋转、缩放和跟随机器人。
- splat 数据只在初始化或资产切换时上传，不得逐帧重复发送。
- 前端渲染不得阻塞物理 step。
- 断开浏览器连接时，仿真可按配置继续或安全暂停。

### PL-FR-3：相机

- 支持固定世界相机。
- 支持跟随机器人 root/body 的相机。
- 支持可交互调整的自由相机；若 MVP 受限，至少先提供固定和跟随模式。
- 相机的 position、look-at、FOV、near/far、分辨率可配置。
- Viser 自由相机由浏览器直接控制，不需要逐帧同步到 Warp 相机。
- reset 和切换环境后相机跟随目标必须正确更新。

### PL-FR-4：播放控制

- 支持播放/暂停。
- 支持单步执行。
- 支持 reset 当前环境。
- 支持速度调整和恢复实时速度。
- 支持选择要显示的环境索引。
- 在 Viser ``Visualization`` 面板提供 ``Collision > Enabled`` 开关，
  运行中实时切换 group 3 碰撞代理；其状态与 ``Groups > Geoms > G3``
  保持同步。
- 适用时保留 reward 与 metric 曲线。

### PL-FR-5：多环境

- Play 可以运行一个或多个 world，但只渲染用户选择或明确配置的 world。
- 切换 world 时同步切换机器人状态、相机目标和 splat group。
- 不允许出现 world A 的机器人配 world B 的 3DGS 背景。
- 未显示 world 的策略推理和仿真行为遵循现有 `play` 配置。

### PL-FR-6：录制

- 支持把 Warp CameraSensor 合成画面保存为视频或图像序列。
- 录制帧率、分辨率、长度和输出路径可配置。
- 实时显示和录制必须共享 splat 数据、颜色、透明度和空间变换语义。
- 录制结束或异常退出时正确关闭编码器，避免损坏输出文件。

### PL-FR-7：前端能力说明

- 文档必须区分 Viser 原生交互式渲染与 Warp CameraSensor 录制路径。
- 碰撞代理默认不应遮挡或替代高保真合成画面中的 3DGS 外观。
- 文档说明 MuJoCo viewer/simulate 版本由外部仓库提供。
- 后续外部 viewer 对接应复用输入契约，不改变 Viser 模块职责。

### PL-FR-8：错误处理

- 没有 3DGS 资产时继续使用现有 Viser 行为，或根据显式模式报错。
- 缺少相机、资产不匹配、checkpoint 不兼容时 fail fast。
- GPU 渲染失败、显存不足和浏览器断连提供可理解的状态信息。
- Viser splat 上传或浏览器渲染错误不能静默终止仿真线程。

## 7. 非功能需求

### PL-NFR-1：性能

- 物理 step、策略推理、初始 splat 上传和浏览器渲染分别记录性能。
- 静态 splat 初始化后不产生持续的图像编码和帧传输开销。
- 基准覆盖分辨率、splat 数量、显示帧率和不同网络条件。

### PL-NFR-2：兼容性

- 没有 3DGS 的 Viser viewer 功能无回归。
- Play 不修改训练 checkpoint 格式，除非通过版本化迁移明确实施。
- 不修改 MuJoCo Warp、Viser 或 mjviser 安装包源码。
- 前端功能在本地浏览器和 SSH 端口转发场景下可用。

### PL-NFR-3：可维护性

- 推理循环、Viser 场景、UI 控制和 MJWarp 录制职责分离。
- Viser 代码不解析原始 PLY，只读取规范化场景接口。
- 公共配置有类型标注并通过 `make check`。
- 用户可见变化更新 viewer 文档和 changelog。

## 8. 验收标准

### PL-AC-1：独立启动

`play` 能在不启动训练模块的情况下加载环境和 checkpoint。zero/random policy
也能用于 3DGS 场景检查。

### PL-AC-2：正确显示

Viser 原生 3D 视口持续显示 splat 与机器人 mesh，允许自由旋转和缩放；
reset 后机器人状态继续更新，静态 splat 不重复上传。

### PL-AC-3：交互

暂停、继续、单步、reset、调速、相机切换和环境切换均不造成状态错配或
资源泄漏。

### PL-AC-4：多环境一致性

切换 world 后显示正确的机器人、相机目标和 splat group，不出现跨 world
混合。

### PL-AC-5：录制

输出视频可以正常播放，帧数、分辨率和目标时长符合配置，画面与实时查看
使用同一渲染语义。

### PL-AC-6：性能与回归

长时间运行时队列和内存保持有界；现有无 3DGS 的 Viser 流程通过回归测试；
`make check` 和最终 `make test` 通过。

## 9. 需求拆解

### Play Phase 0：接口验证

- PL-T0.1：确认训练模块 manifest、环境配置和 checkpoint 的读取契约。
- PL-T0.2：用最小场景验证 Viser 原生 Gaussian Splat API。
- PL-T0.3：验证中心、协方差、颜色、透明度和坐标系转换。
- PL-T0.4：比较 Viser 与 MJWarp 渲染路径的视觉语义。
- PL-T0.5：验证固定/跟随相机与环境 reset。
- PL-T0.6：形成 Viser UI 和线程/队列模型决策记录。

依赖：训练模块至少提供可加载的最小规范化资产和场景。

### Play Phase 1：独立配置与生命周期

- PL-T1.1：定义 3DGS Play/Viser 配置。
- PL-T1.2：实现 manifest、checkpoint 和相机启动校验。
- PL-T1.3：把资源创建和释放接入现有 `play` 生命周期。
- PL-T1.4：支持 zero/random policy 调试。
- PL-T1.5：增加错误配置和资源释放测试。

依赖：Play Phase 0。

当前进度：已实现 schema v1 manifest 的相对路径、SHA-256、坐标系和
统一 transform 校验，并通过 ``--scene-manifest`` 接入 Play/demo；原有
``--gaussian-splat-file`` 与 ``--collision-mjcf-file`` 继续作为互斥的
临时实验入口。checkpoint 与场景兼容性元数据仍待补充。

### Play Phase 2：Viser 原生 3DGS 显示

- PL-T2.1：把规范化 splat 转换为 Viser 中心、协方差、RGB 和透明度。
- PL-T2.2：建立静态 splat scene node，并避免逐帧重复上传。
- PL-T2.3：在同一 3D 视口中渲染 splat 与机器人 mesh。
- PL-T2.4：处理浏览器连接、断连和重连。
- PL-T2.5：增加坐标、尺度、可见性和多 world 测试。
- PL-T2.6：记录初始上传、浏览器显存和交互帧率。

依赖：Play Phase 1。

### Play Phase 3：相机与控制

- PL-T3.1：实现固定和机器人跟随相机。
- PL-T3.2：评估并实现自由相机同步。
- PL-T3.3：接入暂停、单步、reset 和速度控制。
- PL-T3.4：实现环境索引和 splat group 原子切换。
- PL-T3.5：接入 debug、reward 和 metric 控件。
- PL-T3.6：增加 reset、切换和并发控制测试。

依赖：Play Phase 2。

### Play Phase 4：录制

- PL-T4.1：复用 Warp 合成相机输出。
- PL-T4.2：实现帧率、分辨率、长度和输出配置。
- PL-T4.3：实现编码器正常及异常关闭。
- PL-T4.4：验证实时画面和视频语义一致。
- PL-T4.5：增加视频冒烟测试。

依赖：Play Phase 2；可与 Play Phase 3 后半部分并行。

### Play Phase 5：质量收尾

- PL-T5.1：完成长时间运行、断连和压力测试。
- PL-T5.2：完成无 3DGS Viser 回归测试。
- PL-T5.3：补充 `play --help`、viewer 文档和故障排查。
- PL-T5.4：记录外部 MuJoCo viewer/simulate 的未来对接边界。
- PL-T5.5：更新 changelog。
- PL-T5.6：运行 `make check` 和 `make test`。

依赖：前述 Play 阶段。

## 10. 与外部 MuJoCo viewer/simulate 的未来对接

未来对接只要求双方约定以下稳定接口：

- 场景 manifest/schema；
- 规范化 splat 数组语义；
- 碰撞资产和统一坐标变换；
- 当前 world 的仿真状态；
- checkpoint 推理后的控制输出；
- 相机和环境索引语义。

本仓库不会在 Viser 模块中加入外部 viewer 的实现细节。若后续需要适配器，
应建立独立 adapter 模块，并单独编写需求与验收标准。

## 11. 待确认事项

1. 典型浏览器能够稳定渲染的最大 splat 数量是多少？
2. 是否需要为 Viser 单独配置 opacity、数量或 LOD 过滤？
3. 目标实时分辨率、帧率和典型网络环境是什么？
4. 是否需要默认展开 MJWarp Camera Feeds 以便与 Viser 原生渲染对比？
5. 外部 MuJoCo viewer/simulate 仓库未来采用进程内、进程间还是文件接口？

未知时默认采用：Viser 原生交互式 3DGS 视口、完整静态 splat 一次上传、保留
现有调试控件；Camera Feeds 和视频继续使用 MJWarp CameraSensor。
