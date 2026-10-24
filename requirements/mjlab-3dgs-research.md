# mjlab 3DGS 仿真训练调研需求

项目总体进度、已完成证据和后续执行顺序见 `mjlab-3dgs-todo.md`。

## 1. 文档目的

本文档定义 mjlab 3DGS 仿真训练项目的前期调研范围、研究问题、检索方法、
证据标准、实验设计和交付物。调研结果既用于指导工程方案选择，也作为后续
论文的背景、相关工作、方法设计和实验讨论材料。

本文件是“调研工作要求”，不是文献综述正文。所有技术结论必须在完成检索、
阅读和实验后写入独立调研报告，不能仅凭经验或二手资料下结论。

第一轮 prior-art 调研结果见 `mjlab-3dgs-research-report.md`。该报告确认
GaussGym、MuGS、DISCOVERSE、SplatGym 和 SplatTransform 等工作已覆盖本项目
原始设想的多个组成部分，因此后续研究必须以报告中的强基线和重新定位后的
贡献边界为准。

调研工作与以下需求文件对应：

- `mjlab-3dgs-training.md`：3DGS 资产、碰撞代理和训练模块；
- `mjlab-3dgs-play-viser.md`：基于 Viser 的独立策略回放模块。

## 2. 调研目标

1. 建立 3DGS、机器人仿真、碰撞重建和视觉强化学习的技术背景。
2. 确认 MuJoCo Warp 原生 splat 渲染的能力、限制和性能特征。
3. 比较从 3DGS 构造物理碰撞表示的候选方法。
4. 分析视觉模型与物理代理分离带来的对齐误差和 sim-to-sim/sim-to-real 问题。
5. 为资产格式、碰撞算法、训练配置和 Viser 展示方案提供可追溯的选择依据。
6. 设计能够支撑论文论点的基线、消融实验和评价指标。
7. 形成可直接转化为论文背景与相关工作章节的结构化材料。

## 3. 核心研究问题

后续调研和实验必须围绕以下问题展开。

### RQ-1：3DGS 是否适合作为机器人仿真的视觉环境表示？

- 与 mesh、NeRF、点云和传统纹理场景相比，3DGS 的优势和限制是什么？
- 3DGS 在实时渲染、显存占用、视角一致性和动态场景方面表现如何？
- 机器人相机观测需要的 RGB、深度和语义信息分别能否可靠获得？
- 原始 3DGS 的颜色表示、球谐阶数和透明度会怎样影响视觉策略？

### RQ-2：3DGS 如何与物理仿真结合？

- 为什么 3DGS 本身不能直接作为常规刚体碰撞表面？
- 视觉表示和物理表示分离后，如何保证单位、坐标和几何对齐？
- 现有工作采用 mesh、SDF、TSDF、occupancy、heightfield、primitive
  或其他代理的比例和效果如何？
- 是否存在直接对 Gaussian、ellipsoid 或 radiance field 进行接触、距离查询
  或物理求解的方法？这些方法是否适合大规模并行机器人训练？

### RQ-3：如何从 3DGS 生成稳定而高效的碰撞代理？

- 仅使用 Gaussian center 与同时使用 scale、rotation、opacity 的差异是什么？
- voxel occupancy、TSDF、Poisson reconstruction、marching cubes、SDF、
  heightfield 和 convex decomposition 各自适合什么场景？
- 如何处理漂浮 Gaussian、透明表面、薄墙、孔洞、反射区域和观测盲区？
- 网格简化程度如何影响几何误差、接触稳定性和仿真吞吐？
- 如何定义与机器人任务相关的碰撞重建质量，而不是只使用视觉指标？

### RQ-4：3DGS 对强化学习训练有什么影响？

- 对纯本体感觉策略，3DGS 的作用是否仅限于人类观察？
- 对视觉策略，3DGS 与传统 mesh 渲染会造成怎样的 domain gap？
- 相机更新频率、分辨率、splat 数量和 world 数如何影响训练吞吐？
- 视觉随机化、颜色变化、光照变化和几何代理误差应该如何建模？
- 多场景训练是否提高泛化能力，如何保证每个 world 的视觉与碰撞一致？

### RQ-5：如何评价训练后策略？

- sim-to-sim 应比较哪些仿真器、渲染器或物理表示？
- Play 阶段的高保真视觉是否会掩盖碰撞代理误差？
- 策略成功率、回报、稳定性、穿透率、接触误差和计算性能如何联合报告？
- 若未来进行 sim-to-real，哪些误差最可能主导迁移失败？

### RQ-6：本项目的论文贡献应该如何定位？

- 贡献是系统集成、碰撞重建算法、并行训练方法，还是完整评测基准？
- 与“仅把 3DGS 作为背景显示”的方案相比，本项目新增了什么能力？
- 与已有 radiance-field robotics 工作相比，创新点和适用边界是什么？
- 哪些贡献可以被定量实验直接验证？

## 4. 调研主题

### RS-1：3DGS 基础与格式生态

调研内容：

- 3D Gaussian Splatting 的基本表示和渲染原理；
- Gaussian position、covariance、rotation、scale、opacity 和球谐颜色；
- 常见训练和导出工具的 PLY 字段差异；
- 原版实现、gsplat、Nerfstudio 及其他主流工具链的资产兼容性；
- 压缩、LOD、裁剪、分块和流式加载方法；
- 静态与动态 3DGS 的区别。

交付结果：

- 字段语义对照表；
- 支持格式建议；
- 资产转换风险清单；
- 论文背景中 3DGS 原理部分的材料。

### RS-2：MuJoCo Warp 原生 splat 渲染

调研内容：

- 当前锁定版本和上游最新版本的 splat API；
- BVH 构建、ray-splat intersection、alpha compositing 和 depth 语义；
- splat 与 MuJoCo geom 的遮挡关系；
- 多 world、splat group 和内存共享方式；
- CUDA graph 兼容性；
- RGB、深度和 segmentation 的能力边界；
- 版本变化、已知问题和上游测试覆盖。

资料必须优先使用：

- MuJoCo Warp 官方源码、测试和发布记录；
- MuJoCo 与 NVIDIA Warp 官方文档；
- 对应版本的可复现实验。

交付结果：

- 版本化能力矩阵；
- 最小可复现实验；
- API 与性能限制报告；
- 工程实现决策依据。

### RS-3：3DGS 几何提取与表面重建

调研内容：

- 从 Gaussian center 或 covariance 提取点、法线、occupancy 和 density；
- radiance field/3DGS 到 mesh、SDF、TSDF 的研究；
- Poisson、ball pivoting、marching cubes 和神经/优化式表面重建；
- 针对室内、室外、薄结构和大范围场景的差异；
- 去噪、连通分量、孔洞修复和网格简化；
- 有 ground-truth mesh 与无 ground-truth mesh 时的评价方法。

交付结果：

- 方法分类表；
- 候选算法复现难度和依赖分析；
- 适用于本项目的 shortlist；
- 碰撞生成技术验证方案。

### RS-4：机器人碰撞表示与物理稳定性

调研内容：

- MuJoCo/MuJoCo Warp 支持的 primitive、mesh、heightfield 和 SDF；
- 静态三角网格、凸包、凸分解和 signed distance collision；
- 接触点数量、摩擦、接触维度和 solver 参数对稳定性的影响；
- 地面、楼梯、墙体、洞口和小障碍物的表示选择；
- 高面数几何对 broadphase、narrowphase 和整体训练吞吐的影响；
- 穿透、悬空、接触抖动和非流形网格的诊断方法。

交付结果：

- MuJoCo Warp 碰撞能力矩阵；
- 不同代理的物理稳定性实验；
- 面数/误差/吞吐权衡曲线；
- 默认碰撞方案与回退方案。

### RS-5：3DGS 与机器人学习

调研内容：

- NeRF/3DGS 在导航、操作、定位、视觉控制和仿真中的使用；
- photorealistic rendering 对 policy learning 和 domain gap 的影响；
- vision-based RL、imitation learning 和 world model 中的场景表示；
- 视觉观测与 proprioception 的融合方式；
- 多场景、domain randomization、相机随机化和视觉增强；
- sim-to-sim 与 sim-to-real 的评价方法。

交付结果：

- 相关工作分类图；
- 代表性论文对比表；
- 本项目与已有工作的差异说明；
- 可用于论文 related work 的结构化笔记。

### RS-6：查看器与可视化系统

调研内容：

- Viser 的图像组件、相机控制、异步更新和远程访问方式；
- MuJoCo mesh 视口与 Warp 合成画面的功能差异；
- 浏览器图像编码、发送延迟和最新帧队列；
- 训练后回放、视频录制和实验可视化的可复现要求；
- 外部 MuJoCo viewer/simulate 3DGS 实现的未来接口边界。

交付结果：

- Viser 前端方案对比；
- 延迟与带宽基准；
- Play 模块架构选择依据；
- 论文系统实现部分的可视化材料。

## 5. 文献检索规范

### 5.1 数据源

至少覆盖：

- Google Scholar；
- arXiv；
- IEEE Xplore；
- ACM Digital Library；
- Semantic Scholar 或 OpenAlex；
- 会议和期刊官方网站；
- MuJoCo、MuJoCo Warp、NVIDIA Warp、Viser 等官方文档和仓库。

### 5.2 关键词组

检索应组合使用以下关键词，而不是只搜索项目名称：

```text
3D Gaussian Splatting robotics
Gaussian Splatting simulation
Gaussian Splatting reinforcement learning
Gaussian Splatting robot navigation
Gaussian Splatting robot manipulation
Gaussian Splatting mesh extraction
Gaussian Splatting surface reconstruction
Gaussian Splatting collision
Gaussian Splatting signed distance field
radiance field physics simulation
NeRF robotics simulation
photorealistic simulation robot learning
sim-to-sim robot learning
sim-to-real visual locomotion
MuJoCo Warp rendering splat
MuJoCo mesh SDF collision
```

应根据已发现论文的术语继续扩展同义词和引用链。

### 5.3 检索时间范围

- 3DGS 主题从其奠基工作开始检索至调研执行当日；
- 表面重建、碰撞检测和机器人仿真可纳入更早的经典工作；
- 每次论文提交前重新执行最近一年和最近三个月的增量检索；
- 调研报告必须记录检索日期，避免把动态变化的“最新工作”写成永久结论。

### 5.4 纳入标准

- 与至少一个研究问题直接相关；
- 方法、实验或系统描述足够判断其贡献；
- 优先选择已同行评审论文、官方技术报告和公开源码；
- arXiv 工作可以纳入，但必须标注未同行评审状态；
- 工程能力结论优先使用官方源码、文档和可复现实验。

### 5.5 排除标准

- 只有宣传材料而无方法或实验细节；
- 无法确认来源的转载和自动生成摘要；
- 与机器人、几何或仿真没有可解释关联的纯视觉展示；
- 重复版本只保留最终版本，同时记录版本关系；
- 无法验证的性能数字不进入定量对比。

### 5.6 引用链扩展

对每篇核心论文执行：

- backward snowballing：检查其关键参考文献；
- forward snowballing：检查引用该论文的后续工作；
- 查找作者或项目主页上的代码、数据和补充材料；
- 检查是否存在后续期刊版、修订版或撤稿信息。

## 6. 证据与笔记标准

每篇纳入论文至少记录：

- BibTeX key 和完整引用；
- DOI、官方页面或 arXiv id；
- 发表 venue、年份和评审状态；
- 研究问题与应用场景；
- 输入、输出和场景表示；
- 核心方法；
- 数据集、机器人和仿真器；
- 对比基线；
- 评价指标；
- 关键结果；
- 局限性；
- 是否提供代码、数据和许可证；
- 与本项目的直接关系；
- 可复现性备注。

所有数字必须标注来源表格、图或实验设置。引用原文时记录页码或章节，
最终论文中优先释义，避免大段复制。

以下内容必须明确区分：

- 论文作者直接报告的事实；
- 本项目根据资料作出的推断；
- 本项目自行实验得到的结果；
- 尚未验证的假设。

## 7. 文献质量评价

每篇核心工作按以下维度进行定性或分级评价：

- 与本项目的相关性；
- 方法描述完整性；
- 实验规模和基线充分性；
- 是否报告失败案例和限制；
- 代码、数据和环境是否公开；
- 是否能够在合理资源下复现；
- 结论是否超出实验支持范围；
- 是否存在后续工作验证或反驳。

质量评价用于控制证据权重，不能简单用引用数替代。

## 8. 技术对比矩阵

### 8.1 场景表示对比

至少比较：

| 维度 | Mesh | Point cloud | NeRF | 3DGS | 3DGS + collision proxy |
| --- | --- | --- | --- | --- | --- |
| 实时渲染 | 待调研 | 待调研 | 待调研 | 待调研 | 待调研 |
| 几何显式性 | 待调研 | 待调研 | 待调研 | 待调研 | 待调研 |
| 物理碰撞 | 待调研 | 待调研 | 待调研 | 待调研 | 待调研 |
| 编辑难度 | 待调研 | 待调研 | 待调研 | 待调研 | 待调研 |
| 显存/存储 | 待调研 | 待调研 | 待调研 | 待调研 | 待调研 |
| 视觉真实性 | 待调研 | 待调研 | 待调研 | 待调研 | 待调研 |
| 多 world 适配 | 待调研 | 待调研 | 待调研 | 待调研 | 待调研 |

表格只能在完成资料核验和实验后填写，不允许用主观印象填写结论。

### 8.2 碰撞生成方法对比

至少比较：

- 输入信息：center、scale、rotation、opacity、相机或深度；
- 输出表示：mesh、heightfield、SDF、primitive；
- 是否保留完整 3D 结构；
- 薄结构和孔洞表现；
- 噪声鲁棒性；
- watertight 要求；
- 生成时间和峰值内存；
- 输出复杂度；
- MuJoCo Warp 兼容性；
- 接触稳定性；
- 可复现性和许可证。

强制纳入以下公开基线：

- SplatTransform 的 voxel/SVO 与 collision mesh 管线；
- SplatGym 的 OctoMap/FCL 路线；
- center-only 与 covariance-aware occupancy；
- SuGaR、2DGS 或 Gaussian Opacity Fields 中至少一种表面重建路线；
- 有原始图像时，GaussGym 的 VGGT + NKSR 路线；
- 人工或 ground-truth mesh 上界。

Splat-Nav、SAFER-Splat 和 FOCI 属于直接在 Gaussian 表示上做碰撞约束的
相关工作。它们不一定能直接替代 MuJoCo 接触几何，但必须纳入方法讨论，并评估
其距离或 overlap 定义能否作为碰撞代理的监督或评价信号。

### 8.3 系统对比

选择现有最相关系统，比较：

- 使用的机器人任务；
- 使用的场景表示；
- 3DGS renderer 的实际实现，而不是只记录物理后端名称；
- 原生 ray traversal 或独立 renderer + mask compositing；
- 是否具有物理碰撞；
- 是否支持并行训练；
- 是否支持视觉策略；
- viewer 与录制能力；
- PyTorch、CUDA 扩展、JIT、EGL 等安装和运行依赖；
- 是否存在跨框架同步和额外显存副本；
- 开源程度；
- 性能和硬件；
- 本项目新增能力。

系统矩阵至少包含 GaussGym、MuGS、DISCOVERSE、SplatGym、SplatSim、
RoboGSim 和 Splatting Physical Scenes。任何“首次”或“首个”表述都必须在投稿
前重新执行增量检索后才能使用。

## 9. 实验调研需求

文献结论必须与本项目技术验证结合。Phase 0 至少设计以下实验。

### EX-1：原生 splat 渲染正确性

变量：

- splat 数量；
- opacity；
- isotropic/anisotropic scale；
- rotation；
- 机器人和 splat 的相对深度；
- 相机分辨率和 world 数。

指标：

- RGB 正确性；
- 深度误差和语义；
- 初始化时间；
- render time；
- 显存；
- CUDA graph 可用性。

### EX-2：碰撞重建算法比较

至少包含合成场景和真实场景。合成场景应具有可用的 ground-truth 几何。

候选方法：

- center-only voxel occupancy；
- covariance-aware occupancy；
- SplatTransform collision mesh；
- OctoMap/FCL collision query；
- TSDF/marching cubes；
- heightfield + obstacle mesh；
- 至少一种高质量 3DGS surface reconstruction；
- 经验证可用的 SDF 或其他候选。

几何指标：

- Chamfer distance；
- point-to-surface distance 分位数；
- normal consistency；
- occupancy IoU；
- watertight/连通分量；
- 顶点和面数。

物理指标：

- 落体穿透深度；
- 静止接触漂移和抖动；
- 足底接触稳定性；
- 障碍物碰撞漏检率；
- 仿真 SPS；
- 峰值显存。

### EX-3：网格简化消融

控制同一重建结果，改变目标面数或误差阈值，绘制：

- 面数与几何误差；
- 面数与物理吞吐；
- 面数与任务成功率；
- 面数与穿透/接触异常。

### EX-4：训练影响

根据最终任务至少比较：

- 传统 mesh 场景；
- 3DGS 视觉 + 同一碰撞代理；
- 不同质量碰撞代理；
- 纯 proprioception 与视觉 observation；
- 单场景与多场景训练。

指标：

- 学习曲线和样本效率；
- episode return；
- 任务成功率；
- 失败类型；
- 训练 SPS；
- GPU 显存；
- 多随机种子均值与离散程度。

### EX-5：Play 性能

比较不同 splat 数量、分辨率、显示帧率和网络条件下的：

- Warp 渲染延迟；
- 图像编码延迟；
- Viser 发送延迟；
- 浏览器实际 FPS；
- 丢帧率；
- 内存和显存稳定性。

## 10. 论文材料结构

调研输出应能够支持以下论文结构。

### 10.1 Introduction

- 真实场景视觉建模与机器人训练之间的差距；
- 3DGS 的实时高保真视觉优势；
- 3DGS 缺少物理碰撞的核心问题；
- 高保真视觉、简化物理代理和并行训练结合的研究动机；
- 本项目拟解决的问题和可验证贡献。

### 10.2 Background

- 3DGS 表示与渲染；
- MuJoCo Warp 并行仿真；
- 机器人碰撞表示；
- manager-based RL 与视觉观测。

### 10.3 Related Work

- radiance field/3DGS for robotics；
- neural scene representation for simulation；
- 3DGS surface reconstruction；
- collision proxy generation；
- photorealistic robot learning and sim-to-real。

### 10.4 Method

- 资产规范化；
- 统一坐标变换；
- covariance-aware 几何采样或最终选定方法；
- 碰撞重建与简化；
- MuJoCo Warp 原生渲染接入；
- 多 world 训练和 Viser Play。

### 10.5 Experiments

- 数据和任务；
- 基线；
- 几何与物理指标；
- 训练结果；
- 性能；
- 消融和失败案例。

调研阶段不能提前承诺最终贡献表述。贡献必须由实现和实验结果支持。

## 11. 调研交付物

### RE-D1：文献数据库

建议维护机器可读表格或 YAML/CSV，包含第 6 节要求的字段。

### RE-D2：引用库

- 经过去重的 BibTeX；
- 稳定 citation key；
- 论文版本和官方链接；
- 不提交无再分发许可的论文 PDF。

### RE-D3：结构化阅读笔记

按研究主题组织，每个结论可追溯到论文、官方资料或本项目实验。

### RE-D4：相关工作矩阵

至少包括方法、场景表示、碰撞方式、任务、仿真器、指标、代码和限制。

### RE-D5：技术调研报告

报告应包含：

- 执行摘要；
- 检索方法和日期；
- 主题综述；
- 候选方案比较；
- 推荐方案；
- 风险和未解决问题；
- 对训练与 Play 需求的影响。

### RE-D6：实验协议

固定数据、参数、随机种子、硬件、软件版本、指标和统计方法。

### RE-D7：论文材料包

- Background 草稿素材；
- Related Work 草稿素材；
- 方法选择论证；
- 可复用图表数据；
- 失败案例和 limitation 列表。

## 12. 调研任务拆解

### Research Phase 0：研究协议

- RE-T0.1：确认论文目标、目标 venue 和预计时间表。
- RE-T0.2：固定研究问题、关键词和纳入/排除标准。
- RE-T0.3：建立文献数据库和 BibTeX 规范。
- RE-T0.4：定义实验记录模板和硬件信息模板。
- RE-T0.5：确定真实资产的许可证和可发表性。

### Research Phase 1：基础与生态

- RE-T1.1：调研 3DGS 原理和资产格式。
- RE-T1.2：调研 MuJoCo Warp splat API 和版本历史。
- RE-T1.3：调研 MuJoCo Warp 碰撞能力。
- RE-T1.4：完成场景表示和系统能力矩阵初稿。
- RE-T1.5：输出 Background 第一版材料。

### Research Phase 2：几何与碰撞

- RE-T2.1：系统检索 3DGS mesh/SDF/surface reconstruction 工作。
- RE-T2.2：检索点云与 radiance field 几何提取经典方法。
- RE-T2.3：完成碰撞生成 shortlist。
- RE-T2.4：定义并执行 EX-1、EX-2 和 EX-3。
- RE-T2.5：基于文献和实验选择 MVP 碰撞算法。

### Research Phase 3：机器人学习相关工作

- RE-T3.1：检索 3DGS/NeRF robotics 工作。
- RE-T3.2：检索 photorealistic simulation 和 visual RL 工作。
- RE-T3.3：检索 sim-to-sim/sim-to-real 评价方法。
- RE-T3.4：完成相关工作分类与对比矩阵。
- RE-T3.5：输出 Related Work 第一版材料。

### Research Phase 4：系统与训练实验

- RE-T4.1：定义工程系统基线。
- RE-T4.2：执行 EX-4 训练比较和消融。
- RE-T4.3：执行 EX-5 Play 性能测试。
- RE-T4.4：统计多随机种子结果和失败案例。
- RE-T4.5：根据结果修订论文贡献和 limitation。

### Research Phase 5：论文材料整理

- RE-T5.1：完成增量检索并更新最新工作。
- RE-T5.2：核查所有引用、数字和图表来源。
- RE-T5.3：整理 Background 和 Related Work 草稿。
- RE-T5.4：输出方法选择依据和实验协议。
- RE-T5.5：整理可发表图表、caption 和补充材料清单。
- RE-T5.6：归档检索日期、软件版本和复现实验命令。

## 13. 验收标准

### RE-AC-1：可追溯性

每项核心技术结论都能追溯到至少一个一手来源或本项目实验；推断和事实已
明确区分。

### RE-AC-2：覆盖度

六个研究问题均有文献证据、未解决问题和与本项目的关系说明。

### RE-AC-3：方法选择

碰撞算法、资产格式和渲染方案均有对比矩阵和实验依据，不以主观偏好作为
唯一理由。

### RE-AC-4：实验可复现

实验记录包含代码版本、配置、随机种子、资产指纹、硬件、依赖版本和原始
结果文件。

### RE-AC-5：论文可用性

交付物能够支持论文 Introduction、Background、Related Work、Method 和
Experiments，不需要重新查找核心引用来源。

### RE-AC-6：引用质量

引用库无明显重复、伪造或无法访问的条目；关键事实优先引用原始论文和官方
资料；最新工作在投稿前完成增量检查。

## 14. 开放问题

1. 论文的主要贡献最终定位为系统、算法还是 benchmark？
2. 是否拥有可公开或可描述的真实 3DGS 场景？
3. 是否能够获得对应 ground-truth mesh、深度或激光扫描？
4. 目标任务是 locomotion、navigation、manipulation 还是多任务？
5. 是否计划做 sim-to-real，能够使用哪些真实机器人和传感器？
6. 目标 venue 对篇幅、匿名代码、数据开放和补充材料有什么要求？
7. 论文实验可使用的 GPU、训练时间和随机种子预算是多少？

这些问题不阻止基础检索，但必须在实验协议和论文贡献固定前回答。
