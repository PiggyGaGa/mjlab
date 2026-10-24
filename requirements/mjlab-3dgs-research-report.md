# mjlab 3DGS 机器人训练相关工作调研报告

## 1. 报告结论

**检索日期：2026-10-24。** 本报告是面向项目立项和论文选题的第一轮
prior-art 调研，不等同于投稿前的完整系统综述。结论依据论文、作者项目页、
官方代码仓库和 MuJoCo Warp 官方文档；没有把新闻转载或无法核验的性能宣传
作为主要证据。

调研后的核心结论是：**以前已经有多个项目做过“3DGS + 机器人仿真 +
物理/训练”的不同组合，而且与本项目高度相似的工作已经出现。** 因此，下面
这些表述不能再单独作为论文创新点：

- 在机器人仿真器中显示 3DGS；
- 把 3DGS 与 MuJoCo 物理结合；
- 在并行环境中用 3DGS 训练视觉策略；
- 从 splat/点云生成体素、octree 或 collision mesh；
- 用高保真 3DGS 背景缩小视觉 sim-to-real gap。

最直接的重复风险来自以下工作：

1. [GaussGym](https://arxiv.org/abs/2510.15352) 已在 IsaacGym 中实现大规模
   并行 3DGS 视觉训练、四足/人形机器人 locomotion、场景自动生成和
   sim-to-real；其开源流程还从 VGGT 几何通过 NKSR 生成碰撞 mesh。
2. [MuGS](https://github.com/Renforce-Dynamics/MuGS) 已提供 MuJoCo、`mjlab`
   和 `mujoco_warp` 集成，并声明支持 4096 个并行环境的批量渲染；但它使用
   独立 `gsplat` renderer 生成 3DGS 背景，不使用 MJWarp 原生 3DGS renderer。
3. [DISCOVERSE](https://air-discoverse.github.io/) 已把 MuJoCo、3DGS、
   Real2Sim、并行仿真和机器人学习组织成完整开源系统。
4. [SplatGym](https://arxiv.org/abs/2410.19564) 已实现 3DGS 神经仿真环境、
   octree 碰撞检测和视觉导航强化学习。
5. [SplatTransform](https://github.com/playcanvas/splat-transform) 已提供从
   Gaussian splat 直接生成稀疏体素和 collision mesh 的开源命令行管线。

这并不意味着本项目没有必要。它意味着要把目标分成两类：

- **工程目标仍有较高必要性。** mjlab 当前需要一个原生、类型化、可测试且与
  manager-based RL 一致的 3DGS 资产入口；使用 MuJoCo Warp 原生 splat
  ray tracer，可以避免外置 gsplat 渲染器、分割 mask 和二次 alpha 合成。
- **论文目标只有在重新定位后才有必要性。** “系统接入”本身不够。论文需要
  把贡献集中在面向接触动力学的碰撞代理、视觉—物理不一致的量化、任务相关
  简化目标，或可复现的 MuJoCo Warp 训练基准上。

综合判断：**工程项目建议继续；论文选题为有条件继续。** 如果最终只完成 PLY
加载、原生渲染和普通 voxel-to-mesh，论文创新性偏弱；如果能提出并验证一种
面向机器人接触稳定性和训练吞吐的 3DGS 碰撞代理方法，则仍有明确研究空间。

针对碰撞方向的第二轮专题调研进一步表明：PLY 到 voxel/collision mesh、
covariance-aware 表面和 Gaussian 直接碰撞推理均已有先例；当前更清晰的空缺是
系统量化视觉—碰撞代理偏差如何影响足底/末端接触与策略性能，以及在固定面数或
仿真吞吐预算下，用机器人接触分布驱动代理简化。建议先建立误差基准，再开发
task-aware collision proxy，二者可以分别形成 benchmark 与算法贡献。

## 2. 本轮调研的问题边界

本轮重点回答四个问题：

1. 是否已有 3DGS 与物理仿真结合的机器人系统？
2. 是否已有 3DGS 场景中的碰撞、规划或 collision mesh 生成方法？
3. 是否已有在 3DGS 场景中进行并行机器人训练的系统？
4. 在这些工作的覆盖下，mjlab 项目还有哪些必要且可发表的工作？

本轮不对所有 3DGS 表面重建论文做穷尽式排序，也不复现各项目性能数字。
作者报告的性能只能作为后续复现实验的候选目标，不能直接视为公平对比结果。

## 3. 相关工作分类

### 3.1 3DGS + 并行机器人训练

#### GaussGym

[GaussGym 项目页](https://www.escontrela.me/gauss_gym/)和
[开源仓库](https://github.com/escontra/gauss_gym)显示，它把 3DGS 作为
IsaacGym 等向量化物理仿真器的渲染器，用于从像素学习 locomotion 和导航。
它覆盖四足和人形机器人、RGB/深度/heightmap、场景随机化、Viser 训练监控、
独立 play 命令以及 sim-to-real。

其场景构建不是简单地把任意 PLY 中心直接三角化。公开流程使用 VGGT 估计
metric pose、稠密点云和法线，再用 NKSR 重建高分辨率碰撞 mesh，同时训练
3DGS 视觉表示。这说明“高保真视觉 + 独立物理 mesh + 大规模并行训练”的
总体架构已经被验证。

与本项目的差异是：

- 后端是 IsaacGym，而不是 mjlab/MuJoCo Warp；
- 碰撞几何依赖原始图像处理和额外几何估计，不是仅从现成 3DGS PLY 构建；
- 其重点是视觉 locomotion 和 sim-to-real，不是 MuJoCo 接触代理的系统评价；
- 没有使用 MuJoCo Warp 原生 splat ray tracer。

它是本项目训练方向最重要的系统基线。论文中若讨论并行视觉 locomotion，必须
正面对比，而不能只把 Isaac Lab 的传统 mesh 环境作为基线。

#### MuGS

[MuGS 官方仓库](https://github.com/Renforce-Dynamics/MuGS)实现 MuJoCo
前景与 3DGS 背景的混合渲染，并提供 `mugs_mjlab` 集成。仓库文档明确展示
4096 个 mjlab 环境、独立相机姿态和批量 `gsplat` rasterization。

需要特别澄清：**MuGS 不是 MuJoCo Warp 原生 3DGS renderer 的封装。** 在其
mjlab 集成中，MJWarp 负责批量仿真状态以及 MuJoCo 前景渲染，3DGS 背景由另一套
`gsplat`/PyTorch CUDA renderer 生成，之后通过 segmentation mask 做 alpha
compositing。“使用了 mujoco_warp”与“使用 mujoco_warp 的 3DGS 渲染器”是两件
不同的事。

它与本项目的重合点非常高：

- MuJoCo 物理；
- mjlab 环境；
- mujoco_warp 批量仿真状态或 MuJoCo 前景；
- 3DGS 相机观测；
- 视觉策略训练/评价用途。

关键差异在渲染架构。MuGS 采用 MuJoCo 前景、gsplat 背景、segmentation mask
和 alpha blending 的两阶段管线；本项目采用
[MuJoCo Warp 原生 3DGS](https://mujoco.readthedocs.io/en/latest/mjwarp/#d-gaussian-splatting-3dgs)
路径。原生路径让 splat 与物理 geom 在同一 ray traversal/depth 语义下遮挡，
同时支持 per-world splat group，但采用单次 planar slice 近似、每条 ray 有限
splat hit 等实现约束。

两种架构的依赖和部署成本也不同。MuGS 的 3DGS 路径需要 PyTorch、`gsplat`
及其 CUDA/JIT 编译工具链，并要处理 CUDA 架构、PyTorch CUDA 版本和 headless
EGL 等兼容问题；MJWarp 原生路径复用项目已有的 Warp/MJWarp 安装和 render
context，不再引入第二套 3DGS CUDA renderer。因而，本项目在安装复杂度、版本
耦合、设备内存所有权和训练生命周期方面具有明确的工程差异。

因此，“mjlab 可以显示 3DGS”已经不是新贡献，但“把 MJWarp 原生 3DGS renderer
作为 mjlab 的一等场景/传感器能力”并没有被 MuGS 覆盖。仍值得研究的是原生
MJWarp 与外置 gsplat 合成方案在安装复杂度、遮挡正确性、深度、画质、吞吐、
显存和多 world 扩展上的公平比较。

#### DISCOVERSE

[DISCOVERSE](https://arxiv.org/abs/2507.21981)（IROS 2025 Oral）是基于
MuJoCo 和 Gaussian Splatting 的 Real2Sim2Real 仿真框架。它同时支持 3DGS、
mesh 和 MJCF 资产，面向操作任务、数据生成、模仿学习、ROS 接入和零样本
sim-to-real。

DISCOVERSE 证明“MuJoCo + 3DGS + 准确物理 + 并行机器人学习”这一系统级
组合已有同行评审先例。本项目可区别于它的方向是 MuJoCo Warp 原生批量训练、
静态大场景 locomotion、自动碰撞代理的物理质量评价，以及与 mjlab manager
接口的原生一致性。

### 3.2 3DGS + 视觉控制与 Real2Sim

#### SplatGym

[SplatGym](https://arxiv.org/abs/2410.19564)从单段视频构建 3DGS 环境，
提供 ego-view、虚拟物体插入和视觉导航 RL。它从重建结果提取点云，使用
OctoMap 构造 octree，并通过 FCL 进行碰撞检测。

它已经覆盖“从神经场景提取碰撞表示并训练 RL”这一概念，但其碰撞主要服务于
导航体的 collision query，不是把复杂地形作为 MuJoCo rigid contact surface，
也没有验证足底多点接触、摩擦、穿透和高频 solver 稳定性。

#### SplatSim

[SplatSim](https://arxiv.org/abs/2409.10161)（ICRA 2025）将 Gaussian
Splatting 作为 RGB 操作策略的主要视觉表示，使用仿真数据训练并进行零样本
sim-to-real。它说明高保真 splat 视觉对 RGB manipulation policy 有实际价值，
但重点不是从 splat 自动恢复静态环境碰撞，也不是 locomotion 的高吞吐训练。

#### RoboGSim

[RoboGSim](https://arxiv.org/abs/2411.11839)提出由 Gaussian
Reconstructor、Digital Twins Builder、Scene Composer 和 Interactive Engine
组成的 Real2Sim2Real 系统，面向操作数据合成和闭环策略评价。其数字孪生仍需
为对象建立显式可交互几何。

#### Splatting Physical Scenes

[Splatting Physical Scenes](https://arxiv.org/abs/2506.04120)在 MuJoCo
中联合使用 3DGS 外观和显式 object mesh，并通过可微渲染与可微物理共同优化
物体几何、外观、机器人姿态和物理参数。它表明“3DGS 视觉 + mesh 物理”的
混合表示本身已经不是空白。

它主要处理 ALOHA 双臂操作和从不完美机器人数据构建 object-level digital
twin；本项目若聚焦大范围静态场景、训练吞吐和接触友好的低复杂度代理，仍有
不同的问题设置。

### 3.3 从 Gaussian 表示获得碰撞或安全约束

#### SplatTransform

[SplatTransform](https://github.com/playcanvas/splat-transform)是当前最直接的
工程基线。它可以：

- 过滤离群 cluster；
- 把 splat 体素化为 sparse voxel octree；
- 对室内执行 external fill，对室外执行 floor fill；
- 从 seed point carve 可通行空间；
- 输出 watertight voxel-face 或平滑 collision GLB。

这与原需求中“过滤、体素化、填洞、清理和 collision mesh”高度重合。项目可以
把它作为外部基线或可选后端，但不能在不知道该工具的情况下重新实现同类管线
并宣称算法创新。

#### Splat-Nav、SAFER-Splat 与 FOCI

[Splat-Nav](https://arxiv.org/abs/2403.02751)直接从 Gaussian ellipsoid
构造安全 polytope corridor；
[SAFER-Splat](https://arxiv.org/abs/2409.09868)用 control barrier function
在在线 Gaussian map 中做碰撞安全过滤；
[FOCI](https://rffr.leggedrobotics.com/works/foci/)（IROS 2025 Oral）把机器人
和环境都表示为 Gaussian，并以 Gaussian overlap integral 构造 orientation-
aware、可微的碰撞代价。

这些工作说明 Gaussian 的 covariance、orientation 和 opacity 中包含可用于
几何推理的信息，“只使用 center”不是唯一选择。但是，规划中的软碰撞代价或
安全约束不等同于 MuJoCo 离散接触 solver 所需的稳定碰撞面。把这些信息转化成
低复杂度、接触稳定的 mesh/heightfield/SDF，仍是本项目可以研究的问题。

### 3.4 3DGS 表面与 mesh 重建

[SuGaR](https://imagine.enpc.fr/~guedona/sugar/)（CVPR 2024）通过表面对齐
正则化后采样并进行 Poisson reconstruction；
[2DGS](https://surfsplatting.github.io/)（SIGGRAPH 2024）以 2D oriented
disks 改善几何一致性并支持 TSDF mesh extraction；
[Gaussian Opacity Fields](https://arxiv.org/abs/2404.10772)（SIGGRAPH Asia
2024）从 3D Gaussians 构造 view-independent opacity field 并提取表面。

这些方法的目标通常是视觉几何精度，不是机器人接触稳定性或最低仿真成本。
同时，它们常需要训练图像、相机参数或重新优化，而本项目希望支持只有现成 PLY
资产的场景。二者形成了可用性与几何质量之间的明确研究权衡。

### 3.5 四个拟议研究方向的专题调研

本节针对“低面数稳定碰撞、利用 Gaussian 完整属性、量化视觉—物理误差、
任务相关简化”四个方向进行第二轮检索。检索结论中的“未发现”仅表示截至报告
日期，本轮可核验的一手论文和官方项目中没有发现直接覆盖，不代表绝对不存在。

#### 3.5.1 从现成 3DGS PLY 自动生成低复杂度、接触稳定的碰撞代理

**已有工作较多，但尚未完整解决本项目的问题。**

- [SplatTransform](https://github.com/playcanvas/splat-transform) 是最直接的
  PLY-only 工程基线。它对 Gaussian 场做稀疏体素化，支持 opacity threshold、
  floater/cluster 过滤、室内 external fill、室外 floor fill、按 capsule carve
  可通行空间，并输出平滑或 voxel-face collision GLB。它证明从现成 splat
  自动得到可交互碰撞已经可行，也意味着“体素化后 marching cubes”不能单独
  作为论文创新点。其目标主要是游戏/导航碰撞，官方资料没有报告 MuJoCo 足底
  接触、摩擦、solver 稳定性或 RL 策略指标；公开 issue 也显示室内扫描仍可能
  出现 fill 泄漏和不完整几何。
- [GaussGym](https://arxiv.org/abs/2510.15352) 已把真实视频转换为 3DGS 视觉场景
  和用于足式机器人训练的碰撞 mesh。但碰撞路线使用视频、VGGT 几何和 NKSR，
  不是只从已经存在的 Graphdeco PLY 恢复碰撞。
- [SplatGym](https://arxiv.org/abs/2410.19564) 从重建点云建立 OctoMap，并用
  FCL 支持导航碰撞查询。它面向自由空间和导航体，不验证 MuJoCo 刚体接触面。
- [DREAM](https://openreview.net/pdf?id=S0FmCZ6by5)（ICLR 2026）在
  Real2Sim2Real 操作系统中用 2D Gaussian Splatting 生成 collision mesh、用
  3DGS 表示视觉，并进一步做质量辨识和策略学习。它再次验证视觉与物理双表示
  的系统路线，但输入包含视频和重建训练过程，不是任意 PLY 的低面数代理问题。
- [SuGaR](https://openaccess.thecvf.com/content/CVPR2024/html/Guedon_SuGaR_Surface-Aligned_Gaussian_Splatting_for_Efficient_3D_Mesh_Reconstruction_and_CVPR_2024_paper.html)、
  [2DGS](https://arxiv.org/abs/2403.17888)、
  [GOF](https://arxiv.org/abs/2404.10772) 和
  [OMeGa](https://openaccess.thecvf.com/content/WACV2026/html/Cao_OMeGa_Joint_Optimization_of_Explicit_Meshes_and_Gaussian_Splats_for_WACV_2026_paper.html)
  提供更准确的表面或 mesh，但优化目标主要是 Chamfer、normal consistency 和
  novel-view quality；通常还需要图像、相机、重新训练或联合优化，没有把低面数
  MuJoCo 接触稳定性作为目标。

因此，方向 1 的可发表空间不能表述为“自动生成 collision mesh”，而应缩小为：

> 在只有任意训练后 3DGS PLY 的条件下，生成面向足式机器人或接触操作的低复杂度
> MuJoCo 碰撞代理，并在相同几何误差下改善接触稳定性和仿真吞吐。

这里“接触稳定”必须通过穿透、滑移、接触法线、冲量抖动、solver overflow、
机器人跌倒率和 SPS 证明，不能由 mesh 可视化代替。

#### 3.5.2 利用 scale、rotation 和 opacity，而不是只使用 Gaussian center

**已有充分的方法依据，但从渲染几何到刚体接触仍存在转换空缺。**

- SuGaR 通过正则化让 Gaussian 向表面对齐，再利用这种对齐进行 Poisson
  reconstruction。2DGS 直接把表示改成有方向的平面 Gaussian disk，并加入深度
  distortion 和 normal consistency，使几何比普通 3DGS 更一致。
- GOF 从 ray-tracing volume rendering 定义 Gaussian opacity field，并提取其
  level set；这比把中心当作无权点云更充分地利用 density/opacity。它说明
  opacity 可以参与表面定义，而不仅是预过滤阈值。
- [SplatFace](https://openaccess.thecvf.com/content/WACV2025/html/Luo_SplatFace_Gaussian_Splat_Face_Reconstruction_Leveraging_an_Optimizable_Surface_WACV_2025_paper.html)
  提出的 splat-to-surface 距离同时考虑位置和 covariance，提供了“Gaussian
  到候选表面”的 covariance-aware 距离先例，虽然任务限定在人脸重建。
- [Distance and Collision Probability Estimation from Gaussian Surface Models](https://arxiv.org/abs/2402.00186)
  把环境 Gaussian 解释为 ellipsoid，计算其与 ellipsoidal robot 的距离、梯度和
  碰撞概率；[FOCI](https://arxiv.org/abs/2505.08510) 进一步把机器人和环境都
  表示为 Gaussian，并用 Gaussian overlap integral 做 orientation-aware trajectory
  optimization。[Splat-Nav](https://arxiv.org/abs/2403.02751) 和
  [SAFER-Splat](https://arxiv.org/abs/2409.09868) 也直接在 Gaussian map 上构造
  安全约束。它们共同证明 covariance 和 orientation 对碰撞推理是有效信号。

这些方法大多输出 mesh、规划距离或软碰撞代价，并不直接向 MuJoCo 离散接触
solver 提供三角面、heightfield 或 SDF。因而，本项目可以比较以下逐级基线：

1. center-only occupancy；
2. center + opacity 权重；
3. covariance ellipsoid 的截断占据场；
4. orientation-aware surfel/局部平面；
5. opacity/density level set；
6. 在以上表示上做 task-aware mesh、heightfield 或混合代理。

关键消融不是“mesh 看起来更完整”，而是 scale、rotation、opacity 分别对薄墙、
台阶边缘、地面法线、浮点抑制和足底接触产生多少可重复改善。还必须检查 optical
opacity 与 physical occupancy 并不等价：半透明物体、反光面和漂浮 Gaussian
可能使高 opacity 也不是可靠的实体证据。

#### 3.5.3 定量评价视觉表面与碰撞表面的偏差

**相关文献证明几何误差会影响机器人，但本轮未发现针对 3DGS locomotion 的系统
因果基准。这个方向目前具有较强研究空间。**

- DREAM、GaussGym、DISCOVERSE 和 Splatting Physical Scenes 都采用视觉表示与
  显式物理 mesh 的双表示，但重点是完整系统或最终任务成功率，没有系统扫描两者
  的空间偏移、局部缺失和简化率。
- [Contact-Rich Object Insertion](https://openreview.net/pdf?id=Z0yOJaZUK_U)
  明确把几何与接触建模误差视为 contact-rich policy sim-to-real 性能下降的主要
  来源之一，并通过插入任务逐项改进仿真真实性。它不是 3DGS 工作，但直接支持
  “几何代理误差必须用任务结果评价”的论点。
- [Geometric Red-Teaming for Robotic Manipulation](https://openreview.net/pdf?id=ux5EptB7xZ)
  使用 simulator-in-the-loop 搜索很小但物理可行的形状变化，并报告这些变化可使
  抓取、插入和关节操作策略发生灾难性失败；其候选 mesh 还需经过 CoACD 以保证
  碰撞稳定。这说明普通 Chamfer 很小并不保证策略等价。
- 传统 contact-preserving simplification 工作
  [Improving performance of rigid body dynamics simulation by removing inaccessible regions from geometric models](https://doi.org/10.1016/j.cad.2012.05.009)
  通过移除对特定碰撞对象不可达的表面来提升性能，同时保守地保持接触点。它提供
  了“按可能接触集合评价简化”的先例，但没有神经重建或学习策略实验。

建议把视觉—物理误差拆为可控制变量，而不是只比较若干算法的最终 mesh：

- 全局/局部平移、旋转和尺度误差；
- 地面高度 bias、法线噪声和低频起伏；
- 台阶边缘钝化、薄结构删除和孔洞；
- 假障碍、漏障碍和可通行区域拓扑变化；
- uniform decimation 与接触区域定向 decimation；
- 视觉保持不变而只改变 collision，以及 collision 保持不变而只改变视觉。

评价应分三层：几何层报告有符号距离、法线误差、free-space precision/recall；
物理层报告穿透、滑移、接触切换/冲量抖动、solver 迭代和 SPS；策略层报告 return、
跌倒率、落脚成功率、轨迹偏差以及跨代理/真实场景的性能下降。最后一组“视觉固定、
碰撞变化”与“碰撞固定、视觉变化”实验能区分感知误差和动力学误差。

#### 3.5.4 面向机器人任务的碰撞简化

**已有 task/contact-aware 几何思想，但本轮未发现直接以 3DGS、足底接触和 RL
训练结果联合优化碰撞代理的方法。**

- SplatTransform 的 seed、室内/室外 fill 和 capsule carve 已带有有限的 agent
  awareness：它优先保证给定尺寸角色的可通行空间，而不是只追求全局几何误差。
  但 capsule 不能表达足式机器人多接触、机身姿态或操作器 reachability。
- contact-preserving simplification 按碰撞对象的可达表面保留接触，说明可以从
  “所有三角形同等重要”转向 pair-wise contact importance。
- FOCI 利用机器人朝向和 Gaussian covariance 进行规划，说明 robot footprint 和
  orientation 应进入几何重要性计算，而不是只使用固定膨胀半径。
- Geometric Red-Teaming 以策略 rollout 和任务成败寻找最敏感的形状区域，为
  simulator-in-the-loop 的 task-aware importance 提供了方法论依据；它优化的是
  失败形状，而不是代理简化，但可反向用来识别不能被过度简化的区域。

对本项目更合适的 task-aware 目标可以写为：在给定面数或 physics SPS 预算下，
最小化机器人访问分布上的接触与任务损失。区域权重可以来自：

- 训练 rollout 的足底/轮胎/末端执行器接触频率；
- 机器人 body swept volume 和可达集；
- 地面、台阶、扶手、桌面、门框等语义或几何 affordance；
- 策略失败、穿透、滑移和 solver overflow 的空间热图；
- 对碰撞表面扰动的任务指标敏感度。

为了避免把特定策略的错误固化进 mesh，应使用训练/验证场景和策略分离，并比较
geometry-only、contact-frequency、semantic-affordance 和 policy-sensitivity 四类
权重。一个有说服力的结果应在相同面数下提高未见策略/轨迹的任务成功率，或在
相同任务性能下显著提高 MuJoCo Warp SPS。

#### 3.5.5 覆盖度与研究空间汇总

| 拟议方向 | 已有覆盖 | 尚未充分覆盖 | 本项目建议定位 |
| --- | --- | --- | --- |
| PLY 到低面数稳定碰撞 | SplatTransform、SplatGym、GaussGym、DREAM | 任意 PLY 到 MuJoCo 足底/操作接触的稳定性与吞吐联合优化 | 强工程基线之上的接触感知代理 |
| 利用 covariance/opacity | SuGaR、2DGS、GOF、SplatFace、FOCI | 将完整 Gaussian 属性转成硬接触友好的低复杂度表示 | covariance-aware occupancy/surfel 消融 |
| 视觉—碰撞偏差影响 | 双表示 Real2Sim 系统、contact-rich transfer、Geometric Red-Teaming | 3DGS locomotion 中受控误差到接触与策略性能的因果曲线 | 可复现 benchmark，可独立成贡献 |
| task-aware 简化 | capsule carve、contact-preserving simplification、policy red-teaming | 足底/末端接触分布驱动的 3DGS 代理简化 | 在面数/SPS 预算下优化任务损失 |

综合判断：方向 1 单独做成通用转换工具的新颖性有限；方向 2 是方向 1 的必要方法
增量；方向 3 是当前证据最明确、最容易形成可复现实验贡献的研究问题；方向 4
潜在创新性最高，但需要先有方向 3 的指标和基准，否则无法证明“task-aware”确实
优于普通简化。

## 4. 系统对比矩阵

下表中的“碰撞”描述其公开方法，不代表本项目已经复现验证。

| 工作 | 物理/执行后端 | 3DGS 的角色 | 碰撞/几何 | 机器人学习 | 与本项目的主要关系 |
| --- | --- | --- | --- | --- | --- |
| SplatGym, 2024 | 自定义环境 + FCL | ego-view 场景 | 点云转 OctoMap | 视觉导航 RL | 已覆盖 splat 场景、碰撞查询和 RL |
| SplatSim, 2024/2025 | 传统 simulator | RGB 主渲染表示 | 显式仿真几何 | 操作、sim-to-real | 已验证 splat 视觉训练价值 |
| RoboGSim, 2024 | Isaac Sim | 场景/物体外观 | digital twin mesh | 操作数据与评价 | 完整 Real2Sim2Real 系统先例 |
| Splatting Physical Scenes, 2025 | MuJoCo | 与 object mesh 联合 | 优化后的显式 mesh | 双臂操作 | 已覆盖 MuJoCo 混合表示 |
| DISCOVERSE, 2025 | MuJoCo | 统一渲染引擎 | mesh/MJCF 物理资产 | 并行模仿学习 | 已覆盖 MuJoCo + 3DGS + 学习 |
| GaussGym, 2025 | IsaacGym | 向量化视觉 renderer | VGGT + NKSR mesh | 并行视觉 locomotion/RL | 训练方向最接近的学术基线 |
| MuGS, 2026 | MuJoCo + mjlab/MJWarp 状态，独立 gsplat renderer | gsplat 背景 + mask 合成 | MuJoCo 显式几何 | 批量视觉训练/评价 | 工程栈接近，但不使用 MJWarp 原生 3DGS renderer |
| SplatTransform, 2026 | 引擎无关工具 | 输入资产 | voxel/SVO/collision GLB | 无 | 碰撞生成最直接的工程基线 |
| 本项目拟议方案 | MuJoCo Warp + mjlab | 原生 ray-traced splat | 自动低复杂度物理代理 | 并行 RL + 独立 Play | 必须在碰撞质量与原生集成上形成增量 |

## 5. 本项目必要性评估

### 5.1 工程必要性：高

继续实现有以下实际价值：

1. **原生渲染路径。** MuJoCo Warp 已原生支持 splat BVH、物理 geom 遮挡和
   per-world splat group。本项目可以把该能力变成 mjlab 的正式 SceneCfg 和
   camera observation，而不增加一个独立 renderer。
2. **现有任务兼容。** 用户已有 mjlab 任务、manager、reward、termination、
   checkpoint 和训练脚本；迁移到 GaussGym 或 DISCOVERSE 会产生较高成本。
3. **训练与 Play 解耦。** 当前需求明确要求 headless training 与独立 Viser
   Play，这适合 mjlab 的模块边界，也便于集群训练。
4. **只有 PLY 时的资产可用性。** GaussGym 的高质量碰撞流程依赖原始图像和
   VGGT 几何；很多已有资产只有 PLY。本项目可填补“仅有 splat 资产到可训练
   MuJoCo 场景”的工程缺口。
5. **可控依赖。** 原生 MJWarp 路径可以避免 `gsplat`、PyTorch 和另一套 CUDA
   扩展/JIT 工具链的额外版本耦合，保持 mjlab 的数组、设备、显存和生命周期
   模型一致。这是相对 MuGS 的直接工程优势。

工程上应优先复用或对比 SplatTransform，而不是从零开始构建没有差异的体素
工具。项目自己的实现只有在可复现性、MuJoCo 兼容性、任务指标或依赖约束上有
明确优势时才值得保留。

### 5.2 原始论文命题的必要性：低

若论文命题只是“在 mjlab 中加载 3DGS，生成简化碰撞并训练机器人”，必要性和
新颖性仍然不足。GaussGym、DISCOVERSE、SplatGym 和 SplatTransform 已覆盖其中
的主要研究概念；MuGS 则覆盖了 mjlab 中使用 3DGS 观测的工程场景，但没有覆盖
MJWarp 原生 3DGS renderer。原生接入是清晰的系统差异，单独看仍较难支撑算法
论文。

以下内容尤其不宜作为主贡献：

- PLY 字段读取和坐标变换；
- 调用 MuJoCo Warp 已有的 splat API；
- 普通 opacity threshold、voxelization 和 marching cubes；
- 展示机器人在 3DGS 背景中运动的视频；
- 仅报告“可以训练”而没有公平基线和接触质量指标。

### 5.3 重新定位后的研究必要性：中到高，取决于实验

推荐把论文问题改为：

> 如何从只有外观优化目标的 3D Gaussian 场景，构造面向机器人接触动力学的
> 低复杂度物理代理，并量化视觉—物理不一致对大规模策略训练的影响？

该命题与已有 surface reconstruction 的区别不是“mesh 更好看”，而是代理是否：

- 在足底、轮胎或机身接触中稳定；
- 对薄墙、台阶、斜坡、孔洞和浮点噪声保持正确可通行性；
- 以更少面数获得更高 MuJoCo Warp SPS；
- 不因视觉面和碰撞面错位导致策略学习错误行为；
- 能从只有 PLY 的资产中确定性生成；
- 能在多 world、多场景训练时保持视觉—物理 group 严格对应。

如果这些问题能通过方法和实验回答，项目仍有论文价值。

## 6. 建议的论文贡献定位

### 6.1 首选定位：task-aware collision proxy

提出一个利用 position、scale、rotation、opacity，必要时结合语义或机器人
footprint 的 collision proxy 生成方法。优化目标不只包含 Chamfer/IoU，还包含：

- 接触法线误差；
- 机器人可通行区域拓扑；
- 漏碰撞与假碰撞代价；
- 静态接触漂移、穿透和 solver overflow；
- 碰撞复杂度与 simulation SPS。

需要证明该方法优于 center-only voxel、SplatTransform、通用表面重建和人工/
ground-truth mesh，而不仅是展示若干成功案例。

### 6.2 次选定位：视觉—物理代理误差基准

建立一个可复现 benchmark，系统改变碰撞代理分辨率、偏移、孔洞、简化率和
视觉质量，测量其对 locomotion/navigation policy 的训练与迁移影响。该方向的
价值在于连接几何重建指标与机器人任务指标。

单纯发布 benchmark 的要求是：资产许可清晰、场景和任务有代表性、基线足够、
统计方法完整，并提供生成和复现实验代码。

### 6.3 配套定位：原生 MJWarp 渲染评测

将 MJWarp 原生 ray-traced splat 与 MuGS 风格的 gsplat compositor 进行比较：

- 安装步骤、二进制/CUDA 兼容矩阵和首次 JIT 时间；
- 运行时依赖数量、额外显存副本和跨框架同步；
- splat/geom 双向遮挡；
- RGB、depth 和 segmentation 语义；
- 视角外推和高阶 SH 缺失影响；
- 单/多相机、多 world、不同 splat 数的吞吐和显存；
- CUDA graph 和静态 BVH 行为；
- 策略训练中的端到端 FPS，而不是孤立 renderer FPS。

该部分适合作为系统贡献或实验贡献，不建议单独作为论文唯一创新。

### 6.4 论文主任务建议：真实重建地形上的接触丰富运动

推荐把论文主任务定义为：

> G1 在由真实场景 3DGS 重建得到的结构化复杂地形上，根据速度或 waypoint
> 命令完成稳定行走，并比较不同碰撞代理对接触动力学、策略训练和跨代理迁移的
> 影响。

这里 velocity tracking 是控制目标，不是论文贡献本身；场景应包含平地、斜坡、
楼梯、窄门、低矮障碍、离散踏步和小间隙，使机器人必须根据代理几何正确落脚。
可以将任务命名为 ``reconstructed-scene perceptive locomotion`` 或
``contact-aware terrain traversal``，不建议只称为 flat velocity tracking。

选择该任务的原因是：

1. 当前仓库已有 G1 velocity/tracking 任务、训练器和 Play 路径，开发成本最低；
2. 双足机器人的小支撑域对地面高度、法线、台阶边缘和漏碰撞敏感，容易把代理
   误差转化为可测量的穿透、滑移、错误落脚和跌倒；
3. 静态大场景与当前 3DGS 范围一致，不必先解决动态物体分割、质量、关节和抓取；
4. 同一条轨迹可在 ground-truth mesh、SplatTransform、center-only 和
   covariance/task-aware proxy 上重放，适合做因果对比；
5. 相比单纯导航，足底接触能检验 MuJoCo solver 所需的表面质量，而不只是
   free-space occupancy。

建议按以下三级实验降低“策略训练难度”对几何结论的干扰：

1. **固定运动/固定策略重放。** 在各代理上执行相同站立、直行、转弯、上下楼梯
   和定点落脚轨迹，先测穿透、接触法线、冲量、滑移、solver overflow 和跌倒；
2. **代理内训练、跨代理测试。** 分别在各碰撞代理上训练相同 velocity/waypoint
   policy，再交叉测试到 ground-truth 或高质量 mesh，报告 return 和成功率矩阵；
3. **感知运动扩展。** 加入 depth/heightmap 或 RGB，固定 collision 后改变视觉，
   再固定视觉改变 collision，从而分离感知域差异与动力学域差异。

任务选择的取舍如下：

| 候选任务 | 优点 | 主要问题 | 建议角色 |
| --- | --- | --- | --- |
| 平地 velocity tracking | 已有配置、训练稳定、迭代快 | 几乎不检验重建碰撞，难体现 3DGS 必要性 | 冒烟测试和性能基线 |
| 结构化地形 locomotion | 直接放大高度、法线、边缘和孔洞误差；符合静态场景范围 | 需要准备 ground-truth/高质量参考几何 | **论文主任务** |
| 高动态 parkour | 展示效果强，对接触和几何敏感 | 控制难度、动力学和几何误差严重混杂；GaussGym 已覆盖类似展示 | 最终 stress test，不作为唯一主任务 |
| 机器人 manipulation | 接触精度敏感，适合研究可操作表面 | 通常需要动态对象、质量、摩擦、抓取和实例分割，超出当前静态 PLY 范围 | 后续扩展或第二篇工作 |
| 点机器人/移动底盘导航 | 容易构建并可测 free-space | SplatGym、Splat-Nav、SAFER-Splat 等已有较强覆盖，不能验证足底硬接触 | 辅助拓扑/可通行性实验 |

Parkour 可以作为论文视频和极限测试，例如跨越间隙、跳上台阶或连续踏石，但不应
成为唯一证据。若 parkour 策略失败，很难判断原因来自碰撞代理、reward、动作空间、
动力学参数还是训练不足。结构化地形上的可控速度/落脚任务更适合做消融和统计。

Manipulation 的研究价值同样高，尤其是插入、沿表面滑动和放置，但更适合在完成
静态环境碰撞代理后开展。届时可选“桌面/货架重建 + 显式动态物体”的混合场景，
避免错误地把纯视觉 Gaussian 当作具有质量和摩擦的可抓取对象。

如果资源允许，论文可以用 G1 作为主结果，再用四足机器人做小规模跨形态验证。
G1 更能突出落脚与平衡误差，四足则可证明方法并非只对一种支撑结构有效。

## 7. 必须加入的基线

### 7.1 碰撞代理基线

1. center-only voxel occupancy；
2. covariance-aware Gaussian occupancy；
3. SplatTransform voxel/SVO + collision mesh；
4. OctoMap/FCL 风格查询，仅用于导航碰撞基准；
5. TSDF + marching cubes；
6. heightfield + obstacle mesh；
7. SuGaR、2DGS、GOF 或可获得的高质量 surface reconstruction；
8. GaussGym 的 VGGT + NKSR 路线（当原始图像可用）；
9. 人工或 ground-truth mesh 上界。

FOCI/Splat-Nav 不直接输出 MuJoCo 接触 mesh，但应作为“直接使用 Gaussian 几何
进行碰撞推理”的相关方法讨论；若条件允许，可以把其距离/overlap 结果用作代理
生成的监督或评测信号。

### 7.2 系统基线

- 无 3DGS 的 mjlab mesh-only 场景；
- MuGS 或等价的独立 gsplat compositor；
- GaussGym 的公开配置和性能口径；
- DISCOVERSE 的 MuJoCo + 3DGS 系统能力；
- MJWarp 原生 splat + 同一 collision mesh。

系统性能必须统一分辨率、相机数、splat 数、world 数、更新频率、GPU、warm-up
和计时区间。不能把 renderer FPS 与 physics step/s 混为一谈。

## 8. 建议实验与判定门槛

### 8.1 几何与物理实验

至少覆盖：平地、斜坡、楼梯、窄门、薄墙、小障碍物、孔洞、室内真实扫描和
室外真实扫描。

除 Chamfer、normal consistency、occupancy IoU 外，必须报告：

- foot/geom 穿透深度和漏碰撞率；
- 静态物体漂移和接触抖动；
- 台阶高度、斜坡角和通道宽度的物理误差；
- 接触数、constraint 数、solver overflow；
- mesh 面数、初始化时间、峰值内存和 physics SPS；
- 多随机种子任务成功率。

### 8.2 训练实验

至少区分三类问题：

1. 纯 proprioception：3DGS 仅用于观察，不应产生逐 step 渲染成本；
2. RGB/depth policy：比较真实感、遮挡和几何误差对学习的影响；
3. sim-to-sim：同一 checkpoint 在不同视觉 renderer 或碰撞代理中评价。

若目标包含 sim-to-real，必须使用真实机器人结果或明确把它写成未来工作，不能用
Viser 中的视觉效果替代迁移证据。

### 8.3 Go/No-Go 门槛

- **工程 Go：** 原生渲染、资产契约和 headless 训练对现有用户有直接价值，
  即使最终不投稿也值得完成。
- **算法论文 Go：** 至少在两类真实场景和一个接触丰富任务上，相对强基线同时
  改善物理正确性或任务结果，并给出复杂度/吞吐权衡。
- **系统论文 Go：** 必须有 MuGS、GaussGym 或 DISCOVERSE 未提供的可验证能力，
  不能只更换后端名称。
- **No-Go/改投工程报告：** 若最终只是调用原生 renderer 加普通 voxel mesh，
  应定位为开源工程贡献、技术报告或上游功能，不应包装成新算法论文。

## 9. 对当前实现与需求的直接影响

当前仓库已经实现的 Graphdeco PLY 解析、统一变换、过滤、per-world group 和
MJWarp 原生 render context 接入，适合作为工程 Phase 1；它建立了后续碰撞与
训练实验所需的稳定视觉入口。

下一阶段不应立刻固定一个自研 voxel 算法，而应先完成：

1. 用同一 PLY 复现 SplatTransform collision mesh；
2. 建立合成 ground-truth 场景和 MuJoCo 接触冒烟测试；
3. 实现 center-only 与 covariance-aware 两个最小基线；
4. 定义视觉资产与 collision asset 的共同 manifest 和坐标校验；
5. 用实验判断研究增量究竟来自算法、任务指标还是系统性能。

训练和 Play 继续保持独立是正确的，但 GaussGym 已同时提供训练中的 Viser
监控和独立 play 命令，因此“模块分开”是良好架构要求，不是论文创新。

## 10. 风险与未决问题

1. **时间敏感性。** 3DGS 机器人仿真在快速发展，投稿前必须重新检索最近三个
   月的论文、OpenReview 状态和代码更新。
2. **性能可比性。** MuGS、GaussGym 等数字来自不同 renderer、硬件、分辨率、
   Gaussian 数量和计时口径，未经复现不能作排名。MuGS 的 `gsplat` 数字不能
   直接代表 MJWarp 原生 3DGS renderer 的性能。
3. **原生 renderer 画质。** MJWarp 使用 planar slice 近似和有限 hit 数；原生
   集成更干净，不自动意味着视觉质量优于 gsplat。
4. **PLY 几何歧义。** 3DGS 为 photometric loss 优化，透明、反射、漂浮点和
   未观测区域无法仅靠阈值可靠恢复物理表面。
5. **输入公平性。** 只有 PLY 的方法与使用原始图像、深度、相机或人工 mesh 的
   方法输入不同，实验必须明确区分。
6. **动态对象。** 当前静态场景问题不能自然扩展到可动、铰接或可变形对象；
   不应在论文中暗示已解决完整数字孪生。
7. **许可证。** 真实场景 PLY、图像、mesh、机器人模型和第三方算法许可证必须
   在数据发布和论文图示前逐项核验。

## 11. 推荐执行顺序

1. 保留并完善当前 MJWarp 原生 3DGS 视觉接入。
2. 把 SplatTransform 作为第一个外部碰撞基线，而不是竞争性重写目标。
3. 建立 synthetic ground truth、碰撞质量指标和 physics SPS 基准。
4. 实现 center-only 与 covariance-aware 基线，验证 covariance 是否真正改善
   接触任务。
5. 再决定是否投入 task-aware collision proxy 算法。
6. 完成至少一个 proprioceptive 和一个 vision-based 训练任务。
7. 最后实现独立 Viser Play 和论文可视化，不用展示界面替代训练证据。

## 12. 主要一手资料

- [MuJoCo Warp 3DGS 官方文档](https://mujoco.readthedocs.io/en/latest/mjwarp/#d-gaussian-splatting-3dgs)
- [GaussGym 论文](https://arxiv.org/abs/2510.15352)、[项目页](https://www.escontrela.me/gauss_gym/)、[代码](https://github.com/escontra/gauss_gym)
- [MuGS 代码与文档](https://github.com/Renforce-Dynamics/MuGS)
- [DISCOVERSE 论文](https://arxiv.org/abs/2507.21981)、[项目页](https://air-discoverse.github.io/)、[代码](https://github.com/discoverse-dev/DISCOVERSE)
- [SplatGym 论文](https://arxiv.org/abs/2410.19564)、[代码](https://github.com/SplatLearn/SplatGym)
- [SplatSim 论文](https://arxiv.org/abs/2409.10161)、[项目页](https://splatsim.github.io/)
- [RoboGSim 论文](https://arxiv.org/abs/2411.11839)、[项目页](https://robogsim.github.io/)
- [Splatting Physical Scenes](https://arxiv.org/abs/2506.04120)
- [SplatTransform 代码与 collision 文档](https://github.com/playcanvas/splat-transform)
- [Splat-Nav](https://arxiv.org/abs/2403.02751)
- [SAFER-Splat](https://arxiv.org/abs/2409.09868)
- [FOCI 项目页](https://rffr.leggedrobotics.com/works/foci/)、[代码](https://github.com/leggedrobotics/foci)
- [SuGaR 项目页](https://imagine.enpc.fr/~guedona/sugar/)
- [2D Gaussian Splatting 项目页](https://surfsplatting.github.io/)
- [Gaussian Opacity Fields](https://arxiv.org/abs/2404.10772)
- [SplatFace](https://openaccess.thecvf.com/content/WACV2025/html/Luo_SplatFace_Gaussian_Splat_Face_Reconstruction_Leveraging_an_Optimizable_Surface_WACV_2025_paper.html)
- [OMeGa](https://openaccess.thecvf.com/content/WACV2026/html/Cao_OMeGa_Joint_Optimization_of_Explicit_Meshes_and_Gaussian_Splats_for_WACV_2026_paper.html)
- [DREAM](https://openreview.net/pdf?id=S0FmCZ6by5)
- [Distance and Collision Probability Estimation from Gaussian Surface Models](https://arxiv.org/abs/2402.00186)
- [Contact-Rich Object Insertion](https://openreview.net/pdf?id=Z0yOJaZUK_U)
- [Geometric Red-Teaming for Robotic Manipulation](https://openreview.net/pdf?id=ux5EptB7xZ)
- [Contact-preserving model simplification](https://doi.org/10.1016/j.cad.2012.05.009)

