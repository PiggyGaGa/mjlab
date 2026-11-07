# P6 视觉训练方法、论文来源与复现说明

更新日期：2026-11-07。P6 是工程基线：将仓库已有 YAM Vision 路径迁移到
G1 3DGS depth locomotion；不把以下已有算法或编码器作为本项目原创贡献。

## 1. 方法与来源的对应关系

| 组成 | 原始论文/实现来源 | 本项目实际使用与边界 |
|---|---|---|
| PPO clipped policy objective | Schulman et al., *Proximal Policy Optimization Algorithms*, 2017, [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) | 由已安装 RSL-RL 的 PPO 实现优化 actor/critic；不是重新提出 PPO |
| Generalized Advantage Estimation | Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)，预印本 2015 | RSL-RL rollout 中计算 GAE；本配置 gamma=0.99、lambda=0.95 |
| CNN + spatial softmax / expected feature position | Levine, Finn, Darrell, Abbeel, *End-to-End Training of Deep Visuomotor Policies*, JMLR 17(39):1–40, 2016, [正式论文](https://jmlr.org/papers/v17/15-522.html) | 复用其空间特征位置编码思想；本项目使用 PPO，不使用论文中的 guided policy search，也不复现其完整网络或实验 |
| 静态 3D Gaussian 场景表示 | Kerbl, Kopanas, Leimkühler, Drettakis, *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, ACM TOG 42(4), 2023, [作者项目页](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) | 加载已有 PLY，以 MJWarp 渲染 depth；不训练 Gaussian 参数，不把渲染深度等同于真实测距真值 |
| 工程训练框架 | [RSL-RL 官方仓库](https://github.com/leggedrobotics/rsl_rl) 与本仓库 YAM Vision 配置 | 软件依赖，不应冒充独立的视觉 locomotion 论文；实际安装版本记录在运行报告 |

上述 spatial softmax 来源是结构出处；它原用于视觉操作策略，不能由此声称
已经复现其 manipulation 算法或证明 G1 locomotion 有视觉收益。
YAM Vision 在本项目中指仓库已有的任务配置与模型实现，不指定一篇未经核实的
“YAM Vision 算法论文”。

## 2. 对应代码

- G1 depth task：`src/mjlab/tasks/velocity/config/g1/env_cfgs.py` 中
  `unitree_g1_flat_3dgs_depth_env_cfg`。
- G1 PPO/CNN 配置：`src/mjlab/tasks/velocity/config/g1/rl_cfg.py` 中
  `unitree_g1_3dgs_depth_ppo_runner_cfg`。
- 复用起点：`src/mjlab/tasks/manipulation/config/yam/rl_cfg.py` 中
  `yam_lift_cube_vision_ppo_runner_cfg`。
- Spatial softmax 模型：`src/mjlab/rl/spatial_softmax.py`。
- 时序模型薄适配层：`src/mjlab/rl/temporal_spatial_softmax.py`；历史缓冲
  直接复用 `ObservationManager`，未另建项目专用 buffer。
- Depth observation：`src/mjlab/envs/mdp/observations.py` 中 `camera_depth`。
- Runner/检查点：`src/mjlab/rl/runner.py`，继承 RSL-RL OnPolicyRunner。

实现关系为：固定场景 → MJWarp depth camera → CNN → spatial softmax
→ 与本体特征拼接 → MLP actor/critic → PPO。训练梯度更新策略和价值网络，
不穿过 MuJoCo 物理或 Gaussian renderer 反传到场景资产。

## 3. 当前网络与观测定义

相机固定在 `robot/torso_link`，局部位置 `(0.08, 0, 0.42)`，朝前下俯
20 度，垂直 FOV 70 度。图像为 `(B, 1, 48, 64)` float32，控制步更新，
无帧历史。G1 开启 `zero_depth_as_far=True`，将 renderer 的缺失深度零值
映射为 5 米，再计算 `clip(depth, 0.05, 5.0) / 5.0`；有效近端下限为
0.01，并非减去 near 后再缩放。映射后的 1 同时包含缺失和超出 cutoff，
不能解释为测到了 5 米处的表面或已证明前方无障碍。其他任务默认仍使用
旧归一化行为。renderer 的深度定义与诊断见
`validation/p6-depth-quality-2026-10-30.md`。
PNG 仅作诊断：固定灰度范围，白色远、黑色近；NPY 保存归一化浮点值。

CNN 为两层卷积，通道 `[16,32]`，kernel `[5,3]`，stride `[2,2]`，
zero padding、ELU，无 max pooling。输出 32 个 `12×16` 特征图，
temperature=1 的 spatial softmax 每通道生成两个期望坐标，共 64 维。
当前代码的坐标缓冲 `pos_x` 对应图像行、`pos_y` 对应列，输出顺序实际为
每通道 `(row, column)`，均使用 `[-1,1]` 的归一化网格；写论文公式时应
明确该约定，不能只根据变量名把第一项误称为水平坐标。

对通道 c、图像位置 (i,j)，计算：

```text
p[c,i,j] = exp(z[c,i,j] / T) / sum_{u,v} exp(z[c,u,v] / T)
k_row[c] = sum_{i,j} p[c,i,j] * row[i]
k_col[c] = sum_{i,j} p[c,i,j] * col[j]
```

actor 接收 99 维本体 observation 和 64 维视觉特征，critic 接收 109 维
特权 observation 和自己的 64 维视觉特征；两者都有 camera 输入，不能
描述为“critic 不读取视觉”。当前配置创建各自 CNN，而非共享 encoder。
MLP hidden dims 均为 `(512,256,128)`、ELU，并启用 observation normalization。
actor 使用 Gaussian action distribution，输出 29 个动作；动作经 G1
joint-position action 的 scale 和默认 offset 转为关节目标，不是直接电机力矩。

视觉任务移除了 `foot_height_scan`、critic 的 `foot_height` 和依赖该 sensor
的两个奖励项，以免共享 BVH 的隐藏 plane 遮挡 3DGS。对照实验应统一这些
奖励/观测差异，不能直接将与普通任务的全部差值归因于视觉。

## 4. 优化设置与论文写法

独立对照任务 `Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth-Validity` 在相同
depth 后增加 `float(isfinite(raw_depth) & (raw_depth > 0))` 通道，camera 为
`(B,2,48,64)`。大于 5m 的有限正值仍有效；零值缺失编码为 `(1,0)`，
有效远端为 `(1,1)`。这是一项工程观测消融，不改变 PPO 或 Spatial
Softmax 算法，也不引入额外渲染。单通道任务保留独立 task/experiment。
两种编码的首层卷积不同，checkpoint 不直接兼容；详见
`validation/p6-validity-2026-10-31.md`。初步测试只证明可用性区分与训练
连通性，不能用于宣称视觉任务收益。

独立时序任务
`Mjlab-Velocity-Flat-Unitree-G1-3DGS-Depth-Validity-History` 保留连续四帧，
camera shape 为 `(B,4,2,48,64)`，历史顺序为从旧到新。reset 后由仓库现有
history buffer 用首帧回填，避免把全零伪历史暴露给策略。RSL-RL 的 2D CNN
接口只接受 `(B,C,H,W)`，因此薄适配层仅在 encoder 边界把 `(T,C)` 合为
8 个输入通道，后续 Spatial Softmax、MLP 和 PPO 配置不变。该实现是固定
四帧的首个工程基线，尚未实现 stride、延迟或噪声，也没有证明时序输入
改善避障表现。

robust 对照保存七个连续控制步，并由 encoder adapter 以 stride 2 选择
`(t-6,t-4,t-2,t)` 四帧。任务复用主线 observation delay/history，固定增加
一个控制步延迟；项目专用组合 observation term 以两控制步周期
sample-and-hold，并在同一 depth+validity 张量上执行量程噪声、3×3 blur、
随机矩形缺失和整帧 dropout，保证缺失像素仍为 `(1,0)`。随机 corruption
在 play 配置关闭。另有 asymmetric 对照让 actor 读取 camera、critic 只读
109 维特权状态。以上均是鲁棒性接口和消融入口，不是效果结论。

PPO clipped objective 使用 ratio r=pi_new(a|o)/pi_old(a|o)：

```text
L_clip = E[min(r * A, clip(r, 1-epsilon, 1+epsilon) * A)]
delta_t = reward_t + gamma * nonterminal_t * V(next_t) - V(t)
A_t = delta_t + gamma * lambda * nonterminal_t * A_(t+1)
```

实现会处理 termination/time-limit bootstrap；不能忽略掩码照搬示意式。
当前 epsilon=0.2，value loss coefficient=1，entropy coefficient=0.01，
clipped value loss 开启，5 learning epochs，4 minibatches，Adam 初始
learning rate=1e-3，adaptive schedule、desired KL=0.01，gradient norm
上限 1，rollout 默认 24 steps/environment。准确参数以报告中的配置为准，
它们是本工程设置，不声称与上述论文超参数完全相同。

可在论文方法中表述为：

> We train a depth-conditioned G1 policy using PPO with generalized advantage
> estimation. A convolutional encoder followed by spatial softmax and expected
> feature coordinates, following Levine et al. (2016), embeds rendered depth.
> The visual features are concatenated with proprioceptive observations for
> the actor and privileged observations for the critic. Static Gaussian assets
> provide visual observations through the MuJoCo Warp renderer, while separate
> collision geometry governs physical contacts.

以上是基线方法描述，仍需配合实验配置和引用；不能写成已经证明优于 blind
policy。视觉收益需要 P7 的 blind/privileged/mesh-depth/3DGS-depth 对照，
以及冻结帧、遮挡或关闭相机的消融。

## 5. P6 smoke 复现入口

```bash
uv run tests/run_g1_3dgs_vision_smoke.py --num-envs 1
uv run tests/run_g1_3dgs_vision_smoke.py --num-envs 2 \
  --scene-manifest src/mjlab/asset_zoo/3dgs/room1/manifest.json \
  --output /tmp/p6-vision-manifest-2
```

脚本从头构建视觉网络，不使用旧 98 维 velocity checkpoint。默认两次
24-step rollout 和 PPO 更新，保存 depth PNG/NPY、`vision_smoke.pt` 与
`report.json`。无外部服务上传；使用真实 PPO act/process/returns/update
路径并检查 CNN 参数变化、有限 loss/gradient、相机置零后的输出敏感性，
以及 actor/critic/optimizer checkpoint 重载，并继续一次 rollout/PPO 更新。

相机置零敏感性仅证明网络连接有效，不证明学习到了有效视觉行为。
多环境 partial reset 验证同一 splat asset 下的 row 隔离，不覆盖不同
`world_asset_ids` 的资产切换。PyTorch peak allocation 不包含 Warp/GL 的
全部显存，不能作为完整显存 benchmark。

固定 reset 姿态的相机方向、自遮挡分离与解析 depth 语义已验证，但不代表
动态全姿态或所有场景的深度质量合格。本阶段仍需动态自遮挡/空洞处理、不同资产组隔离、
较大规模 CUDA graph/显存/吞吐测试和更长训练实验。单个短测试不能证明
收敛、稳定行走或训练效果。
