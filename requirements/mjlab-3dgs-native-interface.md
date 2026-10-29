# 原生 MuJoCo 3DGS 接口契约与验收边界

接口版本：1；验证日期：2026-10-30；对应 MuJoCo Release：
`https://github.com/PiggyGaGa/mujoco/releases/tag/v1.0`。

## 场景接口

场景入口仍为 schema v1 `manifest.json`。mjlab 负责校验资产指纹、依赖、
统一 transform 和 spawn，构建碰撞 MJCF 模型，并加载规范化 splat 数组。
外部 viewer 不再对导出的 PLY 施加第二次场景变换。

`env.scene.gaussian_splat_data` 为 `GaussianSplatData`，提供：

| 字段 | 形状 | 语义 |
|---|---|---|
| `position` | `(N, 3)` | MuJoCo world 坐标，单位米 |
| `rotation` | `(N, 4)` | 归一化 `wxyz` 四元数 |
| `scale` | `(N, 3)` | 正数，世界尺度下的三个主轴 scale |
| `rgba` | `(N, 4)` | `[0, 1]`，DC 静态颜色与 opacity |
| `adr` | `(A+1,)` | 每个资产在上述数组中的起止索引 |
| `group_id` | `(num_envs,)` | 每个 world 对应的资产编号 |

`mujoco.gaussian.GaussianAssets(data).path_for_world(env_idx)` 将对应资产写为
binary little-endian Graphdeco PLY，反编码 linear scale/opacity 为 log/logit。
端点 opacity 会截断到 float32 epsilon 范围；DC 颜色不含高阶 SH。
同一资产复用缓存文件，`close()` 删除临时文件；须在关闭 viewer 后调用。

`mujoco.viewer.launch_passive(model, data, gaussian_ply=path)` 打开原生窗口。
handle 的 `gaussian_status` 必须等待到 `ready`，`error:` 前缀表示加载失败。
`load_gaussian_ply(path)` 排队切换资产，随后也须等待就绪。当前 mjlab 仅在
资产编号改变时调用；不支持通过原地修改数组自动刷新资产。

## 状态与物理接口

mjlab 的 MJWarp 环境继续负责策略和物理；native viewer 使用同一个 CPU
`MjModel` 和用于显示的 `MjData`。每次同步选择 `env_idx` 后复制 `qpos`、
`qvel`、`ctrl`、mocap pose 和外力，并调用 `mj_forward`；显示端不另行
推进一套机器人动力学。模型随机化字段另由现有 model-field 同步处理。

机器人状态使用模型内的 joint/body 顺序。切换环境时，状态行和
`group_id[env_idx]` 必须对应；当前测试未覆盖不同 collision model 的
跨场景原子切换。Viser 的 world offset 是多环境展示偏移，不能再烘焙进
原生单环境 PLY。

外部碰撞使用 geom group 3。显示开关只修改 `MjvOption.geomgroup[3]`，
不得修改 `geom_contype`、`geom_conaffinity`、质量或接触求解器。
renderer、窗口和导出 PLY 生命周期由 native viewer 管理，环境资源由调用方关闭。

## 策略兼容性

指定的 August G1 velocity `model_10000.pt` 需要 98 维 actor observation，
当前标准 G1 velocity 为 99 维，直接 `play` 会严格加载失败。
验收工具的显式 `--legacy-phase-observations` 仅适用于此旧配置：

1. actor 顺序为 angular velocity、projected gravity、command、phase、
   relative joint position、relative joint velocity、last action。
2. phase 为 period `0.6 s` 的 sin/cos；command norm 小于 `0.1` 时置零。
3. joint position 使用旧版 unbiased 语义，移除当前 actor 的 base linear velocity。

以上依据该 run 的 `params/env.yaml` 和旧仓库的 phase 实现核对。
适配不修改 checkpoint 权重，也不修改生产任务默认配置。其他机器人、动作
顺序、训练动力学和随机化差异不在该适配的兼容承诺内；成功运行不代表策略
迁移质量或零跌倒。critic 未用于此次推理验收。

## 验证与尚缺证据

`tests/test_native_gaussian_contract.py` 使用非对称位置、各向异性 scale 和
非单位旋转，检查 native PLY 往返后的 position、covariance 和 DC RGBA 与
Viser 使用的规范化几何一致。它验证传输数学，不证明最终画面或遮挡一致。

`tests/run_g1_3dgs_acceptance.py` 验证真实窗口上的回调/动作队列、暂停、
单步、reset、继续、相机参数、退出和临时文件清理；重复运行四个窗口。
CPU MuJoCo 接触重放使用相同初始状态与控制，比较碰撞显示开关两种设置
下的 qpos/qvel、接触对和距离；不替代完整 MJWarp 策略轨迹重放。

性能比较复用同一个已加载环境，以 off/on/off/on 顺序分别采样 10 秒，
预热 10 个策略步；记录 viewer setup 时间、Actual RT、viewer 提交频率和
设备总显存。初始化/PLY 导入/checkpoint 加载耗时未计入 viewer setup。
提交频率不是显示器实测 FPS，设备总显存包含桌面及其他进程；此为初测，
不作为严格性能基准或长期无泄漏证据。

仍需人工对照两个前端的坐标、尺度、朝向与遮挡，并实际操作鼠标/键盘。
使用同一旧策略的对照入口（`--viewer native` 或 `--viewer viser`）：

```bash
uv run tests/run_g1_3dgs_acceptance.py \
  --checkpoint /path/to/model_10000.pt \
  --legacy-phase-observations --interactive --viewer native --collision-visible
```

退出交互会话仅记录 `interactive_session_closed`，不自动宣称人工验收通过。
