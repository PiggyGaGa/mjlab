# 原生 MuJoCo 3DGS 接口契约与验收边界

接口版本：1；验证日期：2026-10-30；对应 MuJoCo Release：
`https://github.com/PiggyGaGa/mujoco/releases/tag/v1.0`。

## Native 半透明叠加更新（2026-11-10，本地源码）

- 外部 MuJoCo 仓库的经典渲染器新增可选透明阶段前回调，native viewer
  使用“不透明几何/背景 → 3DGS → 半透明 geom/mesh → 标签/UI”的顺序。
  原 `mjr_render()` 保持空回调；不改透明分类阈值或物理碰撞。
- 这是碰撞调试覆盖层，不保证透明 mesh 与 Gaussian 的逐片元联合排序。
  立体、正交和 scene transform 等原有 3DGS 限制仍在。
- `simulate` 及新增 `mjgs_transparency_test` 编译通过。OpenGL 4.6 NVIDIA
  实测 alpha=0.1/0.5/1 的真实 mesh 混合通过；隐藏/无资产时逐像素等同
  原始渲染；segmentation 不调用外部颜色绘制；截图包含完整透明阶段。
- 原有 `mjgs_render_test` 的遮挡、viewport 偏移、显示切换、GL 状态和空视锥
  回归通过；`mjgs_asset_test`、`mjgs_lifecycle_test` 均通过。
- 以上属于本地修复，不代表 GitHub v1.0 Release 已包含变更。mjlab 必须使用
  由修改后源码构建的 wheel；安装后用 `uv run --no-sync` 保留本地构建版本。
- 已用 GCC 11 构建并安装 Python 3.13 本地 wheel：
  `mujoco/dist/mujoco-3.16.0+3dgs-cp313-cp313-manylinux_2_35_x86_64.whl`。
  安装库导出透明阶段回调；`tools/test_python_3dgs.py --gpu` 的两个窗口均完成
  加载、重载、关闭及临时 PLY 清理。未重新发布远端 wheel。
- Native demo 默认测试入口改为 ArtanimMotionCaptureStudio manifest。已运行
  `uv run --no-sync tests/run_g1_3dgs_native_demo.py --steps 200 --collision-visible`：
  3DGS renderer ready、预训练 G1 策略 200 步及有限状态检查通过。
  当前资产柱子 alpha=0.8、地面 alpha=0.2；未修改这些用户设置。该冒烟测试
  不代表长期策略稳定性或完整场景的像素级遮挡验收。

## Release wheel 校验更新（2026-11-10）

GitHub v1.0 的同名 cp312/cp313 wheel 已替换。本次重新下载两个完整文件，
通过 ZIP CRC 检查并计算 SHA-256，更新 `uv.lock`；下载 URL 和版本保持不变。
uv 使用 SHA-256，而非 MD5。

- cp312（6,675,586 bytes）：
  `1b8d11b7205a240223faefd6262dd1fa5003f554e0578b4688ea65c77e6b21fb`
- cp313（6,680,874 bytes）：
  `ff9e145b2a1b2e259863d51f7270e5fce42d18e31b3a6e88db20132da5b9f5f7`

以上数值记录该次下载，不要求之后同名 wheel 更新时继续手工维护。

## 不依赖锁文件哈希的 wheel 更新流程

当前 uv 0.13.5 强制要求 `uv.lock` 的 URL wheel 具有 hash，删除字段会在解析
锁文件时失败。`--no-verify-hashes` 是 `uv pip install` 选项，不是 `uv sync`
的关闭开关。因此保留有效锁文件，增加独立安装工具：

```sh
uv run --no-sync scripts/tools/update_mujoco_wheel.py
uv run --no-sync tests/run_g1_3dgs_native_demo.py --interactive --collision-visible
```

工具按当前解释器选择 cp312/cp313，使用 `uv pip install --no-verify-hashes`
直接重装同名 Release wheel，不读取 `uv.lock` 的校验值，也不改写仓库文件。
`--no-cache` 避免复用旧文件，`--no-deps` 保持其他依赖不变。首次环境初始化仍
使用正常 `uv sync`；之后对反复替换的 Release 使用此工具和 `--no-sync`。
若再次使用普通 `uv sync` 或 `uv run`，仍会恢复锁文件约束；没有宣称关闭其
强制校验。工具支持 `--dry-run`，仅适用于 Linux x86_64、CPython 3.12/3.13。
Ruff、ty、Pyright 和离线锁文件一致性检查通过。联网试运行未获用户允许，
因此本次没有执行网络验证或更换已安装的 wheel。

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

### 外部场景的碰撞缓存（2026-11-10）

G1 3DGS tracking/velocity 回放将 `sim.nconmax` 提高到至少 256，保留用户
设置的更大值。基础平面任务和训练配置保持原值。原 35 的容量可能在碰撞
初筛时溢出并截断候选对；关闭 `warn_overflow` 只隐藏提示，不能恢复漏掉的
候选。此设置同样作用于 Viser 与 native viewer 后面的 MJWarp 仿真。
256 是回放默认余量，不保证任意复杂场景都足够；若仍溢出需按实际负载增大。

验证：`tests/test_g1_3dgs_contact_capacity.py` 的 3 项配置回归通过；使用
ArtanimMotionCaptureStudio manifest 运行 native demo 1000 步，策略步进与
有限状态检查通过，日志没有 overflow 提示。该测试不等于所有姿态和场景的
容量上界证明。修改文件通过 Ruff、ty 和 Pyright 检查。

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
