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

## 2026-11-25：远程 wheel 校验码刷新

用户替换了 GitHub Release `v1.0` 的 CPython 3.13 Linux x86_64 wheel。
重新下载的文件大小为 6,682,461 字节，实际 SHA-256 与 Release API 的 digest
一致：`c34e3d93895279f93e395dc5707b8067e0b9c426487ebf63b6c986bf88d7b1b8`。
已更新 `uv.lock` 对应 cp313 条目的 hash；URL、版本及其余依赖不变。
ZIP 完整性、`mujoco==3.16.0+3dgs`、cp313 ABI/platform tag 及依赖元数据已核验。
本次只刷新安装校验信息，没有安装 wheel、运行 GPU 或改动驱动。

同一 Release 当前未提供 cp312 wheel，原 cp312 URL 返回 HTTP 404。
没有可验证的新文件，因此保留其既有配置和校验值，避免擅自改变 Python 3.12
支持范围；Python 3.12 安装仍需重新上传对应 wheel 或另行决定回退方案。

### 2026-11-25：再次替换 cp313 wheel 后刷新

Release API 记录该资产于 2026-11-25 21:57:44（Asia/Shanghai）更新。
重新下载的 cp313 wheel 为 6,682,816 字节，实际 SHA256 与 Release digest
一致：`33ef7f10e5dde0697987509f893e92703801c457cd73e5c8e2886099bdd0bc6b`。
`uv.lock` 对应 hash 已从上一轮 `c34e3d...` 更新为此值；版本、URL 和依赖
元数据保持不变。ZIP CRC、包名/版本、cp313 ABI/platform tag 和全部依赖
元数据均已核验。本次未安装 wheel、运行 GPU 或修改驱动，也未改动其它任务。
Release 资产列表仍只有 cp313 文件，cp312 的既有配置与校验值保留。

### 2026-11-25：按用户要求移除 cp312 定制依赖

用户明确要求删除 Python 3.12 的远程 wheel 依赖。移除 `pyproject.toml` 的
cp312 source 和锁文件对应定制包/引用；cp313 保留用户给出的 SHA256
`33ef7f10e5dde0697987509f893e92703801c457cd73e5c8e2886099bdd0bc6b`。
项目原有 `requires-python` 范围保持不变，Python 3.12 改走与其它未覆盖平台
一致的官方 MuJoCo 依赖，不再提供此 fork 的 native 3DGS viewer。
定制 wheel 更新工具也拒绝 Python 3.12；当前使用指南及入口说明已同步。
不修改已安装环境、GPU 或驱动。前文 cp312 保留记录是历史状态，以本节为准。

本地 source、锁文件 cp313 URL 和更新脚本按用户追加要求，统一匹配远程现有
文件名 `mujoco-3.16.0+3dgs-cp313-cp313-manylinux_x86_64.whl`。
Release API 确认该文件大小和 digest 与上一轮已核验 payload 相同；其内部
WHEEL tag 仍为 `cp313-cp313-manylinux_2_35_x86_64`。但 uv 的依赖解析不接受
外部文件名中的 `manylinux_x86_64` 标签，实测 `uv lock` 返回当前 Linux x86_64
无兼容 wheel。按用户要求保留远程同名配置，不伪称已解决安装兼容性。
锁文件中 cp312 定制包和相关引用已定向清理，Python 3.12 的官方包选择 marker
已更新；现有其它包版本、hash 和 Git revision 保持不变。当前 URL/hash/TOML
一致性不等于 `uv lock` 或安装验收通过；网络解析仍受上述标签问题阻断。

本次验证：两项更新工具 CPU 回归通过（3.12 被拒绝且不调用安装，3.13
使用当前远程同名 URL）；`make check`、TOML/marker/URL/hash 一致性和 diff
检查通过。`make docs` 构建成功，1 条 warning 为在线 intersphinx 索引无法访问。
只更新配置和锁文件记录，不声称定制 wheel 的 uv 安装已通过。

### 2026-11-25：恢复标准文件名并完成联网锁文件验证

用户已将远程资产恢复为
`mujoco-3.16.0+3dgs-cp313-cp313-manylinux_2_35_x86_64.whl`。
本地 source、锁文件和更新工具/回归测试均已同步恢复。重新下载的实际文件
SHA256 仍为 `33ef7f10e5dde0697987509f893e92703801c457cd73e5c8e2886099bdd0bc6b`，
大小 6,682,816 字节，ZIP CRC 和内部 cp313/manylinux_2_35 标签核验通过。
此前非标准外部文件名导致的兼容性阻断已解除。

联网 `uv lock` 和 `uv lock --check` 均成功解析 244 个包，包名/版本集合保持
不变；cp312 定制依赖保持移除，3.12 官方包与 3.13 定制包的 marker 路径已逐项
核验。两项更新工具 CPU 回归、`make check` 和 diff 检查通过。
离线复查曾因缓存的 NVIDIA warp-lang 索引候选不足而失败，后续联网复查通过；
这不属于 wheel 文件名/校验错误。本次未安装依赖或执行 GPU/native viewer。

用户随后确认已更新本地安装。CPU 核验实际加载路径、映射库及11个原生文件，
均与33ef7f10 wheel一致；训练相机所用已安装Warp仍为8ffab7a。用户选择先测
此环境，详情及准备好的全量复测协议见
`validation/p10-current-full-2026-11-25.md`。文件一致性不代替GPU性能验收。
