# P3 六项验收记录（2026-10-30）

结论：自动验证部分已推进，P3 不整体关闭。

## 环境与复现

- Linux x86_64，glibc 2.35，Python 3.13。
- MuJoCo `3.16.0+3dgs`，GitHub v1.0 Release wheel。
- Torch `2.14.0+cu130`，NVIDIA RTX 4070 Laptop GPU 8 GiB，驱动 580.159.03。
- room1 manifest：`src/mjlab/asset_zoo/3dgs/room1/manifest.json`。
- checkpoint：用户指定 run 的 `model_10000.pt`。

```bash
uv run tests/run_g1_3dgs_acceptance.py \
  --checkpoint /home/yzt/humanoid/Unitree/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/2026-08-25_12-42-10/model_10000.pt \
  --legacy-phase-observations --output /tmp/p3-velocity-adapted-final.json
```

原始结果：[四轮自动验收](p3-native-2026-10-30.json)、
[未适配时的严格加载失败](p3-velocity-incompatible-2026-10-30.json)。

## 六项结果

| 项目 | 结果 | 限制或下一步 |
|---|---|---|
| velocity 策略联调 | 显式适配后通过，四轮共 1,806 个计时策略步 | 当前标准配置直接加载失败；不代表训练环境完全兼容或零跌倒 |
| Viser/native 一致性 | PLY 传输几何、DC 颜色回归通过 | 尚未完成真实画面、碰撞代理、尺度/朝向和遮挡的人工对照 |
| 碰撞显示不影响物理 | CPU 相同初始状态/控制的 100 步重放通过 | 尚未完成完整 MJWarp 策略重放 |
| 控制与生命周期 | 四轮真实窗口的程序回调、相机参数、退出和 PLY 清理通过 | 尚未实际操作键盘/鼠标；不是长期无泄漏证明 |
| 接口契约 | 版本 1 文档与传输回归完成 | 不承诺不同 collision model 的原子切换或跨版本兼容 |
| 性能 | 四轮初测已记录 | 未测显示 FPS、进程专属显存、完整初始化耗时；尚非正式 benchmark |

旧 checkpoint 的 actor 为 98 维，当前标准任务为 99 维。旧训练保存的配置
包含 `0.6 s` sin/cos phase，actor 无 base linear velocity，且字段顺序与
当前不同。测试工具按旧配置显式恢复 actor 顺序及 unbiased joint position；
严格加载权重通过，没有裁剪、补零或改写权重。适配仅存在于验收脚本中。
详见 [接口契约](../mjlab-3dgs-native-interface.md)。

## 性能初测

同一个已加载环境按 off/on/off/on 打开四次窗口，各预热 10 步、采样
10 秒；下表统计仅包含计时段。Actual RT 为推进的仿真时间除以墙钟时间。
每个实际显示同步帧均核对 CPU qpos/qvel 与 MJWarp 对应行完全一致。

| 轮次 | 3DGS | viewer setup 秒 | 策略步 | Actual RT | 提交频率 Hz | 设备总显存 MiB |
|---|---|---:|---:|---:|---:|---:|
| 1 | 关闭 | 0.368 | 499 | 0.997 | 59.94 | 1751 |
| 2 | 开启 | 0.904 | 442 | 0.882 | 30.53 | 1864 |
| 3 | 关闭 | 0.486 | 499 | 0.998 | 60.00 | 1733 |
| 4 | 开启 | 0.933 | 366 | 0.731 | 28.77 | 1839 |

3DGS 加载 799,083 个 Gaussian。设备总显存包含桌面和其他应用，窗口开始前
为 1012 MiB，全部关闭后为 999 MiB。该结果不能证明长期无泄漏。
viewer setup 包含窗口创建和 renderer readiness，不包含场景初始化、源
PLY 导入或 checkpoint 加载。提交频率不是实际显示帧率；顺序执行、不同
reset 轨迹及 GPU 竞争会影响比较。不能将开关差值视为严格渲染成本。

两轮开启 3DGS 后，CPU 碰撞重放分别记录 680、533 个接触样本，隐藏/显示
group 3 的 qpos、qvel、接触对和接触距离完全一致。仅对显示 mask 进行切换，
没有修改真实碰撞模型或接触参数。

## 检查与剩余人工验收

以下检查通过：

```bash
make check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest \
  tests/test_native_gaussian_contract.py tests/test_native_viewer_gaussians.py \
  tests/test_native_viewer_actions.py tests/test_viser_update_policy.py -q
```

结果：Ruff、ty、Pyright 通过；`13 passed`。禁用 pytest 插件自动发现是为
避免系统 ROS 插件因缺少 `lark` 导致收集失败。未运行完整 `make test`。

人工对照可用同一验收脚本加 `--interactive --viewer native`，再使用
`--interactive --viewer viser`；均保留 `--legacy-phase-observations`，加
`--collision-visible` 显示代理。需记录：场景与机器人/碰撞的尺度、朝向、
对齐和前后遮挡；鼠标 orbit/zoom、键盘暂停/单步/reset/继续、手动退出。
当前未记录这些人工结果，相关 TODO 保持未勾选。
