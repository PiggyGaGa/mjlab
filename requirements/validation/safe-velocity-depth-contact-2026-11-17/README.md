# G1 可见深度停车与 pelvis 接触事件 GPU 验收

状态：已完成，按当前范围收口（2026-11-17，用户决定）。不再追加场景测试或
复测。结果固定为 33/36，三个轻微减速项作为已知限制保留；原始通过标志和
失败记录不变。下文命令仅用于实验归档，不代表安排再次运行。

日期：2026-11-17。设备：RTX 3060 12 GiB，驱动 580.82.07，Warp 1.18.0。
经沙箱外执行验证 GPU 可用；早先沙箱内的 CUDA 不可见记录不代表机器没有 GPU。

## 固定条件

Checkpoint：

```text
logs/rsl_rl/g1_3dgs_safe_velocity_blind/2026-11-12_22-38-51_blind_conservative_seed42_100/model_99.pt
```

SHA-256：`cc2a17909d774af9071633b3d27a2d099a4bd9c1419b611fbbbf31a29d80f417`。

冻结 locomotion actor，nominal command `(0.5, 0, 0)`，stop-only 深度安全层；
stop 0.4 m、全速 1.4 m、前向速度上限 0.5 m/s、7 帧净空中位数及重叠局部窗口。
使用 play 配置，保留原 startup 动力学随机化；seed 为 0、1、2。
本次不修改深度处理、速度阈值、物理资产或 checkpoint。

相机 BVH 启动检查要求 group 仅包含 1/2、无共享 raycast sensor。运行时感知为
3DGS + 可视 mesh，隐藏 group 3 只参与物理与接触检测。主碰撞指标是 pelvis
与非地面障碍物的 ContactSensor；当前 found>0 或历史任意正力均记为接触。
trace 记录 nominal/safe/measured velocity、alpha、净空、接触目标/力和终止原因。

停车接受条件为平面实测速度连续至少 1 s 不超过 0.1 m/s；期间保持制动/停车
模式、无 pelvis 接触、无越界/跌倒/termination。到达该条件后可提前结束。
free/false_alarm 必须完成原 10 s 窗口且没有安全层介入，才能满足连续跟踪判据。

## 首轮诊断与判据修复

原始文件保留为 `apartment-matrix.json` 和 `studio-matrix.json`，不要覆盖。
这两份文件使用旧累计静止时间判据：apartment 得到 15/18，Studio 得到 18/18，
但不能据此关闭连续停车验收。

独立反算末尾连续静止时间发现，apartment 的 12 个停车结果均不足 1 s，
范围为 0.24--0.94 s；Studio 有 2 个不足 1 s。原因是 evaluator 在 BRAKE/BLOCKED
模式内速度超阈值时，没有清除旧的静止时间，错误地累加多段短暂停留。
任务侧 evaluator 现已修复，速度或模式不满足条件立即重置计时。
新报告同时记录 `evaluation.stationary_stop_rule=continuous_hold` 和独立反算的
`summary.final_stationary_hold_s`，以便核查。

另修正接触 telemetry：当 force history 存在时仍检查当前 found，覆盖“已接触
但历史力为零”及“当前已分离但历史力仍为正”的情况。Depth-Stop 的 terminated
run 不得判为 accepted。所有旧实验文件继续保留。

## 严格复验命令

```bash
uv run --no-sync tests/run_g1_3dgs_safe_velocity_adapter.py \
  --controller depth_stop \
  --checkpoint logs/rsl_rl/g1_3dgs_safe_velocity_blind/2026-11-12_22-38-51_blind_conservative_seed42_100/model_99.pt \
  --scenario free left_open right_open both_open dead_end false_alarm \
  --seed 0 1 2 --device cuda:0 --require-pass \
  --output requirements/validation/safe-velocity-depth-contact-2026-11-17/apartment-continuous-hold-matrix.json

uv run --no-sync tests/run_g1_3dgs_safe_velocity_adapter.py \
  --controller depth_stop \
  --checkpoint logs/rsl_rl/g1_3dgs_safe_velocity_blind/2026-11-12_22-38-51_blind_conservative_seed42_100/model_99.pt \
  --scenario free \
  --scene-manifest src/mjlab/asset_zoo/3dgs/ArtanimMotionCaptureStudio/manifest.json \
  --scene-spawn-prefix safe_velocity_generalization_ \
  --nominal-command 0.5 0 0 --allowed-region -10 10 -10 10 \
  --expected-outcome safe_blocked_stop --steps 800 \
  --seed 0 1 2 --device cuda:0 --require-pass \
  --output requirements/validation/safe-velocity-depth-contact-2026-11-17/studio-continuous-hold-matrix.json
```

Studio 的 `--scenario free` 仅复用协议参数，实际出生点来自 manifest 的六个
泛化位置/朝向，目标结果显式指定为 safe_blocked_stop。障碍是额外可视柱体，
此结果不扩大解释为原始 3DGS 中所有真实障碍均已验收。

## 严格复验结果

以下仅引用带 `continuous-hold` 文件名的新报告，首轮文件不替代该结果。

| 矩阵 | 通过 | pelvis 接触 | 跌倒 | termination | 最小统计净空 |
|---|---:|---:|---:|---:|---:|
| Apartment 六场景 × 三 seed | 15/18 | 0 | 0 | 0 | 0.6786 m |
| Studio 六泛化 spawn × 三 seed | 18/18 | 0 | 0 | 0 | 0.7131 m |

合计 33/36 通过。其中障碍停车 30/30，自由/false-alarm 零介入跟踪 3/6。
全部 36 个 run 的实际 camera BVH geom group 都是 `[1,2]`，无共享 raycast。
逐 trace 核查 12,075 行，未发现 pelvis contact event 或 done；30 个停车结果
末尾连续静止时长均为 1.00--1.02 s，与报告独立反算字段一致。

| Apartment 场景 | 通过 | 三 seed 最小统计净空 | 时长范围 |
|---|---:|---:|---:|
| free | 1/3 | 1.3055 m | 10.00 s |
| left_open | 3/3 | 0.7614 m | 2.96--3.38 s |
| right_open | 3/3 | 0.7524 m | 2.92--3.46 s |
| both_open | 3/3 | 0.7245 m | 3.06--3.44 s |
| dead_end | 3/3 | 0.6786 m | 3.26--4.00 s |
| false_alarm | 2/3 | 1.3384 m | 10.00 s |

Studio 路径长 2.2864--3.2962 m，接受时间 6.78--9.64 s。

| Studio 泛化 spawn 后缀 | 通过 | 三 seed 最小统计净空 |
|---|---:|---:|
| pillar_00 | 3/3 | 0.7712 m |
| pillar_01 | 3/3 | 0.7791 m |
| pillar_02 | 3/3 | 0.7695 m |
| pillar_03 | 3/3 | 0.7131 m |
| pillar_04 | 3/3 | 0.7567 m |
| pillar_05 | 3/3 | 0.7745 m |

这里的净空是深度统计量，不是精确物理几何间距；“无接触”来自独立接触传感器。
Apartment 命令因三项行为不符，以 `--require-pass` 正常返回 1，完整报告仍已保存；
Studio 返回 0。按用户决定，本轮场景验证工作完成并收口；free-space 三项轻微
减速作为已知限制归档，未达到零误减速目标的事实保持不变。

## 侧方可视障碍引起的轻微减速

严格复验中 free/seed 1、free/seed 2 和 false_alarm/seed 1 未通过零介入跟踪
要求，均为 `episode_timeout` 分类，无接触或 termination；这里的 outcome 表示
在评估窗口内没有满足预期行为，不是环境 time_out term 触发。
最低 alpha 分别为 0.9373、0.9055、0.9384；首次介入分别为 4.36、4.30、9.92 s，
介入各持续 10、9、5 个采样点；末帧仍保持约 0.42--0.46 m/s 前进。

对 free/seed 1 单独进行 225-step GPU 复现，在首次减速的 step 218/4.36 s
捕获相机深度与位姿，用 renderer 预计算相机射线将 ROI 内 raw depth<1.4 m
的 108 个像素反投影到世界坐标。再用同一状态、同一射线查询可视组 1/2 的
mesh，并比较轴向深度（容差 0.03 m）：108/108 像素均对应 `pillar_01`。
世界坐标中位数约为 `(1.433, 0.402, 0.687)` m；这些点不属于地面，也不属于
机器人自身可视几何。细节见 `depth-source-free-seed1.json`。

该柱体位于前进通道侧方。当前整片前向 ROI 加重叠窗口的规则不区分速度方向
上的通行区域与侧方障碍，因此产生保守减速。相机看到了合法可视柱体，且 camera
BVH 仅包含 group 1/2，不能将此失败解释为隐藏碰撞混入深度。其它两个失败项
仅有轨迹诊断，尚未逐像素确认来源，不据此推断全部由同一物体引起。

速度方向/通行区域敏感性作为历史扩展设想保留，本轮不再执行。全部失败记录
保留，不继续调整 alpha 阈值、不隐藏侧方柱体，也不缩短原 10 s 窗口。

## 回归与变更范围

定向回归为 93 passed、10 slow tests 未选；覆盖 baseline/evaluator、任务配置、
接触 telemetry 和视觉/碰撞隔离。Ruff 格式/lint、Ty、Pyright 与 diff 检查通过。
GPU 矩阵是独立的真实 checkpoint 验收，不以单元测试代替。

本次行为代码仅涉及任务侧 `safe_velocity_baseline.py` 的停车计时修复和专用
`tests/run_g1_3dgs_safe_velocity_adapter.py` 的 telemetry/验收强化。
共享 play、sensor、scene、simulation 无新增修改。旧 checkpoint、数据和首轮
矩阵均保留；所有报告写入本独立目录，未覆盖此前实验。

## 提交前静态检查补充

提交前检查发现已有 smoke 单元测试漏传深度图尺寸，以及 apartment 碰撞生成
工具的 SciPy 返回值类型存在歧义。测试补齐实际的 48×64 尺寸；生成工具仅用
`typing.cast` 收窄返回类型，不改变运行时算法，也不重新生成碰撞资产。
本次额外定向单元测试为 91 passed、10 slow tests 未选；碰撞生成工具的纯数值
单元测试为 4 passed。未重新运行场景矩阵，也未扩展已关闭的场景验收范围。
