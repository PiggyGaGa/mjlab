# 视觉深度与隐藏碰撞隔离验证

日期：2026-11-17。本记录的隔离测试设备为 CPU；沙箱内不可访问 CUDA，后续经
沙箱外检查确认 RTX 3060 可用。GPU 停车矩阵另见
`safe-velocity-depth-contact-2026-11-17/README.md`。

## 实现边界

- 相机渲染 3DGS 与可视 mesh（group 1/2）。
- 教师查询可视/隐藏碰撞（group 1/3），标签不成为视觉输入。
- 任务侧 `CollisionTruthRaySensor` 复用现有 ray frame、buffer 和后处理，调用
  `mujoco_warp.rays(rc=None)`，不进入 camera SensorContext。
- 每次读标签时刷新射线，覆盖 reward → forward → observation 的时序。
- 共享 play、sensor、scene 和 simulation 代码无新增修改。
- Collector 启动时检查实际 camera BVH group，混入非可视组则报错。
- 旧任务配置可通过 `isolate_teacher_rays=False` 复现，collector 对应参数为
  `--legacy-shared-render-context`。历史数据、checkpoint、报告均保留。

## 验证

`tests/test_safe_velocity_visual_isolation.py` 覆盖两个隐藏盒体高度和两个 world：
教师距离随障碍变化，可视平面深度始终为 2 m；查询教师后重新渲染仍无隐藏盒体。
测试还覆盖 forward 前读取、forward 后无 scene.update 直接读取的刷新语义，
以及默认隔离配置和显式 legacy 配置。

真实 G1 smoke 参数：scenario `both_open`、seed 0、2 worlds、2 steps、chunk 2、
deterministic locomotion、camera corruption 开启、startup dynamics randomization
关闭。checkpoint：

```text
logs/rsl_rl/g1_3dgs_safe_velocity_blind/2026-11-12_22-38-51_blind_conservative_seed42_100/model_99.pt
```

新路径产物：`/tmp/mjlab-visual-isolation-20261117-fresh`。
legacy 对照：`/tmp/mjlab-visual-isolation-20261117-legacy`。
这些是临时 smoke 产物，不作为正式训练数据。

两种模式各生成 4 samples、2 chunks，四类 termination 均为 0。
schema-v2 Student loader 可回读两者；camera shape `(4, 7, 2, 48, 64)`，
teacher shape `(4, 4)`。新旧 teacher label 最大差为 0；camera 最大绝对差为
1.0，有 79,520 个 depth/validity 张量值不同。该短 smoke 仅验证接线与隔离，
不构成长期安全行为验收。

新 manifest 的 `visual_input`：

```json
{
  "sources": ["3dgs", "visible_mesh"],
  "camera_geom_groups": [1, 2],
  "teacher_geom_groups": [1, 3],
  "teacher_query": "direct_collision_rays",
  "collision_only_in_camera": false
}
```

原 schema 2 tensor 格式保持兼容；历史 manifest 没有该字段，仍按旧实验解释。
下一步为 GPU 小批量采集和吞吐验证，再决定正式重新采集预算。

## Studio 验收更正

group-1 柱体是额外加入的可视且可碰撞物体，与原始 Gaussian 不同是正常情况。
旧 Gaussian-vs-group-1 的 0/6 只保留为历史测量，撤销其资产验收失败解释。
18/18 停车有效支持“3DGS 背景 + 可视 mesh 障碍”，不能扩大为纯 3DGS 障碍结果。
alignment runner 默认 group 3，其投影仅供离线诊断，绝不进入相机输入。

## Depth-Stop 接触事件链路验收

主线使用可见深度调速，通过 pelvis 与非地面障碍物的 ContactSensor 判断碰撞。
它不挂载教师射线，也没有 `safety_teacher` observation。上述独立教师查询
属于可选监督采集；运行时接触判定不依赖射线或相机中的碰撞代理投影。

新增 CPU 双 world 实验使用原生 Scene、Simulation、CameraSensor 和 ContactSensor。
静态相机位于可视平面上方 3 m；group-3 隐藏墙体处于相机视线内，机器人基座
用名为 pelvis 的可动刚体替代，以精确控制接触，避免将步态误差混入传感验收。

1. 两个 world 初始均与墙分离，`found=0`，相机深度均为 3 m。
2. 仅 world 0 平移 pelvis，与隐藏墙产生 3 cm 穿透：其 `found>0`、接触力
   大于 10 N、`dist<0`；world 1 仍无接触。两者深度与接触前完全相同。
3. 将 world 0 恢复分离，当前接触归零；4 子步的力历史仍保留该次撞击，
   world 1 的力历史始终为零。相机深度仍不改变。

另对 apartment 和 Studio 两种 Depth-Stop 配置验证：没有 RayCastSensorCfg，
相机组为 `(1,2)`，安全速度命令绑定该相机；障碍奖励引用 pelvis 接触传感器，
历史长度等于 decimation。上述检查位于同一
`tests/test_safe_velocity_visual_isolation.py`，使用现有框架配置，无运行时代码改动。

`found` 表示几何接触是否存在；10 N 是当前奖励使用的碰撞力筛选阈值。
正式验收 runner 更保守，将当前 found>0 或历史任意正接触力记为碰撞。
较小接触仍会被检测并计入验收，不能把未超奖励阈值解释为没有接触。
