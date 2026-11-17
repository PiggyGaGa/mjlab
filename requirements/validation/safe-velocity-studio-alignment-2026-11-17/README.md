# 历史测量解释更正

本目录保留最初六个 spawn 的 report 和 overlay，不重新生成或覆盖。

报告比较的是 Gaussian-only 深度和 group-1 合成柱体的投影。柱体具有可视几何和
物理碰撞，是额外加入 Studio 的测试物体；它们无需存在于原始 3DGS 中。因此
`accepted=0/6` 表示两者不同，不代表相机受到隐藏碰撞污染或可视柱体停车失败。

原 18/18 Depth-Stop 结果应表述为“3DGS 背景 + 可视 mesh 柱体”停车验收。
旧 Vision-Data 相机/教师共享上下文引入 group 3 是另一个独立问题。

当前 alignment runner 默认以 group 3 作为隐藏代理的离线参考。需要复现本目录
的 group-1 历史测量时，显式传入 `--reference-geom-groups 1`。离线参考深度
始终不传给机器人或视觉网络。
