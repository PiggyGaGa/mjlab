# Viser 半透明 mesh 验证（2026-11-10）

## 实现与范围

- 最终按用户要求统一采用参考动作的直接 mesh 接口；不再使用此前的 GLB
  alphaMode 修补方案。复用 mjviser 原有几何构建、分组和位置更新，在 mjlab
  适配层将提交转换为顶点、三角面、RGB 与 opacity。
- 固定 geom、动态批量 geom、固定/动态 site 均绕过 GLB；不同面 RGBA 分组，
  分组 handle 转发位置、显隐和移除。纹理/PBR 替换为平面基色，不修改碰撞。
- 通过 `scripts/tools/build_viser_transparency.py` 构建 Viser 1.0.27 客户端，
  将 Gaussian mesh 的透明队列 renderOrder 从 10000 改为 -10000。
  不透明 mesh 仍在独立的不透明阶段先绘制。
- 新客户端已在当前虚拟环境中构建安装。其他环境及重装 Viser 后需重新运行
  构建工具；仅安装 mjlab Python 变更不足以调整前端绘制顺序。
- 使用本地 main 对比确认修改位于现有 Viser 适配边界，没有修改 MuJoCo、
  仿真核心或安装包的 Python 源码。

## 自动检查

```sh
uv run pytest tests/test_viser_simple_mesh.py tests/test_viser_update_policy.py tests/test_viser_conversions.py -q
```

结果：31 passed。变更 Python 文件通过 Ruff、ty 和 Pyright 检查。
客户端通过 TypeScript 编译和 Vite 生产构建。

额外运行 `uv run pytest tests/test_variants.py -k viser -q`：2 passed，
验证每世界模型变体、可视化重建与凸包显示仍兼容直接 mesh handle。

回归覆盖 XML alpha=0.1/0.5/1 的直接传递。测试禁止 `Trimesh.export`，确认
固定/动态 geom 和 site 的创建与重建不经过 GLB；同时检查分组保留三角面数量、
颜色与 alpha 分离、顶点索引有效、子 handle 变换/移除、纹理材质基色转换。
原有 Viser 转换和更新策略回归通过。

## 浏览器像素验证

使用无界面 Chrome、SwiftShader 和本地 Viser 服务。场景由相机前方的
红色 box geom 与后方的蓝色 Gaussian 平面构成；没有策略、GPU 仿真或外部
PLY 的干扰。通过 `ClientHandle.get_render` 获取 256×256 PNG，统计中心
16×16 像素区域的 RGB 中位数。

| Box alpha | 中心 RGB |
| --- | --- |
| 0（隐藏） | (0, 0, 255) |
| 0.1 | (22, 0, 232) |
| 0.5 | (111, 0, 140) |
| 1.0 | (224, 0, 23) |

alpha 增大时红色增强、蓝色减弱，验证透明混合生效且可透视后方 3DGS。
该像素测试使用固定 box；批量动态 handle 由自动直接 mesh 回归覆盖，尚未做
完整机器人动态遮挡的浏览器像素验证。

## 仍有限制

- 透明 mesh 作为 3DGS 后绘制的覆盖层，不能保证与 Gaussian 的逐片元空间
  排序；位于 Gaussian 墙后的透明 mesh 仍可能透出。
- 简单 mesh 不保留纹理、PBR 或纹理透明通道。渐变顶点色转换为面色，可能增加
  分组及绘制调用；本次不声称普遍加速或保证原材质外观一致。
- 未启动完整 G1 策略 demo，也未修改 native viewer。
