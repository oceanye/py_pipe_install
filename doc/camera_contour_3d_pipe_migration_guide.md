# 基于摄像头轮廓与三维模型对比的技术总结及管道迁移指南

> 文档日期：2026-08-28
>
> 参考实现：VISOMetry `codex/relative-cad-semantic-fusion-20260828` 分支，至 `8aadc67`
>
> 目标读者：准备把现有 CAD 轮廓定位能力迁移到“摄像头中主要表现为两条平行外轮廓线”的管道识别项目的算法、客户端与测试人员。

## 1. 先给结论

现有 VISOMetry 的核心不是“先从图像分割出完整物体，再与 CAD 做一次比较”，而是一个**模型驱动的闭环**：

1. 已知相机内参、三维模型和一个近似初始位姿；
2. 把三维模型按候选位姿投影到当前图像；
3. 从投影剪影上采样带模型三维坐标的轮廓点；
4. 沿每个投影点的二维法线，在摄像头图像中寻找真实边缘；
5. 用点到边缘的法向残差优化模型到相机的位姿；
6. 用对应点数、两侧支持度、残差、跳变量和连续弱帧数决定接受、降级或丢失；
7. 帧间光流或 VIO 只提供下一次绝对轮廓匹配的初值，不能独立成为最终定位依据。

这套范式适合迁移到管道项目，但不能直接把通用 6DoF 跟踪器和当前阈值原样复制。管道场景有三个决定性特点：

- 一根足够长的直圆管，其主要图像证据是左右两条视相关的切线轮廓；在弱透视下近似平行，在一般针孔透视下会向管轴消失点轻微会聚。
- 仅靠两条侧轮廓，**沿管轴平移**和**绕管轴旋转**不可观测；把它们硬塞进 6DoF 优化会造成矩阵病态、随机漂移或由阻尼“伪装出来的稳定”。
- 二维检测框只能作为 ROI 或粗门禁。它会丢掉管轴方向、两侧宽度和边缘法向信息，不能作为主要几何残差。

因此推荐的迁移方案是：

- 单根直管、只需定位中心轴：优先输出“相机坐标系下的三维管轴线 + 半径 + 置信度”，不要虚构完整 6DoF。
- 下游接口必须接收 `T_CM`：使用降维增量优化，只更新可观测自由度；轴向位移和绕轴旋转由初始化值、VIO 或外部锚点保持，并明确标记为 `prior_only`。
- 需要完整姿态或区分多根同规格管道：必须增加端面、法兰、弯头、阀门、焊缝、二维码/人工标记、深度或模型拓扑等破除对称性的证据。
- 直管段用解析圆柱轮廓或专用双侧轮廓采样器；弯头、三通和复杂管件继续使用 mesh + z-buffer 可见性。

## 2. 先区分四个容易混淆的任务

| 任务 | 输入 | 输出 | 当前 VISOMetry 是否直接具备 |
|---|---|---|---|
| 轮廓读取 | 单帧图像 | Canny 边、线段、轮廓像素、梯度方向 | 具备通用边缘读取；管道双线配对需新增 |
| 识别/初始化 | 任意图像或 ROI | 目标是谁、初始位姿或初始管轴 | 当前主要依靠人工粗放置/外部初值；不是零先验检测器 |
| 连续跟踪 | 已初始化视频 | 每帧位姿或管轴状态 | 具备模型轮廓闭环、运动先验和质量状态机 |
| 数模对比 | 图像 + 可信位姿 + CAD | 支持度、偏差、缺失/存在证据 | 具备通用物理边支持分析；管道应改为双侧/分段语义支持 |

迁移时最常见的误判是：看到现有系统可以稳定跟踪，就认为它已经能在任意画面中自动找到目标。事实上，局部法线搜索的收敛域受 `max_edge_search_px` 限制；初值若远离真实管道，算法很可能吸附到背景中另一组平行边。因此，新项目必须单独实现“管道双线检测与初始关联”，再进入模型驱动跟踪。

## 3. 当前实现的数据流

```text
CameraX / ARCore CPU image
        │
        ├─ YUV_420_888 的 Y 平面 → 8-bit 灰度图
        ├─ 同帧 timestamp / 图像尺寸 / rotation metadata
        └─ 与当前分辨率严格一致的相机内参 K
                         │
                         ▼
              高斯滤波 + Sobel + Canny
                         │
             梯度幅值/方向 + 距离变换
                         │
运动先验 ───────► 候选 T_CM ◄────── 初始放置/重定位
                         │
                         ▼
       CAD mesh 软件/GPU 光栅化：mask、depth、triangle_id
                         │
                         ├─ 外剪影等弧长采样
                         └─ 有图像支持的实体物理边采样
                         │
                         ▼
       沿投影法线局部搜索真实边缘 + 方向/统计门禁
                         │
                         ▼
      point-to-line 残差 + Huber IRLS + LM/GN 位姿优化
                         │
                         ▼
     点数 / 支持率 / 内点率 / 残差 / 跳变 / 连续帧状态机
                         │
              GOOD / WEAK / LOST + T_CM
                         │
          接受帧才播种 mesh-bound 短基线 LK 点
```

### 3.1 相机输入

- Android 端以 `ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST` 丢弃积压旧帧，避免算法延迟不断增长。
- YUV 图像只使用 Y 平面完成几何跟踪，不依赖 RGB 转换。
- 算法坐标保持在**原始图像像素系**；显示旋转由单独的 VIEW↔IMAGE 映射处理。若迁移项目选择物理旋转图像，必须同步旋转内参、主点、图像尺寸和所有二维观测。
- 内参必须与实际送入 native 的灰度分辨率一致；降采样时 `fx/fy/cx/cy` 必须同比缩放。
- 当前 C++ 投影核心不施加畸变，接口中的 `dist` 仅保留。迁移项目必须二选一并全链一致：先对图像 `remap/undistort` 后使用针孔投影，或在模型投影与残差雅可比中显式加入畸变。不能“原始畸变图像 + 无畸变投影”混用。

对应实现：[CameraController.kt](../android/app/src/main/java/com/visometry/tracker/camera/CameraController.kt)、[cad_tracker.cpp](../cpp/src/cad_tracker.cpp)。

### 3.2 图像边缘场

当前边缘读取顺序为：

```text
gray → GaussianBlur → Sobel(gx, gy)
     → gradient magnitude / atan2 direction
     → Canny binary edge
     → distance transform + distance-field gradient
```

这些中间量分别服务于：

- Canny：限制候选必须靠近真实边缘；
- 梯度方向：要求图像边缘法向与模型轮廓法向一致；
- 梯度幅值：亚像素峰值精化；
- 距离场：快速评分候选模型轮廓与最近图像边缘的距离；
- 灰度统计：概率对应线在法线两侧比较前景/背景强度分布，抑制杂乱背景的错误边。

对应实现：[edge_extractor.cpp](../cpp/src/edge_extractor.cpp)、[correspondence.cpp](../cpp/src/correspondence.cpp)。

### 3.3 三维模型投影与可见性

模型点使用统一位姿：

```text
P_C = T_CM · P_M
u = fx · X_C / Z_C + cx
v = fy · Y_C / Z_C + cy
```

其中长度单位为 mm，相机系采用 OpenCV 约定：X 向右、Y 向下、Z 朝前。

软件光栅器逐三角形生成：

- `mask`：模型在画面中的可见区域；
- `depth`：每像素最近模型深度；
- `triangle_id`：每像素对应的可见三角面；
- `normal_map`：可见面法线。

外轮廓从 mask 提取，按弧长重采样；每个二维轮廓点再通过 `triangle_id + 透视正确重心插值` 绑定到模型坐标三维点。因此优化器处理的不是无语义二维像素，而是“模型三维点 ↔ 当前图像边缘”的约束。

对应实现：[soft_render.cpp](../cpp/src/soft_render.cpp)。

### 3.4 对应搜索与优化

对每个预测轮廓点 `p_i`，沿其二维单位法线 `n_i` 搜索图像边缘 `q_i`。核心残差是：

```text
r_i = n_iᵀ · (q_i - π(K, T_CM, P_i^M))
```

这是一维点到线残差，只惩罚垂直于轮廓的误差，不惩罚沿轮廓切向滑动。它比二维最近点欧氏距离更符合轮廓几何，也允许解析求导。

当前默认对应器还会执行：

- 搜索窗与图像边界门禁；
- 图像梯度方向一致性；
- 前景/背景局部灰度可分性评分；
- 亚像素边缘位置精化；
- MAD 外点剔除；
- 低支持场景下的法线方向均衡；
- 对应置信度作为优化权重。

位姿优化使用 SE(3) 左乘增量、Huber IRLS 和 LM 阻尼：

```text
T_new = exp(δξ^) · T_old

min Σ_i w_i · Huber(r_i) + motion/anchor prior
```

对应实现：[pose_optimizer.cpp](../cpp/src/pose_optimizer.cpp)。

### 3.5 时序与质量门禁

- ARCore VIO 或内部匀速模型提供下一帧 `T_CM` 预测；VIO 不直接覆盖模型位姿。
- 接受帧可在渲染 mask 内检测 Shi-Tomasi 点，通过 z-buffer 绑定到 CAD 表面，再用双向金字塔 LK 估计下一帧初值。
- LK 必须通过前向误差、前后向一致性、像素位移、运动先验重投影、位姿修正量和最终残差门禁。
- LK 结果只作为下一次绝对 CAD 轮廓优化的初值；拒绝帧清空播种，禁止长时间纯光流积分。
- 最终状态由有效对应点数、匹配率、内点率、平均残差、相对预测跳变和连续弱帧数共同决定。LOST 后停止发布位姿，要求显式重初始化。

对应实现：[relative_tracker.cpp](../cpp/src/relative_tracker.cpp)、[quality.cpp](../cpp/src/quality.cpp)。

## 4. 管道双轮廓的几何特点

### 4.1 “两条平行线”是工程近似，不是一般透视下的严格结论

无限长圆柱的两条可见侧轮廓来自过相机中心、与圆柱相切的两个平面。轮廓在三维中是两条与管轴平行的母线，其针孔投影是两条图像直线。平行三维直线通常投影到同一个管轴消失点，因此：

- 管轴大致平行于像平面、视场较小或 ROI 较短时，两条线看起来近似平行；
- 近距离、广角或管轴明显朝向相机时，两条线会发生可见会聚；
- 强制两条线绝对平行会把真实透视会聚误解释成姿态或宽度误差。

建议把“平行度”作为带容差的候选门禁。MVP 可以使用近似平行线模型；精度阶段应允许一般两直线或直接拟合解析圆柱投影。

### 4.2 管端是否入画会改变问题

- 管道两端都不在画面：只有两条侧轮廓，轴向位置不可观测。
- 一个端面入画：圆/椭圆端面和侧轮廓交点可约束轴向位置与朝向。
- 法兰、弯头、三通、阀门或焊缝入画：这些是打破圆柱对称性的语义锚点。
- 管道穿过画面边界：通用 closed-contour sampler 会把图像边界处的“闭合边”误当真实物理轮廓，必须改成两条 open polyline，并显式忽略边界闭合段。

### 4.3 推荐的二维观测参数化

在弱透视 MVP 中，把两条线统一方向并写成法式：

```text
L_left : nᵀx = ρ_left
L_right: nᵀx = ρ_right
||n|| = 1

ρ_center = (ρ_left + ρ_right) / 2
h_image  = |ρ_right - ρ_left| / 2
```

观测由三类独立信息组成：

- 方向 `θ`：管轴在图像中的方向；
- 中心偏移 `ρ_center`：管道中心线的二维位置；
- 半宽 `h_image`：已知真实半径和内参时提供距离/尺度约束。

当圆管轴大致平行像平面、位于主点附近且透视变化较小时，可用下面的式子提供**初始化深度近似**：

```text
Z ≈ 2 · f · radius_mm / width_px
```

该式不能作为一般斜视管道的最终测量公式。正式优化应把圆柱或 mesh 按完整针孔模型投影后求残差。

### 4.4 可观测性边界

| 状态量 | 只有两条侧轮廓时 | 建议处理 |
|---|---|---|
| 图像中心线位置 | 可观测 | 直接由双线中线约束 |
| 图像管轴方向 | 可观测；近消失点时深度方向可能病态 | 使用线方向/消失点并输出协方差 |
| 垂直管轴的三维位置 | 已知 K 与半径时通常可观测 | 用解析圆柱投影或受约束优化 |
| 相机到管道的尺度/距离 | 已知真实半径时可估；半径未知时与尺度耦合 | 半径必须来自模型或其他传感器 |
| 沿管轴平移 | 不可观测 | 冻结为初值/外部锚点；不得宣称由轮廓测得 |
| 绕管轴旋转 | 圆管完全不可观测 | 冻结或从非对称附件/纹理获得 |
| 多根同半径直管的语义身份 | 仅靠局部双线通常不可判定 | 使用拓扑、邻接管件、全局位置或人工 ID |

LM 阻尼只能让病态法方程“可求解”，不能创造缺失的观测信息。迁移实现应检查 Hessian 特征值或条件数，发布 `observable_dofs` 和协方差，而不是只看优化器是否返回数值。

## 5. 推荐的管道系统架构

```text
相机帧 + K + 畸变策略 + timestamp
                   │
                   ▼
       灰度/对比度处理 + 定向边缘/LSD
                   │
       运动预测 ROI ───► 双线候选生成与配对
                   │                │
                   │        初始识别/重捕获候选
                   ▼                ▼
   解析圆柱或 mesh 投影 → 预测左右侧轮廓 + 可见区间
                   │
                   ▼
     左右侧身份关联 + 沿法线概率对应搜索
                   │
                   ▼
     双侧均衡 point-to-line + 中线/宽度/方向残差
                   │
                   ▼
     可观测子空间优化 + VIO/运动先验 + 鲁棒核
                   │
                   ▼
 双侧支持率 / 残差 / 可观测性 / 跳变 / 时序状态机
                   │
      axis track / constrained T_CM / presence evidence
```

### 5.1 模型数据契约

建议不要只给一个通用 OBJ。每个直管段增加显式语义几何：

```json
{
  "schema_version": 1,
  "coordinate_system": "model",
  "length_unit": "mm",
  "segments": [
    {
      "id": "pipe_A_01",
      "axis_origin_model": [0.0, 0.0, 0.0],
      "axis_direction_model": [0.0, 0.0, 1.0],
      "radius_mm": 50.0,
      "s_min_mm": -1000.0,
      "s_max_mm": 1000.0,
      "neighbors": ["elbow_A_02"],
      "tracking_enabled": true
    }
  ]
}
```

要求：

- 管轴方向必须归一化；
- `axis_origin_model` 的轴向零点要有工程语义，例如法兰端面或设计里程；
- 半径使用实际外轮廓半径，保温层、涂层或套管存在时不能继续使用裸管半径；
- `s_min/s_max` 未知时可为空，但此时接口不得把轴向位置标记为已测量；
- 复杂管件保留 tracking mesh，用于遮挡、可见性和全局数模对比；
- 视觉模型与跟踪几何必须同单位、同原点、同坐标系。

### 5.2 图像观测契约

推荐让检测器输出结构化观测，而不是只输出矩形框：

```text
PipeContourObservation
  timestamp_sec
  image_size
  left_line / right_line       # 归一化齐次线或法式参数
  left_interval / right_interval
  left_support / right_support
  left_gradient_polarity / right_gradient_polarity
  parallel_error_deg
  width_px_median / width_px_mad
  overlap_ratio
  roi
  confidence
```

左右标签以预测中心线的有符号法向距离确定，不要依赖检测器返回顺序。没有预测时先建立一个一致的图像法线方向，再按 `ρ` 排序。

### 5.3 输出状态契约

首选输出：

```text
PipeAxisTrack
  direction_camera[3]          # 单位方向，符号需用拓扑或上一帧保持
  closest_point_camera_mm[3]   # 到相机原点的轴线上最近点，满足 c·d = 0
  radius_mm
  observable_dofs
  covariance
  status                       # INITIALIZING / GOOD / WEAK / LOST
  left_support / right_support
  center_residual_px
  width_residual_px
  angle_residual_deg
```

`direction + closest_point` 是四自由度三维直线表示，避免给无意义的轴向原点。若业务接口必须返回 `T_CM`，还要同时返回：

```text
pose_source_by_dof:
  transverse_translation: measured
  axis_direction: measured
  axial_translation: prior_only
  roll_about_axis: prior_only
```

## 6. 各算法模块如何迁移

### 6.1 相机与线程：基本可直接复用

可复用原则：

- 最新帧优先、旧帧不排队；
- native/核心跟踪器单线程串行调用；
- 图像、内参、时间戳和运动先验必须来自同一采集基准；
- UI 只消费冻结后的结果快照；
- 分辨率变化时同步缩放 K，并重建依赖图像尺寸的缓存；
- 记录原始算法结果，显示平滑不能写回算法状态或评估日志。

迁移优化点：若只使用灰度，不必先构造完整 NV21 再截取 Y；可直接复制/映射紧凑 Y 平面并配合缓冲池，减少每帧分配。

### 6.2 边缘与线段读取：在通用 EdgeExtractor 上增加定向线段层

推荐流程：

1. 在预测 ROI 内做轻度高斯滤波；曝光跨度大时可评估 CLAHE，但必须用 A/B 数据证明不会放大噪声边。
2. 保留 Sobel 幅值和方向，供法向一致性与边缘极性判断。
3. 使用 Canny 形成稀疏候选。
4. 使用 LSD、EDLines 或受 ROI/方向约束的 Hough 得到亚像素线段。
5. 对每条线段保存长度、方向、支持像素、残差、梯度方向分布和强度对比，不要只留两个端点。

反光管道常有沿轴方向的高光线或涂层接缝，它们也会形成长平行线。真正外轮廓必须同时满足：位于预测直径两侧、与模型预期宽度一致、两侧都有支持、与背景/前景强度变化相符，并在时间上连续。

### 6.3 双线配对：这是新项目必须新增的初始化层

候选线对至少检查：

- 方向差小于配置阈值；一般透视模式改为“消失点一致”而非严格平行；
- 两条线具有足够的轴向重叠长度；
- 中位宽度落在模型投影允许范围；
- 沿有效区间的宽度变化符合透视圆柱预测；
- 两侧梯度法向与预测外法向兼容；
- 两侧各自支持率达标，禁止一条强线复制成两侧；
- 中心线落在预测 ROI 或允许的重捕获搜索区；
- 与上一帧方向、中心、宽度变化满足运动上限；
- 背景边缘密度过高时降低置信度或拒绝自动初始化。

可以使用如下组合分数进行候选排序，但每一项都应先归一化并由数据标定：

```text
score_pair =
    w_len     · overlap_support
  + w_edge    · min(left_support, right_support)
  + w_model   · model_width_agreement
  + w_motion  · temporal_agreement
  + w_photo   · foreground_background_separation
  - w_angle   · perspective_direction_error
  - w_width   · width_nonuniformity
```

不要只取 Hough 投票最高的两条线；在厂房、吊顶、墙缝和支架场景中，这通常会选中背景结构。

### 6.4 模型轮廓生成：直管优先解析，复杂件使用 mesh

有三种实现策略：

| 策略 | 适用场景 | 优点 | 风险 |
|---|---|---|---|
| 解析圆柱切线轮廓 | 单根/多根直圆管 | 精确、稳定、无网格棱化、计算量低 | 需要单独实现投影和雅可比；端面/复杂遮挡需扩展 |
| 圆柱 mesh + z-buffer | 快速复用、弯头/三通/遮挡 | 可直接复用现有渲染与可见性 | 低分段圆柱会产生轮廓跳变；闭合采样会混入端面和画面边界 |
| 混合模式 | 工程管网 | 直段用解析双侧线，复杂管件用 mesh，语义统一 | 数据关联和权重管理更复杂 |

推荐混合模式。特别注意：圆柱的光滑外轮廓是**随视角变化的 apparent contour**，不是 STEP 中固定的一条 BRep 物理边。不能把某两条固定母线写入 `semantic_edges.json` 并期望在所有视角都代表外轮廓。固定语义边适用于法兰边、接缝、弯头边界等真实物理特征；直圆管侧轮廓应由解析切线或当前视角的 mesh 剪影实时生成。

若暂时复用 mesh：

- 圆周方向使用足够分段并做精度/耗时评测，不能为减面牺牲外径；
- 新增 `PipeContourSampler`，输出左右两条 open polyline；
- 采样按两侧分别等弧长分配，左右点数相等；
- 排除端面、图像边界闭合边和不可见段；
- z-buffer 继续负责遮挡和前后管段选择。

### 6.5 对应与残差：保留法线搜索，增加双侧结构约束

现有 point-to-line 对应器可直接复用大部分机制，但必须增加：

- `side_id = LEFT/RIGHT`；
- 每侧最小有效点数和最小覆盖率；
- 左右侧等权或封顶，防止一侧因纹理/高光产生更多边缘而支配优化；
- 每侧独立的残差中位数/MAD，再做全局鲁棒核；
- 中线、宽度和方向的成对诊断；
- 检测到一侧被遮挡时进入 WEAK，而不是用另一侧假装完整双线约束。

推荐同时记录三个成对残差：

```text
r_center = (r_left + r_right) / 2      # 管道整体横向偏移
r_width  = (r_right - r_left) / 2      # 投影直径/深度不一致
r_angle  = angle(predicted_axis, observed_axis)
```

优化目标可以保留逐点 Huber 项，并把成对项作为门禁或低权重辅助：

```text
E = Σ side-balanced Huber(r_point)
  + λ_center · Huber(r_center)
  + λ_width  · Huber(r_width)
  + λ_angle  · Huber(r_angle)
  + motion prior
```

二维框的 IoU、中心和宽高可用于候选排序，但不应替代上述残差。框对沿轴滑动、左右边错误关联和透视方向变化都不敏感。

### 6.6 优化器：从无约束 6DoF 改成可观测子空间

有两种推荐实现：

#### 方案 A：直接优化三维轴线

状态使用相机系单位方向 `d_C` 和到相机原点最近点 `c_C`，约束 `c_C·d_C=0`。这是四自由度状态，与无限圆柱的几何一致。半径由模型固定，或作为有强先验的附加变量。

优点是不会产生轴向/roll 假观测，接口语义最清晰。若相机在运动、管道静止，可在世界系维护轴线，用每帧相机位姿投影。

#### 方案 B：保留 T_CM，但只更新可观测增量

令完整 SE(3) 增量由低维参数生成：

```text
δξ = B · δη,    B ∈ R^(6×k), k < 6
```

`B` 去掉沿管轴平移和绕管轴旋转两个方向。优化法方程变成：

```text
(Bᵀ H B) δη = -Bᵀ g
```

不可观测分量保持初值或外部先验。每帧检查 `BᵀHB` 的最小特征值、条件数和协方差；低于门槛时降级，不发布“高置信完整位姿”。

不要仅靠给不可观测方向增加很大的 LM damping。那会把先验值包装成看似由图像收敛得到的数值，且难以向下游解释误差来源。

### 6.7 初始化与重定位

单根已知半径直管的 MVP 初始化：

1. 在全图或业务 ROI 中检测双线候选；
2. 由平均方向得到初始图像管轴；
3. 由中线得到横向位置；
4. 由已知半径和像素宽度得到粗深度；
5. 对可能的前后倾角和方向符号建立少量多假设；
6. 用解析圆柱/mesh 完整投影评分每个候选；
7. 只有双侧支持、残差、宽度和可见长度同时过门禁才初始化；
8. 轴向位置与 roll 标记为 prior，不由双线初始化。

若场景有多根相同管道，仅凭局部双线不能完成可靠语义识别。应优先引入管网拓扑和邻近管件，再考虑通用目标检测网络。深度学习可以帮助提供 ROI/类别，但不能自动消除同规格无限圆柱的几何不可观测性。

### 6.8 帧间跟踪

推荐顺序：

```text
上一接受状态
  → VIO/匀速预测
  → 预测左右轮廓和窄 ROI
  → 当前帧重新检测/匹配两侧绝对边缘
  → 可观测子空间优化
  → 质量门禁
```

mesh-bound LK 可以作为可选初值通道，但管道往往纹理少、反光高光会随视角移动，因此应比通用构件更保守：

- 只在预测管道内部、远离高光饱和区和轮廓边缘处播种；
- 使用前后向一致性和模型重投影门禁；
- 不足点时直接退回运动预测；
- 每个接受帧重新播种；
- 不能用纯光流维持长期轴向位置或绕轴旋转。

### 6.9 数模对比与管道存在性

若业务还要判断某一管段是否存在，建议把每个管段当成语义实体，并对左右轮廓分别计算：

```text
left_support  = supported_left_samples / assessable_left_samples
right_support = supported_right_samples / assessable_right_samples
pair_support  = min(left_support, right_support)
```

`min` 比平均值更安全：一条背景强边不能补偿另一侧完全缺失。只有以下条件同时满足才提交负证据：

- 当前位姿可靠；
- 该管段在画内有足够可见长度；
- z-buffer/场景模型认为没有被其他模型实体遮挡；
- 两侧均有足够可评估采样；
- 负证据跨时间、最好跨视角重复出现。

画外、自遮挡、外物遮挡、曝光失败和整体边缘退化应返回 `UNKNOWN/UNASSESSABLE`，不能直接判缺失。存在/缺失事实状态与“是否暂时从跟踪约束中排除”应分开，并使用迟滞，避免缺失实体的背景边把位姿拖走后形成自证循环。

## 7. 当前模块到管道项目的迁移矩阵

| 当前模块 | 参考文件 | 迁移建议 | 管道专用改造 |
|---|---|---|---|
| 相机帧接入 | [CameraController.kt](../android/app/src/main/java/com/visometry/tracker/camera/CameraController.kt) | 基本复用 | 直接 Y 平面/缓冲池；严格处理 K、畸变和 rotation |
| 坐标/投影约定 | [coordinate_conventions.md](coordinate_conventions.md) | 原样保留 | 增加管轴线表示和轴向 gauge 说明 |
| 边缘提取 | [edge_extractor.cpp](../cpp/src/edge_extractor.cpp) | 复用 | 增加定向线段检测、饱和高光/低对比诊断 |
| mesh 光栅与 z-buffer | [soft_render.cpp](../cpp/src/soft_render.cpp) | 复杂管件/遮挡复用 | 直管增加解析投影或双 open-polyline sampler |
| 概率法线搜索 | [correspondence.cpp](../cpp/src/correspondence.cpp) | 高度复用 | side ID、双侧最低支持、每侧 MAD、等权预算 |
| 6DoF 优化 | [pose_optimizer.cpp](../cpp/src/pose_optimizer.cpp) | 复用投影雅可比/Huber/LM 框架 | 改为轴线状态或 `δξ=Bδη`；增加秩/条件数检查 |
| 多尺度精化 | [cad_tracker.cpp](../cpp/src/cad_tracker.cpp) | 复用 coarse-to-fine 框架 | 每层重新检测双线；搜索窗按预测不确定度设定 |
| mesh-bound LK | [relative_tracker.cpp](../cpp/src/relative_tracker.cpp) | 可选复用 | 纹理不足自动关闭；禁止作为绝对定位权威 |
| 运动预测/状态机 | [quality.cpp](../cpp/src/quality.cpp) | 复用设计 | 指标改为双侧支持、中心/宽度/角度残差和可观测性 |
| 语义实体边 | [cad_tracker.h](../cpp/include/visometry/cad_tracker.h) | 管件固定边可复用 | 圆柱视相关侧轮廓不能伪装成固定 BRep 边 |
| 移动配置 | [tracker_mobile_fast.yaml](../configs/tracker_mobile_fast.yaml) | 仅参考配置结构 | 所有阈值重新用管道数据标定，禁止照搬数值 |

## 8. 建议配置骨架

下面数值仅用于启动离线实验，不是生产验收阈值；最终应由目标摄像头、距离、管径、材质和背景数据标定。所有阈值进入配置，不应写死在代码中。

```yaml
schema_version: 1

camera:
  image_width: 1280
  image_height: 720
  distortion_mode: pre_undistort   # pre_undistort | distorted_projection

image_processing:
  gaussian_kernel: 5
  canny_threshold_low: 50
  canny_threshold_high: 140
  use_clahe: false
  saturation_reject_value: 250

line_detection:
  method: lsd
  roi_margin_px: 40
  min_segment_length_px: 80
  max_parallel_error_deg: 3.0
  min_axis_overlap_ratio: 0.60
  max_width_mad_ratio: 0.10
  gradient_direction_tolerance_deg: 30.0

model_matching:
  representation: analytic_cylinder   # analytic_cylinder | mesh | hybrid
  normal_search_px_coarse: 16
  normal_search_px_fine: 6
  huber_delta_px: 2.0
  min_support_per_side: 0.60
  min_points_per_side: 24
  max_center_residual_px: 3.0
  max_width_residual_px: 3.0
  max_axis_angle_residual_deg: 2.0
  equalize_side_weights: true

optimizer:
  state: axis_line                    # axis_line | constrained_se3
  freeze_axial_translation: true
  freeze_roll_about_axis: true
  max_iterations: 10
  damping: 0.001
  min_observable_eigenvalue_ratio: 1.0e-5
  max_condition_number: 1.0e6

motion:
  use_vio_hint: true
  constant_velocity_fallback: true
  max_center_jump_px: 30
  max_axis_angle_jump_deg: 5.0
  max_width_change_ratio: 0.15

quality:
  weak_consecutive_frames: 3
  lost_consecutive_frames: 6
  require_both_sides_for_good: true
  one_side_state: weak

presence:
  enabled: false
  min_assessable_length_px: 120
  missing_support_threshold: 0.35
  negative_confirmations: 3
  positive_confirmations: 3
  min_evidence_interval_sec: 0.5
```

配置还应携带版本号/哈希并写入每次测试结果，保证阈值变化可追溯。

## 9. 核心伪代码

### 9.1 初始化

```text
initialize(frame, K, pipe_model):
    gray = prepare_geometry_image(frame, K, distortion_policy)
    edge = extract_edges_and_gradients(gray)
    segments = detect_oriented_segments(edge)
    pairs = build_pipe_pairs(segments, pipe_model, no_motion_prior)

    hypotheses = []
    for pair in top_k(pairs):
        coarse_state = infer_axis_and_depth(pair, K, pipe_model.radius)
        for tilt_hypothesis in expand_perspective_hypotheses(coarse_state):
            predicted = project_pipe_contours(pipe_model, tilt_hypothesis, K)
            match = match_both_sides(predicted, edge, pair)
            state = optimize_observable_subspace(tilt_hypothesis, match)
            score = evaluate_both_sides_and_observability(state, match)
            hypotheses.append(state, score)

    best = select_unambiguous_hypothesis(hypotheses)
    if best fails any hard gate:
        return INITIALIZING without pose
    return accepted axis state, with axial/roll marked prior_only
```

### 9.2 连续跟踪

```text
track(frame, previous_state, motion_hint):
    predicted = propagate(previous_state, motion_hint)
    predicted_contours = project_pipe_contours(model, predicted, K)
    roi = expand_by_uncertainty(predicted_contours)

    edge = extract_edges_and_gradients(frame, roi)
    observed_pair = detect_or_associate_pair(edge, predicted_contours)
    matches = normal_search_by_side(predicted_contours, observed_pair, edge)

    if either side is unassessable:
        return WEAK or LOST according to hysteresis; do not invent full update

    candidate = optimize_observable_subspace(predicted, matches)
    quality = evaluate(
        left_support, right_support,
        center_residual, width_residual, angle_residual,
        Hessian condition, motion jump
    )

    if quality accepted:
        publish candidate
        reseed optional short-baseline features
    else:
        preserve last accepted state
        clear relative seeds when required
```

## 10. 推荐实施顺序

### P0：坐标、标定和数据契约

交付物：

- `camera.json` 与残余畸变验证；
- `pipe_model.json`；
- 原始视频、逐帧时间戳和标注规范；
- 原始图像坐标与显示坐标的单元测试；
- 已知三维轴/半径投影到图像的像素级闭环测试。

门禁：已知真值状态投影的双侧轮廓与人工标注在受控图像上误差达到项目设定值；未过此门禁前不要调跟踪器。

### P1：纯二维双线检测

交付物：

- 定向线段检测器；
- 双线配对与置信度；
- 左右侧身份；
- 背景平行线、高光线、单侧遮挡测试集。

门禁：先评估线对检测 precision/recall、角度、中心和宽度，不混入三维优化误差。

### P2：三维模型投影与单帧匹配

交付物：

- 解析圆柱或专用双侧 mesh sampler；
- z-buffer 可见区间；
- 双侧法线对应；
- 单帧 axis-line 或 constrained-SE3 优化；
- Hessian 可观测性报告。

门禁：合成图和受控真机图上，单帧从规定初值范围收敛；不可观测分量保持不变且被正确标记。

### P3：视频跟踪与质量状态机

交付物：

- VIO/匀速预测；
- ROI 动态扩张；
- GOOD/WEAK/LOST 与重初始化；
- 原始结果日志和独立显示平滑；
- 性能降级策略。

门禁：遮挡、模糊、快速运动和重新入画时不发布错误高置信状态。

### P4：多管段语义与数模对比

交付物：

- 管网拓扑和实体 ID；
- 管段/管件联合关联；
- 每侧、每段 support；
- PRESENT/MISSING/UNKNOWN 多帧证据；
- 缺失实体对跟踪约束的迟滞排除/恢复。

门禁：同规格平行管、多重遮挡和局部出画时不串 ID、不把不可评估当缺失。

## 11. 测试与评价指标

### 11.1 单元测试

- K 随分辨率缩放后投影不变；
- 畸变/去畸变路径互斥且闭环；
- 图像 rotation 与 K/坐标转换一致；
- 两条线的方向统一、左右排序、中心/宽度计算；
- 一般透视会聚线不被错误强制成平行；
- 圆柱解析投影与高分段 mesh 投影一致；
- 图像边界闭合边不进入侧轮廓；
- 每侧采样预算相等；
- 优化雅可比通过有限差分检查；
- 轴向平移和 roll 被识别为不可观测；
- 只有一侧支持时不能输出 GOOD；
- LOST 帧不覆盖 last-good。

### 11.2 合成数据

至少覆盖：

- 不同半径、距离、焦距和主点；
- 管轴平行像平面、斜向相机、明显透视会聚；
- 一个/两个端面入画；
- 10%～70% 遮挡；
- 运动模糊、欠曝、过曝、噪声；
- 背景中存在更多、更长的平行线；
- 两根或多根相同直径平行管；
- 法兰、弯头、支架和阀门；
- 管道部分出画和重新入画。

### 11.3 真机数据

每轮保存：原始/压缩视频、相机内参来源、分辨率、时间戳、逐帧观测线、每侧 support、残差、状态、运动先验、可观测性、算法配置 revision 和人工动作标记。

建议指标：

| 层级 | 指标 |
|---|---|
| 双线检测 | 两侧同时检出率、错误线对率、中心线像素误差、宽度误差、方向误差、有效长度覆盖率 |
| 单帧三维 | 管轴角误差、轴线横向距离误差、深度误差、初始化成功率、收敛域 |
| 视频跟踪 | GOOD/WEAK/LOST 占比、错误高置信帧率、抖动、重捕获时间、p50/p95 耗时 |
| 语义关联 | 管段 ID 切换率、同规格管混淆率、拓扑不一致率 |
| 存在性 | 缺失 precision/recall、确认延迟、遮挡误报率、回装恢复延迟 |

轴向平移和绕轴旋转只有在存在相应锚点时才进入精度指标；否则应报告为不可观测，而不是用初始化值计算出虚假的零误差。

## 12. 典型失败模式与排查

| 症状 | 常见原因 | 修正方向 |
|---|---|---|
| 跟到墙缝/支架的两条平行线 | 只按长度/Hough 票数选线 | 加模型宽度、双侧极性、ROI、时序和管网拓扑 |
| 跟到管道高光或中缝 | 内部亮线比外轮廓更强 | 使用两侧外法向、预期直径、前景/背景统计和饱和区门禁 |
| 深度随像素宽度剧烈跳动 | 直接逐帧使用 `2fr/w`；宽度受模糊/斜视影响 | 只用作初值；完整投影优化 + 时序协方差 |
| 轴向位置缓慢漂移 | 用两条侧线优化完整 6DoF | 冻结轴向自由度，或增加端面/法兰/深度/拓扑锚点 |
| 绕轴角度随机变化 | 圆柱旋转对称 | 标记为不可观测；增加非对称特征 |
| 管道穿出画面后残差异常 | closed contour 把图像边界当物理边 | 使用左右 open polyline，排除 border closure |
| 一侧被挡仍显示 GOOD | 总支持率被另一侧大量点稀释 | 每侧独立下限，以 `min(left,right)` 作双侧门禁 |
| 快速运动后吸附错误平行线 | 搜索窗扩大但没有候选级几何验证 | LOST/重定位分离；扩大 ROI 时重新做全套双线配对 |
| 换分辨率后整体偏移/缩放 | K 未同步缩放或主点基准错误 | 图像尺寸、K、ROI 和显示映射作为原子配置更新 |
| 画面边缘误差系统性增大 | 残余畸变未处理 | 棋盘格验证；预去畸变或畸变投影二选一 |
| 同规格管道 ID 来回切换 | 局部轮廓无语义区分度 | 结合拓扑、世界位置、邻接管件和多目标数据关联 |

## 13. 不建议直接照搬的内容

- 不要照搬当前移动配置中的 Canny、搜索窗、点数、残差和 LOST 阈值；它们来自另一种构件、另一台相机和特定真机数据。
- 不要把通用模型的 `largest external closed contour` 直接用于穿屏长管。
- 不要把两条固定 CAD 母线当成圆柱在所有视角下的外轮廓。
- 不要用二维框 IoU 代替法向点到线残差。
- 不要让一侧强边或背景线代表完整管道。
- 不要让纯 LK、VIO 或平滑结果成为绝对数模对比依据。
- 不要把优化器返回有限数值等同于六自由度都可观测。
- 不要在没有已知管径、深度或外部尺度时承诺单目毫米级三维距离。
- 不要在多根相同直管场景中仅凭局部双线宣称完成语义识别。

## 14. 推荐的第一版最小实现

如果新项目第一阶段只有“一根已知外径的长直管，管端通常不入画”，建议暂时不迁移全部通用 CAD 跟踪器，而是按以下最小闭环实现：

1. 标定相机并固定畸变策略；
2. 检测两条近似平行/同消失点的长边；
3. 用模型半径、K、中心线和宽度建立三维管轴初值；
4. 解析投影左右切线轮廓；
5. 沿预测法线在当前图像重新找边；
6. 优化四自由度三维轴线；
7. 左右侧独立门禁，输出管轴、半径、协方差和 GOOD/WEAK/LOST；
8. 明确不输出由图像测得的轴向原点与绕轴角；
9. 等真机闭环稳定后，再增加管件 mesh、VIO、语义 ID 和缺失分析。

若第一阶段就需要与完整管网 CAD 对齐，则采用混合架构：全局 `T_CM` 由法兰/弯头/阀门/端点等非对称语义特征约束，直管双轮廓只提供高精度横向和管轴方向残差。

## 15. 可参考的现有资料

- [坐标系与投影约定](coordinate_conventions.md)
- [Android 模型、相机和初始化流水线](android_model_pipeline.md)
- [轮廓/区域/概率对应线技术路线](frontier_research_and_roadmap.md)
- [C++ 跟踪器公开接口](../cpp/include/visometry/cad_tracker.h)
- [内部模块和配置结构](../cpp/include/visometry/tracker_core.h)
- [移动端融合配置示例](../configs/tracker_mobile_fast.yaml)
- [Android 运行与诊断说明](../android/README.md)

迁移验收时最重要的原则是：**图像只证明它真正观测到的自由度；先验只负责未观测部分，二者必须在接口和日志中明确区分。**
