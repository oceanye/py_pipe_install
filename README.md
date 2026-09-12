# py_pipe_install

2026-09-11 实测复核：已复算 GitHub 上传的四份标定诊断，并检查现场照片与算法。结论、修复和补拍步骤见 [实测素材与标定复算评估](doc/FIELD-20260911-实测素材与标定复算评估.md)。当前素材尚未通过标定质量检查，旧现场包使用的演示标定不能用于真实精度验收。

基础立面评估默认采用自动匹配：双击 `run_gui.bat`，点击左上角 **基础立面评估**。程序从双目有效深度拟合可见管道的局部圆柱，将其与 STL 的管径和截面排列匹配，自动定位检查区域；斜视时可指定模型管长方向，不要求逐管框选或二维码定位。完整操作见 [基础立面深度模式](doc/基础立面深度模式.md)。

局部测量 GUI：双击 `run_gui.bat`，或运行 `.\.venv\Scripts\python.exe -m pipe_twin gui`。支持已连接 USB 双目相机的实时左右预览与同步抓拍、CAD 侧立面四方向快速定位、管轴自动轻微倾斜校正、1:1 二维码完整位姿定位、现场模型/历史照片录入、局部管径与位置测量、中心距/净距/前后关系、实测样本校正和逐管状态导出。操作步骤及测量口径见 [局部测量工作台使用说明](doc/局部测量工作台使用说明.md)。默认 ±1 mm 为对照阈值，真实相机精度需实测验证。

基于 CAD 先验和视觉证据的管道安装状态识别与数字孪生项目。
基础模式的首次操作：

1. 选择 STL 并指定原坐标单位；
2. 确认自动识别的管道直径颜色映射，例如 22=#E74C3C;50=#3498DB；
3. 选择真实双目标定和左右已矫正照片，或点击“连接双目并抓拍”；
4. 保持默认自动模式，点击“保存并评估”；必要时在三维预览中指定管长方向，或为有歧义的局部观测指定少量模型对应。

程序自动保存现场包，在照片和表格中逐根显示“安装 / 未安装 / 遮蔽不确定”。固定机位后再次使用只需抓拍并评估。“未安装”需要可信匹配后预期位置的双目空位深度，且至少两次连续独立采集一致；匹配有歧义、遮挡或深度不足时保持不确定。管长方向的绝对位置不参与判定。旧的区域框选流程保留为手工兼容模式；二维码定位和三维姿态设置用于完整三维模式。
当前交付 **M0 单层离线演示基线**、**M1 两层 CAD 合成双目/遮挡拓扑基线**，以及 **M2 真实双目静态照片 + 3DM/3MF/STL + 本地 GUI 软件候选版**。M2 已能输出逐管三态并完成合成回归；真实相机阈值、标定精度和现场验收仍要用到场实物确认。多机位和 3DGS 属于后续阶段。

## 生产采集主线：固定相机定时拍照

生产系统采用固定相机定时拍照，不录制连续视频，也不进行实时连续跟踪。巡检间隔可在 **5～60 分钟**内配置，建议初值为 **30 分钟**；周期只决定状态发现延迟，不代表左右相机可以相隔数分钟拍摄。

每次定时触发形成一个 `CaptureGroup`：每个机位通常直接拍摄一张原始照片；需要抵抗偶发模糊、曝光波动或临时遮挡时，可在 2～5 秒窗口内进行少量重拍。组内数据始终是有限数量的离散照片，不得编码成连续视频后再作为生产权威输入。

代码保留 manifest 显式绑定的 M0 **单目照片**入口，并新增 M2 **同步双目照片**入口。两条入口都校验照片原始字节的 SHA-256、解码宽高、带时区的 ISO-8601 拍摄时间，以及 `RAW_PIXELS_NO_EXIF_TRANSFORM` 原始像素方向策略。单目 M0 不推断安装状态；双目 M2 在 CAD 网格、模型哈希、对象绑定、标定、CAD 配准、同步、图像质量、左右一致视差、颜色和宽度门禁均满足时才输出确定状态。

双目每张照片必须由 manifest 显式记录 `capture_group_id`、机位/相机 ID、左右角色、配对关系、时间戳和同步健康。左右目须在每次 `CaptureGroup` 内同步，禁止依据文件名、目录顺序或时间邻近关系猜测配对。后续多机位沿用这一显式绑定规则。

## 当前状态

M0 使用以下历史回归资产：

- `test_model/管道群.3mf`：单位为毫米的单层三管网格模型；
- `test_model/管道群.mkv`：固定视口的 CAD 桌面录屏，不是真实相机或双目数据，仅用于历史 M0 回放；
- `test_model/manifest.json`：历史模型、视频、对象映射、阈值和验收口径；
- NumPy + OpenCV 离线回放入口：读取 manifest，默认输出模型/视频审计、量测摘要、显隐时间线和验收报告；可选输出逐帧 JSONL 证据。

M1 使用 `管道群2.3mf` 的 9 根前后分层管道，生成一组可重复的虚拟双目 RGB、精确 CAD 深度、稳定实例 ID、正交立面及有向遮挡图。`管道群2.mkv` 只是固定 Fusion 视口中的对象显隐参考，不含相机运动，也不作为深度或双目真值。合成输出验证的是投影、Z-buffer 所有权和遮挡拓扑；它不代表真实深度相机性能。

M2 新增统一 CAD 网格层：3MF 使用 object ID，3DM 使用 Rhino 对象 GUID，STL 使用按连通闭合网格计算的稳定组件 ID；仅 manifest 绑定的管道参与现场分析，辅助 Curve/Point/Text 可保留。直接 Mesh 可以读取，Brep/Extrusion 必须在 3DM 中带完整 Render Mesh 缓存；Block 实例、绑定对象无网格或 GUID 缺失会明确失败。STL 不保存单位、颜色和业务 ID，因此导入时必须明确选择原坐标单位，并逐管核对自动拆分的组件、设计外径、颜色和业务 ID；相接或共享顶点的多个实体会被视为同一组件。所有坐标统一换算为毫米。分析器从实际 CAD 三角网格生成逐相机 amodal 投影和设计遮挡关系，再将 OpenCV 左右一致视差、颜色和独立宽度指标与 CAD 证据融合。

仓库中的 `field_stereo_demo_manifest.json` 把合成左右照片作为 M2 端到端回归：结果应为 8 根 `INSTALLED`、0 根 `NOT_INSTALLED`、1 根因左右目均完全遮挡而 `UNKNOWN`。这只证明软件链和安全状态机可运行，不是现场验收。

每根 `pipe_id` 在一次左右目融合后只输出一个三态结果：`INSTALLED（安装）`、`NOT_INSTALLED（未安装）` 或 `UNKNOWN（不确定）`。GUI 统一显示“不确定”；历史 M0/M1 报告中的兼容字段 `installation_state_zh` 可能写作“不明”，英文枚举和状态语义不变。安装状态与逐视点的 `FULLY_VISIBLE/PARTIALLY_OCCLUDED/FULLY_OCCLUDED/OUT_OF_FRUSTUM` 相互独立；逐视点只保存证据类型，不另设管级安装状态。当前九根管的合成场景真值均为安装；基于精确合成实例掩码的双目参考判定为 ID 1～6、8、9“安装”，ID 7“不确定”，没有“未安装”样本。正交立面不参与双目融合，因此立面中 ID 9 虽完全遮挡，仍可因左右目可见而在融合结果中判为安装。该结果不是对真实 RGB 识别算法或现场安装状态的验证。

当前 M0 识别策略是“**颜色识别 + 投影直径二次校验**”：颜色用于生成管道身份候选，画面中的投影宽度/直径桶用于排除明显不一致的候选。历史录屏没有相机内参、外参或尺度标定，因此原始宽度 `diameter_px` 属于像素域的**非标定量测**。报告中的 `projected_diameter_estimate_mm` 只是结合模型标称长度得到的归一化分类特征，必须同时标记 `diameter_source=model_prior`、`metric_calibrated=false`；它不是现场毫米量测，也不得用来宣称管径测量精度。

当前 M0 的逐管枚举为 `visibility=VISIBLE/NOT_OBSERVED`、`decision=MATCHED/UNKNOWN`，且 `installation_state` 恒为 `UNKNOWN`。候选冲突不会被强制匹配，但独立的 `AMBIGUOUS` 和无效帧状态仍是下一阶段契约。核心边界为：

```text
not_observed != not_installed
```

未观测到某根管只说明该帧没有形成有效视觉证据，可能来自 CAD 显隐、遮挡、出画、画质或算法失败；不得自动推导为“未安装”或 `MISSING`。

## 当前模型口径

| pipe_id | 颜色 | 3MF 标称外径 | 当前层 |
|---|---|---:|---|
| `pipe-white-d20` | 白 | D20 | `L0` |
| `pipe-red-d40` | 红 | D40 | `L0` |
| `pipe-blue-d45` | 蓝 | D45 | `L0` |

三个直径来自 3MF 几何和 manifest，是模型先验；MKV 中的投影宽度只参与相对二次校验。颜色、直径和 3MF object ID 均不单独充当永久业务主键，业务关联以 manifest 中的稳定 `pipe_id` 为准。

这组三维样件是当前算法回归资产，不替代后续实物目标。已提出的 D22/D50、约 1 m 相机距离仍需用真实双目原始帧、完整标定和可追溯尺度真值单独验收。

管道群2把颜色和直径保留为两个独立属性，实例身份仅由 manifest 的 `instance_id/pipe_id` 决定。其正交立面真值包含 3 根前层和 6 根后层管道：后层实例 7、9 被前层完全覆盖，其余错位重叠形成可量化的部分遮挡关系。单独依据该正交立面，ID 7、9 都不能形成安装正证据；加入合格的左右视点后，当前 ID 9 可评估为“安装”，ID 7 仍为“不确定”（报告枚举 `UNKNOWN`）。

## 仓库结构

```text
doc/
  需求文档.txt
  camera_contour_3d_pipe_migration_guide.md
  管道数字孪生识别系统测试开发计划.md
  现场双目识别与GUI使用说明.md
test_model/
  manifest.json
  pipe_group2_manifest.json
  pipe_group2_synthetic_stereo/
  field_stereo_demo_manifest.json
  管道群.3mf
  管道群.mkv
  管道群2.3mf
  管道群2.mkv
pipe_twin/
  __init__.py
  __main__.py
  cad_model.py
  cli.py
  detector.py
  gui.py
  model_3mf.py
  photo_capture.py
  pipeline.py
  state.py
  stereo_analyzer.py
  synthetic_stereo.py
tests/
  test_cad_model.py
  test_detector.py
  test_gui.py
  test_model_3mf.py
  test_photo_capture.py
  test_pipeline_safety.py
  test_repository_smoke.py
  test_state.py
  test_stereo_3dm_integration.py
  test_stereo_analyzer.py
  test_synthetic_stereo.py
```

## 干净环境安装

运行时依赖为 `numpy`、无 GUI 后端的 `opencv-python-headless`、McNeel `rhino3dm` 和二进制 DXF 支持库 `ezdxf`，版本由根目录 `requirements.txt` 固定。桌面 GUI 使用 Python 标准库 Tkinter，不引入 Qt/VTK；Windows 官方 Python 通常自带 Tk，精简 Linux 环境若需打开 GUI 应另行安装系统 Tk 包。CI 只做无桌面测试。

### 运行日志

CLI 和桌面 GUI 启动时会自动写入结构化 JSONL 日志，默认位置为项目根目录的 `logs/pipe_twin.log.jsonl`。日志包含运行 ID、命令、模型加载、GUI 数据载入、分析成功/失败和异常堆栈；文件按 10 MiB 轮转并保留 5 个备份。可通过 `PIPE_TWIN_LOG_DIR` 指定日志目录。日志会过滤疑似令牌、密码和连接字符串字段，不记录原始图像内容。

### DXF 侧立面导入

在 GUI 工具栏点击“导入DXF侧立面”，选择 DXF 文件后切换到 `dxf` 视图。支持 `LINE`、`LWPOLYLINE`、`ARC` 和 `CIRCLE`；其中 `CIRCLE`/`ARC` 会绘制侧立面外轮廓圆弧。实体默认采用 DXF 对象自身颜色（ACI 或 true color），没有对象颜色时才回退到图层颜色。需要统一调整时，点击“管径颜色配置”，同一管径会使用同一种颜色，并随 manifest 保存。当前导入用于侧立面显示和人工核对。

也可以直接点击“DXF自动建档”：程序会串联导入、图层颜色读取、默认流水号、初始图元映射和 manifest 草稿保存，减少重复操作。

如果 DXF 图元没有业务编号，界面会显示 `P001`、`P002` 等默认流水号。先点击 DXF 图元，再在右侧管道列表选择目标管道，点击“绑定DXF图元”，最后点击“保存DXF映射到manifest”；映射会写入 manifest 的 `elevation.entity_bindings`，并绑定 DXF SHA-256。保存后的 manifest 报告需要重新分析。

没有现场清单时，导入 DXF 后可点击“从DXF生成manifest草稿”。程序会用图元流水号生成可编辑的管道目录；直线/折线的设计外径先用示意值 `1 mm`，圆的直径由半径计算，需在“现场数据录入”中核对并修改。该草稿用于建立身份和颜色映射，不能直接作为现场双目分析清单，仍需补充真实 CAD 模型、标定和照片。

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

### Linux/macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Windows/Python 3.12 的验证锁文件为 `requirements-lock-windows-py312.txt`。新增或升级依赖时必须同步更新 `requirements.txt`、锁文件、干净环境安装结果和 smoke/test 结果。

## 分析定时单目照片

当前照片入口复用 M0 的单层颜色/投影直径规则。可以保留 `test_model/manifest.json` 中的 `scope`、`model`、`acceptance` 和 `limitations`，删除整个 `video` 对象，将 `schema_version` 更新为 `1.1`，再加入下列最小 `capture` 段。manifest 和照片路径均按 manifest 所在目录解析。

```json
{
  "capture": {
    "kind": "still_capture_set",
    "camera_layout": "mono",
    "capture_group_id": "inspection-20260901-0001",
    "calibration_id": null,
    "interval_minutes": 30,
    "analysis": {
      "analysis_roi_xyxy": [300, 250, 1400, 780],
      "minimum_component_area_px": 1000,
      "minimum_long_side_px": 500,
      "minimum_aspect_ratio": 8.0,
      "diameter_absolute_tolerance_mm": 5.0,
      "diameter_relative_tolerance": 0.2,
      "delta_e76_tolerance": 35.0,
      "color_rules": {
        "red": [[0, 12, 120, 255, 120, 255], [170, 179, 120, 255, 120, 255]],
        "blue": [[95, 135, 80, 255, 80, 255]],
        "white": [[0, 179, 0, 60, 190, 255]]
      }
    },
    "captures": [
      {
        "capture_id": "capture-000001",
        "expected_visible_pipe_ids": null,
        "views": {
          "mono": {
            "camera_id": "field-camera-01",
            "path": "captures/capture-000001.png",
            "sha256": "替换为照片原始字节的64位十六进制SHA-256",
            "expected_width": 1920,
            "expected_height": 1080,
            "captured_at": "2026-09-01T00:30:00.123+08:00",
            "timestamp_source": "MANIFEST_OPERATOR_CONFIRMED",
            "orientation_policy": "RAW_PIXELS_NO_EXIF_TRANSFORM"
          }
        }
      }
    ]
  }
}
```

`sha256` 必须替换为照片文件原始字节的实际哈希；`expected_width/expected_height` 是不应用 EXIF 自动旋转时的原始解码宽高。照片路径必须使用正斜杠、相对于 manifest 且解析后仍位于其目录内；当前只接收 PNG/JPEG，单文件不超过 100 MiB，声明/编码尺寸不超过 5000 万像素。`timestamp_source` 只接受 `CAMERA_HARDWARE_CLOCK`、`CAMERA_SYSTEM_CLOCK`、`EXIF_DATETIME_ORIGINAL`、`HOST_SYSTEM_CLOCK` 或 `MANIFEST_OPERATOR_CONFIRMED`。生产数据通常将 `expected_visible_pipe_ids` 设为 `null`，表示没有人工 oracle，此时报告会区分“输入审计通过”和“算法尚未评价”，不会伪造 PASS；即使测试 oracle 得到 `passed=true`，`field_acceptance_passed` 仍为 `null`，不能解读为现场验收通过。静态现场可能产生字节完全相同的照片，因此当前只报告重复 SHA 计数、不据此判失败；冻结/旧图需由后续调度器结合 slot、设备序号和可信时间戳判断。

从仓库根目录运行：

```powershell
python -m pipe_twin analyze --manifest path/to/capture_manifest.json --output outputs/photo_report.json --observations outputs/photo_observations.jsonl
```

`analyze` 是兼容入口，仍只接受单目照片或历史视频；双目必须使用下面独立的 `analyze-stereo` 命令，避免旧 M0 契约被误升级。

## 运行 3DM/3MF/STL + 双目安装状态识别

现场标定与启动已自动化：GUI 内置**棋盘格双目标定向导**——生成 A4 打印棋盘格，用已连接的双目相机抓拍 ≥10 组姿态多样的照片，自动完成左右目内参/畸变、双目联合标定和极线矫正；产出的标定写入清单格式文件，原始 K/D 与矫正映射存入工作台配置档案，抓拍时实时把相机原始帧转换为矫正图。标定、管件目录（含逐管颜色，可从照片点选取色）、二维码定位、相机方向修正、设备索引和最近清单都保存在 `outputs/measurement_workbench/workbench_profile.json`，下次启动自动恢复；工具栏“一键抓拍并分析”把抓拍、建档和识别合并为一步。向导只解决内参与极线矫正，CAD 配准仍需一次二维码定位或方向设置。详细的操作步骤和门禁口径见[局部测量工作台使用说明](doc/局部测量工作台使用说明.md)。

先审计 CAD。报告会列出对象或组件 ID、网格来源、毫米包围盒和网格闭合性；3DM 还会列出 GUID、对象名、图层和颜色：

```powershell
python -m pipe_twin inspect-model path/to/pipes.3dm --output outputs/model_audit.json
python -m pipe_twin inspect-model test_model/管道布置.stl --stl-unit millimeter --output outputs/stl_model_audit.json
```

`--stl-unit` 是 STL 必填项，可选 `millimeter/centimeter/meter/inch`。仓库样例 `管道布置.stl` 的自动审计结果为 12 个闭合管件组件、576 个顶点和 1104 个三角面，自动目录识别出的设计外径约为 26、41、51 mm 三组。

如果 Rhino 文件还包含未网格化的 Curve/Point/Text 或辅助 BRep，未带过滤参数的审计会安全拒绝（不会静默漏掉对象）；可按 Rhino 中看到的 GUID 重复传入 `--object-id`，例如 `--object-id 1234... --object-id 5678...`。`analyze-stereo` 会直接从 manifest 的 `cad_object_id` 集合过滤绑定管道，并仍对每个绑定对象严格校验。

现场 manifest 使用 `schema_version=2.0`，显式绑定模型哈希、每根管的 `pipe_id ↔ cad_object_id`、中心线/外径/颜色、双目标定与 CAD 外参，以及按时间排序的 `capture.capture_groups[].views.left/right`。当前 M2 的 SGBM 与 `Z=fx·B/d` 门禁只接收已经完成共同极线矫正的左右图（`stereo_calibration.rectified=true`）；若相机输出原始未矫正图，先用同一组内参/双目标定执行 `cv2.stereoRectify` 和 `initUndistortRectifyMap`，再把矫正后的图及其哈希写入 manifest。完整字段和实物采集清单见[现场双目识别与 GUI 使用说明](doc/现场双目识别与GUI使用说明.md)。先把 GLM/OpenCV 的原始标定转换为 manifest 标定（必须显式提供 translation_unit 和 left_camera_pose，适配器不会猜单位或 CAD 外参）。

命令：
  # 自动棋盘格标定：左右目录按排序后一一配对，默认 9×6 内角点、25 mm 方格、至少 8 对
  python -m pipe_twin calibrate-stereo --left-dir calibration/left --right-dir calibration/right --output calibration/auto_stereo.json --expected-baseline-mm 95
  python -m pipe_twin adapt-calibration --source calibration/opencv_stereo.json --output calibration/manifest_stereo_calibration.json --calibration-id field-rig-202609 --validated --registration-validated
  python -m pipe_twin validate-calibration --calibration calibration/manifest_stereo_calibration.json

`calibrate-stereo` 会统一使用 classic 检测器处理所有棋盘照片，固定分别求得的两目内参后求双目外参和 `stereoRectify`，并输出同时包含标定与原始帧矫正配方的便携 JSON。照片质量、姿态跨度、RMS、内参、基线、极线残差或视差方向任一不合格都会拒绝输出；需要不同规格时显式传 `--board-columns/--board-rows/--square-size-mm`，建议用 `--expected-baseline-mm` 填写实测镜头中心距。输出会在质量门禁通过后标记 `validated=true`，但仍保持 `registration_validated=false`；下一步在 GUI 中使用 QR 配准到 CAD，配准通过后才可用于现场分析。适配器会调用 stereoRectify，保留 R1/R2/P1/P2、统一毫米基线，并把原始 K/D/R/T 与 CAD 世界坐标位姿写进审计字段。`validated` 代表棋盘与极线几何通过，`registration_validated` 代表 CAD 配准通过；两者都通过才进入现场测量。

建议的自动标定顺序是：打印一张已知方格边长的棋盘格；让棋盘在近/中/远距离、画面四角和不同倾角各拍一组左右同步照片（建议 8–15 组）；把左目照片放入 `calibration/left`、右目照片放入 `calibration/right`，两边按同一序号命名；运行上面的命令后，在快速双目评估窗口使用二维码配准。程序会报告每一组被接受或剔除的原因和 RMS 重投影误差，失败时只需补拍提示的照片。
运行分析：

现场数据录入窗口可选择相机位于 CAD 的 ±X/±Y/±Z 方向，也可沿用当前外参并输入双目组中心、yaw、pitch、roll；它们作为刚性相机—CAD 外参调整参与完整三维投影、管件定位、角度匹配和前后关系判定。修改后的外参应使用固定控制点再次验证，不能用角度输入替代双目图像的极线矫正。

```powershell
python -m pipe_twin analyze-stereo `
  --manifest path/to/field_stereo_manifest.json `
  --output outputs/installation_status.json `
  --evidence-dir outputs/stereo_evidence
```

打开本地 GUI：

```powershell
python -m pipe_twin gui `
  --manifest path/to/field_stereo_manifest.json `
  --report outputs/installation_status.json
```

GUI 提供 CAD 状态立面/等轴示意、逐管清单、左右目当前照片和选中对象投影框。安装/未安装/不确定分别使用绿/红/琥珀，同时显示文字、英文枚举和原因码。手工载入报告时会核对模型字节哈希、manifest 哈希、model revision、采集批次、标定 ID、pipe ID 集合和 CAD 对象绑定；任一不符则全部安全降级为 `UNKNOWN`，避免用旧报告给新模型着色。

仓库自带的软件回归可直接运行：

```powershell
python -m pipe_twin analyze-stereo `
  --manifest test_model/field_stereo_demo_manifest.json `
  --output outputs/field_stereo_demo_report.json `
  --evidence-dir outputs/field_stereo_demo_evidence

python -m pipe_twin gui `
  --manifest test_model/field_stereo_demo_manifest.json `
  --report outputs/field_stereo_demo_report.json
```

预期汇总为 `installed=8, not_installed=0, unknown=1`；`PG2-B-WHITE-D20-Y103` 被前层管道在左右目完全遮挡，必须保持不确定。示例只有一个拍摄时刻，所以不会产生“未安装”；该状态的自动测试使用两个不同内容、时间间隔合格且当前仍为自由空间的左右照片组验证。相同照片副本、同一时刻的快速重拍或历史缺失但当前不再缺失都不能累计成“未安装”。

## 运行 M0 历史视频回归

从仓库根目录运行：

```powershell
python -m pipe_twin analyze --manifest test_model/manifest.json --output outputs/current_demo_report.json
```

如需保存每帧的逐管证据，追加：

```powershell
python -m pipe_twin analyze --manifest test_model/manifest.json --output outputs/current_demo_report.json --observations outputs/current_demo_observations.jsonl
```

报告记录输入资产和 manifest 哈希、model revision、软件版本、模型/视频审计、非标定量测摘要、显隐时间线与验收结果。manifest 哈希间接绑定本次阈值配置；当前尚未记录 Git commit。该命令只保留为历史 M0 回放与回归入口，不是生产采集方式，也不代表真实管道安装验收。

## 生成管道群2模拟双目与立面

仓库已提交一套生成后的黄金样例。要从 CAD 和 manifest 重新生成：

```powershell
python -m pipe_twin simulate-stereo --manifest test_model/pipe_group2_manifest.json --output-dir test_model/pipe_group2_synthetic_stereo
```

核心输出包括平滑与人工纹理版左右 RGB、深度/实例预览、保存 `float32 depth_z_mm/disparity_px`、`uint16 instance_id` 和双向可对应 mask 的 `*_truth.npz`，以及 `camera.json`、逐视角遮挡 JSON、`installation_status.json` 三态汇总、amodal 立面 PNG/SVG、有向拓扑 SVG、像素遮挡矩阵和连续几何重叠 CSV。深度/视差背景为 `NaN`，实例背景为 `0`；完整文件清单和 SHA-256 记录在 `dataset_manifest.json`。

当前虚拟双目为 1920×1080、95 mm 水平基线、约 40° 水平视场，相机距前层表面约 1 m。参数是为了建立受控算法真值，不是对现有实体双目标定参数的确认。详细口径见[管道群2模拟双目与遮挡拓扑说明](doc/管道群2模拟双目与遮挡拓扑说明.md)。

## Smoke 测试

```powershell
python -m unittest discover -s tests -v
```

Smoke 测试验证仓库交付、定时单目照片的 manifest 绑定与输入校验、历史 3MF/MKV 回放、M1 合成几何/遮挡契约、3DM GUID/单位/缓存网格读取、STL 单位/连通组件/稳定 ID、相机方向与倾斜外参、M2 双目视差与三态安全门禁、GUI 的报告绑定，以及棋盘格向导的标定数学（合成刚体恢复已知内参与基线）、极线矫正配方 fail-closed 门禁和工作台配置档案的校验/持久化；不代表真实相机、标定、物理管径或业务状态指标已经验收。

## 后续能力边界

- **多层**：M1 已建立两层 CAD 合成真值；真实图像阶段仍必须使用完整三维中心线、层标识、逐视点遮挡/可见性以及同色同径歧义处理，不能只靠颜色和投影直径确定身份。
- **双目/真实相机**：M2 已提供显式配对、内参/畸变、基线、CAD 外参、同步和 OpenCV 左右一致视差入口；仍必须用真实标定板、独立尺度真值和 D22/D50 实物完成现场阈值/精度验收，软件回归结果不能替代验收。
- **多机位**：各机位先独立产生证据，再按标定健康、可见性和时间同步进行 late fusion；增加机位不得改变核心状态机。
- **3DGS**：只作为冻结场景 epoch 的漫游、复核和覆盖分析派生资产，无权写入权威安装状态，也不得用于毫米级验收。
- **LiDAR**：不纳入当前开发和采购范围。

## 核心文档

- [需求文档](doc/需求文档.txt)
- [CAD 轮廓与三维模型迁移指南](doc/camera_contour_3d_pipe_migration_guide.md)
- [测试开发计划](doc/管道数字孪生识别系统测试开发计划.md)
- [现场双目识别与 GUI 使用说明](doc/现场双目识别与GUI使用说明.md)
- [管道群2模拟双目与遮挡拓扑说明](doc/管道群2模拟双目与遮挡拓扑说明.md)

## 数据与安全

- 不要提交虚拟环境、密钥、令牌、现场人员原始影像或大体积采集数据；
- 后续原始单目/双目照片、深度、模型权重和大体积派生资产应保存到受控对象存储，Git 中只保存 manifest、哈希和必要的小型黄金样例；
- 所有 CAD、历史录屏、照片、标定、配置、模型和数据集都应记录版本与内容哈希；
- CLI 会拒绝报告/JSONL 与 manifest、3MF、历史 MKV、manifest 绑定照片或彼此使用同一路径；输出先在各自目录完整暂存，再逐文件原子替换，任一成组提交失败会恢复两份旧输出；
- 任何健康失败、完全遮挡或不可评估场景都必须安全降级为 `UNKNOWN/UNDETERMINED/ERROR`。

## 协作

提交变更前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。安全问题请按 [SECURITY.md](SECURITY.md) 处理。
