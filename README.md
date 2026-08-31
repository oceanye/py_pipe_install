# py_pipe_install

基于 CAD 先验和视觉证据的管道安装状态识别与数字孪生项目。当前同时交付 **M0 单层离线演示基线**与 **M1 两层 CAD 合成双目/遮挡拓扑基线**；真实双目现场量测、多机位和 3DGS 属于后续阶段。

## 生产采集主线：固定相机定时拍照

生产系统采用固定相机定时拍照，不录制连续视频，也不进行实时连续跟踪。巡检间隔可在 **5～60 分钟**内配置，建议初值为 **30 分钟**；周期只决定状态发现延迟，不代表左右相机可以相隔数分钟拍摄。

每次定时触发形成一个 `CaptureGroup`：每个机位通常直接拍摄一张原始照片；需要抵抗偶发模糊、曝光波动或临时遮挡时，可在 2～5 秒窗口内进行少量重拍。组内数据始终是有限数量的离散照片，不得编码成连续视频后再作为生产权威输入。

当前代码只实现 manifest 显式绑定的**单目照片**入口，即 `camera_layout=mono` 且每条记录只有 `views.mono`。入口校验照片原始字节的 SHA-256、解码宽高、带时区的 ISO-8601 拍摄时间，以及 `RAW_PIXELS_NO_EXIF_TRANSFORM` 原始像素方向策略；当前质量阈值尚未经过现场相机验收，安装状态仍保持 `UNKNOWN`。

未来接入双目或多机位时，每张照片必须由 manifest 显式记录 `capture_group_id`、机位/相机 ID、左右角色、配对关系、曝光时间和同步健康。左右目仍须在每次 `CaptureGroup` 内同步，禁止依据文件名、目录顺序或时间邻近关系猜测配对。

## 当前状态

M0 使用以下历史回归资产：

- `test_model/管道群.3mf`：单位为毫米的单层三管网格模型；
- `test_model/管道群.mkv`：固定视口的 CAD 桌面录屏，不是真实相机或双目数据，仅用于历史 M0 回放；
- `test_model/manifest.json`：历史模型、视频、对象映射、阈值和验收口径；
- NumPy + OpenCV 离线回放入口：读取 manifest，默认输出模型/视频审计、量测摘要、显隐时间线和验收报告；可选输出逐帧 JSONL 证据。

M1 使用 `管道群2.3mf` 的 9 根前后分层管道，生成一组可重复的虚拟双目 RGB、精确 CAD 深度、稳定实例 ID、正交立面及有向遮挡图。`管道群2.mkv` 只是固定 Fusion 视口中的对象显隐参考，不含相机运动，也不作为深度或双目真值。合成输出验证的是投影、Z-buffer 所有权和遮挡拓扑；它不代表真实深度相机性能。

每根 `pipe_id` 在一次左右目融合后只输出一个三态结果：`INSTALLED（安装）`、`NOT_INSTALLED（未安装）` 或 `UNKNOWN（不明）`。安装状态与逐视点的 `FULLY_VISIBLE/PARTIALLY_OCCLUDED/FULLY_OCCLUDED/OUT_OF_FRUSTUM` 相互独立；逐视点只保存证据类型，不另设管级安装状态。当前九根管的合成场景真值均为安装；基于精确合成实例掩码的双目参考判定为 ID 1～6、8、9“安装”，ID 7“不明”，没有“未安装”样本。正交立面不参与双目融合，因此立面中 ID 9 虽完全遮挡，仍可因左右目可见而在融合结果中判为安装。该结果不是对真实 RGB 识别算法或现场安装状态的验证。

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

管道群2把颜色和直径保留为两个独立属性，实例身份仅由 manifest 的 `instance_id/pipe_id` 决定。其正交立面真值包含 3 根前层和 6 根后层管道：后层实例 7、9 被前层完全覆盖，其余错位重叠形成可量化的部分遮挡关系。单独依据该正交立面，ID 7、9 都不能形成安装正证据；加入合格的左右视点后，当前 ID 9 可评估为“安装”，ID 7 仍为“不明”。

## 仓库结构

```text
doc/
  需求文档.txt
  camera_contour_3d_pipe_migration_guide.md
  管道数字孪生识别系统测试开发计划.md
test_model/
  manifest.json
  pipe_group2_manifest.json
  pipe_group2_synthetic_stereo/
  管道群.3mf
  管道群.mkv
  管道群2.3mf
  管道群2.mkv
pipe_twin/
  __init__.py
  __main__.py
  cli.py
  detector.py
  model_3mf.py
  photo_capture.py
  pipeline.py
  state.py
  synthetic_stereo.py
tests/
  test_detector.py
  test_model_3mf.py
  test_photo_capture.py
  test_pipeline_safety.py
  test_repository_smoke.py
  test_state.py
  test_synthetic_stereo.py
```

## 干净环境安装

当前 M0 运行时依赖为 `numpy` 和无 GUI 的 OpenCV 包 `opencv-python-headless`，版本由根目录 `requirements.txt` 固定。

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

该入口当前只接受单目照片。把 `camera_layout` 改成 `stereo`，或同时放入 `video` 与 `capture`，都会被拒绝；双目/多机位需等显式配对和同步契约实现后再接入。

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

Smoke 测试验证仓库交付、定时单目照片的 manifest 绑定与输入校验、历史 3MF/MKV 回放、M1 合成几何/遮挡契约，以及人工纹理双目的 OpenCV StereoSGBM 可用性；不代表真实相机、标定、物理管径或业务状态指标已经验收。

## 后续能力边界

- **多层**：M1 已建立两层 CAD 合成真值；真实图像阶段仍必须使用完整三维中心线、层标识、逐视点遮挡/可见性以及同色同径歧义处理，不能只靠颜色和投影直径确定身份。
- **双目/真实相机**：必须补充定时拍摄的左右原始照片、显式配对关系、内参与畸变、基线、外参、同步和独立尺度真值后，才允许输出毫米级几何结论。
- **多机位**：各机位先独立产生证据，再按标定健康、可见性和时间同步进行 late fusion；增加机位不得改变核心状态机。
- **3DGS**：只作为冻结场景 epoch 的漫游、复核和覆盖分析派生资产，无权写入权威安装状态，也不得用于毫米级验收。
- **LiDAR**：不纳入当前开发和采购范围。

## 核心文档

- [需求文档](doc/需求文档.txt)
- [CAD 轮廓与三维模型迁移指南](doc/camera_contour_3d_pipe_migration_guide.md)
- [测试开发计划](doc/管道数字孪生识别系统测试开发计划.md)
- [管道群2模拟双目与遮挡拓扑说明](doc/管道群2模拟双目与遮挡拓扑说明.md)

## 数据与安全

- 不要提交虚拟环境、密钥、令牌、现场人员原始影像或大体积采集数据；
- 后续原始单目/双目照片、深度、模型权重和大体积派生资产应保存到受控对象存储，Git 中只保存 manifest、哈希和必要的小型黄金样例；
- 所有 CAD、历史录屏、照片、标定、配置、模型和数据集都应记录版本与内容哈希；
- CLI 会拒绝报告/JSONL 与 manifest、3MF、历史 MKV、manifest 绑定照片或彼此使用同一路径；输出先在各自目录完整暂存，再逐文件原子替换，任一成组提交失败会恢复两份旧输出；
- 任何健康失败、完全遮挡或不可评估场景都必须安全降级为 `UNKNOWN/UNDETERMINED/ERROR`。

## 协作

提交变更前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。安全问题请按 [SECURITY.md](SECURITY.md) 处理。
