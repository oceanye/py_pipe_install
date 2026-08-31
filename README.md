# py_pipe_install

基于 CAD 先验和视觉证据的管道安装状态识别与数字孪生项目。当前交付为 **M0 单层离线演示基线**；固定双目、真实现场量测、多层遮挡、多机位和 3DGS 属于后续阶段。

## 当前状态

M0 使用以下成套资产：

- `test_model/管道群.3mf`：单位为毫米的单层三管网格模型；
- `test_model/管道群.mkv`：固定视口的 CAD 桌面录屏，不是真实相机或双目数据；
- `test_model/manifest.json`：模型、视频、对象映射、阈值和验收口径；
- NumPy + OpenCV 离线分析入口：读取 manifest，默认输出模型/视频审计、量测摘要、显隐时间线和验收报告；可选输出逐帧 JSONL 证据。

当前识别策略是“**颜色识别 + 投影直径二次校验**”：颜色用于生成管道身份候选，画面中的投影宽度/直径桶用于排除明显不一致的候选。录屏没有相机内参、外参或尺度标定，因此原始宽度 `diameter_px` 属于像素域的**非标定量测**。报告中的 `projected_diameter_estimate_mm` 只是结合模型标称长度得到的归一化分类特征，必须同时标记 `diameter_source=model_prior`、`metric_calibrated=false`；它不是现场毫米量测，也不得用来宣称管径测量精度。

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

## 仓库结构

```text
doc/
  需求文档.txt
  camera_contour_3d_pipe_migration_guide.md
  管道数字孪生识别系统测试开发计划.md
test_model/
  manifest.json
  管道群.3mf
  管道群.mkv
pipe_twin/
  __init__.py
  __main__.py
  cli.py
  detector.py
  model_3mf.py
  pipeline.py
  state.py
tests/
  test_detector.py
  test_model_3mf.py
  test_pipeline_safety.py
  test_repository_smoke.py
  test_state.py
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

## 运行 M0 演示

从仓库根目录运行：

```powershell
python -m pipe_twin analyze --manifest test_model/manifest.json --output outputs/current_demo_report.json
```

如需保存每帧的逐管证据，追加：

```powershell
python -m pipe_twin analyze --manifest test_model/manifest.json --output outputs/current_demo_report.json --observations outputs/current_demo_observations.jsonl
```

报告记录输入资产和 manifest 哈希、model revision、软件版本、模型/视频审计、非标定量测摘要、显隐时间线与验收结果。manifest 哈希间接绑定本次阈值配置；当前尚未记录 Git commit。该命令是当前约定的稳定入口，不代表真实管道安装验收。

## Smoke 测试

```powershell
python -m unittest discover -s tests -v
```

Smoke 测试只验证仓库交付、3MF/MKV/manifest 基本可读性和 M0 最小分析链，不代表真实相机、标定、物理管径、遮挡推理或业务状态指标已经验收。

## 后续能力边界

- **多层**：必须引入完整三维中心线、层标识、逐视点遮挡/可见性以及同色同径歧义处理；不能继续只靠颜色和投影直径确定身份。
- **双目/真实相机**：必须补充原始左右帧、内参与畸变、基线、外参、同步和独立尺度真值后，才允许输出毫米级几何结论。
- **多机位**：各机位先独立产生证据，再按标定健康、可见性和时间同步进行 late fusion；增加机位不得改变核心状态机。
- **3DGS**：只作为冻结场景 epoch 的漫游、复核和覆盖分析派生资产，无权写入权威安装状态，也不得用于毫米级验收。
- **LiDAR**：不纳入当前开发和采购范围。

## 核心文档

- [需求文档](doc/需求文档.txt)
- [CAD 轮廓与三维模型迁移指南](doc/camera_contour_3d_pipe_migration_guide.md)
- [测试开发计划](doc/管道数字孪生识别系统测试开发计划.md)

## 数据与安全

- 不要提交虚拟环境、密钥、令牌、现场人员原始影像或大体积采集数据；
- 后续原始双目数据、深度、模型权重和大体积派生资产应保存到受控对象存储，Git 中只保存 manifest、哈希和必要的小型黄金样例；
- 所有 CAD、录屏、标定、配置、模型和数据集都应记录版本与内容哈希；
- CLI 会拒绝报告/JSONL 与 manifest、3MF、MKV 或彼此使用同一路径，并通过同目录临时文件原子提交输出；
- 任何健康失败、完全遮挡或不可评估场景都必须安全降级为 `UNKNOWN/UNDETERMINED/ERROR`。

## 协作

提交变更前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。安全问题请按 [SECURITY.md](SECURITY.md) 处理。
