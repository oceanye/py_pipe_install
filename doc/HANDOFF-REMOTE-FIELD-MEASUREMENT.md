# 远程现场三管实测要求与工作交接

更新时间：2026-10-02（Asia/Shanghai）。

本文件用于 `main` 合并后的下一轮办公室现场复测。目标是从 DXF 的 12 个候选管位中，识别现场实际安装的 3 根管，并取得可以和独立参照比较的管径与距离。它把“采集完成”“流程跑通”和“量测精度通过”分开记录；没有完整证据时，结果必须保持 `UNKNOWN` 或 `NOT_RUN`。

## 当前基线

- 主线：`main`，合并提交 `ffee4e5b902a3bd6b7ad21bc1e08be4dfa2385f3`。
- 模型：`test_model/管道布置.dxf`，12 个完整圆形管位；矩形边框不是管道。对应 `管道布置.stl` 可用于交叉核对，但现场自动识别不要求 3DM/3MF。
- 模型外径：蓝色 51 mm、红色 41 mm、白色 26 mm。颜色只作候选和辅助判定，不能单独决定实例身份。
- 现场实物：已知 3 根，彼此平行；具体对应哪 3 个 CAD 管位仍需由双目外径、距离和截面布局确定。不要默认“一蓝一红一白”。
- 相机：历史现场配置为索引 `0`、并排左右流、每目 `1920×1080`，左右顺序为 `left_right`；实际设备回读必须写入报告。
- 标定板：`8×6` 内角点（9×7 方格），用户实量方格边长约 17 mm。旧的 15 姿态标定只能作为候选配置，不能因为文件存在就宣称本轮现场精度已验收。
- 几何口径：距离必须注明是左目矫正光心到中心线、同一截面最近表面，还是轮廓切点；局部截面不能推断整根管的端点、总长或沿轴向安装位置。

## 现场测量点

复测前在现场照片和 `reference_measurements.json` 中使用固定的临时标签 `R1`、`R2`、`R3`。标签只表示当次实物，不把颜色当标签，也不要在拍摄期间移动管道或相机。

| 点位 | 数量 | 放置/记录要求 | 用途 |
|---|---:|---|---|
| `M0` 控制点 | 1 组 | 棋盘格完整进入左右视野；记录内角点规格、实际方格边长、摆放位置和是否移动 | 检查图像尺寸、角点、局部尺度和左右几何 |
| `S1`、`S2`、`S3` 管道截面点 | 每根 1 个 | 在 R1/R2/R3 的可见中段贴可移除标记，三点尽量处在同一轴向站位；两目都能看到至少一段连续外轮廓 | 双目局部圆柱拟合和 3/12 身份匹配 |
| `D1`、`D2`、`D3` 直径参照 | 每根 1 组 | 在 S1/S2/S3 同一截面用卡尺测外径至少 3 次，记录每次读数、仪器分辨率和操作者 | 独立验证 `measured_diameter_mm`；模型名义直径不能代替实测 |
| `Q1`、`Q2`、`Q3` 距离参照 | 每根 1 组 | 用独立尺具/测距仪从固定相机基准记录距离；明确基准是左目光心、镜头外壳基准还是其他点，并分别记录中心线/最近表面口径 | 验证软件输出的 Z 距离和空间距离 |
| `V1` 全场视点 | 至少 1 组 | 三根实物、部分候选空位和 M0 同时可见，无遮挡，保持相机与管道相对姿态稳定 | 识别三根实物和判断未安装候选 |
| `V2` 斜视视点 | 可选 1 组 | 相机相对管轴有明显俯视/斜视但 S1–S3 仍可见；若移动相机，必须作为新的采集批次并重新记录姿态 | 验证顶部/最近外表面和斜视深度渐变 |

最低可接受的现场参照是每根管一组直径读数和一组距离读数。没有 Q1–Q3 时可以做流程回归，但 `independent_metric_ground_truth` 必须写 `NOT_RUN`，不得写“精度通过”。

## 办公室端准备

在办公室项目根目录执行。先确认没有 GUI 相机预览、标定窗口或其他程序占用设备；主 GUI 可以保留，但相机预览必须关闭。8765 文件服务可以继续运行，8770 控制服务只在确实需要远程触发时启动。

```powershell
git fetch origin
git switch main
git pull --ff-only origin main
git rev-parse HEAD
python --version
python -m pip check
python -m pipe_twin capture-stereo --help
```

应记录完整代码 SHA、Python/OpenCV 版本、相机索引、实际并排流尺寸、左右顺序、快门请求值和驱动回读值。不能把 `AUTO` 的回读写成已经满足 1/200 秒；Windows 驱动可能以约 1/256 秒表示短快门。

推荐直接双击仓库根目录 `run_gui.bat`，它会启动 GUI、8770 控制服务和 8765 只读文件服务；命令行等价入口是 `python -m pipe_twin office-client`。客户端自动检测 Tailscale 地址，默认使用 Tailscale 内网免令牌，并尝试创建仅限 `100.64.0.0/10` 的防火墙规则；需要额外认证时再加 `--require-token`。若启动状态为 `LOCAL_ONLY` 或 `NEEDS_ADMINISTRATOR`，远程端不能开始正式采集，先按 [远程抓拍控制通道](REMOTE-CAPTURE-AGENT.md) 修正网络/权限。不要把 token 放入仓库或证据包。

首次启动服务后，先从办公室本机做一对小抓拍，再从家中验证状态查询和下载。不要先提交 30 对任务来排查端口或设备问题。

## 家中端远程采集

在确认办公室服务健康、没有预览占用后，家中端先提交一组小任务：

```powershell
python -m pipe_twin remote-capture `
  --agent-url http://<办公室-Tailscale-IP>:8770 `
  --count 3 `
  --interval-s 1 `
  --detect-chessboard `
  --board-columns 8 `
  --board-rows 6 `
  --wait
```

小任务成功并能通过 8765 下载校验后，再提交正式批次。正式批次建议 30 对、2 秒间隔、120 秒时限；这是采集数量上限，不是要求把静止照片误当成 30 个独立姿态：

```powershell
python -m pipe_twin remote-capture `
  --agent-url http://<办公室-Tailscale-IP>:8770 `
  --count 30 `
  --interval-s 2 `
  --duration-s 120 `
  --detect-chessboard `
  --board-columns 8 `
  --board-rows 6 `
  --wait
```

任务返回 `job_id` 后，使用同一运行目录的 `evidence_manifest.json` 下载并逐文件校验：

```powershell
python -m pipe_twin fetch-stereo `
  --url http://<办公室-Tailscale-IP>:8765/<job_id>/ `
  --output-dir outputs/remote_fetch/<job_id>
```

若 8770 不可用，远程不能触发相机；办公室端必须由现场人员运行 `capture-stereo`，完成后再通过 8765 提供该目录。8765 能访问只说明文件服务可读，不能说明实时相机可控。

## 采集批次和参数纪律

1. `attempt-00-health`：1–3 对，确认设备索引、实际分辨率、左右顺序、非黑帧和文件落盘；失败也保留日志。
2. `attempt-01-daylight`：光照充分时的正式批次，优先原始灰度/默认匹配，保留 30 对原图和完整清单。记录手动短快门的请求与回读；不能用曝光不足的图像替代白天数据。
3. `attempt-02-low-light`：只有在需要比较时才做，场景、相机和模型姿态保持一致，作为单独批次；弱光处理改善覆盖率不等于深度精度通过。
4. `attempt-03-oblique`：若执行 V2 斜视视点，建立新目录并记录相机是否移动。改变机位后不能沿用上一批次的历史对齐或空位证据。

每一批都必须保留原始 PNG、`capture.json`、`job.json`、`evidence_manifest.json`、stdout/stderr、结构化日志和实际参数。禁止覆盖前一次目录，禁止生成空白照片或用历史照片冒充当前采集。

## 分析与验收门槛

### 流程通过

- 至少一批完整左右图，原始尺寸为 1920×1080，左右图成对且哈希可复算。
- 标定分辨率、相机布局和实际设备一致；棋盘检测结果、角点垂直残差和深度审计均写入报告。
- `surface_audit.status` 为 `COMPLETE`，没有 `LOCAL_SURFACE_SEARCH_TRUNCATED`；如果仍截断，只能报告诊断结果，不能报告管径/距离。
- 双目两目都形成可接受的局部圆柱，3 个观测与 12 个模型候选得到唯一的三管对应；对应依据以外径、三维距离和截面布局为主，颜色只作辅助。
- 三根实物均有 `measured_diameter_mm` 和命名的距离输出；未安装候选没有有效表面时仍为 `UNKNOWN`，不凭单帧空白证明未安装。

### 精度通过

精度验收必须把 D1–D3、Q1–Q3 的独立参照和软件输出逐项对齐。首次现场可以把以下作为建议工程门槛，现场仪器和项目规范另有要求时应在报告中覆盖它们：

- 直径误差目标：`abs(双目外径 - 卡尺外径) <= 2 mm` 且记录相对误差；
- 距离误差目标：在同一距离定义下 `abs(软件距离 - 独立参照) <= 10 mm`；
- 三根管的身份在重复批次中保持一致，不能依靠颜色变化后仍改配管号。

这些是验收目标，不是代码自动颁发的 `validated=true`。若参照定义不一致、标定未重新验证、只有单一固定姿态或任一管的可见弧/有效深度不足，报告写 `PARTIAL` 或 `NOT_RUN`。

## 必须回传的证据包

远程批次完成后，按运行 ID 回传一个目录或压缩包，至少包括：

```text
<RUN_ID>/
  attempt-*/
    capture.json
    job.json
    pair_*_left.png
    pair_*_right.png
  calibration/
    calibration.json 或实际使用的标定副本
  reference_measurements.json
  setup_photo_left.png        # 可选，但应能看出 M0、S1–S3 和遮挡情况
  setup_photo_right.png       # 可选
  environment.json
  commands.txt
  checks.json
  report.md
  logs/
  evidence_manifest.json
```

`reference_measurements.json` 至少包含：

```json
{
  "run_id": "field-YYYYMMDD-HHMMSS-xxxx",
  "physical_pipe_count": 3,
  "board": {"inner_corners": [8, 6], "square_mm": 17.0},
  "pipes": [
    {"physical_id": "R1", "color_observed": "", "diameter_readings_mm": [],
     "distance_reference": {"definition": "", "centerline_mm": null, "nearest_surface_mm": null}},
    {"physical_id": "R2", "color_observed": "", "diameter_readings_mm": [],
     "distance_reference": {"definition": "", "centerline_mm": null, "nearest_surface_mm": null}},
    {"physical_id": "R3", "color_observed": "", "diameter_readings_mm": [],
     "distance_reference": {"definition": "", "centerline_mm": null, "nearest_surface_mm": null}}
  ]
}
```

照片、token、私有标定源文件和完整日志不要直接提交 Git。大文件通过受控文件服务或 GitHub Release 传递；主线只提交小体积的 `report.md`、`checks.json`、`environment.json`、`upload.json` 和哈希索引。

## 回传模板

远程执行者完成后按下面结构回复，并把同样内容写入 `report.md`：

```text
RUN_ID:
CODE_SHA:
CAPTURE: PASS | PARTIAL | FAIL
ANALYSIS: PASS | PARTIAL | UNKNOWN | NOT_RUN
UPLOAD: PASS | FAIL
PAIR_COUNT:
ACTUAL_LAYOUT_AND_SIZE:
SHUTTER_REQUEST_AND_READBACK:
CALIBRATION_ID_AND_SHA256:
BOARD_DETECTION:
PIPE_REFERENCE_IDS: R1/R2/R3
PIPE_TO_MODEL_MAPPING: physical_id -> DXF/STL pipe_id
DIAMETER_REFERENCE_STATUS:
DISTANCE_REFERENCE_STATUS:
SURFACE_AUDIT_STATUS:
REJECTION_REASON_COUNTS:
DOWNLOAD_SHA256:
LIMITATIONS_AND_NEXT_ACTION:
```

只要出现相机占用、黑帧、左右尺寸不一致、标定与分辨率不匹配、候选搜索截断、身份歧义或缺少独立距离参照，就要在 `LIMITATIONS_AND_NEXT_ACTION` 中明确写出，不得用“流程已完成”代替现场量测结论。

## 代码和分支约定

- `main` 是办公室部署基线；现场先拉取并记录 SHA，再开始采集。
- 新代码必须从 `main` 建分支并通过 PR；不要在现场机器上直接改代码后覆盖部署。
- 原始照片和 token 不进入 Git；报告索引和哈希可以进入 Git。
- 现场失败也要回传失败报告、stderr、日志和证据清单，不能只回传一张报错截图。
- 下一轮分析端先执行 `git pull --ff-only origin main`，再用 `fetch-stereo` 下载对应运行目录，核验 `evidence_manifest.json` 后才开始离线分析。
