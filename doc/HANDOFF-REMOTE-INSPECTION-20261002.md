# 2026-10-02 远程办公室相机检查交接

更新时间：2026-10-02（Asia/Shanghai）。执行基线为 `main` 的
`1886f736fc49a32d4eaaa93f17cd0cff3b5467ad`。本交接用于下一次办公室端检查，
重点确认连续读帧变黑、稳定曝光和可量测管面；不得把本次失败回放改写成识别成功。

## 当前证据结论

本机已从办公室 `8770` 控制服务提交并从 `8765` 文件服务取回：

| 批次 | 结果 |
| --- | --- |
| `remote-20261002-145837-4b644e75` | 1 对，`USABLE`；原始照片可解码 |
| `remote-20261002-152358-4cda1f0b` | 8 对，1 对可用、7 对近黑，`PARTIAL` |
| `remote-20261002-152710-8cc8d28d` | 1 对，`USABLE`；左右棋盘均检测到 48 点 |

连续批次第 2～8 对左右目均为 `p50=0`、`p95=1`、`max=1`，命中
`LOW_LUMINANCE_*` 和 `NEAR_BLACK_FRACTION`。原图已保留，但这些对不能进入量测。

当前使用的候选标定是 `field-chess-fb5dc0f71ee5`，基线约 60.6446 mm、
矫正焦距约 2998.7002 px、每目 1920×1080。它的来源声明为
`REUSED_CANDIDATE_NOT_NEWLY_VALIDATED_FIELD_CALIBRATION`，且
`registration_validated=false`；不得在报告中称为本轮重新验证的生产标定。

可用棋盘对的局部检查为：矫正后垂直残差 RMS 1.441 px、P95 2.017 px、最大
2.237 px；三角测量深度中位数 1291.70 mm；棋盘格边长中位数 17.010 mm。
这些数值只说明当前视野的局部几何，不能作为管道距离真值。

用 `test_model/管道布置.dxf`（12 个候选，标称 51/41/26 mm，现场实际总数约 3）
执行自动立面流程后：

- `INSTALLED=0`、`NOT_INSTALLED=0`、`UNKNOWN=12`；
- 局部圆柱观测数为 0，配准原因是 `NO_LOCAL_SURFACE_OBSERVATIONS`；
- 有效深度约 10.5%～11.0%，并带 `LOW_STEREO_COVERAGE`；
- 候选预算临时增至 384 的诊断仍得到 0 个圆柱观测，不能靠放宽门禁制造测量值。

51/41/26 mm 在当前报告中仍是模型标称值，不是现场实测直径；约 1291.7 mm
是棋盘局部深度统计，不是任何一根管的中心线或最近表面距离。

## 办公室端执行

相机预览、标定窗口必须关闭，只保留主 GUI、8770 控制服务和 8765 文件服务。
默认免令牌只允许本机或 Tailscale 地址；不要把 token 写入仓库或证据包。

```powershell
git fetch origin
git switch main
git pull --ff-only origin main
git rev-parse HEAD
python --version
python -m pip check
python -m pipe_twin capture-stereo --help
```

先从办公室本机确认服务：

```powershell
Invoke-RestMethod http://127.0.0.1:8770/v1/health
Invoke-WebRequest http://127.0.0.1:8765/ -UseBasicParsing
```

随后在**没有其他程序占用相机**的条件下做一轮 8 对连续读帧。可以在控制端提交：

```powershell
python -m pipe_twin remote-capture `
  --agent-url http://<办公室-Tailscale-IPv4>:8770 `
  --count 8 --interval-s 0.5 `
  --detect-chessboard --board-columns 8 --board-rows 6 `
  --wait --poll-s 1 --timeout-s 90
```

若直接在办公室端运行，使用等价的 `capture-stereo`，把输出放入新的
`outputs/<RUN_ID>/attempt-01/`，不能覆盖旧目录：

```powershell
python -m pipe_twin capture-stereo `
  --output-dir outputs/<RUN_ID>/attempt-01 `
  --left-index 0 --layout side_by_side_left_right `
  --eye-width 1920 --eye-height 1080 `
  --count 8 --interval-s 0.5 `
  --detect-chessboard --board-columns 8 --board-rows 6
```

### 本轮必须检查

1. `capture.json` 的 `status`、`quality_status`、`pair_count`、`usable_pair_count`、
   `unusable_pair_count`。
2. 每一对左右 `image_health` 的 `p50/p95/max/dark_fraction/reason_codes`。
3. 相机请求曝光、驱动实际曝光和分辨率；不能把 `AUTO` 写成已满足 1/200 秒。
4. 8 对中只要出现后续近黑帧，标记 `PARTIAL`，保留全部原图和日志，先查相机/驱动
   的曝光状态、自动曝光切换、USB/设备占用，不要送入管径结论。
5. 若 8 对均可用，再检查棋盘是否左右各 48 点；棋盘检测通过仍不等于完成多姿态标定。

## 分析门槛

只有以下条件全部满足，才可以继续自动 DXF 识别：

- 左右原图成对、1920×1080、哈希可复算，且 `quality_status=USABLE`；
- 使用与当前分辨率和设备匹配的矫正配方；
- `surface_audit.status=COMPLETE`，不能有 `LOCAL_SURFACE_SEARCH_TRUNCATED`；
- 左右目都拟合出局部圆柱，至少形成 3 个现场观测；
- 12 个 DXF 候选的对应由外径、三维距离和截面布局确定，颜色只作辅助；
- 三根实物各有卡尺直径和独立距离参照，且明确距离定义。

不满足时，报告必须保持 `UNKNOWN` 或 `NOT_RUN`。不允许复制模型直径、降低残差门槛、
插值填洞、用棋盘深度代替管道距离，或把颜色直接当作实例身份。

## 必须回传

远程检查完成后提交或上传一个不含 token 的小报告，并通过 8765 或受控 Release 传回大文件：

```text
RUN_ID:
CODE_SHA:
CAPTURE: PASS | PARTIAL | FAIL
PAIR_COUNT:
USABLE_PAIR_COUNT:
UNUSABLE_PAIR_COUNT:
ACTUAL_LAYOUT_AND_SIZE:
EXPOSURE_REQUEST_AND_READBACK:
BLACK_FRAME_COUNT_AND_REASON_CODES:
CALIBRATION_ID_AND_SHA256:
BOARD_DETECTION:
SURFACE_AUDIT_STATUS:
OBSERVATION_COUNT:
PIPE_TO_MODEL_MAPPING:
DIAMETER_REFERENCE_STATUS:
DISTANCE_REFERENCE_STATUS:
ANALYSIS: PASS | PARTIAL | UNKNOWN | NOT_RUN
LIMITATIONS_AND_NEXT_ACTION:
```

至少保留 `capture.json`、`job.json`、左右 PNG、实际标定副本及 SHA-256、
`environment.json`、完整命令、`checks.json`、`report.md`、日志和
`evidence_manifest.json`。原始照片、私有标定源和日志不要提交 Git 历史。

## 本机回放入口

远程回传后，本机先按清单核验下载：

```powershell
python -m pipe_twin fetch-stereo `
  --url http://<办公室-Tailscale-IPv4>:8765/<RUN_ID>/ `
  --output-dir outputs/remote_checks/<RUN_ID> --workers 4
```

只有 `*.fetch.json.status=PASS`、文件哈希一致且清单未变化，才开始离线分析。
分析报告须区分“采集完成”“流程通过”和“量测精度通过”。
