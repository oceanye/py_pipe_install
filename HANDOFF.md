# 测试与远程开发交接

更新时间：2026-10-02（Asia/Shanghai）。PR #10 已 squash 合并到 `main`；本文件保留开发和测试背景，下一轮远程现场执行以 [远程现场三管实测要求与工作交接](doc/HANDOFF-REMOTE-FIELD-MEASUREMENT.md) 为准。用户最新分工：**本机侧重点完成测试、数据复核和汇报；现场端按 handoff 完成采集与回传。**

## 最新远程检查入口（2026-10-02）

**长曝光诊断更新（本机提交 `ebb1e08`）**：手动曝光上限已扩展到 `0.1–2000 ms`，GUI 增加 1 秒/2 秒档位，工作台配置校验与远程请求契约保持一致。全套回归为 `500 passed, 1 skipped, 256 subtests passed`，`pip check` 通过。办公室当前两次 20 秒预热（手动 33.333 ms 与 AUTO）都未得到可用帧：手动任务 `remote-20261002-181356-3bb5d6cc`、AUTO 任务 `remote-20261002-181608-01761cc0` 均为 `0/3`；单帧诊断 `remote-20261002-181903-221da220` 显示目标 33.333 ms、驱动回读 31.25 ms，但左右图 P50=1、近黑像素约 99.98%。这证明请求链已到驱动，但不能证明传感器实际曝光或光路正常；当前不能把问题归因于曝光时长，也不能输出测距/管径结论。

办公室客户端需拉取并重启到 `ebb1e08` 后，才可执行 1 秒/2 秒实拍。先做 `--warmup-s 20 --count 1` 的 1000 ms、2000 ms 对照，检查 `capture.json.camera_exposure` 的 `status/readback_native/reported_ms` 与图像统计；若驱动回读已接受而图像仍近黑，应优先检查镜头/遮挡、现场灯光实际照射、USB/驱动状态和摄像头原厂控制面板，不把长曝光强行送入识别。

**办公室已执行并回传（16:34 起）**：[黑图复核报告](field_reports/field-20261002-163419-0b0305c7/report.md)。采集基线 `9823119`：本机复现手动短快门 1/8 可用，AUTO 对照 8/8 可用；原远程黑图下载与源文件哈希一致。随后本机已在 `0bf1cec` 贯通远程曝光参数，新增 [1/30 秒曝光交接](doc/HANDOFF-REMOTE-EXPOSURE-1-30.md)。办公室旧进程需重启后才能接收 `exposure_ms=33.333333`；现有测距/管径结论仍为 UNKNOWN。

用户授权合并本地未提交的文件服务代码，兼容实现为 `9318e3f`；最终 489 项测试和 254 子测试通过。该服务入口现已跟踪，GUI 服务生命周期沿用最新 office-client；详见报告的合并说明。执行阶段 8770 无监听，故实拍使用办公室等价 CLI，8765 下载正常。已有进程未被强制重启。

下一次办公室检查请优先执行 [远程办公室相机检查交接](doc/HANDOFF-REMOTE-INSPECTION-20261002.md)。当前 `main` 基线为 `1886f736fc49a32d4eaaa93f17cd0cff3b5467ad`。本机最近一次复核确认：连续 8 对中只有 1 对可用、7 对近黑；棋盘单组局部检查通过，但 DXF 自动流程仍为 12 根 `UNKNOWN`、0 个局部圆柱观测。远程端的首要任务是查清曝光在连续读帧后变黑，并回传逐对 `image_health`、曝光读回、日志和哈希清单；在 `surface_audit=COMPLETE` 且形成局部圆柱之前，不输出管径或距离结论。

## 2026-10-02 现场回传后的主线调整

本轮现场证据已并入 `main`：[`field-20261002-113318-fef83aa4/report.md`](field_reports/field-20261002-113318-fef83aa4/report.md)。该报告绑定现场执行时的 `CODE_SHA=a7dc2244ecd202b883d8c0ee2c761636f38895f0`，记录的是调整前基线；它确认 28 对 AUTO 图像可稳定读到棋盘，但短快门变暗，六次模型回放均为 `TRUNCATED`、0 根有效圆柱、12 个候选 `UNKNOWN`，独立量测未执行。现场报告中的失败结论没有被改写成量测成功。

随后提交 `943096b` 将 handoff 的三个开发项落地，并保留资源上限和证据不足拒判：

- `pipe_twin/remote_capture.py` 为每个保存的双目对记录左右目灰度分位数、暗像素比例、饱和比例和原因码；`capture.json` 增加 `usable_pair_count`、`unusable_pair_count`、`quality_status`。原图仍全部保存，`status=COMPLETED` 只表示抓拍流程完成，不能替代可用曝光判断；`capture_agent` 会把质量状态带到 job 状态。
- `pipe_twin/local_surface.py` 在颜色来源额度未用完时回流深度候选，已尝试的连通域不会重复尝试；每只眼总候选仍不超过 96，预算耗尽仍标为 `TRUNCATED`。审计增加按 `role/source/color` 的拒绝原因计数，以及“缺失深度或遮挡信号”（明确不是物理遮挡证明）。
- `pipe_twin/remote_fetch.py` 在并行下载前顺序建立目录树，并按已存在路径逐段做链接/越界校验，避免 Windows 并发 `mkdir`/`resolve` 造成的偶发 `evidence path leaves output directory`。新增嵌套目录 `workers=4` 回归。
- `pipe_twin/office_client.py` 和 `run_gui.bat` 现在把 GUI、8770 控制服务、8765 只读证据服务合并为一个办公室客户端；自动检测 Tailscale 地址、验证已运行的 8765 是否指向同一目录，并尝试建立仅限 Tailnet 的 Windows 防火墙规则。默认采用 Tailscale 内网免令牌；可用 `--require-token` 显式恢复 Bearer token。防火墙若返回 `NEEDS_ADMINISTRATOR`，需要管理员运行一次。

验证：免令牌/令牌兼容针对性测试 38 项、3 个子测试通过；仓库全量为 **481 passed、1 skipped、247 subtests passed**（`python -m pytest tests -q`）。这些代码变更尚未在办公室相机上重新执行，现场报告仍是 `a7dc224` 基线；下一次现场回传必须注明新的代码 SHA 和新的 `quality_status` / 审计字段。

### 下一次现场执行要求

1. 现场电脑切到 `main` 并拉取包含 `943096b` 的最新提交；先运行 `python -m pytest tests -q`，再关闭预览窗口，只保留主 GUI 和文件服务。
2. 采集后同时检查 `capture.json.status`、`capture.json.quality_status`、`usable_pair_count` 和每一对的 `image_health.reason_codes`。短曝光即使流程返回 `COMPLETED`，只要质量为 `PARTIAL/UNUSABLE` 就不能送入管径结论；保留 AUTO 恢复前后的过渡帧。
3. 用 `workers=4` 从现场文件服务下载完整证据并保存 `*.fetch.json`；若失败，保留失败报告和原始错误，不用串行续传覆盖问题。回传新清单 SHA、下载报告和本次代码 SHA。
4. 对三根实物补 S1–S3 物理标签、D1–D3 卡尺直径、Q1–Q3 相机基准距离和分辨率；让红/蓝/白管在左右目都有未被棋盘截断的连续侧面。未具备这些条件时，结果继续保持 `UNKNOWN`，不能用模型的 51/41/26 mm 代替实测。

## 接手入口

- **本机已执行主线复测（2026-10-02）**：[运行 `field-20261002-113318-fef83aa4` 的测试与实拍报告](field_reports/field-20261002-113318-fef83aa4/report.md)。基于 `a7dc224`，475 项本地测试通过；完成多档快门/重开检查和 28 对 AUTO 新采集。手动短快门变暗已复现，六次新照片回放均仍截断、0 根有效圆柱，独立参照未执行。后续开发请先读此报告，旧弱光汇总保留作对照。

- 仓库：`oceanye/py_pipe_install`
- 开发交接分支：`feature/model-aware-pipe-distances-20261001`（历史分支，代码已合并）
- [PR #10](https://github.com/oceanye/py_pipe_install/pull/10)：已合并，合并提交 `ffee4e5b902a3bd6b7ad21bc1e08be4dfa2385f3`。
- 已测试的历史实现提交：`94532b6ab419592bb333fe3214f7265ca6bebd7a`（候选来源调度基线）。
- 当前主线：`main`，已含现场证据提交 `4ffd2ac` 和本轮实现提交 `943096b`；另含 DXF 支持、距离定义、现场数量约束、候选来源回流、抓拍质量审计和并行取证修复。
- [本轮测试报告](test_reports/20261002-model-distance/report.md)、[检查清单](test_reports/20261002-model-distance/checks.json)、[模型与距离定义](doc/管道模型尺寸与距离定义.md)。后续 handoff 提交只增加文档和数值证据，不增加应用功能。
- [下一轮现场点位、独立参照和远程控制交接](doc/HANDOFF-REMOTE-FIELD-MEASUREMENT.md)。

远程在干净 checkout 中接续：

```text
git fetch origin
git switch main
git pull --ff-only origin main
python -m pip install -r requirements.txt
python -m pip install pytest==9.1.1
python -m pytest tests -q
```

已有同名本地分支时切换后使用 `git pull --ff-only`。不要覆盖未提交修改。后续代码通过 PR 交付；回传完整提交 SHA、测试结果、变更说明和剩余问题。本轮代码已进入 `main`，现场实测仍须按新 handoff 回传证据后再验收。

## 已确认的现场事实

| 项目 | 当前事实及边界 |
| --- | --- |
| 实物数量 | 用户确认总共 3 根，彼此平行，没有相对倾斜；不等于相机与管轴垂直 |
| 模型目录 | STL / DXF 各 12 个候选管位；具体哪 3 个对应实物尚未确定 |
| 颜色及设计外径 | 用户回忆蓝 51、红 41、白 26 mm；DXF 实体颜色与直径确实一致 |
| 各色数量 | 按蓝/红/白准备显示；只确认总数为 3，不将各色各一根当作已验证的分类约束 |
| 模型身份 | STL 与 DXF 管号顺序不同；按截面 XY 位置和直径关联，不能按同号关联 |
| 棋盘 | 8×6 内角点，即 9×7 方格；打印名义 19 mm，用户实量 17 mm，当前用 17 mm |
| 图像 | 每目 1920×1080；左右来自一个并排 UVC 帧，不等于另行验证过硬件同步 |
| 快门 | 保留多档可调；目标 1/200 s，不支持时用户接受更快的约 1/256 s；弱光原片记录为 AUTO |
| 标定 | 沿用既有 15 姿态标定 `field-chess-fb5dc0f71ee5`；本次 6 对固定姿态照片只做检查，没有新求解完整标定 |
| 当前距离 | 约 1.29 m 涉及棋盘/匹配点深度，不能认定为蓝管中心线、顶点或最近表面的实测值 |

用户目标是先跑通流程，不要求当前就达到高精度。仍需区分模型尺寸、实测量、模型预测和原始匹配点统计；不能把模型直径复制成测量结果。

## 已完成、可接续的实现

`43c0b15` 已推送到交接分支：

- `pipe_twin/pipe_geometry.py`：版本 `pipe-section-distances-v1`，在同一指定截面输出中心线、最小表面 Z、最近表面空间距离和轮廓切点。原点为左目矫正相机光心。
- `elevation_auto.py`：模型与观测按同一观测轴向位置比较，避免 CAD 中点及 DXF 预览管长改变测距；实测/模型预测分列，健康失败保持未知。
- `elevation_dataset.py`：新增可选 `present_pipe_count`；检出数量超过已知实物数量时拒绝本次配准，不强选 3 根，也不补造观测。
- `elevation_depth.py`：手工区域的深度中位数明确标成诊断量，没有轴线/半径时不输出实测截面。
- `elevation_gui.py`：增加实际管数和“尺寸与测距”，保存后可恢复。模型候选数与实物数分开显示。
- `local_surface.py`、`metrology.py`：附带明确的距离定义；圆柱仍由双目表面拟合。

代码已做本地全量及实际模型回放，但**现场量测目标尚未达到**。本地测试与 CI 的精确版本和结果分别记录在测试报告中。

## 本轮远程开发回传（2026-10-02）

针对 P0 候选预算问题，提交 `94532b6ab419592bb333fe3214f7265ca6bebd7a` 已推送到本分支并进入 PR #10：

- 每只眼仍保留 `maximum_component_candidates=96` 的硬上限；默认 72 个额度给深度几何候选，剩余额度按颜色来源独立分配，避免深度碎片耗尽后颜色来源整段跳过。
- 深度和颜色组件按可解释的细长/管状优先级排序，背景大块不会仅凭面积抢占全部候选额度；审计中保留各来源的分配、尝试和跳过数量。
- 新增确定性测试：在真实管道周围加入大量背景深度碎片，仍能让真实细长组件进入双目圆柱拟合，同时总尝试数不超过 96；搜索截断仍标记 `TRUNCATED`，上层继续拒绝输出现场直径/距离结论。

验证结果：本地 `467 passed, 1 skipped, 247 subtests passed`；GitHub Windows CI 同结果，Ubuntu CI 为 `451 passed, 17 skipped, 247 subtests passed`。本轮没有重新取得现场原图，因此尚未把弱光回放结论改写为现场量测成功。

下一步仍按原验收边界执行：先用白天现场图复放并记录每管候选来源、有效点和拟合原因，再做持续开流、关闭重开、多档快门和独立距离参照；只有搜索完整、标定/深度健康且有独立参照时，才评估毫米级直径和距离。

## 远程优先处理的任务

### P0：候选预算与候选来源被跳过

两种模型回放得到相同的明确失败链，详见 [field_replay.json](test_reports/20261002-model-distance/field_replay.json)：

- 左右目各尝试 96 个深度候选，额度耗尽，另外分别跳过 306 / 304 个深度组件；
- 三种颜色的候选来源在左右目均被整段跳过，不能把结果解释为“颜色检测过但没发现管道”；
- 192 次已尝试拟合中，164 次残差过大、26 次局部轴向支持不足、2 次截面直径不稳定；另有 8 条预算事件，共 200 条拒绝记录；
- `pair_healthy=true`，但 `surface_audit.truncated=true`、`analysis_healthy=false`，0 根有效圆柱，12 个候选均 UNKNOWN。

重点检查 `local_surface.extract_local_pipes` 中深度优先与共享预算、候选排序和背景碎片处理。可研究保留各来源额度、几何预筛或合理去重，具体方案由远程实现。不能仅放大预算、放宽圆柱残差或删除截断拒判来制造成功。

验收：增加“背景深度碎片很多、仍有真实细长管道”的确定性测试，证明有效候选有机会进入拟合；同时维持资源上限、来源统计、歧义/截断拒判。回传各来源尝试数、跳过数、拟合原因、耗时和内存变化，再由本机复放真实数据。检测数量增加本身不算精度提升。

### P1：弱光数据诊断与可靠性

既有 6 帧比较中，256 px 搜索加 NLM+CLAHE 使蓝管区域有效深度均值从 2.11% 增至 3.16%，棋盘参考错配仍约 34.82%。亮度中位数左右仅 11/255、21/255，角点矫正后垂直残差 P95 约 2.10–2.23 px。

算法工作先用保存数据，记录每帧和每管结果，不只汇报全图平均覆盖率。白天数据由本机采集验收；固定单姿态不能冒充多姿态标定。禁止用插值填洞或 CAD 合成深度充当实测。

### P1：三根实物与十二个模型管位

总数约束已存在；具体模型身份仍不明确，也未确认实物间距与模型布局相同。远程评估匹配歧义、平行轴和模型子集对应的交互与报告，不硬编码某三个管号。无法唯一配准时保留观测编号及诊断，不把颜色直接当实例身份。沿管轴的位置、端点、整管最近点仍不可由局部截面推定。

### 后续设备回归由本机执行

白天检查持续开流、关闭重开、多档快门请求与驱动实值、原始/弱光模式同照片对比。远程可交付设备测试脚本与日志格式；本轮没有重新打开相机，不能据此宣称此前“看到画面后黑屏/无法打开”的现场症状已验收消除。

## 数据与复现

Git 内已有实际模型、合成测试和本轮数值报告。`test_reports/20261002-model-distance/inputs.json` 列出模型、6 对原片、标定及回放输入的相对路径与 SHA-256，并明确 `GIT` / `LOCAL_ONLY`。

**本轮原图与完整现场包仍只在本机，未随 handoff 上传；不能从远程 checkout 直接重放这些本地路径。** 这轮只推送文档、数值统计和哈希。远程先跑仓库合成回归，数据相关改动交回本机复放；需要原片时另行安排数据交付。

本机路径（相对仓库根目录）：

- `outputs/quick-calibration-20261001/capture-070621/`：6 对原片、角点和曝光记录。
- `outputs/quick-calibration-20261001/camera_calibration.json`：实际沿用的标定。
- `outputs/low-light-20261001/`：6 帧比较、HTML 和旧手工区域包。
- `outputs/model-distance-20261002/verification.json`：STL / DXF 包路径、模型映射与 GUI 核验；`open_model_check.bat` 可打开实际三管设置的 DXF 包。

获得同一现场包后，回放命令：

```text
python -m pipe_twin analyze-stereo --manifest <现场包>/manifest.json --output <新目录>/report.json
python scripts/compare_stereo_processing.py <现场包>/manifest.json --output <另一个新目录> --num-disparities 256 --board 8 6
```

`field_reports/latest.json` 现在指向 2026-10-02 的 `field-20261002-113318-fef83aa4` 采集包；该包仍是 `a7dc224` 基线，不代表 `943096b` 已完成现场复测。旧 [远程采集 handoff](doc/HANDOFF-REMOTE-CAPTURE.md) 只作采集流程参考，当前实物数量与 17 mm 格长以本文件为准。

## 工作区与交付约定

历史 PR #10 未包含本机独立文件服务改动。2026-10-02 下午用户授权后，剩余未跟踪的 `pipe_twin/remote_files.py`、`tests/test_remote_files.py` 已在 `9318e3f` 纳入版本管理，并适配主线 office-client；旧 GUI 自动启动接线不再恢复。已有服务进程未被强行替换，本机配置和工具/打印文件仍保留本机。详见最新黑图复核报告。

远程每轮交付：实现 SHA / PR、复现命令、测试结果、逐项指标变化、未完成与失败原因。新功能通过单元测试后由本机做实际数据和设备验收；测试通过、流程跑通、量测精度通过三者分别汇报。下一次本机侧拉取远程成果后，优先测试和汇报，开发问题回写 handoff。
