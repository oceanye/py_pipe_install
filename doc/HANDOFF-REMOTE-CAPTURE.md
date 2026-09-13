# 远程主机交接任务：固定棋盘格无人值守采集与结果回传

执行仓库：`https://github.com/oceanye/py_pipe_install`，主分支：`main`。

请在连接了双目相机的现场主机上执行本任务。用户不在现场，但可以部署并启动程序；棋盘格已经放在相机视野内。本次目标是取得可回放的真实照片、采集元数据、诊断结果和完整日志，并上传到同一 GitHub 仓库，供另一台分析电脑的助手下载检查。

**交付完成的条件是结果已上传且下载校验通过，不是程序已经启动，也不是文件仅留在现场硬盘。无论采集成功、部分失败还是完全失败，都必须回传报告和日志。**

## 1. 授权范围与工作边界

- 允许部署依赖、运行有限时长采集、检查日志、修复必要的软件问题，并将相关代码、结果索引和证据包上传到上述仓库。
- 不需要用户逐张抓拍或移动棋盘格。不要等待现场人员进行多姿态操作。
- 沿用现场现有相机和镜头设置；不要删除现场配置、标定文件、历史照片或日志，不使用 `git reset --hard` / `git clean` 清空现场。
- 本次先完成原始证据采集与健康检查，不以固定单姿态棋盘替代完整标定，不修改质量阈值使结果强行通过，不把未知参数标为 `validated=true`。
- 基础立面模式不要求事先人工测量 CAD 相机位姿。只有相机标定、深度和自动 STL 配准通过后才做状态评估；未通过时报告原因并保留“不确定”。
- 本任务无需发邮件、聊天消息或改变仓库可见性。上传范围是这次采集所需成果；不包含 `.venv`、`.git`、`.env`、密码、访问令牌或私钥。

## 2. 先确认现场配置，禁止混用新资料示例

现场历史证据见：

- `doc/FIELD-20260911-实测素材与标定复算评估.md`
- `doc/现场问题记录.md`

| 项目 | 现场历史配置（优先核实并沿用） | 新资料中的示例配置（不能直接套用） |
| --- | --- | --- |
| 相机编号 | 现场记录曾为 `0`，以本机现有配置为准 | 旧示例脚本为 `1` |
| 并排流尺寸 | `3840×1080` | `1280×480` |
| 每目尺寸 | `1920×1080` | `640×480` |
| 棋盘内角点 | `8×6`，即 `9×7` 个方格 | `11×7`，即 `12×8` 个方格 |
| 方格边长 | 历史记录 `19 mm`，未确认当前板相同前只作为历史值 | 资料未确认 |
| 镜头中心距 | 历史记录 `60 mm` | 示例 T 范数约 `120.33`，单位未确认 |

先读现场相机配置并拍一对原图核实。配置变更必须另开一次尝试目录，记录请求值与实际返回值。不要缩放照片来伪装成目标分辨率。若当前板规格不明，可以先完成不带棋盘检测的原图采集，再离线判断；检测失败不应阻止保存原图。

## 3. 部署和运行前检查

1. 在现场项目目录记录 `git status --short --branch`、当前提交和分支。工作区干净时使用 `git fetch origin`、`git switch main`、`git pull --ff-only origin main`；有未提交改动时先保留并检查，避免覆盖现场配置。
2. 记录执行代码的完整 SHA；若执行期间改了代码，保存 diff、测试结果和最终提交，明确每次尝试用了哪一版。
3. 使用 Python 3.12 环境优先部署（仓库有 Windows/Python 3.12 的依赖锁定清单）；已有可用环境可复用。检查并保存 `python --version`、`python -m pip check`、`python -m pip freeze`、OpenCV 版本和系统平台。首次安装按 `requirements.txt`，同版本 Windows 可按 `requirements-lock-windows-py312.txt`。不要假定已有虚拟环境就是可复现的依赖说明。
4. 执行 `python -m pipe_twin capture-stereo --help`。本版本采集入口为 `capture-stereo`，会保存原始左右 PNG 和 `capture.json`。若不存在，说明未拉到包含本任务的代码，先解决版本问题。
5. 确认没有其他预览程序占用设备。若被占用，记录并处理本项目已知的占用进程；不要盲目终止未知服务。
6. 确认磁盘空间，检查当前 Git/GitHub CLI 鉴权是否能够向 `oceanye/py_pipe_install` 推送并上传 Release 附件。不要将凭据内容写入日志。若不能上传，仍应保存本地成果并明确报告上传失败，不能宣称任务完成。

## 4. 无人值守采集

每次任务生成唯一的 `RUN_ID`，例如 `field-20260913-210000-abcdefgh`。同一任务中的设备/分辨率重试分别放在 `attempt-01`、`attempt-02`，不得覆盖上次目录。

以下为 PowerShell 示例。先在仓库根目录执行，`$fieldRunRoot` 等变量只在本任务中使用。示例配置来自现场历史记录，执行前按第 2 节核实。

```powershell
$fieldRunId = 'field-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + ([guid]::NewGuid().ToString('N').Substring(0, 8))
$fieldCodeSha = (git rev-parse HEAD).Trim()
$fieldRunRoot = Join-Path (Get-Location) ('outputs/' + $fieldRunId)
New-Item -ItemType Directory -Force -Path $fieldRunRoot | Out-Null
$env:PIPE_TWIN_LOG_DIR = Join-Path $fieldRunRoot 'logs'

python -u -m pipe_twin capture-stereo `
  --output-dir (Join-Path $fieldRunRoot 'attempt-01') `
  --left-index 0 `
  --layout side_by_side_left_right `
  --eye-width 1920 --eye-height 1080 `
  --count 30 --interval-s 2 --duration-s 120 `
  --detect-chessboard --board-columns 8 --board-rows 6 `
  1> (Join-Path $fieldRunRoot 'capture.stdout.txt') `
  2> (Join-Path $fieldRunRoot 'capture.stderr.txt')
$fieldCaptureExitCode = $LASTEXITCODE
$fieldCaptureExitCode | Set-Content (Join-Path $fieldRunRoot 'capture.exitcode.txt')
```

运行要求：

- 优先取得 30 对完整左右原图；保存 PNG，保留原始像素、原始尺寸和左右配对。
- 棋盘不动也照常采集，不要使用标定向导的“姿态去重接受数量”代替本任务的拍照数量。
- `--duration-s` 只限制循环中的采集时段，不能中断卡死的相机驱动。现场执行器必须另外设置约 5 分钟的进程级超时；若超时，先保存现有输出和日志，再终止本次采集子进程，记录 `TIMEOUT`，随后继续打包上传。
- 需要后台运行时使用隐藏窗口、重定向 stdout/stderr，并记录进程退出码；不要在用户离线时弹出必须点击的 GUI。
- 首次打不开相机时只做有界重试：先复核本机已有设备编号及占用情况，再尝试已确认的设备。每次重试保留参数、退出码和日志，不无限扫设备、不无限重拍。
- `capture.json` 记录每对图像的 SHA-256、左右角色、拍摄时间、时间来源、同步差、设备/布局与实际分辨率。并排流来自同一个 UVC 帧不等于已验证传感器硬件同步，报告中保持这一区别。
- 拍照失败、棋盘检测失败、没有有效标定均不允许伪造照片或用历史图冒充当前图。原始照片必须保留，失败信息也属于交付成果。

## 5. 自动评估范围

无有效标定时必须做：开流/读帧统计、实际尺寸、图像可解码性、左右齐全性、SHA-256 校验、棋盘检测结果和错误统计。固定视野的多次图像相同可能只是场景静止，不得仅凭图像哈希相同认定传感器提供了过期帧。

如果现场有与当前设备、分辨率和镜头设置匹配的标定文件，则复制实际使用的标定到证据包，记录来源和 SHA-256，进一步检查角点矫正后的垂直视差（中位数、RMS、P95）及固定平面深度稳定性。精度/尺寸结论必须说明参考值的来源。

当前 `capture-stereo` 的可选棋盘检测不等于完成上述全部几何评估。需要离线脚本时补齐并提交脚本；做不了的项目明确写 `NOT_RUN` 及原因，不写“通过”。缺少有效标定时也应立即上传已拍照片，供分析端后续复算。

固定单姿态棋盘只能验证该视野的局部表现。完整 K/D/R/T 标定、全视场精度、可靠毫米尺度和完整现场安装状态验收不能由这次采集单独证明。

## 6. 必须交付的文件

运行目录（之后整体打包到 Release）：

```text
outputs/<RUN_ID>/
  attempt-01/
    capture.json
    pair_0001_left.png
    pair_0001_right.png
    ...全部成功落盘的原始左右图...
  attempt-02/                   # 有重试时保留
  calibration/                 # 实际使用的标定文件，如有
  logs/                        # 本次运行相关完整日志及轮转分卷
  capture.stdout.txt
  capture.stderr.txt
  capture.exitcode.txt
  environment.json             # OS/Python/OpenCV/依赖/代码 SHA
  pip-freeze.txt
  commands.txt                 # 完整执行参数，不含凭据
  code.diff                    # 执行时有未提交代码改动则必须提供
  checks.json                  # 逐项 PASS/FAIL/NOT_RUN、指标和原因
  report.md                    # 中文结论和失败原因
  evidence_manifest.json       # 包内每个文件相对路径、大小、SHA-256（不含自身）
```

相机完全打不开时，`capture.json` 可能只能记录失败、没有照片；若崩溃早于清单生成，补充失败报告解释缺失原因。禁止为了凑齐目录而生成空白照片。检测角点图、深度图和其他派生成果如果生成，也一并上传并标记为派生文件。

`report.md` 至少包含：任务 ID、开始/结束时间和时区、执行代码 SHA、现场配置的来源、请求与实际尺寸、成功照片对数、棋盘检测率和检测方法、几何指标/未执行原因、是否可进入下一步分析，以及完整错误摘要。失败时保留 stderr 和堆栈，不只给一张报错截图。

## 7. 上传到 GitHub，确保分析端能取得

采用“主线报告索引 + Release 完整附件”：

1. 在 `main` 提交小体积报告到 `field_reports/<RUN_ID>/`：`report.md`、`checks.json`、`environment.json`、`upload.json`。其中 `upload.json` 记录运行 ID、执行代码 SHA、采集/分析状态、Release 标签和 URL、每个附件的文件名/大小/SHA-256、上传和回读校验状态。附件中保留完整日志，主线报告可以另外摘录错误摘要。
2. 完整运行目录（所有原始照片、成果、日志和清单）压缩成 `<RUN_ID>.zip`，计算 ZIP 的 SHA-256，上传到同仓库名为 `field-capture-<RUN_ID>` 的 GitHub Release。大包按完整图像对分卷，每卷分别登记哈希和包含的文件；不要把大量原始图像直接塞入 Git 历史。
3. 保留项目 `.gitignore` 规则。`outputs/`、`logs/` 被忽略不表示可以不上传；它们必须包含在 Release 证据包内。不要 `git add .` 混入现场私有文件。
4. 使用既有授权上传，Release 标记为预发布且 `--latest=false`，避免成为软件“最新正式版本”。使用唯一标签，不覆盖或删除历史成果。Release 指向实际采集代码提交；报告索引可在后续提交中更新。
5. 上传后，从 GitHub 下载到新的校验目录，计算 SHA-256 并与本地包一致；解压检查左右对数、文件清单和逐图哈希。仅看到上传命令退出码 0 不能替代回读校验。
6. 更新 `field_reports/latest.json` 指向本次报告与 Release，并提交推送 `main`。如果本次失败也应指向本次实际失败报告，而不是继续让分析端误读上一次成功结果。保留旧运行目录和报告。
7. 上传认证、网络或分支冲突导致失败时保留本地产物，明确标为 `UPLOAD_FAILED`；不得写“用户已能看到”。有限重试后说明具体阻塞点。

GitHub CLI 上传/回读示意（请先生成 report、ZIP 和哈希；路径变量按本次实际值设置）：

```powershell
$fieldReleaseTag = 'field-capture-' + $fieldRunId
$fieldArchivePath = Join-Path (Get-Location) ('outputs/' + $fieldRunId + '.zip')
Compress-Archive -LiteralPath $fieldRunRoot -DestinationPath $fieldArchivePath
$fieldArchiveHash = (Get-FileHash -LiteralPath $fieldArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()

gh release create $fieldReleaseTag $fieldArchivePath `
  --repo oceanye/py_pipe_install --target $fieldCodeSha `
  --title ('现场采集 ' + $fieldRunId) --prerelease --latest=false `
  --notes-file (Join-Path $fieldRunRoot 'report.md')
if ($LASTEXITCODE -ne 0) { throw 'Release 上传失败：保留本地成果并记录 UPLOAD_FAILED' }

$fieldVerifyDir = Join-Path (Get-Location) ('outputs/verify-' + $fieldRunId)
New-Item -ItemType Directory -Path $fieldVerifyDir | Out-Null
gh release download $fieldReleaseTag --repo oceanye/py_pipe_install `
  --pattern ([System.IO.Path]::GetFileName($fieldArchivePath)) --dir $fieldVerifyDir
if ($LASTEXITCODE -ne 0) { throw 'Release 回读失败：不得标记上传验收通过' }
$fieldDownloadedHash = (Get-FileHash -LiteralPath (Join-Path $fieldVerifyDir ([System.IO.Path]::GetFileName($fieldArchivePath))) -Algorithm SHA256).Hash.ToLowerInvariant()
if ($fieldDownloadedHash -ne $fieldArchiveHash) { throw '下载附件与原包 SHA-256 不一致' }
```

正文不要硬编码尚不存在的成功 URL 或哈希。以实际远程返回值填写索引；在提交前执行 `git diff --check`，只暂存本次报告和审查过的代码。普通推送被拒时解决具体冲突，不强推或自动绕过分支规则。

## 8. 最终回复与验收

最终回复用户必须给出：

- `RUN_ID`、采集/分析/上传三个分别独立的状态；
- 实际保存的左右照片对数和核心结论；
- `main` 上本次报告的可访问链接和提交 SHA；
- Release 页面链接、证据包下载链接和 SHA-256；
- 下载回读校验结果；
- 若失败，完整日志的链接、具体原因以及能否仅靠软件继续处理。

分析端的助手随后可执行 `git pull --ff-only` 读取 `field_reports/latest.json`，再通过 `gh release download` 获取完整证据包并核验哈希。用户只需在原对话说“拉取最新现场结果”。**GitHub 上传不会自动推送进原对话，也不代表原对话助手已查看数据；完成上传并提供可下载链接，才能在后续对话中继续评估。**

验收清单：

- [ ] 未要求现场人员移动棋盘或逐张抓拍。
- [ ] 当前现场配置与新资料示例分开记录，没有套错分辨率和内角点。
- [ ] 成功图像左右成对、原像素保存、元数据与 SHA-256 一致；失败也留证。
- [ ] 实际做了什么、没做什么、哪些通过/失败均有明确报告。
- [ ] 所有本次成果、原图、完整日志均已包含在上传附件中。
- [ ] `field_reports/latest.json` 和报告已推送到 `main`。
- [ ] Release 附件已上传并经下载、哈希和包内文件检查。
- [ ] 最终回复包含真实报告/附件链接，没有将本地保存冒充远程可见。
