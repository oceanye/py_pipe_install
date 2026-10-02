# 远程抓拍控制通道（开发版）

推荐直接双击仓库根目录的 `run_gui.bat`：它会在同一个进程中启动 GUI、8770 控制服务和 8765 只读文件服务，默认免令牌。命令行等价入口是 `python -m pipe_twin office-client`。

## 设计

控制与文件传输分开：

- 8770：绑定办公室的 Tailscale 地址，默认提供 Tailscale 内网免令牌的 JSON 控制接口；也可显式启用 Bearer token；
- 8765：继续提供 D:\pipe_twin_runs 的 GET 下载；
- 相机只在办公室本机打开，家中电脑只提交有限参数；
- 每个任务使用唯一目录，写入左右 PNG、capture.json、job.json、evidence_manifest.json 和 SHA-256；
- 相机访问使用每个设备索引的 OS 文件锁，GUI、命令行和 agent 不能同时占用同一设备。

agent 不接受远程输出路径、PowerShell、任意 shell 或任意 Python 代码。客户端只能指定拍摄数量、间隔、时长和可选棋盘格检测；相机索引、布局、分辨率和输出根目录由办公室端固定配置。

## 黑帧修复后的拍摄设置（2026-10-03）

普通 `remote-capture` / `capture-stereo` 拍照默认使用 AUTO，避免每次远程请求都强制切回 1/256 秒。保留 `--exposure-ms 5`、`--exposure-ms 33.333333` 等显式手动请求；`--auto-exposure` 可明确选回 AUTO。自动曝光不表示满足 1/200 秒防运动模糊要求，手动请求失败时也不会暗中切换 AUTO。

Windows DirectShow 通过原生 `IAMCameraControl.GetRange/Set/Get` 同时设置曝光值和模式标志，再核验模式读回。设备不支持的值不会写入，`UNCONFIRMED` 与驱动范围一起返回；例如本机驱动范围 -11～-2，1 秒/2 秒请求不会再被误认为已执行。API 依据：[Microsoft Set](https://learn.microsoft.com/en-us/windows/win32/api/strmif/nf-strmif-iamcameracontrol-set)、[GetRange](https://learn.microsoft.com/en-us/windows/win32/api/strmif/nf-strmif-iamcameracontrol-getrange)。驱动读回不是传感器时序的独立测量。

远程仍默认预热 20 秒、要求末尾连续 3 对通过原有亮度检查；AUTO 首次失败时最多重开同一设备一次，再完整预热，手动模式不自动重开。每次失败都保存末对原始诊断 PNG 和健康记录，放在 `startup_warmup_attempts` 中，不算进合格照片数；预热最终失败返回 `FAILED + UNUSABLE`。请给含重试的调用预留至少 120 秒超时。

`GET /v1/health` 及 `job.json.runtime` 会给出服务启动时的代码 SHA、工作树状态、OpenCV 后端能力和 `capture_contract=remote-auto-native-dshow-v1`。只有重启客户端才加载新代码；本机 headless OpenCV 未编入 MSMF 时，会明确提示后端不可用，不再误报设备占用。

无需曝光参数的验证命令：

```powershell
python -m pipe_twin remote-capture --agent-url http://<办公室-Tailscale-IP>:8770 --count 8 --interval-s 0.5 --detect-chessboard --board-columns 8 --board-rows 6 --wait --timeout-s 150
```

检查 `quality_status=USABLE`、8 对全部可用、连续照片有实际内容，再通过 `fetch-stereo --workers 4` 核验原图。这只验收远程拍照，不等于重新完成标定或量测精度验收。

## 办公室端首次试运行

### 一键启动

直接双击 `run_gui.bat`。客户端会自动：

1. 检测本机 Tailscale IPv4 地址；未检测到时只绑定 `127.0.0.1`，并在启动状态中标记 `LOCAL_ONLY`；
2. 启动仅限本机或 Tailscale 访问的 8770 抓拍接口，默认不要求 token；
3. 如带 `--require-token`，才创建或复用 `C:\Users\<用户>\.pipe-twin\capture-agent.token` 并启用 Bearer token；
4. 启动只读 8765 证据服务，或用临时文件确认后复用已经运行且指向 `D:\pipe_twin_runs` 的服务；
5. 尝试创建只允许 `100.64.0.0/10` Tailscale 网段访问 8770 的 Windows 防火墙规则；如果状态为 `NEEDS_ADMINISTRATOR`，请以管理员身份再启动一次客户端，或由管理员预先创建该规则；
6. 最后打开 GUI。关闭 GUI 时两个集成服务一起停止。

默认免令牌模式下不需要复制任何密钥；同一 Tailnet 内、能通过 ACL 和防火墙连接该端口的设备可以提交有限抓拍任务。程序同时检查免令牌监听地址和连接来源，只允许回环或 Tailscale IPv4。旧 token 文件即使存在，也不会自动开启认证。

从旧版升级时，先关闭旧客户端，在项目目录运行 `git pull --ff-only origin main`，再双击 `run_gui.bat`。已运行的进程不会因拉取代码而自动切换模式；更新后 `GET /v1/health` 应直接返回 200，且 `auth_required=false`。如果仍返回 401，请检查是否还运行着旧客户端或手动启动的 `serve-capture`。

若启用 `--require-token`，才需要把 token 文件通过安全方式复制到家中控制端。GUI 的相机预览不打开时，远程抓拍可直接使用；预览或标定窗口占用相机时，任务会返回 `CAMERA_BUSY`，不会强制抢占。

客户端也支持无界面服务模式：

```powershell
python -m pipe_twin office-client --no-gui
```

### 兼容独立文件服务

已有的 `python -m pipe_twin.remote_files serve` 入口现已纳入版本管理，适合需要在关闭 GUI 后继续下载证据的办公室配置。它只提供文件下载，不打开相机；新版 GUI 不会自动再启动它。`office-client` 会校验文件根目录后复用现有服务，并且停止客户端时不关闭该独立服务。

该入口仍要求明确的 `--bind`、`--port`、`--directory` 和 `--control-ip`；只接受 Tailscale 或回环监听地址，GET/HEAD 仅允许配置的控制端与本机监听地址。它复用主线的只读、路径越界和链接检查。Windows 监听使用独占端口，防止另一个服务再次绑定同一端口；客户端校验目录后的探测文件清理，会短暂等待 HTTP 文件句柄释放。

这些兼容改动需要在下次重启相应服务后生效；拉取代码不会替换内存中的旧进程。配置保留在本机 `outputs/measurement_workbench/remote_files.json`，不随代码提交。

如需恢复额外认证：

```powershell
python -m pipe_twin office-client --require-token
```

### 手动独立服务（保留令牌认证的兼容方式）

通常只需使用上面的 `office-client --no-gui` 排查启动问题。以下 `serve-capture` 是保留的独立服务入口，仍要求 token。它沿用现场已经验证过的索引 0、并排左右、每目 1920×1080 配置；实际参数改变时要显式修改。

~~~powershell
$token = & .\.venv\Scripts\python.exe -c "from pipe_twin.capture_agent import generate_token; print(generate_token())"
$tokenPath = Join-Path $env:USERPROFILE '.pipe-twin\capture-agent.token'
New-Item -ItemType Directory -Force (Split-Path -Parent $tokenPath) | Out-Null
Set-Content -LiteralPath $tokenPath -Value $token -Encoding ascii -NoNewline
$env:PIPE_TWIN_LOG_DIR = 'D:\pipe_twin_runs\service_logs'

.\.venv\Scripts\python.exe -m pipe_twin serve-capture `
  --bind 100.103.31.118 `
  --port 8770 `
  --token-file $tokenPath `
  --output-root D:\pipe_twin_runs `
  --file-base-url http://100.103.31.118:8765 `
  --left-index 0 `
  --layout side_by_side_left_right `
  --eye-width 1920 `
  --eye-height 1080 `
  --default-count 1 `
  --max-count 30 `
  --max-duration-s 300
~~~

第一次先以前台方式运行，确认日志和相机释放正常，再注册为当前用户的计划任务或 Windows 服务。不要把 token 提交 GitHub，也不要把 8770 暴露到公网。
token 文件特意放在 `D:\pipe_twin_runs` 之外，因为 8765 会把该目录作为只读下载根目录提供。
服务事件写入 `D:\pipe_twin_runs\service_logs\pipe_twin.log.jsonl`；可用 `Get-Content D:\pipe_twin_runs\service_logs\pipe_twin.log.jsonl -Tail 20` 检查启动、请求、抓拍和失败原因。

办公室 Windows 防火墙只允许家中 Tailscale 地址访问 8770。下面的地址只是当前控制端地址，若 Tailscale 地址变化必须重新核对：

~~~powershell
New-NetFirewallRule `
  -DisplayName 'Pipe Twin capture agent from home' `
  -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8770 `
  -RemoteAddress 100.92.137.53 -Profile Any
~~~

## 家中端测试

默认免令牌模式下，家中端直接提交一次小任务：

~~~powershell
python -m pipe_twin remote-capture `
  --agent-url http://100.103.31.118:8770 `
  --count 3 `
  --interval-s 1 `
  --exposure-ms 33.333333 `
  --warmup-s 20 `
  --wait
~~~

`--exposure-ms 33.333333` 是 1/30 秒请求；办公室 Windows DirectShow 可能在 `capture.json` 中回报 31.25 ms（1/32 秒）。手动范围为 0.1–2000 ms，`1000`/`2000` 可用于 1 秒/2 秒静止场景诊断；是否真的接受要看 `capture.json.camera_exposure` 的状态和回读。远程任务默认会在相机打开后预热 20 秒，并在末尾要求连续 3 对可用帧；`--warmup-s` 可显式调整。需要由驱动自行控制时可改用 `--auto-exposure`。远程请求会把曝光目标、预热统计和驱动回读保存在证据中，不能只看 job 的 `COMPLETED` 判断亮度合格。

只有办公室显式启用 `--require-token` 或使用独立 `serve-capture` 时，才给远程命令加 `--token-file <本机令牌文件>`。

返回的 JSON 中会有 job_id 和 run_url。任务完成后，通过现有 8765 下载并校验：

~~~powershell
python -m pipe_twin fetch-stereo `
  --url http://100.103.31.118:8765/<job_id>/ `
  --output-dir outputs/remote_fetch/<job_id>
~~~

另一个远程任务正在运行时，提交接口返回 409 `CAMERA_BUSY`。如果 GUI 或本地命令行已经持有相机锁，本次请求会先接受并随后以 `FAILED`、`error.code=CAMERA_BUSY` 结束；此时先关闭相机预览后再重试，服务不会强行抢占设备。

## 接口边界

- GET /v1/health：查看服务和当前任务；
- POST /v1/captures：提交有限抓拍任务；
- GET /v1/captures/<job_id>：查询任务；
- 删除任务、删除文件和任意命令执行均被禁用。

当前实现基于 Python 标准库 HTTP 服务，适合 Tailscale 内的受控试运行。远程实测前先检查健康接口，再拍 1 对照片，最后从家中执行完整下载校验；服务连通和照片成功传输不能代替现场量测精度验收。
