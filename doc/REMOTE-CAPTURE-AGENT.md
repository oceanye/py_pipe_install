# 远程抓拍控制通道（开发版）

办公室电脑关闭期间只做本地开发；本文描述后续开机后的部署和实测流程。当前办公室的 8765 仍然是只读文件服务，不接收控制请求。

## 设计

控制与文件传输分开：

- 8770：绑定办公室的 Tailscale 地址，提供带 Bearer token 的 JSON 控制接口；
- 8765：继续提供 D:\pipe_twin_runs 的 GET 下载；
- 相机只在办公室本机打开，家中电脑只提交有限参数；
- 每个任务使用唯一目录，写入左右 PNG、capture.json、job.json、evidence_manifest.json 和 SHA-256；
- 相机访问使用每个设备索引的 OS 文件锁，GUI、命令行和 agent 不能同时占用同一设备。

agent 不接受远程输出路径、PowerShell、任意 shell 或任意 Python 代码。客户端只能指定拍摄数量、间隔、时长和可选棋盘格检测；相机索引、布局、分辨率和输出根目录由办公室端固定配置。

## 办公室端首次试运行

在办公室项目根目录打开 PowerShell。以下示例沿用现场已经验证过的索引 0、并排左右、每目 1920×1080 配置；实际参数改变时要显式修改。

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

将同一个 token 安全地保存到家中电脑，例如 C:\Users\<user>\.pipe-twin\capture-agent.token，然后提交一次小任务：

~~~powershell
python -m pipe_twin remote-capture `
  --agent-url http://100.103.31.118:8770 `
  --token-file C:\Users\<user>\.pipe-twin\capture-agent.token `
  --count 3 `
  --interval-s 1 `
  --wait
~~~

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

当前实现基于 Python 标准库 HTTP 服务，适合 Tailscale 内的受控试运行；长期无人值守部署还应加 Windows 服务生命周期、密钥轮换、ACL 自动检查和更严格的服务监控。远程实测前必须先在办公室电脑上运行健康检查，再拍 1 对照片，最后从家中执行完整下载校验。
