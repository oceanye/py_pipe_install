# 远程 1/30 秒曝光交接

更新时间：2026-10-02（Asia/Shanghai）。实现提交：`0bf1cec`，基于最新现场检查分支 `2caff4e`。

## 已完成

- `remote-capture` 支持 `--exposure-ms`，办公室请求契约允许 `exposure_ms`，并把参数传到 `StereoCameraSession`。
- 远程任务默认先连续预热 20 秒，再要求末尾连续 3 对可用双目帧；可用 `--warmup-s` 覆盖，传 0 仅适合诊断。
- GUI 快门下拉框增加“1/30 秒（Windows 1/32）”。
- 每次 `capture.json` 保存 `requested_exposure_ms`、`exposure_policy` 和每只眼的驱动回读；默认行为仍为 5 ms，显式 `null` 表示 AUTO。
- 参数限制为 0.1–250 ms，拒绝布尔值、非有限值和越界值。
- DirectShow 的 1/30 请求值为 33.333333 ms，驱动整数 log2 档位通常回报 31.25 ms（1/32 秒）；报告必须使用回读值，不得写成精确 1/30。

验证结果：`494 passed, 1 skipped, 254 subtests passed`；`pip check` 通过。针对曝光、请求校验、CLI、采集记录和 GUI 解析的新增测试已覆盖。

## 办公室部署与复测

办公室当前已打开的客户端不会热加载此提交。现场先关闭旧客户端，在干净工作树拉取包含 `0bf1cec` 的 `main`，再重新启动客户端；保持相机预览和标定窗口关闭，只保留主 GUI、8770 和 8765。

家中端请求 8 对 1/30 曝光：

```powershell
python -m pipe_twin remote-capture `
  --agent-url http://100.103.31.118:8770 `
  --count 8 --interval-s 0.5 `
  --exposure-ms 33.333333 `
  --warmup-s 20 `
  --wait --poll-s 1 --timeout-s 120
```

任务完成后，从 8765 下载 `run_url` 对应目录，并核对：

1. `capture.json.requested_exposure_ms` 约为 33.333333；
2. 每一对左右 `provenance.capture_exposure_ms` 的实际回读，Windows 通常为 31.25；
3. 8 对的 `image_health`、`quality_status` 和完整原图哈希；
4. 如果仍有近黑帧，保留原图并标记 `PARTIAL/UNUSABLE`，不要进入管径、距离或 3/12 身份识别。

`capture.json.startup_warmup` 会记录预热读帧数、稳定连续帧数、丢弃原因和实际耗时。预热期间出现第一张亮帧不能提前结束预热。

1/30 是更慢快门，可能增加运动模糊；即使亮度恢复，也必须先通过连续帧质量和棋盘/管面门槛。
