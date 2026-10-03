# 最新主线办公室 AUTO/DirectShow 与 AI 回放

执行时间：2026-10-03（Asia/Shanghai）。本机与办公室运行代码均为 `11322bb8dcb4d9fddbbdf7ece9904b4cd7dc294b`。本轮验证的是 GitHub 最新 `main` 的现场部署、原生曝光控制、远程抓拍和 DXF AI 安全门禁。

## 部署状态

办公室 `GET /v1/health` 返回 `READY`，并明确回传：

- `capture_contract=remote-auto-native-dshow-v1`；
- `code_sha=11322bb8dcb4d9fddbbdf7ece9904b4cd7dc294b`；
- 工作树干净；
- OpenCV `4.13.0`；DirectShow `700` 可用，Media Foundation `1400` 不可用；
- 相机索引 `0`、并排左右、每目 `1920×1080`。

因此这次不是只更新了本机代码，办公室控制服务已经重启并运行最新提交。

## 远程 8 对 AUTO 抓拍

任务：`remote-20261003-092718-6ddaf287`。请求为 8 对、0.5 秒间隔、20 秒预热、AUTO、8×6 棋盘检测。

| 指标 | 结果 |
| --- | --- |
| 流程状态 | `COMPLETED` |
| 图像质量 | `USABLE` |
| 可用对数 | `8/8` |
| 文件取回 | `PASS`，18 个文件，42,203,263 字节 |
| 左右尺寸 | 均为 `1920×1080` |
| 棋盘检测 | 8 对左右均为 48 点，SB detector |
| 左目统计范围 | P50 `16–17`，P95 `154`，暗像素比例 `0.168–0.171` |
| 右目统计范围 | P50 `33–34`，P95 `165–166`，暗像素比例 `0.047–0.055` |

本轮 AUTO 使用原生 `IAMCameraControl`：设备曝光范围 `-11..-2`、步长 `1`、默认 `-6`、能力标志 `3`，AUTO 回读为原生值 `-2`、标志 `1`。这证明此前的黑帧主要问题已经通过“默认 AUTO + 原生模式/范围控制 + AUTO 失败重开”得到修复；驱动仍没有提供 1 秒或 2 秒真实曝光证据。

## DXF/AI 回放

把这 8 对原图接入当前现场 DXF manifest 并运行最新 `analyze-stereo`：

- 8/8 对的 `pair_healthy=true`；
- `surface_audit.status=TRUNCATED`，每对局部圆柱观测为 `0`；
- 注册状态 `INSUFFICIENT_OBSERVATIONS`，主要拒绝为 `LOCAL_AXIS_SUPPORT_TOO_SHORT` 和 `CYLINDER_FIT_RESIDUAL_TOO_LARGE`；
- 最终 `INSTALLED=0`、`NOT_INSTALLED=0`、`UNKNOWN=12`。

AI 没有误把图像可用当作管道可识别，也没有把模型的 51/41/26 mm 当成实测。原因是本批照片仍是固定棋盘标定画面，棋盘遮挡了目标管道，不能证明三根现场管的身份、直径或距离。8 对稳定重复的棋盘图也不是 8 个独立标定姿态。

## 验证结论

本轮可以确认：

1. GitHub 最新代码已部署到办公室；
2. AUTO 默认和原生 DirectShow 曝光范围读取已生效；
3. 办公室相机可以稳定输出 8/8 可用双目图；
4. 远程取证和哈希校验通过；
5. AI 安全拒判门禁有效。

本轮尚不能确认三管识别、直径测量或距离精度。下一批应关闭棋盘检测或把棋盘移出视野，保证蓝、红、白三根管在左右目有连续无遮挡侧面；取得新的 `USABLE` 批次后再进入 DXF 匹配，并补齐 D1–D3 卡尺直径和 Q1–Q3 距离参照。

本机检查：`python -m pip check` 通过；`python -m pytest tests -q` 为 **517 passed、1 skipped、256 subtests passed**。
