# 双目参考脚本

此目录只纳入可审查的参考 Python 脚本：

- `BM_find_distance（测距）/camera_config.py`
- `BM_find_distance（测距）/main.py`
- `Depth（深度图）/main.py`
- `two_vision_calibration（标定）/calibration_code/capture.py`

原始棋盘照片、相机照片、示例图片、`__pycache__` 和打包 zip 不进入 Git。已复核 zip 中的 4 个 Python 脚本与此目录逐字节一致；现有 30 对左右照片混有房间画面和手机屏幕上的棋盘，使用 11×7 的 SB 检测器可以找到部分左右棋盘对，但整批数据没有通过当前的姿态、尺度、RMS 和极线质量门禁。`R-C.jpg` 是能被 SB 检测器识别的单张 11×7 内角点示例。脚本保留用于追溯早期双目算法；当前项目的受控采集入口是仓库根目录的 `python -m pipe_twin capture-stereo`，远程受控采集入口见 [远程抓拍控制通道](../REMOTE-CAPTURE-AGENT.md)。这些旧脚本没有纳入当前自动化测试，也不应被当作现场标定结果。
