# 采集快门：不慢于 1/200 秒

棋盘格标定、GUI 双目抓拍和无人值守采集在每次打开相机后，统一请求手动曝光，快门目标不超过 5 ms（1/200 秒）。先设置分辨率和图像格式，再设置快门，避免切换视频模式重置曝光。两台独立相机分别设置；并排双目流使用其设备提供的共同曝光控制。

Windows 默认使用 DirectShow。该接口采用整数 log2(秒) 档位，因此请求 -8，即 1/256 秒（3.90625 ms）；不会把 -7 的 1/128 秒误当成 1/200 秒。Linux V4L2 使用 100 微秒单位，请求值 50，即 5 ms。程序关闭自动曝光，不通过延长快门来补偿亮度；画面过暗时应补光或通过相机工具调整增益。

标定向导和相机预览显示驱动回读的曝光。驱动拒绝控制、仍报告慢快门、读取异常或使用未支持的接口时，界面明确显示“未确认”，预览仍可打开，需在相机工具中手动核实设置。日志记录请求值、驱动返回值及状态。采集 manifest 保存目标时间、回读时间和状态；标定原始照片档案的相机信息也保存这些设置。

`DRIVER_REPORTED` 表示手动命令获驱动接受且回读曝光不超过 5 ms，不代表独立测量了传感器实际曝光，也不保证画面绝不模糊；仍需保持棋盘稳定并通过原有清晰度门禁。DirectShow 的 OpenCV 4.13 实现不提供自动曝光状态读取，因此不把其 `-1` 返回值当作实际模式。软件实现 revision 由本文件所在 Git 提交记录。

接口依据：[微软 CameraControlProperty](https://learn.microsoft.com/en-us/windows/win32/api/strmif/ne-strmif-cameracontrolproperty)、[OpenCV 4.13 DirectShow 实现](https://github.com/opencv/opencv/blob/4.13.0/modules/videoio/src/cap_dshow.cpp)、[V4L2 曝光控制单位](https://docs.kernel.org/userspace-api/media/v4l/ext-ctrls-camera.html)。
