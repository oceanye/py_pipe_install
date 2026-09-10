# 2026-09 真实双目硬件调试素材

本目录是现场双目相机素材的只读快照，用于离线复现相机接入、标定和管件识别问题。运行程序产生的新数据仍写入 `outputs/`；确认需要长期保留后，再建立新的归档快照提交到 GitHub。

## 硬件与标定输入

- 双目基线：60 mm
- 标称视场角：80°（厂家规格未说明是水平、垂直还是对角视场角）
- 镜头标称焦距：3.0 mm
- 棋盘格内角点：9 × 7
- 棋盘格打印文件名标称方格：20 mm；本次实际打印测量并录入：19 mm
- 目标精度：毫米级

`hardware_profile.json` 是调试时工作台配置的快照。这些厂家规格用于约束和审计，不能单独替代相机内参；计算仍以棋盘格角点和实测方格尺寸为准。

## 目录

- `camera_probe/`：同一 UVC 双目设备在多种请求分辨率下的原始拼接帧及棋盘格预览。
- `measurement_workbench/camera_intake/`：5 次原始左右目拆分结果，每个拍摄目录包含 `left.png` 和 `right.png`。
- `measurement_workbench/field_sessions/`：4 个完整现场数据包，包含左右图、STL、清单和已有诊断证据。
- `measurement_workbench/calibration_diagnostics/`：标定角点、参数和失败审计的 NPZ；`calibration_autosave.npz` 是当前可恢复的 13 组角点。
- `targets/`：现场使用的二维码与棋盘格打印素材。
- `SHA256SUMS.csv`：所有归档文件的相对路径、字节数和 SHA-256，用于检查传输完整性。

## 已知复用关系

现场数据包里的左右图来自前一刻的相机接入拍摄，因此部分文件内容重复：

| 相机接入拍摄 | 对应现场数据包 |
| --- | --- |
| `camera-20260907-210844-6d970019` | `field-20260907-210854-5e80e125` |
| `camera-20260907-214620-1958141e` | `field-20260907-214705-8a677f63` |
| `camera-20260907-215147-c4b63ad0` | `field-20260907-215201-4dc21b59` |
| `camera-20260907-215644-2ce1c2c7` | `field-20260907-215717-c94fa4a4` |

Git 按内容保存相同文件，重复路径不会生成多份不同的图像对象。

## 标定照片说明

2026-09-10 的标定拍摄发生在原图归档功能加入之前，当时程序只保存检测后的左右角点，未保存原始左右画面。因此现有 13 组可以通过 `calibration_autosave.npz` 继续求解和补拍，但无法重新运行角点检测。这个限制无法从 NPZ 反向恢复原图。

从提交 `d99201e` 起，向导每接受一组画面都会立即保存无损原图到 `outputs/measurement_workbench/calibration_captures/<会话>/left|right/`，并在 `session.json` 中记录状态。删除或自动剔除只改变状态，不删除图片。

复算本次诊断可运行：

```powershell
.\.venv\Scripts\python.exe scripts\replay_calibration_diagnostic.py test_data\real_hardware_archive\2026-09-field-test\measurement_workbench\calibration_diagnostics\calibration_20260910_170037_316212.npz --expected-baseline-mm 60
```
