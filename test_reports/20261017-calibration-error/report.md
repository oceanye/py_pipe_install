# 2026-10-17 标定错误修复与基线核对

## 异常原因与修复

实际日志 `2026-10-14T07:05:51.349694+00:00`（本地 15:05:51）记录 `elevation_gui_run_failed`，错误为 `validated rectified calibration is required`。保存的现场包是 `outputs/elevation_workbench/captures/elevation-20261014-150423-df1c14c6/manifest.json`，使用已获用户确认的暂用标定，含 20 对连续照片。

问题在后端 `local_surface._validate_arrays`：GUI 和现场包允许明确确认的暂用记录，解析后的 `StereoCalibration` 却丢失该状态，识别器仍只允许 `validated=true`。新增完整 manifest → 实际 SGBM → 识别 → 报告回归，在修复前重现相同异常。

现在解析器从标定 ID 绑定的确认记录派生 `relative_elevation_provisional`，默认 false，不能用 manifest 中同名布尔字段替代确认记录。局部相对几何识别认可该状态，分区裁剪保留它；`validated=false`、留出失败证据、校正与参数结构检查及完整 CAD 量测要求保持原样。

回归 **109 passed、20 subtests passed（95.44 秒）**，包括真实报告导出、五种后端分区识别、缺失确认及错误几何拒绝。`compileall`、`git diff --check` 通过。

```powershell
.venv/Scripts/python.exe -m pytest tests/test_calibration_provisional.py tests/test_local_surface.py tests/test_elevation_zones.py tests/test_elevation_auto.py tests/test_metrology.py tests/test_stereo_analyzer.py -q
```

## 真实现场重评

用原 manifest 和 20 对保存照片重新执行 `analyze_stereo_capture`，101.11 秒完成，报告及叠图写入原现场目录的 `report.json` / `evidence/`。没有重新打开物理相机，没有修改照片、manifest 或活动标定。报告仍保留 `validated=false` 及全部质量审计。

实际结果：**0 安装 / 0 未安装 / 12 不确定、0 合格管段**。20 帧中 17 帧通过逐帧检查，但整段因 `BURST_REFERENCE_OR_SCENE_UNSTABLE` 判为 INVALID；候选因有效深度不足被拒绝。不能把接受 17 帧表述为融合成功，亦不能据此断言相机或管道实际移动，图像质量也会影响稳定性检查。本次修复解决异常中断，没有得到可靠的实测管径。

GUI 重启后加载修复代码。可以打开该已保存现场重新评估，或直接查看其 `report.json`，无需重新拍摄。

## 用户核对的标定内容

| 项目 | 当前记录 |
| --- | --- |
| 实测棋盘格长 | 17 mm |
| 棋盘角点与方格 | 8×6 内角点，对应 9×7 方格 |
| 输入 | 31 对照片，25 对训练，6 对固定留出 |
| 每目原始/校正分辨率 | 1920×1080 |
| 原始左目 fx、fy、cx、cy | 4195.778、4138.493、966.630、578.978 px |
| 原始右目 fx、fy、cx、cy | 4175.613、4133.193、872.069、516.229 px |
| 校正后两目 fx=fy、cx、cy | 4411.373、1013.138、528.806 px |
| 用户确认的实物基线 | **60 mm** |
| 当前活动候选求解基线 | **55.498616 mm**，与实物不一致 |
| 留出跨目 RMS / 垂直 P95 | 1.995919 / 0.489995 px |
| 状态 | 暂用，未收敛，精度未验收 |

上述焦距是像素单位的等效焦距，不是镜头标签上的毫米焦距。活动标定包的 `rectification_recipe.K1/K2/D1/D2/R1/R2/P1/P2` 保存原始内参、非零畸变及校正矩阵；`stereo_calibration.left_camera/right_camera` 描述已经校正的图像，因此其中畸变系数为零。标定没有直接规定相机到现场管道的距离或相机到 CAD 的绝对位姿。

可人工核对：横跨 8 格应为 136 mm，纵跨 6 格应为 102 mm；同一格长下不应有横纵打印缩放差异。已确认 60 mm 双目基线应作为独立几何约束。随后需要用独立测得的管径和两根管轴心距检查尺度，不能拿输入的 CAD 公称尺寸充当算法实测结果。

## 60 mm 基线一致性对照

活动候选较已知基线少 4.501384 mm，即 7.502%。在焦距及视差不变时，基线由 55.498616 改为 60 mm 会让重建尺度乘以 1.081108。这只是敏感性计算，不是测量修正结论。

只读对照将候选 T 归一到 60 mm，保持 K/D/R 与 17 mm 棋盘不变，重新执行 stereoRectify 并检查原留出 6 组（5、9、13、18、22、26）：跨目 RMS **1.995919 → 11.983490 px**。这说明其余参数也需要联合重新估计；此对照没有进行联合优化，没有替换活动标定，不能用于宣称 60 mm 标定完成。之后的联合解算须保持留出组独立，不能以留出误差选参数。

数值摘要与输入哈希见 [evidence.json](evidence.json)。本机完整回执、pytest XML、脚本和复核日志在 `outputs/calibration-error-20261017/`；活动标定 SHA-256 未改变，仍为 `b52e5f311b9dd750cfd022afe603e62780d933799d78fc32c64baeb124e1e83c`。原始影像不提交 Git。
