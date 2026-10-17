# 2026-10-14 暂用标定替换记录

用户明确要求“基本可以用，替换现有标定”。已在本机备份两个原文件，再同步替换活动标定及工作台配置。GUI 和基础立面评估允许本次明确确认的暂用参数；没有将失败验证改写成通过。

| 项目 | 当前值 |
| --- | --- |
| 活动标定 ID | `field-chess-provisional-20261014-5c62d5b4c50a` |
| 旧标定 ID | `field-chess-fb5dc0f71ee5` |
| 每目分辨率、实测格长 | 1920×1080、17 mm |
| 候选求解基线 | 55.498616 mm，尚未独立核验 |
| 训练 RMS / 留出跨目 RMS | 1.604048 / 1.995919 px |
| 留出垂直 P95 | 0.489995 px |
| 质量状态 | `validated=false`、`holdout.status=FAILED`、`converged=false` |
| 暂用范围 | 双目抓拍、立面相对管径与管间尺寸评估 |

GUI 显示“暂用标定（用户已确认）；精度未验收”，报告保留完整质量审计及暂用记录。记录须绑定当前标定 ID，包含明确同意、用途、带时区的时间和非空原因；缺失记录不能自动放行。参数结构、校正配方、合成示例等检查保持不变；标定向导的自动验收和完整 CAD 绝对测量要求保持不变。暂用参数仍可能产生系统性尺寸偏差，不能将候选尺寸当作已经验收的实物真值。

## 文件与恢复

- 活动标定：`outputs/measurement_workbench/calibration_current.json`。
- 活动配置：`outputs/measurement_workbench/workbench_profile.json`，校正配方与标定 ID 一致，旧位姿调整已重置。
- 原文件备份：`outputs/calibration-20261014/activate-provisional/backup-20261014-102955/`，两个文件均按原始字节保留。
- 替换回执：同目录上级 `activation.json`，包括来源结果、备份和新文件的 SHA-256；仓库副本见 [activation-evidence.json](activation-evidence.json)。

需回退时先关闭 GUI，将备份目录的两个同名文件一起还原到 `outputs/measurement_workbench/`，再重启 GUI；不能只回退其中一个，否则标定和校正配方会不一致。此次未改动历史现场包或原始照片。

工作台自动恢复最近现场时优先采用当前活动标定，保留模型和观察方向，清空旧标定下的矫正照片、框选、连续帧和评估结果。用户主动打开历史现场包时仍完整保留历史标定。新测量步骤：重开基础立面评估 → **③ 双目抓拍 → 连续采集 20 对并使用 → 左图框选 → ④ 保存并评估**。

## 验证

- 相关测试 **118 passed、8 subtests passed（50.07 秒）**，覆盖暂用准入、无效记录拒绝、结构错误及合成标定拒绝、现场包质量证据保存、GUI 自动恢复及显式历史打开、抓拍和立面评估回归。
- 实际 Tk 加载本机配置及原历史现场：新标定和校正器正常，12 管模型恢复，旧照片清空，`validated=false` 保留，原活动文件及历史指针没有被检查过程改动。
- 使用两张已保存的实际棋盘原图构建新校正图：两目均为 1920×1080，像素均值分别 122.553、117.606，输出正常。此项只确认校正链路，不证明毫米精度。
- 本轮未打开物理相机、未采集新的管道照片、未重新测量现场管径和间距。

测试命令：

```powershell
.venv/Scripts/python.exe -m pytest tests/test_calibration_provisional.py tests/test_elevation_gui.py tests/test_elevation_dataset.py tests/test_calibration_quality.py tests/test_capture_gui.py tests/test_elevation_auto.py -q
```

完整日志、JUnit、实际 Tk 检查、替换脚本及新旧标定包保留于本机 `outputs/calibration-20261014/activate-provisional/`。原始影像不提交 Git。
