# 标定九宫格引导验证

实现用户要求的九区定位、稳定计时及采集完成变绿。左目显示编号网格，上方同序九宫格显示中文区域名、进度或已采集组数；右目保留完整图与检出状态，采集仍要求两眼都合格。

## 采集规则

- 稳定时间下限由 0.8 秒调整为真实 6 秒，至少 3 次检测；原有中位数 / P95 / 最大角点位移阈值不变。累计时间只来自检测，不用空转计时器或旧帧自动完成进度。
- 两眼缺失、模糊、检测器切换、区域切换、过大运动或检测间隔超过 5 秒重新开始；自动路径补齐左右帧相同 / 同步检查，与手动路径一致。
- 只有成功进入保留列表的双目照片才点绿，原始 PNG 保存失败不计入。删除最后一组后该格恢复待采集；恢复历史组重新计算覆盖。手动模式达到稳定条件后仍需点击抓拍。
- 绿色只表示位置覆盖。同区可以改变距离和倾斜继续采集，静止的重复姿态不会自动重复增加组数。仍保留 20～30 组建议及原有训练、留出验证和多样性检查。
- 网格只画在缩放后的预览副本，不改变检测或归档原始像素。参数默认折叠，顶部显示当前关键值，可展开修改；停止、抓拍、保存和区域提示均在主视图可见。

## 自动回归

```powershell
$env:PYTHONUTF8='1'
.venv/Scripts/python.exe -m pytest tests/test_calibration_guidance.py tests/test_calibration_capture_stability.py tests/test_chessboard_calibration.py tests/test_calibration_wizard.py tests/test_calibration_quality.py tests/test_camera_controls.py -q
```

**110 passed，4 subtests passed，50.36 秒**。最后调整预览适配宽度、已采集网格颜色与快门文案后，重复相关状态回归：**24 passed，0.35 秒**。`git diff --check` 通过。

新增覆盖：只识别或计时走完不点绿、区域切换重计、检测停滞不推进、缺眼/模糊/检测器不一致/相同帧/不同步重计、归档失败不点绿、手动模式等待按钮、覆盖随保留照片变化及预览不修改原图。

## 真实窗口和实机

`verify_gui.py` 用 10 月 30 日已归档的第 1 对照片，真实检测角点，在独立输出目录回放生产采集路径。时间由测试控制，不能把这次回放当成新现场姿态或标定。检查计时、采集、删除、最小 960×700 窗口和展开参数的状态与控件边界；人工查看截图确认预览完整。证据：`gui-validation.json`，本机截图 `outputs/calibration-guidance-20261030/`。旧原图、会话、活动标定哈希不变。

`verify_live.py` 实际打开相机约 28 秒，仅预览，关闭自动采集。**9 次检测均同时检出左右棋盘**，中心区域稳定时间为 0、2.140、4.734、7.328 秒，首次在 **7.328 秒**满足要求，随后持续稳定；无相机错误。证据：`live-validation.json`。现场照片组新增 **0**，未触发标定解算或替换活动标定。

复现命令：

```powershell
.venv/Scripts/python.exe test_reports/20261030-calibration-guidance/verify_gui.py
# 下列命令会实际打开相机，但不会采集标定照片或替换标定：
.venv/Scripts/python.exe test_reports/20261030-calibration-guidance/verify_live.py
```

当前标定仍是此前暂用版本，SHA-256 `b52e5f311b9dd750cfd022afe603e62780d933799d78fc32c64baeb124e1e83c`。本次完成采集交互改进，尚未完成新一轮多姿态标定。
