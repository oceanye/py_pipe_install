# 单一管段评估区域验证（2026-10-28）

用户明确不需要多个分区并行。此前 Z01/Z02 是旧现场选区配置，主界面画框追加 Z03 导致新旧范围混用；本次将当前 GUI 改为一个区域，重画替换，不再保留停用草稿或分区管理入口。

## 操作与兼容

- 一个绿框可以包含多根管道，与橙色汇聚范围取交集；紫色地面区独立。
- 合法重画替换当前全部选区并清除旧报告；取消、过小或越界框保留原区域与报告。橙色范围、地面区和右目搜索扩展参数保留。
- 打开历史文件时，若只有一个有效区域就沿用并忽略停用草稿；若有多个有效区域则恢复全图，提示重新画一个。旧文件保持原样，新保存另建现场。
- 冲突提示旁改为“重画当前区域”；设置页仅显示当前范围和“清除当前区域（恢复全图）”。主图不再显示递增的 Z 编号。
- 旧多分区数据格式、离线分析和相关回归保留；当前 GUI 保存/评估拒绝未经迁移的多个区域。不修改识别算法、标定或质量门限。

## 验证

相关回归：`pytest tests/test_elevation_gui.py tests/test_gui_entry.py tests/test_elevation_zones.py tests/test_model_zones.py -q --disable-warnings`，**77 passed、3 subtests passed**。覆盖真实 Tk 拖框替换、取消/无效输入、反复重画、旧场景迁移、源文件不变、保存重开、空交集拦截、全图恢复及历史分析兼容。

全量回归：`pytest tests -q --disable-warnings`，**863 passed、1 skipped、267 subtests passed**，321.68 秒。日志：`outputs/single-region-20261028/full-tests.txt`。`compileall -q pipe_twin` 和 `git diff --check` 均通过。

生产回放执行：

```powershell
.venv/Scripts/python.exe test_reports/20261028-single-region/verify_scene.py --publish-resume
```

来源为上一轮恢复的 `elevation-20261027-175505-a2a82c73`。本次使用原有 **2026-10-27 17:46:42.296 +08:00** 的同步末帧，没有新抓拍，也没有补造其他 19 帧的元数据。

新现场：`outputs/single-region-20261028/captures/elevation-20261028-025336-3820630f/manifest.json`，已设为默认恢复。旧默认指针备份在 `outputs/single-region-20261028/previous_last_scene.json`。

加载时只保留原 Z03，矩形 `[1112,295,439,461]`；当前 manifest 内不再有 Z01/Z02。实际 GUI 保存、评估、重开通过。**完整断面结果、地面结果、安装计数和就位结果与源报告相同**（忽略重新保存生成的 capture ID）。仍为两个条件断面、12 个模型位置全部 `UNKNOWN`；本次界面整理不代表毫米精度或安装验收通过。

真实 Tk 窗口测量：1024×720 主窗口及模型设置页控件均未超出窗口边界，当前设置页仅有清除区域按钮；未作截图视觉验收。源 manifest、源 report 和活动标定 SHA-256 不变，照片哈希、拍摄时间与时间来源不变。详见 [数字证据](field-validation.json) 与 [回放脚本](verify_scene.py)。

上一轮 `20261027-region-conflict/recover_scene.py` 记录的是提交 `6735231` 的追加/停用行为，仅用于该版本的历史复盘；当前验证应使用本目录脚本。
