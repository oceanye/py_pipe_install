# 固定双目 Kimi 断面评估与本地 / 远端同步

当前已有两条可运行的固定端路径：A 使用本地双目与图像算法；B 使用同一组参考双目照片和 CAD 横断面调用 Kimi。两者分别产生证据和候选。B 可以在 A 未运行或三维求解不足时独立运行，不修改严格安装 / 就位验收状态。

## GUI 操作

1. 导入 STL/DXF 和当前标定，完成双目抓拍，或手动“打开现场”。
2. 在左图指定管长方向和橙色汇聚范围；需要时框选一个绿色评估区域。在模型窗口应用相机观察侧，并勾选左右同步拍摄确认。
3. 首次使用打开 **Kimi 设置**。默认模型 `kimi-k3.1-pro`、中国区官方接口；填写 key 并保存。Windows 将 key 用当前用户的 DPAPI 加密保存，重启后可继续使用。key 留空表示保留已有凭据。
4. 在 **⑤** 选择 **保存并评估** 执行 A，或 **AI 断面评估** 执行 B。两者无需固定先后；输入完全一致时，AI 复用现场并保留已绑定的 A 报告，方便对照。
5. 到 **AI 断面候选** 页查看模型管号、固定断面点位、候选状态和原因。选行可联动模型管号。详情列出模型、生成时间和照片时间；左图紫红虚线是 AI 采用的图像控制截线。
6. 点击该页下方 **查看 AI 断面图**，在已有模型窗口右侧查看各设计管位的 AI 判断。也可用右图上方 **本地结果 / AI 候选** 切换；圆图与列表双向联动。实线为图像支持，长虚线为有歧义，点线为未解，点划线为建议重采，每根管另有状态文字；紫色外环仅表示选中。

AI 断面图按模型管号关联既有 CAD 圆图，圆心和直径表示设计尺寸，不是 AI 实测尺寸，也不表示安装验收通过。未应用的方向预览不改变 AI 图的判定方向；真正应用方向或修改照片、范围后，旧 AI 状态失效并清除。未运行、被拒绝和输入失效分别显示，不沿用旧的肯定候选。

AI 页隐藏 A 的黄色 IMG 候选框，避免将本地图像候选当作 AI 输出。`SUPPORTED_VISIBLE` 表示图像支持该模型断面候选；`AMBIGUOUS` 表示有多种对应；未观察到不等于未安装。候选不等同于精密尺寸合格或正式验收通过。

**取消 AI** 会释放界面，迟到响应不会保存。网络请求可能已经由提供方处理，不自动重试。同一现场任务失败 / 取消保留已有 A 结果；变更照片、方向、范围或其他输入后旧结果失效。

结果独立保存在现场旁的 `ai_section_result.json`。AI 不覆盖原始 `report.json`、照片或标定。手动重新打开现场会核对来源后恢复有效 A/B 结果；恢复旧 A 报告时也执行当前无效序列身份门禁。默认启动仍清除旧选区，不自动恢复旧 AI 判断。

## 输入和坐标

- B 输入为**最新参考双目对**的左右图，最长边不超过 1920 px；有评估 ROI 时额外提供左图局部图；另有带点位编号的中性 CAD 横断面图。它不代表全部连拍帧融合，也没有重新拍照。
- 真实横断面垂直于 CAD 共同管轴，相机观察侧在该平面内定义近侧。本项目例子的共同轴是 CAD Z，相机来自 +Y；不能用相机朝向当横断面法向，否则会压扁层位。
- 图像控制截面取橙色汇聚带与绿色 ROI 有效轴向交集的中点垂线。未设绿色 ROI 时使用全图，空交集会要求调整范围。截线以左图像素表示，**不是已标定的毫米空间站位**。
- CAD 图上方表示朝相机的一侧，不代表照片上方。鼠标轴线不自动确定 CAD 轴正反方向。镜像、重复布局、遮挡或无法区分的层位保留歧义。
- STL 自动配色不用于物理颜色匹配。显式采购直径 / 颜色先验可供参考，但不能冒充独立测得直径。
- 不发送 A 匹配管号、历史安装答案或严格状态。响应需回填本次输入和模型点位哈希；引用必须属于本次图片，肯定候选必须引用实拍，不能仅引用设计图。

## 本地与远端应用

两端使用同一仓库和接口。GitHub 推送只更新代码，不会重启运行中的 GUI，也不会携带现场数据和密钥。先提交或妥善保留自己的工作区修改；无本地分叉时：

```powershell
git pull --ff-only
.venv/Scripts/python.exe -m pip install -r requirements-lock-windows-py312.txt
.venv/Scripts/python.exe -m pipe_twin ai-status
.venv/Scripts/python.exe -m pipe_twin gui
```

已有 GUI 需关闭后重启。Windows 设置位于 `%LOCALAPPDATA%/PipeTwin/ai-settings.json`，密钥由当前 Windows 用户加密，不应复制到其他机器作为通用密钥文件。其他机器通过 GUI 自行配置，或设置 `MOONSHOT_API_KEY` 环境变量；该变量优先于本机加密凭据。密钥不进入 Git、场景包或命令行参数。

无桌面的远端也可执行：

```bash
python -m pip install -r requirements.txt
# 由运行环境配置 MOONSHOT_API_KEY
python -m pipe_twin ai-status
python -m pipe_twin ai-section-evaluate --manifest /path/to/scene/manifest.json --prepare-only
python -m pipe_twin ai-section-evaluate --manifest /path/to/scene/manifest.json
```

`--prepare-only` 仅核验和汇总输入，不调用模型或保存结果。正式调用会产生提供方用量；同一现场目录建议只由一个任务写 AI 结果。远端运行需要完整场景包（manifest、模型、图片），仅拉代码不能获得本机历史照片。

适配器支持 Kimi 官方中国区及国际区 HTTPS 接口，需与 key 所属区域一致。模型和参数依据 [官方模型列表](https://platform.kimi.com/docs/models) 与 [Chat Completions API](https://platform.kimi.com/docs/api/chat)；K3 使用 `max_completion_tokens`、`reasoning_effort` 和 JSON 输出。

## 完成边界

本轮接通固定双目 B 路径及 GUI / CLI 闭环。真实请求成功不等于管号能唯一识别；当前验收照片的 Kimi 结论仍有歧义。相机运动跨帧配准、手机第三路径和毫米级安装验收仍需各自实现或验证。
