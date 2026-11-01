# 2026-11-01 Kimi 断面路径接入与本地验收

实现提交：`7bd319f73238ca541a1b1dee72ee60293cb0713d`，已推送 GitHub。**固定端 B 路径已从真实图像请求接通到 GUI 接纳、保存和重开；本次照片的模型管号仍有歧义，不能将功能接通写成安装识别成功。**

本轮开始时，本地与 `origin/main` 都是 `b00c1702e99b984c7d16d8c761b1d00539891eaa`，没有分叉或待合并的生产改动。之前的问题是两端共同缺少真实 Kimi 接入，而非本机没有拉到另一套已经完成的实现。本轮补齐这套共同实现，并纠正本机 NumPy 与仓库锁文件不一致的问题。

## 已完成的调用链

1. 从当前现场包核验模型、标定、左右照片、观察侧、汇聚范围及评估区域，构造独立 AI 请求。提供最新参考左右图、必要的局部图及带固定点位的 CAD 横断面图；不提供 A 的匹配管号或历史安装答案。
2. 通过官方 Kimi 接口调用 `kimi-k3.1-pro`，解析结构化响应；处理超时、取消和错误，使用有界响应读取，避免把凭据或提供方原始错误体写入日志。请求不自动重试。
3. 校验输入哈希、模型点位哈希及图片引用，限制候选状态。缺少观察侧或控制截面、无法区分对应、证据不足时保留未解或歧义。
4. GUI 在检查任务代次、取消状态及来源文件后才接纳结果，原子写入独立 `ai_section_result.json`。手动重开现场再次核验；迟到响应或输入变化不能覆盖当前现场。
5. 同一输入可复用不可变现场，保留来源一致的 A 报告供并列查看。照片、方向、ROI、参数或采集来源变化时建立新现场，不把旧 A 报告附到新输入。AI 任务失败或取消保留同一现场的有效 A 结果。

新增模块为 `ai_inputs.py`、`ai_provider.py`、`ai_settings.py`、`ai_workflow.py`、`ai_gui.py`；CLI 提供 `ai-status` 和 `ai-section-evaluate`，其中 `--prepare-only` 不触网。GUI 的 **AI 断面评估 / 取消 AI** 位于⑤，与本地“保存并评估”并列；**Kimi 设置**配置模型、官方服务地址、超时和凭据。

当前本机凭据配置已验证，来源为 `windows_dpapi`。Windows 使用当前用户 DPAPI 加密保存，`MOONSHOT_API_KEY` 环境变量优先。密钥不进入 Git、场景包、报告或命令行参数；其他机器需要独立配置凭据。

## 本地验证结果

| 检查 | 结果 | 证据与边界 |
|---|---|---|
| 最终全量回归 | **1238 passed、1 skipped、272 subtests passed，454.04 s** | 本机独立锁定环境，OpenCV 2 线程；正式环境的 14 个包版本已与其对齐，未放宽质量阈值 |
| 独立复跑跳过项 | **1 passed，0.18 s** | Tk 懒加载检查；全量中因其他测试已导入 Tk 而跳过 |
| 环境一致性 | **14 个包版本一致** | 正式 `.venv` 与干净安装环境一致，NumPy 均为 **2.5.2**，`pip check` 通过 |
| 已保存真实 Kimi 结果的实窗验收 | **15/15 通过** | 1024×720 / 1280×860 布局、来源、旧 A 门禁、模型选择与输入失效 |
| 真实 GUI 点击到 Kimi、保存、重开 | **17/17 通过** | 一次实际提供方调用，复用同一 manifest，A 严格状态和原报告文件不变 |
| 两次真实 Kimi 评估 | 两次均返回 **12 个 AMBIGUOUS** | 0 个 `SUPPORTED_VISIBLE`；12 指设计模型点位，不是检出 12 根现场管道 |
| 其余 CI 命令的本机 smoke | 全部通过 | M0 **1727/1727 PASS**；simulate **9 instances**；field-demo **8 installed / 1 unknown**。这些是测试 fixture，不是现场安装结论 |

全量与环境证据：[validation-evidence.json](validation-evidence.json)、[pytest-output.txt](pytest-output.txt)、[requirements-validated.txt](requirements-validated.txt)。CLI smoke 见 [cli-smoke-evidence.json](cli-smoke-evidence.json)。

### 首轮全量失败与修复

首轮为 **4 failed、1230 passed、1 skipped、272 subtests passed**。其中 3 项是升级到锁定 NumPy 2.5.2 后，既有 A 路径的轴向覆盖计数成为 NumPy 标量，直接写 JSON 失败。`image_regions.filter_disparity` 现在显式转换为 Python `int`；计算和质量判定不变。

另外 1 项是 Windows 远程采集服务拒绝未授权 POST 时，客户端仍在发送请求体，连接提前关闭，导致预期的 HTTP 401 被 `WinError 10053` 掩盖。服务现在仅对一个有效 `Content-Length`、且不超过既有大小上限的未读请求体做有界排空，然后返回拒绝；不解析或执行该请求，不处理分块或长度含糊的输入，不增加重试或放宽超时。

独立探针在相同的 120 次本地回环未授权请求中，修复前出现 **85 次连接中止**，修复后 **120 次均为 HTTP 401**。相关回归及最终全量通过，详见 [auth-rejection-evidence.json](auth-rejection-evidence.json)。该问题与 Kimi key 本身无关。

## 真实照片与 GUI 验收

两次真实推理均使用 **2026-10-30 18:32:50.882 +08:00 的参考双目对**，没有重新打开相机或新拍摄。本次发送参考左右图和模型断面图，并非将整个 10 对序列融合后交给模型。照片仍可看出不同粗细及颜色，但重复模型布局、近远层遮挡和镜像对应不足以唯一确定模型管号，Kimi 两次都保留了歧义。

第一次由实际工作流调用、校验、保存并重开，证据：[live-evidence.json](live-evidence.json)。第二次经真实 Tk 的 `start_ai_evaluation()`、后台 worker、提供方、Tk 接纳和侧文件保存，再手动重开验证，证据：[gui-live-evidence.json](gui-live-evidence.json)。总实网推理请求 **2 次**，没有自动重试。

最终 GUI 验收现场：

```text
outputs/kimi-integration-20261101/gui-live-20261101-111026-285679/scene/manifest.json
```

原场景、照片、标定、原 A 报告及默认现场指针的受保护哈希均未改变。实网 GUI 运行只写隔离现场的 AI 侧文件和隔离输出目录的恢复指针；测试窗口在验收后均已关闭。真实窗口截图在 `outputs/kimi-integration-20261101/gui/` 与上述 `gui-live-.../` 目录。

验收后已用正式 `.venv/Scripts/pythonw.exe -m pipe_twin gui --manifest ...` 打开上述现场。生产窗口标题为“管道安装分析 · 双目断面与状态”，可见⑤的本地 / AI 评估按钮及“AI 断面候选”页，窗口正常响应且无标准错误输出。截图为 `outputs/kimi-integration-20261101/gui-applied.png`；这次打开不发新 AI 请求、不启动相机。[本机应用记录](local-application-evidence.json) 同时确认默认现场指针未改。

本轮还修复了上轮实窗失败：显式 `NOT_RUN` 空 AI 命名空间现在保持“未运行”，不再误显示“证据不足”。AI 页隐藏 A 的 IMG 候选框，详情区分**模型 ID、推理生成时间、实拍时间、仅参考双目对的输入范围**。AI 采用的紫红截线以左图像素表示，不被标成已解出的毫米空间站位。

历史 A 报告即使来源哈希正确，也可能来自旧算法。恢复时现已执行当前无效序列身份门禁：该归档的 **4/10、整组 INVALID** 不会恢复旧版四个 `LIKELY` 管号，而是显示证据不足、全部未观察到。原尺寸诊断和严格安装状态保留，历史文件本身不改写。

## 本地与远端如何直接应用

代码与操作接口已经统一；两端拉取同一提交并按锁文件安装后可以使用相同 GUI / CLI。推送 GitHub 不会自动重启正在运行的 GUI，也不会同步本机图像、标定包或凭据。另一台机器需要完整现场包及自己配置的 key；没有桌面时可运行 CLI。

具体拉取、安装、凭据配置、GUI 步骤和 CLI 命令见 [Kimi 工作流说明](../../doc/KIMI-SECTION-WORKFLOW.md)。本机正式环境已经纠正为 NumPy 2.5.2；不再沿用上轮 NumPy 2.2.6 的验收成绩。

旧主线 `b00c170` 的 [GitHub CI](https://github.com/oceanye/py_pipe_install/actions/runs/41264697249) 已核对原始日志：Windows 与 Ubuntu 都安装 NumPy 2.5.2，且都只有上述 3 项 `image_regions` 测试因 `int64` 写 JSON 失败。两边鉴权测试当时通过，因此本轮 Windows POST 连接时序缺陷是另一项本机发现的问题，不能列为旧 CI 根因。见 [远端原始失败证据](github-ci-before-evidence.json)。

**实现提交远端 CI 已全部通过。** `7bd319f` 的 [CI 运行](https://github.com/oceanye/py_pipe_install/actions/runs/41269041835) 为 **SUCCESS**：Windows **1238 passed、1 skipped、272 subtests passed（458.44 s）**；Ubuntu **1161 passed、78 skipped、267 subtests passed（256.99 s）**。两个平台的三个 CLI fixture 步骤也全部成功，旧版失败的 3 个 JSON 测试均已通过。[完整远端验证证据](github-ci-after-evidence.json) 保存提交号、任务、步骤和日志摘要。

Ubuntu 跳过项按日志节点及源代码守卫核对为 76 项 Tk 显示初始化相关、1 项 Windows DPAPI 专用和 1 项已加载 Tk 的进程隔离检查；CI 未使用 `-rs`，没有捕获具体 TclError 原文。桌面和真实 DPAPI 行为由本机 Windows 验证覆盖，不能把 Ubuntu 的跳过项目称为已执行通过。

本报告和 handoff 在实现提交之后单独归档，不改动已验证的 `pipe_twin/`、`tests/`、依赖或 CI 工作流；报告推送触发的后续 CI 与上述实现验证记录分开。

## 完成边界与后续工作

- 已完成固定端 B 的真实调用与保存闭环，A/B 独立运行和同场景对照；未实现自动认定安装，也没有将歧义候选的并集点亮为已安装。
- 本次没有新增现场照片，没有证明全部连拍帧已配准。A 的无效序列门禁继续生效；跨帧相机运动补偿仍是独立待做事项。
- 图像控制截线与 CAD 横断面用于候选对照，不能替代精密标定、相机完整外参或毫米级安装验收。地面修正的既有试验边界未被这次 AI 调用消除。
- 手机手持第三路径 C 仍未实现。本轮不能算三条路径全部完成。
- 当前识别瓶颈已从“没有真实 AI 接入”转为“同径重复管位、遮挡和观察方向信息不足导致多解”。后续应针对能区分候选的新增证据和约束做验证，而非将这批歧义响应强制转成唯一管号。

## 复现入口

`gui-verify.py` 只读恢复已保存真实响应，不发新请求。`gui-live-verify.py` 默认仅输出准备计划；只有显式传 `--execute-live` 才通过 GUI 进行一次新推理。`live-verify.py` 为工作流实网验证，运行前应阅读脚本范围，避免重复产生用量。原始输入摘要见 [prepared-input.json](prepared-input.json)。

本轮最终自动化命令：

```powershell
$env:PYTHONUTF8='1'
outputs/kimi-integration-20261101/venv/Scripts/python.exe -c "import cv2,pytest;cv2.setNumThreads(2);raise SystemExit(pytest.main(['tests','-q','-rs']))"
.venv/Scripts/python.exe -m pytest tests/test_gui.py::DashboardModelTests::test_module_import_does_not_eagerly_import_tkinter -q -rs
.venv/Scripts/python.exe -m pipe_twin ai-status
```
