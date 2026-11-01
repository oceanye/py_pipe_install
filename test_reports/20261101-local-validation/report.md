# 2026-11-01 本地完整测试与 Kimi 接入核验

测试的生产提交：`1ccf126e59b570b72f26d46dbed650e97dc0d1e2`，与本轮查询的 GitHub `main` 一致。本次只增加测试记录、复现脚本和交接说明，没有修改生产代码。

**结论：并非只差 AI key。用户提供的 Kimi 凭据已通过官方模型列表接口鉴权；项目仍缺真实 AI 请求、输入绑定、结果持久化及 GUI 任务入口。现有自动化回归通过，但实窗验收发现一项状态展示错误。**

## 本次实际执行

| 检查 | 本机结果 | 能证明的范围 |
|---|---|---|
| 全量 `tests` | **1159 passed、1 skipped、272 subtests passed**，408.12 s | 当前生产提交在本机环境的自动化回归 |
| 单独复跑跳过项 | **1 passed**，0.19 s | Tk 懒加载检查在独立进程通过；全量跳过源于其他测试已导入 Tk |
| 依赖 / CLI | `pip check` 通过；`python -m pipe_twin --help` 通过 | 当前环境包依赖兼容、CLI 可启动 |
| 原图完整生产回放 | 完成，**76.656 s** | 从历史 manifest、10 对原图、标定经过实际生产分析器生成独立报告 |
| 真实 Tk 窗口 | **17/18 项通过，1 项失败** | 布局、模拟 AI 契约结果显示、模型选择联动、失效清理；不代表真实 AI 推理 |
| Kimi 官方接口 | 带 key **HTTP 200**，不带 key **HTTP 401** | 模型列表接口鉴权和网络连通；未调用推理、未验证余额或图片识别效果 |

测试日志：[pytest-output.txt](pytest-output.txt)。环境及 JUnit 摘要：[validation-evidence.json](validation-evidence.json)。

## 原始照片回放

源现场：`outputs/installation-assist-20261031/scenes/elevation-20261031-071031-189194c4/manifest.json`。实际照片拍摄于 **2026-10-30 18:32:43～18:32:50**；现场目录时间不是重新拍摄时间。

本轮报告：`outputs/local-validation-20261101/replay-20261101-101238-4a4b3a2e/report.json`。原现场、照片、标定、配置和默认现场指针共 **52 个文件**的 SHA-256 均未变化。没有打开相机、重新采集或将现场图片发送到外部 AI。

- 多帧：10 对中接受 4 对，整组 `INVALID`。
- 辅助身份：`INSUFFICIENT / TEMPORAL_SEQUENCE_INVALID_FOR_IDENTITY`，0 个 `LIKELY`、0 个 `AMBIGUOUS`、12 个 `NO_OBSERVATION`；不再把上一版本从参考帧生成的四管候选当成本轮有效结果。
- 严格安装与就位：均为 **12 UNKNOWN**；没有据此将任何管子判为未安装。
- 控制断面：`DEFINED`，四个条件断面的共同轴向支持区间 **81.458～200.302 mm**，控制站位 **140.880 mm**。这是当前条件轴下的公共截面，不代表已经完成与 CAD 的全局定位。
- 自由求解直径如下，与旧归档数值变化均为 **0 mm**。这些仍是条件解，不是已验收尺寸，也没有用采购规格覆盖求解值。

| 候选 | 原图颜色 | 条件直径 / mm |
|---|---|---:|
| S1 | 上方红管 | 38.8378 |
| S2 | 下方红管 | 40.3657 |
| S3 | 白管 | 20.9309 |
| S4 | 蓝管 | 46.2246 |

地面参考帧检查得到 `CONSISTENT_GROUND_PLANE`，轴向修正为 `TRIAL_APPLIED`，但 `accepted_measurement=false`。当前试验分支绕过了边界一致性检查的一项失败，不能将“地面平面一致”写成“整组标定、相机姿态和管道尺寸均已验收”。序列仍只做运动检查与拒判，没有跨帧相机运动配准或补偿。

复现与详细证据：[replay.py](replay.py)、[replay-evidence.json](replay-evidence.json)。

## GUI 实窗结果

使用真实 Tk 窗口打开历史场景，在内存中注入明确标记为 `LOCAL_UI_FIXTURE_NO_NETWORK` 的模拟 AI 返回值。模拟管号仅用于检查界面，不是外部模型判断。

通过：1024×720 / 1280×860 无控件裁切，照片高度分别为 194 / 334 px；没有 AI 字段时显示未运行；模拟候选、重叠歧义、缺项未解、控制断面文本可见；选择 AI 候选可选择模型管号，但不修改模型几何和安装状态；修改输入后清理结果；过期 worker 结果丢弃；无 Tk 回调错误。所打开窗口已关闭，28 个来源文件保持不变。

**失败项 `explicit_not_run_namespace_shows_not_run`：** 当报告包含由 `build_ai_section_assist(point_map)` 产生的显式 `status=NOT_RUN`、空候选命名空间时，详情显示“未运行”，表格却显示 `INSUFFICIENT / 证据不足`。原因在 `pipe_twin/ai_section_display.py` 的空候选统一回退分支。远程应保留顶层运行状态，并分别覆盖无命名空间、显式未运行、已运行但证据不足三种输入。

截图复核还发现两项非阻断展示问题：切到 AI 页后，照片仍显示 A 路径的黄色 IMG 候选框，需要明确标注来源以免误认成 AI 结果；模型位置列的较长坐标文字有轻微截断，可通过列宽或详情改善。它们不等同于整窗控件裁切。

证据：[gui-evidence.json](gui-evidence.json)，脚本：[gui-verify.py](gui-verify.py)。本机截图位于 `outputs/local-validation-20261101/gui/`；未将整组现场图像追加到 Git。

## Kimi 和应用接入状态

官方地址 `https://api.moonshot.cn/v1/models` 的只读探针使用临时进程内凭据，经不回显输入传入；没有将凭据写入文件、环境配置或 Git。该接口返回的模型包括 `kimi-k3.1-pro` 等 6 项，凭据检查证据为 [kimi-connectivity.json](kimi-connectivity.json)，复现脚本为 [kimi-connectivity.py](kimi-connectivity.py)。

仓库 fixture 写的是 `kimi-3.1-pro`，仅是手工元数据；实际返回的模型 ID 为 **`kimi-k3.1-pro`**，与 [Kimi 官方模型文档](https://platform.kimi.com/docs/models) 一致。后续适配器须使用真实模型 ID。[官方快速开始](https://platform.kimi.com/docs/get-api-key) 给出了服务地址和凭据配置方式。本次未做 Chat Completions 或视觉推理请求，因此不能据此宣称推理余额、视觉效果或应用闭环已通过。

| 环节 | 当前实现 |
|---|---|
| 固定模型断面 / P01…Pn 点位构造 | 函数和测试存在；生产分析链尚未调用 |
| AI 结构化响应校验 | 已有；限制候选状态，不写严格安装判定 |
| AI 展示页及模型选择 | 已有；读取报告中已有的 AI 字段 |
| 提供方设置、凭据读取、网络适配器 | **生产代码未实现** |
| 图片 / 固定断面图 / 方向 / 点位 / 提示词打包 | **生产代码未接入** |
| GUI 发起独立 AI 任务 | **未接入**；“保存并评估”仍执行本地 A 路径 |
| AI 结果附加、落盘、重开现场恢复 | helper 只返回深拷贝字典，**生产保存链未调用** |
| 手机第三路径 | 仍为规划探索，不在本次已完成范围 |

另外用最小内存输入复现三项接入层缺口：任意 `input_hash` 原样接受；不存在的 `observation_refs` 不核实；`control_section=UNRESOLVED` 仍可返回 `CANDIDATES_READY`。这些是将来接入时需要补的接纳条件，不代表当前程序已经发起不安全的 AI 调用。静态调用链、源文件 SHA-256 及探针结果见 [ai-readiness.json](ai-readiness.json)，脚本见 [ai-readiness.py](ai-readiness.py)。

## 环境与完成边界

本机 Python 3.12.10、OpenCV 4.13.0、SciPy 1.18.1、Tk 8.6。**实际 NumPy 2.2.6，仓库 requirements 与 Windows 锁文件均为 2.5.2。** 本次未升级现用环境。官方 PyPI 的 `--dry-run --no-deps --only-binary=:all:` 检查可解析所有锁定包的 Windows/Python 3.12 wheel 元数据，但这不等于干净环境安装及完整依赖解析通过；也不代表本次测试覆盖了 NumPy 2.5.2。

本轮完成了软件回归、归档生产回放、实窗检查、AI 接入审计及提供方鉴权。没有重新验证现场采集硬件、没有完成真实 AI 图像判断、没有解决旧序列配准、没有完成手机路径。

## 交给远程的下一步

1. 接通独立 Kimi 适配器与 GUI 任务入口：绑定当次原图、固定断面、观察方向、点位图和任务指纹，使用真实模型 ID，处理超时 / 错误 / 取消，不依赖 A 已经解出合格管段才能运行 B。
2. 补齐响应接纳、图片引用核实、控制断面资格、过期结果丢弃，以及独立 AI 字段的原子保存和重开；保留不改严格安装状态的边界。
3. 修复已复现的显式 `NOT_RUN` 空候选展示错误。
4. 使用固定同一组照片完成真实 Kimi 与 A 的独立对照，明确有效判断、歧义、遮挡和未观察；不把候选并集自动点亮为已安装。旧无效序列的运动配准问题另行验收，不放宽当前拒判门禁。
5. 在隔离环境按锁文件安装并验证 NumPy 2.5.2，统一开发 / 本地 / CI 的测试环境记录。

## 复现命令

在仓库根目录使用 PowerShell：

```powershell
$env:PYTHONUTF8='1'
.venv/Scripts/python.exe -c "import cv2,pytest;cv2.setNumThreads(2);raise SystemExit(pytest.main(['tests','-q','-rs','--junitxml=outputs/local-validation-20261101/pytest.xml']))"
.venv/Scripts/python.exe -m pytest tests/test_gui.py::DashboardModelTests::test_module_import_does_not_eagerly_import_tkinter -q -rs
.venv/Scripts/python.exe test_reports/20261101-local-validation/replay.py --execute
.venv/Scripts/python.exe test_reports/20261101-local-validation/gui-verify.py
.venv/Scripts/python.exe test_reports/20261101-local-validation/ai-readiness.py
```

GUI 复现脚本目前因已记录的展示错误返回非零，这是本轮验收发现，不应删除该检查来获得全绿。Kimi 脚本需在交互终端使用 `--interactive`，按隐藏输入提示提供凭据；不要把凭据放在命令行参数或报告里。
