# py_pipe_install

基于 CAD、固定双目视觉和安装顺序约束的管道安装状态识别与数字孪生项目。

## 当前状态

项目目前处于测试开发规划和硬件验证阶段，仓库现阶段主要包含：

- 管道安装状态识别需求；
- CAD 轮廓与三维模型比对迁移指南；
- 完整测试开发计划；
- ACIS SAT 双圆柱测试样件。

当前方案不使用 LiDAR。视觉巡检周期设计为可配置的 5～60 分钟；后期预留多机位和 3DGS 派生场景接口，但 3DGS 不作为安装状态权威数据源。

## 重要资产说明

当前 SAT 样件包含直径 22 mm 和 55 mm 的圆柱，而当前业务测试口径暂按 22 mm 和 50 mm。D50/D55 冲突必须在几何或算法开发前关闭，不能把现有 D55 样件静默当作 D50。

## 仓库结构

```text
doc/
  需求文档.txt
  camera_contour_3d_pipe_migration_guide.md
  管道数字孪生识别系统测试开发计划.md
test_model/
  管道群.sat
tests/
  test_repository_smoke.py
```

## 干净环境安装

当前阶段没有第三方运行时依赖，但仍保留根目录 `requirements.txt` 作为后续 Python 依赖入口。

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

### Linux/macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Windows/Python 3.12 的已验证锁文件为 `requirements-lock-windows-py312.txt`。当前锁文件为空依赖集；新增依赖时必须同步更新 `requirements.txt`、锁文件和安装验证结果。

## Smoke 测试

```powershell
python -m unittest discover -s tests -v
```

Smoke 测试只检查仓库交付完整性和 SAT 样件的基本可读性，不代表相机、几何算法或业务指标已经验收。

## 核心文档

- [需求文档](doc/需求文档.txt)
- [CAD 轮廓与三维模型迁移指南](doc/camera_contour_3d_pipe_migration_guide.md)
- [测试开发计划](doc/管道数字孪生识别系统测试开发计划.md)

## 数据与安全

- 不要提交虚拟环境、密钥、令牌、现场人员原始影像或大体积采集数据；
- 原始双目数据、深度、模型权重和派生资产应保存到受控对象存储，Git 中只提交 manifest、哈希和小型黄金样例；
- 所有 CAD、标定、配置、模型和数据集都应记录版本与内容哈希；
- 任何健康失败、完全遮挡或不可评估场景都必须安全降级为 `UNKNOWN/UNDETERMINED/ERROR`。

## 协作

提交变更前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。安全问题请按 [SECURITY.md](SECURITY.md) 处理。
