# 贡献指南

## 分支与提交

- 从 `main` 创建短生命周期分支，例如 `feature/camera-contract`、`fix/unit-mismatch`；
- 提交信息应说明结果和范围，避免把无关改动混入同一提交；
- 通过 Pull Request 合并，PR 中说明测试证据、风险和回滚方式；
- 不得提交虚拟环境、密钥、现场人员原始影像或未经授权的大型数据集。

## 本地检查

```powershell
python -m pip install -r requirements.txt
python -m pip install pytest==9.1.1
python -m pytest tests -v
```

涉及依赖变更时：

1. 更新根目录 `requirements.txt`；
2. 在干净 Python 3.12 环境验证依赖解析；
3. 更新对应锁文件；
4. 在 README 中同步安装或 smoke 命令变化。

## 测试与文档要求

- 修改数据契约、状态机或单位换算时必须添加自动测试；
- 修改 CAD、相机标定、算法阈值或模型时必须记录 revision/hash；
- 安全关键场景不得用总体平均指标掩盖回退；
- 完全遮挡、健康失败或证据不足时必须维持安全拒判语义；
- 3DGS、VLM 和其他辅助插件不得获得权威状态写权限。

## 数据管理

原始数据应保存在受控存储中。Git 只保存：

- 数据 manifest 与校验和；
- 小型、去敏的黄金测试样例；
- 可复现生成或获取数据的说明；
- 标注规范与评估报告。
