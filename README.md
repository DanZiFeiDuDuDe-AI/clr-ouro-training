# CLR / Ouro：持续隐状态调制研究归档

这里汇总 CLR、PLAA 与 Ouro 循环模型实验的代码、论文、训练权重、特征缓存、诊断结果和历史记录。2026 年 10 月 20 日的整理将分散在本地磁盘、造物工作室阶段目录、训练服务器及本仓库原有版本中的资料放到同一套可追溯目录中。

研究关注：在冻结语言模型中，持续隐状态是否产生可验证的行为影响；以及这种计算能否从外部 GRU–FiLM 调制迁移到循环模型内部。不同阶段使用不同模型、数据和验收协议，不能把各时期的数字直接拼成统一实验结论。

## 从哪里开始

| 目的 | 入口 |
|---|---|
| 阅读论文、历史稿与报告 | [论文目录](papers/README.md) |
| 了解所有目录与来源 | [目录导航](docs/目录导航.md) |
| 了解当前 VGen2 状态与停止条件 | [实验状态与限制](docs/实验状态与限制.md) |
| 找到训练代码和环境说明 | [复现与环境](docs/复现与环境.md) |
| 下载权重、数据集与特征缓存 | [数据、权重与恢复](docs/数据权重与恢复.md) |
| 核对归档是否完整 | [来源清单](manifests/files.jsonl) · [统计](manifests/archive_summary.json) |

## 目录结构

```text
papers/                         论文源码、PDF、参考文献、历史稿与阶段报告
docs/                           中文导航、实验状态和恢复说明
research/
  looplm/                       本地 LoopLM 工作区快照，保留原阶段子目录
  clr/                          CLR 早期实验、结果与基座模型配置
  plaa/                         PLAA 相关代码、适配器和论文来源
  studio/                       从造物工作室找回的 v2.2 / v2.3 / single-slot 阶段资料
  migrations/                   D 盘迁移副本；与 H 盘版本分别保留
archive/
  github-2026-10-20/             整理前 GitHub 工作树的历史快照
  server-scnet/                 当前训练服务器项目目录的归档
tools/                          权重恢复、归档检查与远程传输工具
manifests/                      文件来源、SHA-256、脱敏记录和 Release 分卷索引
third_party/                    基座模型的官方模型卡、许可和来源记录
```

同名文件在不同阶段可能有不同含义，因此保留来源目录，不以文件名覆盖不同版本。内容完全相同的大文件在 Release 内只保存一份，再由文件清单映射回各个原路径。

## 下载完整资料

先取得代码、论文和文件索引：

```bash
git clone --depth 1 https://github.com/DanZiFeiDuDuDe-AI/clr-ouro-training.git
cd clr-ouro-training
```

权重和特征缓存保存在 [归档 Release](https://github.com/DanZiFeiDuDuDe-AI/clr-ouro-training/releases/tag/archive-2026-10-20)。安装 Python 3.11 或更新版本与 GitHub CLI 后执行：

```bash
python -m pip install zstandard
python tools/restore_artifacts.py
```

恢复脚本流式验证分卷和对象的 SHA-256，每个对象校验通过后才写入正式路径；默认拒绝覆盖内容不同的本地文件。只想浏览论文或实验报告时，不需要下载模型产物。完整恢复方法、空间需求和离线校验见[恢复说明](docs/数据权重与恢复.md)。

## 实验状态

O3-REBASE2 的既有结论是 **INCOMPLETE**。当前 rebase 的 3K 失败权重，与 10K 训练的第 3000 步快照逐 tensor 一致；较长训练已有 standalone validation 通过记录。Raw linear probe 的 single-write 为 96.66%，UPDATE latest 为 96.42%，不支持把当前不足解释为“原始 carrier 完全没有足够 value 信息”。但这并不等于找到了并修复了一个 VGen2 实现缺陷，也不等于恢复了历史 passing run 的完整契约。

本次归档在造物工作室额外找回了 `a1_v24_vgen2.py` 和 `a1_v24_o21a_mr_cache.py`，作为新的源码证据保留。原始历史 passing checkpoint 的认证仍未完成；本次没有重新训练、修改历史报告数字或运行 integrated O3 evaluation。

Candidate B 保持冻结；Rule A threshold 固定为 `0.8504285216331482`。Standalone gate 通过不自动授权 allocator、router/memory reconnect 或集成测试。详见[状态说明](docs/实验状态与限制.md)。

## 归档边界与许可

本次整理保留研究相关源文件、论文、结果、权重和缓存，排除虚拟环境、字节码、Git 内部目录、凭据和与研究无关的游戏工程。原始本地工作区未被移动或改写。当前 SCNet 服务器已逐文件校验归档；旧 cheems 服务器因网络不可达，仅能保留已有本地与 GitHub 镜像，见[范围与已知缺口](docs/归档范围与安全处理.md)。

历史文件中的硬编码连接密码已从发布副本移除；原始文件哈希与发布哈希分别记录。模型、第三方代码和研究文稿保留各自的权利与许可，不因汇总归档而统一改成一种新许可证。基座模型的官方来源和许可文件位于 [third_party](third_party/README.md)。
