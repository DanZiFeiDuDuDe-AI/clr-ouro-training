# MONOPAPER_STATUS.md

> 本轮 monopaper P0 阶段的状态快照。每完成一个阶段更新（§50）。
> 日期：2026-08-24。维护者：Claude（P0 接手）。

## 核心研究问题

> **Can persistent computation be internalized into a recurrent substrate
> instead of being supplied by an external GRU–FiLM controller?**

本轮从 Ouro-1.4B 开始（不从 Qwen3.5-0.8B、不先实现 JEPA / sequence-valued state / adaptive halting）。

## P0 实验矩阵（§7/§8/§27）

| 条件 | Ouro Loop | Internal Recurrent Substrate | External CLR | 状态 |
|---|:---:|:---:|:---:|---|
| A | ✓ | ✗ | ✗ | 历史基线，已有 |
| B | ✓ | ✗ | ✓ | 历史 v0/v1，已有 |
| C-reset | ✓ | ✓（reset/turn） | ✗ | **本轮待做** |
| C-persist | ✓ | ✓（persist/turn） | ✗ | **本轮待做** |
| D-reset | ✓ | ✓（reset/turn） | ✓ | **本轮待做** |
| D-persist | ✓ | ✓（persist/turn） | ✓ | **本轮待做** |

## 阶段进度（§48）

| 阶段 | 产出 | 状态 |
|---|---|---|
| 0A Audit | `OURO_BASELINE_AUDIT.md` | ✅ 完成 |
| 0B Reproduce | v1_s42 with_S=76/150, without_S=0 on 4060 | ✅ 完成（2026-08-24，精确复现历史金标准） |
| 0C Substrate Design | `P0_SUBSTRATE_DESIGN.md` | ✅ 完成（retention+6-layer hybrid + 6 项修订锁死） |
| 0D Weight Migration Design | `P0_WEIGHT_MIGRATION.md` | ✅ 完成（exact_copy ~100%，γ 唯一新参数，无 structured_init） |
| 0E Minimal Converted Model | forward-only, no train | ✅ 完成（6 层 retention forward 跑通，Amendment 7 gate 通过 diff<1e-3，VRAM 2.55GB） |
| 0F Migrate + Manifest + Verify | `weight_migration_manifest.json` | ✅ 完成（269 tensor exact_copy max_abs_diff==0，γ 96 params 唯一新） |
| 0G Pre-distill Eval | teacher vs migrated student | ✅ 完成（M1 受损未崩：logit cos 0.77，NLL+15.2，可进 0H 蒸馏） |
| 0H Distillation | offline cache + D1/D2/D3 | 🟡 norm-only 600步: logit cos 0.853 / hidden 0.561 / NLL+6.2；norm-only 续训到1800步(v3): 0.883 / 0.607 / +5.06；γ-only fixed 600步: 0.746 / 0.260 / +10.68（γ 已可训但主修复在 norm 接口）；γ+norm fixed 待跑 |
| 0I Persistent Task Training | C-reset / C-persist | ☐ 未开始 |
| 0J External + Internal | D-reset / D-persist | ☐ 未开始 |
| 0K Causal Interventions | zero/random/freeze/swap + per-iter | ☐ 未开始 |
| 0L Analysis | `P0_RESULTS.md` + `P0_RESOURCE_REPORT.md` | ☐ 未开始 |

## 基础设施现状

### cheems 训练机（RTX 4060 8GB）
- 代码 + Ouro 小文件：✅ 已同步 `D:\training`
- `model.safetensors`：✅ 2,869,336,434 B，大小校验通过（SHA256 后台校验中）
- transformers 4.54.1：✅ 已装（Ouro 官方推荐版本，干净环境无需 5.x 补丁）
- bitsandbytes：✅ 已装 0.50.1（4-bit NF4 加载已跑通）
- torch 2.13.0+cu126：✅ 已有

### 阻塞
- 分类器间歇性超时（Anthropic 侧临时故障，非命令问题），影响所有 Bash 调用。
- `clr_ouro.py` `MODEL_PATH` 硬编码 `D:/LoopLM_models/Ouro-1.4B` → 需加 `OURO_MODEL_PATH` 环境变量覆盖，远程指向 `D:/training/Ouro-1.4B`。
- 无 git 仓库（§41）。

## 已冻结的历史事实（不得静默修改，§2）

- 3 seed 均值 +50.4pp [43.6,56.7]（fast_eval）
- k=1 candidate-scoring Δ=+9.33pp（17.33% vs 8.0%）
- k=4 candidate-scoring Δ=+38.67pp（46.67% vs 8.0%）
- 交换测试：换内容 S₄ 不影响（排 memory）
- 逐圈遮蔽：需连续两圈自然递归重建（D/I 满血，G/H/E 崩溃）
- loop1-3 跨轨迹 cos≈1.000，loop4 cos≈0.25-0.31
- S₂ 载体态跨轨迹 cos 两种子均精确 1.0000

详见 `OURO_BASELINE_AUDIT.md` §5 + `papers/PAPER1_FROZEN.md`。

## 不变量与纪律

- 历史数字不得静默修改（§2）。
- candidate-scoring 作主结果，fast_eval 仅 debug（§30）。
- 每套干预带阳性对照（zero-ablation 预期掉分，不掉=干预未生效，整批作废）。
- 实验目标不是证明新架构优，是判断 persistent computation 最合理的 substrate（§52）。
- monopaper 在新实验完成前只允许 we ask/we test/we hypothesize（§42）。
