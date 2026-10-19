# MONOPAPER_PLAN.md

> 本轮 monopaper 的执行计划。P0 阶段聚焦"internal recurrent substrate 能否复现/取代/补充外挂 CLR"。
> 日期：2026-08-24。关联：`MONOPAPER_STATUS.md`（进度）、`OURO_BASELINE_AUDIT.md`（事实）。

## 核心问题与 P0 锚点

> Can an internal recurrent substrate reproduce, replace, or complement
> the persistent-computation effect currently supplied by external CLR?

P0 只回答这一个。不从 Qwen3.5-0.8B 起步、不先做 JEPA / sequence-valued state / adaptive halting（§0）。

## 实验矩阵（§7/§8/§27/§28）

| 对比 | 回答什么 |
|---|---|
| A vs B | 复现历史 external persistence effect（基线门） |
| A vs C-reset | 仅换 token-mixing / recurrent substrate 是否提能力 |
| **C-reset vs C-persist** | **cross-turn internal persistence 是否产生额外 causal effect（最关键）** |
| C-persist vs D-persist | external CLR 在 internal substrate 上是否仍有独立价值 |
| B vs C-persist | internalized persistence 能否取代 external controller |

决策判据（§29）：
- C-persist > C-reset ⇒ internal cross-turn persistence matters（核心 positive）
- C-persist ≈ B ⇒ external GRU–FiLM 可被 internal 取代
- D-persist > C-persist ⇒ 两者互补
- C-reset ≈ C-persist ⇒ 查 state 是否真保留/被读/被 detach（§29 Result D 排查清单）

## 执行顺序（§48，与 STATUS 进度表对齐）

0A Audit ✅ → 0B Reproduce ⏳ → 0C Design → 0D Migration Design → 0E Minimal Converted Model → 0F Migrate+Verify → 0G Pre-distill Eval → 0H Distill → 0I C training → 0J D training → 0K Interventions → 0L Analysis

### 关键设计约束（来自审计）

1. **24 层共享权重**（§0 审计）：替换第 k 层 = 替换所有 UT step 的第 k 层。selective/hybrid conversion（§14）不能按普通 Transformer 理解。conversion-depth ablation 的"比例"指被替换的层号集合占比，不是独立层。
2. **MHA 非 GQA**（16 heads = 16 KV heads）：迁移 attention→SSM 时源结构是全注意力。
3. **state_module 抽象已存在**（`exp_seqstate_s0/state_modules.py`）：C/D 的外挂 CLR 分支复用此接口 + `turn_forward_v1(state_module=...)`，无需重写 forward。但 internal substrate 本身是新机制（迁移 backbone 权重，非外挂）。
4. **梯度语义对齐**（§22）：历史 CLR 用 turn-boundary detach。internal recurrent condition 沿用同样语义，公平性优先。
5. **8GB 约束**（§34）：4-bit frozen teacher + bf16 recurrent module + gradient checkpointing + offline teacher cache。避免 teacher+student 同时驻留。

## 候选 recurrent substrate（§9，Phase 0C 要比较选定）

1. Linear recurrent attention（如 RetNet/linear attention）
2. SSM-style token mixer（如 Mamba-style）
3. Gated linear recurrence（如 RWKV-style gate）
4. Hybrid attention + recurrent blocks
5. Recurrent replacement of selected attention layers
6. Recurrent replacement of selected token-mixing blocks

选定优先级：controlled weight migration > reproducibility > small implementation surface > 8GB feasibility > interpretability of intervention（§9）。不追求最新最复杂。

## 权重迁移三分类（§10/§11/§40，Phase 0D）

- **Class A exact_copy**：shape/语义/归一化兼容的 tensor（embedding, LM head, RMSNorm, MLP projections, 未改 attention blocks）。验证 `max_abs_diff==0`（§12）。
- **Class B structured_init**：Transformer 投影可合理初始化新 recurrent mechanism 但非精确语义对应。必须标记 `structured_init`，记录 source/target/mapping/reshape/transpose/scaling/reason。
- **Class C new_parameter**：SSM 的 A/B/C/decay/transition/state gate/recurrent kernel 等无对应语义。显式初始化 + distill/adapt，不硬迁。

产出 `weight_migration_manifest.json`（§11 schema）+ exact-copy 验证报告（§12）+ UNCHANGED_PARAMETER_CHECK（§13）。

## 阶段分离（§15）

M0 原始 Ouro → M1 架构转换+迁移初始化（不蒸馏）→ M2 转换+蒸馏 → M3 任务适配/persistent-state 训练。每阶段存 checkpoint。

## 干预电池（§24/§25/§26，Phase 0K）

normal / zero / random / freeze / reset-between-turns / persist-between-turns / swap-between-trajectories +（internal state）shuffle / scale / selective reset。尽量 per-iteration diagnostics（real/zero/rebuilt）检验 internalized recurrence 是否仍有 two-step bootstrap 阈值。candidate-scoring 作 final protocol（§30）。

## monopaper skeleton（§43）

1 Intro / 2 Framework / 3 Establishing / 4 Characterizing / 5 From State to Computation / 6 Internal Recurrent Substrate / 7 Sequence-Valued [P1] / 8 Predictive Transition [P2] / 9 Accuracy-Compute [P3] / 10 Discussion / 11 Limitations / 12 Conclusion。

实验完成前只允许 we ask / we test / we hypothesize（§42）。不写 "internal recurrence improves..."。

## Stop conditions（§49）

baseline 无法复现 / checkpoint mismatch / candidate evaluator mismatch / migration 破坏未改层 / persistent state 偷偷 reset / data leakage / cache reuse / teacher-student sample mismatch → 停，先修基础设施。

## 纪律

- 失败是结果（§37）：migration 毁能力 / 蒸馏失败 / internal persistence 零增益 / external CLR 仍赢 / hybrid 不稳定——全部记录。
- 目标不是证明 Linear RNN 更好，是判断 persistent computation 最合理的 substrate（§52）。
- 不为预期故事改实验。
