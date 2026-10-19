# P0I_TRAINING_DESIGN.md — Phase 0I: Cross-Turn Retention State Persistence

Date: 2026-08-25
Author: Marvis (executing 网友's Phase 0I spec)
Canonical root: `D:\training`（网友主目录，不动 `D:\training_marvis` replica）

## 0. 核心问题

> Does cross-turn persistence of the internal retention state **causally** improve multi-turn reasoning?

主实验只跑两个条件，唯一核心变量是 **turn-boundary state lifetime**：

| 条件 | turn 边界状态语义 |
| --- | --- |
| C-reset | retention state 每 turn 重置为零 |
| C-persist | retention state 跨 turn 持续传递 |

其余训练条件完全一致（同一 teacher cache、同一 seed 42、同一可训练参数集、同一 loss、同一 600 步、同一候选评测）。

## 1. 模型初始化：M2_CANONICAL

- 基座：Ouro-1.4B（4-bit NF4 冻结，`clr_ouro.load_ouro`）
- 替换层：REPLACED = [3, 7, 11, 15, 19, 23]（与 0H 一致）
- 初始化 checkpoint：`checkpoints/ouro_retention_m2_d3/best_model.pt`
  - variant: `ouro_retention_m2_d3_continue`（norm1800 总步数）
  - 0H 收尾基准（N=150 确认）：logit cos **.886** / hidden cos **.607** / ΔNLL **+5.12**
  - N=150 无系统性崩坏 → 冻结为 **M2_CANONICAL**（0H 第 3 件事）
- gamma：加载 d3 的 `retention_gamma`（sigmoid_logit 参数，bf16），**冻结**。
  - 理由：0H 已证明 decay-only adaptation 几乎不能恢复 teacher function（γ-only +14.27 ΔNLL），
    且 sigmoid_logit 参数在饱和区梯度消失；继续训 gamma 无收益且引入精度噪声。
- 可训练参数：12 个 post-norm weight（`input_layernorm_2` + `post_attention_layernorm_2`，REPLACED 每层 2 个），
  从 d3 的 `post_norm_weights` 初始化，继续微调。理由：0H 结论 post-norm adaptation 是主要可恢复通道。

## 2. 状态语义（必须钉死）

区分四层 recurrence，各自独立控制：

| 层级 | 语义 | 实现 |
| --- | --- | --- |
| token recurrence | retention 在单次 forward 内逐 token 递归（decay 矩阵/scan） | `RetentionAttention.recurrent_forward` 天然行为 |
| Ouro loop recurrence | 同一层在 4 个 UT loop 间（共享权重、深度方向）的状态 | **loop 边界 reset**：`current_ut>0` 时强制从零开始，不跨 loop 传播 |
| turn persistence | turn 边界状态 | C-reset → 每 turn 从零；C-persist → 用上一 turn 的 loop-0 结束状态 |
| episode persistence | trajectory 边界 | 始终 reset（新 traj 前清空 `_state`） |

**明确排除**：cross-loop persistence 不混入主实验。

### 2.1 实现（`p0i_loop_retention.py`）

- 包装 `_RetentionAttentionAdapter`：`current_ut==0` 时用外部注入的 `_state`；
  `current_ut>0` 时强制 `_state=None`（loop 边界 reset）。
- 每次 `current_ut==0` forward 后记录 `_turn_state = _last_state`（turn 结束状态，供 C-persist 下一 turn 使用）。
- turn 结束读取 `adapter._turn_state`；C-persist 下一 turn 前写 `adapter._state = _turn_state`；
  C-reset 与 episode 边界写 `adapter._state = None`。

### 2.2 Position policy（关键约束）

> RoPE/position 不能 state persist 但 position 偷偷归零。

- C-persist：position_ids 跨 turn **连续**——`offset = sum(prev turn lens)`，turn t 传
  `position_ids = offset + arange(L_t)`。OuroModel.forward 已支持外部 `position_ids`
  （modeling_ouro.py L559/L597-598），RoPE 随之连续。
- C-reset：每 turn `position_ids = arange(L_t)`（从 0 开始，与状态重置一致）。
- 校验：C-persist 下 offset 必须严格等于历史长度累计，训练中 assert。

## 3. 训练配置

| 项 | 值 | 说明 |
| --- | --- | --- |
| 数据 | `teacher_cache/traj_*.pt` + `generate_dataset(len(shards))`（seed 42 配对） | 与 0H 同一 cache，多 turn trajectory |
| 步数 | 600 | 与 0H gammanorm 一致 |
| LR | 3e-4（Adam） | 与 0H 一致 |
| Loss | `LAM_BLOCK=1.0` 的 block(cos+nmse) + `LAM_LOGIT=0.5` 的 logit KL | 与 0H 一致 |
| 可训练 | 12 × post-norm weight（24672 params） | gamma 冻结 |
| 输出 | `checkpoints/p0i_creset_s42/`、`checkpoints/p0i_cpersist_s42/` | best_model.pt + progress.json + distill_log.json |
| 备注 | 每 300 步存中间 ckpt；模型 eval 模式（无 dropout），seed 42 约定与数据配对一致 | |

## 4. 训练前自动测试（`p0i_state_lifetime_tests.py`）

验证（通过才允许训练）：

```
C-reset:
  turn2_start == zero            （每 turn 起始状态为零）
C-persist:
  turn2_start == turn1_end       （跨 turn 状态真传递）
new episode:
  state == initial state         （episode 边界重置）
new Ouro loop:
  state reset                    （current_ut>0 时从零开始）
no trajectory/batch leakage      （不同 traj / batch 之间状态隔离）
```

测试方法：构造两个短 trajectory（2 turns），对 6 个 REPLACED 层逐一检查
`adapter._state` 在 turn/loop/episode 边界的实际取值，输出 PASS/FAIL 矩阵，
并把结果写入 `P0I_STATE_LIFETIME_TESTS.md`。

## 5. 主评测（`p0i_eval.py`）

- 数据集：canonical CLUTRR candidate-scoring，`N=150` trajectory，**全部 turns**（不只 turn 0）。
- 推理模式：C-reset（状态每 turn 重置）vs C-persist（状态跨 turn 传递），
  与训练条件一一对应；position policy 与训练一致（persist 连续 / reset 归零）。
- 指标：
  - Acc(candidate-scoring) per condition，**point estimate + 95% CI**（bootstrap，B=1000）；
  - Δ_internal = Acc(C_persist) − Acc(C_reset)；
  - state norm（每层每 turn 结束）、state delta（turn 间状态差）、cross-turn cosine；
  - per-layer state statistics（norm 均值/方差、delta 均值/方差）；
  - VRAM（峰值）、latency（每 turn 平均耗时）。
- 输出：`p0i_results.json` → `P0I_RESULTS.md`、`P0I_RESOURCE_REPORT.md`。

## 6. 判读规则（预注册）

- `C-persist >> C-reset`：支持 internal cross-turn persistence 具有 causal task effect。
- 接近：先确认 state 真持久、真变化、真被读取、timescale 足够、position semantics 正确，再接受 negative result。
- `C-persist 更差`：检查 stale-state interference / state explosion / position mismatch，但结果本身保留。

## 7. 交付物与停止条件

完成后输出：
1. `P0I_TRAINING_DESIGN.md`（本文件）
2. `P0I_STATE_LIFETIME_TESTS.md`
3. `P0I_RESULTS.md`
4. `P0I_RESOURCE_REPORT.md`

然后**停止**，不自动进入 0J。
