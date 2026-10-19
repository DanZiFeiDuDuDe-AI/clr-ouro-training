# P0_WEIGHT_MIGRATION.md

> Phase 0D：weight migration 逐参数族设计（§10/§11/§40）。
> 关联：`P0_SUBSTRATE_DESIGN.md`（选型 + 6 项修订硬约束）。
> 日期：2026-08-24。

## 0. 迁移总览

本选型（retention + 6-layer hybrid）的迁移面极小：

| Class | 参数族 | 占比 | 说明 |
|:-:|:--|:-:|:--|
| A exact_copy | q/k/v/o_proj(全24层) + 4 RMSNorm(全24层) + MLP gate/up/down(全24层) + embed_tokens + lm_head + model.norm + early_exit_gate + rotary_emb | ~100% | tensor 精确拷贝，max_abs_diff==0 |
| B structured_init | — | 0% | retention 保留 Q/K/V/O 投影语义，无需 structured_init |
| C new_parameter | per-head decay γ（替换 6 层 × 16 head） | 96 params | multi-scale init，Amendment 1 |

**没有 structured_init 是选 retention 的红利**——弃选 Mamba 的原因之一就是 Q/K/V 无对应、大量 Class C。

## 1. Class A — Exact Copy

### 1.1 适用判据（§10）
source 与 target tensor 在 **shape / semantic role / normalization / activation path** 上明确兼容时 direct copy。

### 1.2 逐族清单

| 参数族 | source shape | target shape | 兼容性判据 | mode |
|:--|:--|:--|:--|:--|
| `layers.{i}.self_attn.q_proj.weight` | [2048,2048] | [2048,2048] | 替换层与未替换层均保留投影，shape+role 不变（mixing 变了，proj 没变） | exact_copy |
| `layers.{i}.self_attn.k_proj.weight` | [2048,2048] | [2048,2048] | 同上 | exact_copy |
| `layers.{i}.self_attn.v_proj.weight` | [2048,2048] | [2048,2048] | 同上 | exact_copy |
| `layers.{i}.self_attn.o_proj.weight` | [2048,2048] | [2048,2048] | 同上 | exact_copy |
| `layers.{i}.input_layernorm.weight` | [2048] | [2048] | RMSNorm，全层保留 | exact_copy |
| `layers.{i}.input_layernorm_2.weight` | [2048] | [2048] | post-attn norm，全层保留（Amendment：post-norm 在 retention 层是否解冻训练见 §1.4） | exact_copy |
| `layers.{i}.post_attention_layernorm.weight` | [2048] | [2048] | pre-MLP norm | exact_copy |
| `layers.{i}.post_attention_layernorm_2.weight` | [2048] | [2048] | post-MLP norm | exact_copy |
| `layers.{i}.mlp.gate_proj.weight` | [5632,2048] | [5632,2048] | SwiGLU gate，全层不动 | exact_copy |
| `layers.{i}.mlp.up_proj.weight` | [5632,2048] | [5632,2048] | SwiGLU up | exact_copy |
| `layers.{i}.mlp.down_proj.weight` | [2048,5632] | [2048,5632] | down | exact_copy |
| `embed_tokens.weight` | [49152,2048] | [49152,2048] | embedding 不动 | exact_copy |
| `lm_head.weight` | [49152,2048] | [49152,2048] | untied，不动 | exact_copy |
| `model.norm.weight` | [2048] | [2048] | final RMSNorm（每个 UT step 后应用） | exact_copy |
| `early_exit_gate.weight` | [1,2048] | [1,2048] | gate，不动 | exact_copy |
| `early_exit_gate.bias` | [1] | [1] | 全模型唯一 bias | exact_copy |

### 1.3 exact-copy 验证（§12）
所有 exact_copy tensor 必须验证 `max_abs_diff == 0`（bf16 同 dtype 无精度损失）。
报告：tensor count / parameter count / mismatch count / maximum difference。
mismatch 不得静默继续（§12）。

### 1.4 post-norm 是否解冻（Amendment 留的口子）
`input_layernorm_2` / `post_attention_layernorm_2`（post-attn/post-MLP）在替换层：
- M1（迁移初始化，不蒸馏）：保持 frozen exact_copy。
- M2（蒸馏）：**可选**解冻这 8 个 post-norm（6 层×... 见 P0_SUBSTRATE_DESIGN §9），让 retention 层的 post-norm 适配新 mixing 的幅度分布。解冻则从 manifest 标 `exact_copy → trainable_post_distill`，记录。
- 不解冻也能跑（retention 的 z_n normalization 已控制幅度）；解冻是 if-needed。

## 2. Class B — Structured Initialization

**无。** retention 的 Q/K/V/O 投影语义与原 attention 完全对应（都是 hidden→Q/K/V，O→hidden），不需要从 Transformer 投影"启发式"初始化新机制。这是选 retention 而非 Mamba/RWKV 的关键收益。

## 3. Class C — New Parameters

### 3.1 唯一新增：per-head decay γ

| target 参数 | shape | per | init | 说明 |
|:--|:--|:--|:--|:--|
| `retention.gamma_logit`（替换层） | [16] | 每替换层 16 个 | `θ_h = logit(1-2^{-5-h})`，h=0..15 | Amendment 1 multi-scale |

- **参数化**：`γ_h = σ(θ_h)`，无约束 θ_h 可训练。
- **init 值**：h=0 γ≈0.969，h=15 γ≈≈1.0。覆盖 short→long 多时间尺度。
- **不硬迁**：原 Transformer 无对应语义，显式初始化 + distill/adapt（§10 Class C）。
- runtime 的 z_n accumulator **非 parameter**，不进此表。

### 3.2 不硬迁的判据
γ 控制 retention decay，原 attention 的 softmax mixing 没有对应衰减参数。强行从某投影初始化会是无依据的 heuristic——正确做法是 multi-scale 先验 + 训练适配。

## 4. Migration Manifest Schema（§11）

`weight_migration_manifest.json`：
```json
{
  "source_model": "ByteDance/Ouro-1.4B",
  "target_model": "Ouro-1.4B-retention-6layer-hybrid",
  "source_checkpoint_hash": "58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af",
  "target_initial_checkpoint_hash": "<sha256 of migrated ckpt, M1>",

  "config": {
    "replaced_layers": [3, 7, 11, 15, 19, 23],
    "replacement_ratio": "6/24 = 25%",
    "mixing": "linear_retention_with_rope",
    "state_reset": "per_ouro_loop_boundary (Amendment 5)",
    "position_semantics": "episode_global (Amendment 4)",
    "gamma_init": "multi_scale_1-2^(-5-h)"
  },

  "layers": [
    {
      "source": "layers.3.self_attn.q_proj.weight",
      "target": "layers.3.self_attn.q_proj.weight",
      "mode": "exact_copy",
      "source_shape": [2048, 2048],
      "target_shape": [2048, 2048],
      "rule": "direct copy; projection role preserved, mixing changed (softmax->retention)",
      "checksum_before": "<sha256>",
      "checksum_after": "<sha256>",
      "notes": "projection parameters admit exact tensor migration despite changed mixing operator (Amendment 3)"
    },
    ...
    {
      "source": null,
      "target": "layers.3.self_attn.retention.gamma_logit",
      "mode": "new_parameter",
      "source_shape": null,
      "target_shape": [16],
      "rule": "multi-scale init gamma_h=sigmoid(theta_h), theta_h=logit(1-2^(-5-h))",
      "checksum_before": null,
      "checksum_after": "<sha256>",
      "notes": "only new trainable param; per-head decay (Amendment 1)"
    }
  ],

  "summary": {
    "pct_exact_copy": "<~100% by param count>",
    "pct_structured_init": "0%",
    "pct_new_parameter": "96 params (gamma only)",
    "exact_copy_tensor_count": "<N>",
    "exact_copy_param_count": "<N>",
    "new_param_count": 96,
    "mismatch_count": 0,
    "max_abs_diff_exact_copy": 0
  }
}
```

## 5. UNCHANGED_PARAMETER_CHECK（§13）

迁移后验证：**所有未改 layer 的全部 tensor** + 替换 layer 的 Q/K/V/O（exact_copy 部分）必须与 source checkpoint 逐 tensor `max_abs_diff==0`。
- 未替换 18 层：全 attention + MLP + norm，逐 tensor 验。
- 替换 6 层：Q/K/V/O + 4 norm + MLP 逐 tensor 验（这些是 exact_copy）。
- 唯一允许差异：6 个 `retention.gamma_logit`（new_parameter）。
- 目标：changed architecture only where the experiment intends to change it（§13）——本例 intended change = mixing operator（非参数）+ γ（96 params），其余全冻结一致。

## 6. 阶段分离（§15）

- **M0**：原始 Ouro（已校验，Phase 0B 复现通过）。
- **M1**：架构转换 + 迁移初始化，**不蒸馏**。生成 manifest + exact-copy 验证 + UNCHANGED check。先过 0E parallel/recurrent equivalence gate（Amendment 7），再进 0G pre-distill eval。
- **M2**：转换 + 蒸馏（§0H，offline teacher cache + D1/D2/D3）。
- **M3**：persistent-state 训练（§0I/0J，C/D 矩阵 reset/persist）。
- 每阶段存 checkpoint + run_manifest.json（§33）。

## 7. stop conditions（§49）

- exact-copy tensor 出现 mismatch（max_abs_diff≠0）→ stop，先修迁移。
- UNCHANGED_PARAMETER_CHECK 失败（未改层有差异）→ stop，迁移破坏了 backbone。
- parallel/recurrent equivalence 不过（Amendment 7）→ stop，实现 bug，先修再蒸馏。
- persistent state 偷偷 reset（state 实际未跨 turn 保存）→ stop，causal intervention 无效。
