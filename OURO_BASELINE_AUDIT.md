# OURO_BASELINE_AUDIT.md — Phase 0A

> 产出要求（§48 Phase 0A）：完全审计 Ouro pipeline，不改模型。
> 本文档冻结审计事实，作为 Phase 0B 复现与 Phase 0C/0D 迁移设计的权威依据。
> 审计日期：2026-08-24。审计人：Claude（接手 monopaper P0 阶段）。

---

## 1. Canonical Substrate — Ouro-1.4B

| 项 | 值 | 来源 |
|:--|:--|:--|
| 仓库名 | `ByteDance/Ouro-1.4B`（非 ByteDance-Seed） | DESIGN.md §0 |
| 本地路径 | `D:\LoopLM_models\Ouro-1.4B` | clr_ouro.py L28 |
| 权重文件 | 单文件 `model.safetensors`（非分片） | 本地 ls |
| 文件大小 | 2,869,336,434 字节（~2.67 GiB） | 本地 stat |
| **SHA256** | `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af` | 本地 sha256sum，与 DESIGN.md 记录一致 |
| 配置 | config.json + configuration_ouro.py（trust_remote_code） | 本地 |
| 建模代码 | modeling_ouro.py（含 5 个 transformers-5.x 兼容补丁） | DESIGN.md §0 |
| tokenizer | GPT-2 BPE（vocab.json + merges.txt + tokenizer.json） | 本地 |

**校验结论**：本地权重 SHA256 与金标准精确匹配，可作同步源。H 盘副本已坏区禁用（DESIGN.md §0 事故记录）。

---

## 2. Ouro 架构事实（config.json + modeling_ouro.py 确认）

| 项 | 值 |
|:--|:--|
| 架构类 | `OuroForCausalLM`（`modeling_ouro.py` L656） |
| 内模型 | `OuroModel`（L529），含 UT loop |
| 层数 | 24 层 `OuroDecoderLayer` |
| **权重共享** | 24 层跨 `total_ut_steps=4` 次 UT loop **共享**（DESIGN.md §0：层权重跨 loop 共享 ⇒ FiLM 也天然作用于每个 UT step） |
| 注意力 | MHA：`num_attention_heads=16` = `num_key_value_heads=16`（**非 GQA**），head_dim 128 |
| hidden / intermediate | 2048 / 5632 |
| vocab | 49,152 |
| dtype | bfloat16 |
| max_position_embeddings | 65,536 |
| rope_theta | 1,000,000.0 |
| **循环参数** | `total_ut_steps=4`（UT loop 次数） |
| 早期退出 | `early_exit_gate = nn.Linear(2048,1)`；sigmoid λ 几何退出 PDF；`early_exit_threshold=1.0` 默认不早退 |
| 参数量 | ~1.3–1.4B（与命名一致） |

### 2.1 forward 签名（移植命门）

`OuroModel.forward` 返回 **三元组** `(BaseModelOutputWithPast, hidden_states_list, gate_list)`，非标准 output。直接调内层模型必须解包。

循环体（`modeling_ouro.py`，审计 agent 确认）：
```python
for current_ut in range(self.total_ut_steps):
    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(...)
    hidden_states = self.norm(hidden_states)
    hidden_states_list.append(hidden_states)
    gate_list.append(self.early_exit_gate(hidden_states))
```

KV cache 按 `ut_step × 24 + layer_idx` 编址（96 槽）。训练与 CLUTRR 评测 `use_cache=False` 全绕开。

### 2.2 transformers 兼容性

- README 要求 `<4.56`，推荐 4.54.1。
- 本地用 5.13+ 靠 modeling_ouro.py 的 5 个补丁跑通（pad_token_id / ROPE_INIT_FUNCTIONS default / compute_default_rope_parameters / create_causal_mask kwargs / inv_freq 自愈）。
- **cheems 现状（Phase 0B 后）**：Python 3.13.15，transformers 4.54.1，bitsandbytes 0.50.1，accelerate 1.14.0，torch 2.13.0+cu126。干净环境，用 4.54.1 推荐版，无需 5.x 补丁。两处 `create_causal_mask` 已打动态签名适配补丁（modeling_ouro.py mask_kwargs + clr_ouro.py turn_forward_v1），4.54.1/5.x 双向兼容。

---

## 3. CLR 注入路径（clr_ouro.py 确认）

### 3.1 FiLM 注入

| 项 | 值 | 代码位置 |
|:--|:--|:--|
| 注入层 | `FILM_LAYERS = [3,7,11,15,19,23]`（6 层） | clr_ouro.py L31 |
| 调制公式 | `h ← h * (1 + 0.1·tanh(W·S))`，乘性，调制层输入 | FiLMWrapper L36-57 |
| ALPHA | 0.1 | L32 |
| 注入函数 | `inject_film(model)`：替换 6 层为 FiLMWrapper | L131-139 |
| 属性转发 | FiLMWrapper 必须暴露 `.attention_type`（OuroModel.forward 读此属性，nn.Module `__getattr__` 不转发普通属性） | L45 |

### 3.2 状态更新（PlaaCore / GRU）

| 项 | 值 |
|:--|:--|
| 状态维度 DS | 64（已验证充分；16 不行，256 是 4B 配置） |
| read_proj | `Linear(2048→64)` |
| GRU | `GRUCell(64,64)` |
| init_s | `Parameter(zeros(1,64))` + 0.01 噪声初始化 |
| 更新 | `S_{t+1} = GRU(read_proj(h_last), S_t)` |
| 可训练参数 | ~0.95M（6×2048×64 + 2048×64 + GRU） |

### 3.3 两档前向协议

| 协议 | S 更新粒度 | 函数 | 用途 |
|:--|:--|:--|:--|
| v0 | 每轮一次（完整 4-loop 前向后） | `turn_forward_v0` L148 | 与 0.8B CLR 协议逐行同构 |
| v1 | 每个 UT step 后 | `turn_forward_v1` L182 | per-loop 粒度，状态演化分析 |

**唯一通道原则**：S 是跨轮唯一信息通道（每轮截断上下文）。轮内 S 与残差流天然冗余——v1 若无增益是预期结果。

### 3.4 前向细节（turn_forward_v1）

- 外部复刻 `OuroModel.forward` 的 UT loop（`use_cache=False` 路径）。
- 每个 UT step k 后：`S = updater.update(S, hidden_normed[0, last])`，下一个 UT step 的 FiLM 见到更新后的 S。
- 关键参数：`ablate_s`（zero 协议）、`state_module`（外部状态模块）、`inject_override`（swap 实验注入）、`halt_gate`（Paper 2 Direction C）、`max_ut`（深度扫描）、`intervene`（注入式干预）。
- `state_module is not None` 时走通用路径：`to_modulation(state) → set_state`，`plaa_core` 被忽略。

---

## 4. StateModule 抽象（exp_seqstate_s0/state_modules.py）

**这是本轮 C/D 条件外挂分支可复用的现成接口**，但**不是 internal recurrent substrate**。

### 4.1 接口

```python
class StateModule(nn.Module):
    mod_dim = DS  # = 64
    def init_state(self, B, device) -> state
    def update(self, state, h_last) -> state
    def to_modulation(self, state) -> [B, 64]
```

### 4.2 现有实现

| 类 | state 形状 | 更新 | 性质 |
|:--|:--|:--|:--|
| `VectorStateModule(PlaaCore)` | [B,64] | GRU | **外挂**，state_dict 键兼容旧 ckpt |
| `SequenceStateModule` | [B,K,d]（K*d=64） | tiny-transformer | **外挂** sequence 状态（P1 雏形） |

### 4.3 ⚠️ 层次区分（Phase 0C 命门）

现有 StateModule（含 SequenceStateModule）**全部是外挂的**：通过 `to_modulation() → FiLM` 注入**冻结 backbone**，backbone 内部计算不变。S0 实验是"换一个更好的外挂状态模块"，属 B 的变体，**不属 C/D**。

本轮 §6/§9 的 **internal recurrent substrate** = 把 backbone 自身 token-mixing（attention/MLP）迁移成 Linear-RNN/SSM，使持久计算成为 backbone 内部机制。这是不同层次的工作，Phase 0C 必须新做，不能与 S0 混淆。

---

## 5. 评测协议

### 5.1 两套读出（§30，不得混在同一主表）

| 读出 | 实现 | 性质 |
|:--|:--|:--|
| fast_eval / argmax | 内联在各训练/评测脚本的 `eval_*()`：`logits.argmax() == target_token_id` | **格式敏感**（依赖模型精确产出训练格式 token） |
| candidate-scoring | `_candidate_eval_v2.py` `score_candidates()`：按 14 关系词的 mean token log-prob 排名 | **格式无关**，P0 主结果用此 |

**金标准数字（PAPER1_FROZEN.md，冻结，不得静默修改）**：

| k | with_S | without_S | Δ | 读出 |
|:--:|:--:|:--:|:--:|:--|
| 1 | 17.33% [12.11,24.19] | 8.0% [4.64,13.46] | **+9.33pp** | candidate-scoring, n=150 |
| 4 | 46.67% [38.87,54.63] | 8.0% [4.64,13.46] | **+38.67pp** | candidate-scoring, n=150 |
| 4 (argmax) | 52.0% [44.1,59.8] | 0/150 | +52.0pp | fast_eval（方向印证，非主表） |

三种子 M2（fast_eval）：均值 **+50.4pp [43.6,56.7]**；五类零效应对照（static-trained/zeroed/random/frozen-eval）全精确 0。

### 5.2 历史读出协议坑（已记录，勿重蹈）

1. fast_eval 只比对答案首 token ⇒ grand-* 四关系共享前缀，系统性宽容充水。
2. 候选带前导空格 vs 训练目标无空格——读出必须匹配训练格式（`_candidate_eval_v2.py` 用无空格候选）。
3. `_candidate_eval_v2.py` 曾有"断点续算"硬编码跳过逻辑，改 N_TRAJ 时未删导致旧值混入——已删除。**§32 verified-weights 教训：容错代码参数一变就要复查所有硬编码跳过。**

---

## 6. 干预套件（§24/§25/§26 现有实现）

| 干预 | 脚本 | 条件 |
|:--|:--|:--|
| M2 eval-time | `_interventions_clr_ouro.py` | full/zeroed/random/frozen/dose(α∈{0.25,0.5,0.75,1.0}) |
| 交换测试 | `_swap_test_clr_ouro.py` | A 正常/B 换内容 S₄/C 换载体/D 清空入口 |
| 逐圈遮蔽 | `_loop_zero_matrix.py` | A/D/E/G/H/I 六条件 |
| 动力学 | `_analyze_s_dynamics.py` | per-transition cos, ‖ΔS‖, 跨轨迹 cos |

**关键动力学发现（跨种子复现）**：
- loop1-3 跨轨迹 cos ≈ 0.999-1.000（trajectory-invariant plateau）
- loop4 跨轨迹 cos ≈ 0.25-0.31（trajectory-dependent break）
- 连续过渡 cos(S3,S4) ≈ 0.35（真表示变换，非缩放）
- S₂（载体态）跨轨迹 cos 两种子均精确 1.0000
- 逐圈遮蔽矩阵：需连续两圈自然递归重建（D=51.3%、I=44%），单圈真实（G/H）=0%，全零（E）=0%

---

## 7. Checkpoints & Raw Artifacts

### 7.1 Checkpoint 目录（18 个，每个 best_model.pt ~1.9MB）

仅含 CLR 可训练权重（read_proj + GRU + 6×scale_proj），**不含冻结 backbone**。加载时 backbone 从 `D:\LoopLM_models\Ouro-1.4B` 单独加载。

关键 checkpoint：
- `clr_ouro_v0_s42/`（+ s130/s520）：v0 三种子，含 interventions.json/shortcut_controls.json/candidate_eval.json
- `clr_ouro_v1_s42/`（+ s130）：v1，含 s_dynamics.json/s_dynamics_full.json/swap_test.json/loop_zero_matrix.json/donor_coincidence.json
- `clr_ouro_v0_ut{1,2,3}_s42/`：循环深度扫描
- `clr_ouro_v0_{math,recall}_s42/`：M4 内容传输任务

### 7.2 结果 JSON

| 文件 | 内容 |
|:--|:--|
| `checkpoints/m2_three_seed_summary.json` | 三种子分层 bootstrap CI |
| `checkpoints/candidate_eval_v2.json` | 金标准 candidate-scoring 数字 |
| `checkpoints/clr_ouro_v1_s42/loop_zero_matrix.json` | 逐圈遮蔽矩阵 |
| `checkpoints/clr_ouro_v1_s42/s_dynamics.json` | per-iteration 状态动力学 |

---

## 8. 无 Git 仓库（§41 阻塞）

**`D:\LoopLM` 下无 `.git`**。§41 要求每个独立能力单独 commit。Phase 0B 前应 `git init` 并做首次 commit 锁定当前状态（或明确用户对版本控制的偏好后再定）。

---

## 9. 复现安全门（§31 现成实现）

`exp_seqstate_s0/verify_baseline.py`：
- 加载 `checkpoints/clr_ouro_v1_s42/best_model.pt`
- 跑 150 轨迹 v1 评测（per-loop update, argmax readout）
- Check 1：generic path（VectorStateModule）with_S t2p = 76±2、without_S = 0
- Check 2：generic path == legacy path bucket-for-bucket（40 轨迹子集，同 RNG）
- 输出 `verify_baseline.passed` 标记

**Phase 0B 已通过（2026-08-24）**：cheems（4060 + tf4.54.1 + bnb0.50.1）上跑此脚本，
- with_S t2p = **76/150**（精确命中历史金标准 76±2）
- without_S = **0/150**（零效应对照精确 0）
- generic path == legacy path bucket-for-bucket（40 轨迹）
- exit code 0，`verify_baseline.passed` 已写
- §31 全部门通过：方向 / 量级 / evaluator 语义 / checkpoint loading。也间接证明远程权重完整（损坏不可能命中 76/150）。

---

## 10. 当前阻塞与下一步

### 阻塞
（Phase 0B 前的阻塞已全部解除）原阻塞记录存档：
- 2.87GB 权重同步：Tailscale IPv6 直连 <0.3MB/s 不可用 → 改让 cheems 直连 hf-mirror 自拉（用户手动下载完成，大小校验通过 + baseline 命中间接验证完整性）。
- cheems 环境：transformers 4.54.1 / bitsandbytes 0.50.1 / accelerate 1.14.0 已装齐。

### Phase 0A/0B 完成判定
本文档已覆盖 §51 要求的全部审计项：baseline 数字、checkpoint、loop 实现（k=1..4）、CLR 注入路径、evaluator（candidate-scoring vs fast_eval）、intervention 代码、raw artifacts。Phase 0B 复现通过。

### 下一步（§48 Phase 0C）
设计 internal recurrent substrate——比较 §9 六候选，选定一个，产出完整 `P0_SUBSTRATE_DESIGN.md`（16 项）。先写设计，再实现。当前 `P0_SUBSTRATE_DESIGN.md` 骨架已建，等 `modeling_ouro.py` 逐参数 weight shape 审计返回后补全 ⏳ 项。

---

## 11. 对迁移设计的直接约束（Phase 0C/0D 预备）

1. **"迁移部分层"在 Ouro 上语义特殊**：24 层共享权重 ⇒ 替换第 k 层 = 替换所有 UT step 的第 k 层。不能按普通 Transformer 的"替换第 3 层"理解，会同时改 4 个 loop 位置。影响 §14 selective/hybrid conversion 设计。
2. **state_module 抽象可复用**：C/D 的"外挂 CLR"分支复用 `StateModule` 接口与 `turn_forward_v1(state_module=...)`，无需重写 forward。但 internal substrate 本身是新机制（迁移 backbone 权重，非外挂）。
3. **公平性对齐**：历史 CLR 用 turn-boundary detach（v0/v1 均按 turn 推进，跨 turn 携带但 GRU 内部按 loop）。internal recurrent condition 应沿用同样梯度语义（§22）。
4. **MHA 非 GQA**：迁移 attention 为 SSM/recurrent 时，源结构是 16 头全注意力，非 grouped-query，weight shape 与 Qwen 等不同。
5. **早期退出 gate**：`early_exit_gate(2048→1)` 是独立可干预点，HaltGate 已实现 S-conditioned 偏置（Paper 2 Direction C 雏形）。
