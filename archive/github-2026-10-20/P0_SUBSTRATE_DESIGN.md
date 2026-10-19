# P0_SUBSTRATE_DESIGN.md

> Phase 0C：internal recurrent substrate 选型与设计（§39 16 项必填）。
> 先写设计，再实现（§39）。本文档未完成——weight-shape 相关项待
> `modeling_ouro.py` 架构审计返回后补全（标注 ⏳）。
> 日期：2026-08-24。

## 0. 与 Paper 2 的边界（§42 纪律先行）

Paper 2（`paper2.tex`）三方向（A 显式状态机 / B 学习转移 / C 自适应预算）**均未实现**，且自承 built entirely on Level 2 evidence。

本轮 P0 的 **internal recurrent substrate** 是 Paper 2 未直接覆盖的**新轴**：
- Paper 2 reviewer debt（`paper2.tex` §debt）：state-update operator × injection mechanism 两个自由度从未被联合消融。
- 本轮把它操作化为 **A/B/C/D 矩阵**（§7/§8），把外挂 controller **内化进 backbone**。
- 本轮**不跑** Paper 2 的 A/B/C 三方向（§0：不先做 JEPA/sequence-valued state/adaptive halting）。

若 C-persist > C-reset，本轮产出 Level 1-2 证据，**不自动**升级 Paper 2 的 Level 2 观察。措辞用 we hypothesize（§42）。

---

## 0a. Amendments（2026-08-24 review，0D 前写死，不可静默改）

原 0C 选型（retention + 6-layer hybrid + Q/K/V/O exact_copy + γ 唯一新参数）**保留**。
下列 6 项 implementation semantics 修订为硬约束，写入对应章节：

1. **Multi-scale γ init**（→§6）：不统一 0.95。`γ_h = σ(θ_h)` trainable，但 θ_h 初值使
   `γ_h ≈ 1 - 2^{-5-h}`（h=0..15 head index），覆盖多时间尺度——避免全 0.95 在读完下一
   turn prompt 前 state 已衰减、人为削弱 C-persist。
2. **Non-parametric recurrent normalization accumulator**（→§6）：不开裸 `S_n=γS_{n-1}+KᵀV`。
   runtime state 升为 `(S_n, z_n)`，z_n 是**非训练** scale accumulator（RetNet 式），控制
   accumulated KV 幅度随序列长度的 confound。trainable 参数**仍只有 γ**，不破坏极小迁移面。
3. **exact_copy 措辞严格化**（→§13）：manifest 标 q/k/v/o_proj = exact_copy 指的是
   **tensor 被精确拷贝**，不指 attention 计算函数保持。论文/设计文档措辞：
   "projection parameters admit exact tensor migration despite a changed mixing operator"，
   不写 "attention weights can be exactly migrated" 暗示函数保持。
4. **RoPE 跨 turn position semantics**（→§8）：episode-global position counter（跨 turn 不归零）。
   由此 C 条件拆为 `C-reset + local position` 与 `C-persist + global position`，并**显式承认
   positional semantics 在两条件间不同**（这是诚实报告，不是 bug）。
5. **Ouro loop boundary reset**（→§7，最关键）：retention state 在**每个 Ouro loop 边界
   reset**。三轴 token→Ouro loop→turn 中，P0 只把 turn/episode boundary persistence 作实验轴；
   persist-across-Ouro-loops 留作后续独立 ablation。否则 C-persist 涨分无法区分是 cross-turn
   还是 cross-loop persistence 在起作用。
6. **"structurally analogous" 降级 + state-capacity accounting**（→§4/§9）：不写
   "mathematically isomorphic"。retention state 是 per-layer per-head R^{d_k×d_v} × 6 层，
   bandwidth 远大于 64-d CLR。共同点只是 `state_{n+1}=f(state_n,x_n); y_n=g(state_n,x_n)`
   的结构。实验必须报告 persistent state elements / bytes per layer / total persistent-state
   bandwidth。若 C-persist 大赢，P1 补 low-rank / capacity-controlled retention 对照。

**额外 0E gate**（→§15/0E）：parallel/recurrent equivalence test——同序列、同 γ、同初态，
`O_parallel ≈ O_recurrent` 逐层误差。retention 的定义性质是 parallel/recurrent/chunkwise
三种表示算同一种 computation；自家实现连这关都不过不得开始蒸馏。

---

## 1. Actual Ouro architecture

24 层 `OuroDecoderLayer`，**共享权重**跨 `total_ut_steps=4` 次 UT loop。
MHA（num_attention_heads=16 = num_key_value_heads=16，非 GQA），head_dim 128。
hidden 2048 / intermediate 5632 / vocab 49152 / bfloat16 / **1.435B 参数**。
`OuroModel.forward` 循环体：`for current_ut in range(4): for layer in layers[:24]: hidden = layer(..., current_ut=current_ut); hidden = norm(hidden); hs_list.append; gate_list.append`。
早期退出 `early_exit_gate(2048→1)` sigmoid 几何 PDF，threshold=1.0 默认不早退。
embed_tokens(49152,2048) + lm_head(2048,49152) untied。

**单层参数清单（审计确认）**：每层 51.39M 参数，全 bias=False（除 early_exit_gate）。
- attention：q/k/v/o_proj 各 [2048,2048]（4×4.19M=16.78M，32.6%）
- 4× RMSNorm [2048]（8K，每子块 pre+post 双 norm——**非标准 pre/post-norm**）
- MLP SwiGLU：gate/up [5632,2048] + down [2048,5632]（34.60M，67.3%）

**forward 顺序（双 norm）**：
```
residual=h; h=input_layernorm(h); h=self_attn(h); h=input_layernorm_2(h); h=residual+h
residual=h; h=post_attention_layernorm(h); h=mlp(h); h=post_attention_layernorm_2(h); h=residual+h
```
**每子块 pre+post 双 RMSNorm**——迁移时 post-norm 在标准 recurrent 架构里无对应，需决策（保留/置 identity）。

**current_ut 仅用于 KV cache 索引**（`current_ut*24+layer_idx`，96 槽）；use_cache=False 时完全绕开。

## 2. Candidate recurrent replacements（§9）

| # | 候选 | token mixer 替换 | state 语义 | 迁移可行性 |
|:-:|:--|:--|:--|:--|
| 1 | Linear recurrent attention（RetNet/linear-attn） | attention → retention-style recurrent | per-token recurrent state | Q/K/V 可 structured_init |
| 2 | SSM-style token mixer（Mamba-style） | attention → SSM scan | hidden state h_t | A/B/C/DT 多为 new_parameter |
| 3 | Gated linear recurrence（RWKV-style） | attention → gated linear recurrent | time-mixing state | W 可 structured_init，gate new |
| 4 | Hybrid attention + recurrent blocks | 替换部分层为 1/2/3，其余保留 | 视替换层而定 | 未改层 exact_copy |
| 5 | Recurrent replacement of selected attention layers | 同 4 但只换 attention | 视层 | attention→recurrent |
| 6 | Recurrent replacement of selected token-mixing blocks | 同 5，含 MLP 也可换 | 视层 | 视映射 |

## 3. Selected architecture

**候选 1：Linear recurrent attention（保留 RoPE 的 retention 风格）**，以 hybrid 方式（候选 4）替换部分 attention 层。

具体形式——把选中层的 `OuroAttention` 的 softmax(QK^T/√d)V **mixing** 替换为 **linear retention**：
```
S_n = γ · S_{n-1} + K_n^T V_n          # per-head recurrent state, γ∈(0,1) decay
O_n = Q_n · S_n                         # output = Q 读取 retention state
```
- **保留**：q_proj/k_proj/v_proj/o_proj（shape+role 不变）、4 RMSNorm、MLP、RoPE、residual。
- **替换**：attention 内部的 softmax mixing → retention scan（mixing 是计算图，不是参数）。
- **新增**：decay γ（per-head，shape [16]，或 global [1]）——**唯一的 Class C 新参数**。

配合 §14 hybrid：第一版只替换一个 intermediate ratio 的 attention 层（见 §5），其余层保持原 attention exact_copy。

## 4. Why selected

按 §9 优先级（controlled weight migration > reproducibility > small implementation surface > 8GB feasibility > interpretability of intervention）逐维论证：

| 维 | 候选 1 retention | 候选 2 Mamba | 候选 3 RWKV |
|:-:|:--|:--|:--|
| 迁移面 | **最大**：Q/K/V/O 全 Class A，仅 γ 新 | 最小：Q/K/V 无对应，A_log/conv1d/D 等全 C | 中：K/V/O Class A，Q→receptance Class B，time_decay/first C |
| 实现面 | **最小**：替换 softmax→retention scan，其余不动 | 大：conv1d + SSM scan + 多新模块 | 中：WKV time-mixing + 新 gate |
| 8GB | **最优**：O(seq) 递归，retention state 固定大小 | O(seq) 但 conv1d+scan 更重 | O(seq) |
| 干预可解释性 | **最优**：retention state S 直接逐 token 可干预（zero/swap/reset），与旧 CLR 的 S_t 同构——可直接复用 §24/§25/§26 电池 | state h_t 可干预但语义异 | time-mixing state 可干预 |
| 复现 | 高：未改层 exact_copy，改层 mixing 简单可验 | 中：迁移面大，student 能力损失风险高 | 中 |

**弃选候选 2（Mamba）**：Q/K/V 无对应映射，迁移面最小，student 一迁移就崩的风险最大——违反 §16"一迁移就崩就先改 migration design"。候选 3（RWKV）是备选，但 q_proj 只能 structured_init（语义从 query 变 receptance），不如候选 1 的 q_proj exact_copy。

**候选 1 的关键优势**（措辞按 Amendment 6 降级）：retention state `S_n` 与旧外挂 CLR 的 `S_t` 是 **structurally analogous recurrent state**——共同点只是 `state_{n+1}=f(state_n, x_n); y_n=g(state_n, x_n)` 的结构形式，**非 mathematically isomorphic**。retention 是 per-layer per-head `R∈R^{d_k×d_v}` × 6 层，persistent bandwidth 远大于 64-d CLR（见 §9 capacity 表）。这不妨碍 P0——P0 问的是 substrate，不是 matched state capacity。但实验必须报告 persistent state bandwidth；若 C-persist 大赢，P1 补 low-rank/capacity-controlled retention 对照（避免 reviewer 说"就是塞了巨大 memory matrix"）。这让 C/D 矩阵的 internal state 干预能直接对标旧 CLR 的 zero/swap/reset 电池（§25），是回答"internal 能否取代 external"的最公平比较。

## 5. Which layers are replaced

约束：24 层共享权重 → 替换第 k 层 = 替换所有 UT step 的第 k 层。conversion-depth ablation 的"比例"指被替换层号集合占比。

**第一版（P0 首轮）**：替换 **6 层 = 25%**，选层号 `{3, 7, 11, 15, 19, 23}`（即 FiLM 注入层同集）。

理由：
1. 这 6 层正好是现有 CLR 的 FiLM 注入点——D 矩阵（internal substrate + external CLR）时，FiLM 与 retention 在同层叠加，直接测"外挂+内化是否互补"。
2. 均匀分布（每 4 层 1 个）保证 retention 状态贯穿深度，不集中在浅/深端。
3. 25% 是 §14 推荐的 intermediate ratio——足够检验 effect，又不至于一次改太多导致不稳。
4. 其余 18 层 attention + 全部 MLP + 全 norm exact_copy，UNCHANGED_PARAMETER_CHECK 可验证。

后续可 scan：0% / 25% / 50% / 75%（代码结构允许，§14）。

## 6. State equation

线性 retention（per-head），对替换层。**Amendment 1+2 写死**：
```
# 投影（与原 attention 完全相同，权重 exact_copy）
Q = x @ q_proj^T   # [B, H, L, d_h]   H=16, d_h=128
K = x @ k_proj^T   # [B, H, L, d_h]
V = x @ v_proj^T   # [B, H, L, d_h]
# RoPE（保留，与原 attention 相同；跨 turn position 语义见 §8）
Q, K = apply_rotary(Q, K, cos, sin)
# retention mixing（替换 softmax(QK^T)V）——带非参数 normalization accumulator
S_0 = 0;  z_0 = eps                       # [B,H,d_h,d_h] state + [B,H,1] scale acc
for n in range(L):
    S_n = γ_h · S_{n-1} + K_n^T · V_n      # γ_h per-head multi-scale (Amendment 1)
    z_n = γ_h · z_{n-1} + ‖K_n‖²            # non-trainable scale accumulator (Amendment 2)
    O_n = (Q_n · S_n) / sqrt(z_n)          # normalized read, 控制随 seq 长的幅度 confound
# 输出投影（与原 attention 相同，权重 exact_copy）
out = O @ o_proj^T
```

### γ 初始化（Amendment 1）
`γ_h = σ(θ_h)`，θ_h trainable per-head [16]，**初值使 `γ_h ≈ 1 - 2^{-5-h}`**（h=0..15 head index）：
- h=0: γ≈0.969，h=15: γ≈≈1.0（越深 head 越长记忆）。
- 不统一 0.95——避免所有 head 同一时间尺度，让 retention 覆盖 short→long 多时间尺度。
- 尤其保护 C-persist：长记忆 head 让上一 turn 的 state 在读完下一 turn prompt 后仍存活。

### runtime state（Amendment 2）
- **trainable 仍只有 γ**（16 参数/替换层）。
- runtime state 升为 `(S_n, z_n)`：S_n 是 KV retention matrix [B,H,d_h,d_h]；z_n 是**非训练** scale accumulator [B,H,1]，RetNet 式 normalization，消除 accumulated KV 幅度随序列长度的 confound。z_n 不进 manifest（非 parameter）。

### parallel / recurrent equivalence（Amendment 7，0E gate）
retention 的定义性质：parallel（矩阵分解）/ recurrent（scan）/ chunkwise 三种表示算同一种 computation。0E 实现后、蒸馏前，**强制 parallel/recurrent equivalence test**：同序列、同 γ、同初态，`O_parallel ≈ O_recurrent` 逐层 max_abs_err < 1e-4（bf16 容容差）。不过不得蒸馏。

注：S_n 与旧 CLR S_t 是 **structurally analogous recurrent state**（§4 Amendment 6），非 mathematically isomorphic——capacity 差异巨大（见 §9）。

## 7. State lifetime semantics（§20，Amendment 5——最关键）

三轴：**token → Ouro loop → turn**。

- **token state**：retention `(S_n, z_n)` 在 token 轴上递归，每序列从 init 开始。
- **Ouro loop state**：**retention state 在每个 Ouro loop 边界 reset（S=0, z=eps）**。
  原 Ouro attention 在下一 loop 是重新对当前 hidden sequence 做 attention，不存在"上一 loop 的 KV accumulator"。若让 retention loop1→loop2→loop3→loop4 跨 loop 保存，就额外加入了 **cross-depth persistence**——以后 C-persist 涨分无法区分是 cross-turn 还是 cross-Ouro-loop 在起作用。
  **P0 主条件：retention state resets at each Ouro loop boundary。** persist-across-Ouro-loops 留作后续独立 ablation（不进 P0 主表）。
- **turn state**：跨 turn。**这是 C-reset vs C-persist 的区别点**，也是 P0 的核心实验轴。
- **episode state**：整个轨迹（C-persist 的 state 跨所有 turn 保存）。
- state shape：S [B,H,d_h,d_h]=[B,16,128,128]，z [B,H,1]。dtype bf16。device cuda。detach 行为：turn boundary detach（与历史 CLR 对齐，§22 公平性优先）。gradient-through-turn：P0 用 detach（BPTT 跨 turn 留 future）。

**关键**：token 与 Ouro loop 都 reset，**只有 turn/episode boundary 的 persist 是实验轴**。这让 C-persist 的任何效应可干净归因到 cross-turn persistence。

## 8. reset/persist + position semantics（§8/§21，Amendment 4）

### turn/episode boundary
- **C-reset**：retention state `(S, z)` 在每 turn 开始重置（S=0, z=eps）→ 测 token-mixing 变化，非 cross-turn persistence。
- **C-persist**：retention state 跨 turn 保存 → 真测 internalized persistent computation。
- forward API：`forward(..., recurrent_state=None) → logits, new_recurrent_state`。episode driver 决定 persist 或 reset（§21），**不能让模型内部偷偷按 batch boundary 决定**（否则 causal intervention 无法保证）。

### RoPE 跨 turn position（Amendment 4——必须现在钉）
单序列内保留 Ouro 原 RoPE。但跨 turn 有 position basis 错配问题：turn1 留下的 retention state 里的 rotated K 处于上一 turn 的 positional basis，turn2 的 Q 又从 position 0 重新开始 → C-persist 跨 turn state 语义混乱。

**P0 选择：episode-global position counter**（最简单、最可审计）。position_ids 跨 turn 不归零，整个 episode 连续递增。
- 由此 C 条件拆为：
  - `C-reset + local position`（每 turn position 归零 + state 归零）
  - `C-persist + global position`（position 跨 turn 连续 + state 跨 turn 保存）
- **显式承认 positional semantics 在两条件间不同**：这是诚实报告，不是 bug。C-reset 用 local position 是因为它本就不跨 turn（state 都归零了，position basis 错配不存在）；C-persist 用 global position 是为了让跨 turn 的 rotated K/Q 处于同一 basis。
- 备选（position re-basing / state 不携带 absolute rotation）记为 future work，P0 不取。

## 9. parameter count + state capacity（Amendment 6）

### trainable parameters
- 替换层（6 层）：每层 attention Q/K/V/O = 16.78M 全部 exact_copy 保留（frozen）；新增 γ = 16 参数/层。6 层新增 96 参数。
- 未替换层（18 层）：exact_copy，无新增。
- **trainable**：仅 γ（96 参数）+ 4 RMSNorm post-norm 若放开训练（6×4×2048=49K，可选）。其余全 frozen。
- 若做 distillation（§0H），可解冻 γ + 选中层 post-norm，仍 <50K trainable——远小于旧 CLR 的 0.95M。

### persistent state capacity accounting（必须报告）
| 条件 | persistent state | elements/layer | total elements | bf16 bytes |
|:-:|:--|:--:|:--:|:--:|
| 旧 CLR (B) | S_t ∈ R^64 | 64 | 64 | 128 B |
| C/D retention (6 层) | R ∈ R^{d_k×d_v} per head × 16 heads × 6 层 | 16384/层 | 98304 | 196 KB |

- retention persistent bandwidth ≈ 98304 / 64 = **1536× CLR**。
- 这是 retention 的固有优势，也是 reviewer 风险点。P0 如实报告，不调和。
- P1 若 C-persist 大赢：补 capacity-controlled retention（low-rank R，使 elements≈64）作对照，隔离"substrate"与"bandwidth"两个轴。

## 10. expected compute

- 冒烟实测：原 Ouro 4-loop 单次前向 0.9s。retention scan 是 O(L·H·d_h²) = O(L·16·128²) ≈ O(L·262K)，比 softmax attention 的 O(L²·H·d_h)=O(L²·327K) 在长序列更省、短序列相当。CLUTRR 序列短（~12 token），前向成本持平。
- baseline 340 轨迹 ~80s（实测）。迁移后预计持平或略快。
- distillation：offline teacher cache（§17）后，student 前向 + 反向比纯前向慢 ~3-5×。

## 11. expected VRAM

- 4-bit frozen Ouro 加载：2.51 GB / 8.59 GB（冒烟实测）。
- retention state S_n：[B, 16, 128, 128] bf16 = B×0.5 MB。B=1 时 0.5 MB，可忽略。
- trainable γ + post-norm：<1 MB。
- **distillation 时不装 teacher**（offline cache 已存 hidden/logits），student 4-bit backbone + bf16 小 trainable 层，峰值预计 <4 GB——8GB 充裕。
- gradient checkpointing 备用（若 peak 超预算）。

## 12. intervention design（§24/§25/§26）

对 internal state（retention S_n）：
- **normal**：S_n 自然递归演化。
- **zero**：S_n = 0（全程），测 retention 是否必要。
- **random**：S_n 注入随机张量。
- **freeze**：S_n 冻在初始/某步值。
- **reset-between-turns**（C-reset）：每 turn 开始 S_0=0。
- **persist-between-turns**（C-persist）：S_n 跨 turn 保存（**核心对比**）。
- **swap-between-trajectories**（§25）：两条轨迹同 turn 交换 S_n——直接对标旧 CLR 的 S₄ swap 实验。
- per-iteration diagnostics（§26）：retention 在每个 UT loop 可观测 S_n 的 real/zero/rebuilt，检验 internalized recurrence 是否仍有 two-step bootstrap 阈值。
- candidate-scoring 作 final protocol（§30），fast_eval 仅 debug。

## 13. migration feasibility（Amendment 3 措辞严格化）

详见 Phase 0D `P0_WEIGHT_MIGRATION.md`。概览：
- **Class A exact_copy（绝大多数）**：全部 24 层的 q/k/v/o_proj、4 RMSNorm、MLP gate/up/down、embed_tokens、lm_head、model.norm、early_exit_gate、rotary_emb buffer。
- **Class B structured_init**：无。retention 保留 Q/K/V/O 的投影语义，不需要 structured_init。
- **Class C new_parameter（唯一）**：per-head decay γ（shape [16]/替换层，multi-scale init，Amendment 1）。显式初始化 + distill/adapt。

### exact_copy 措辞（Amendment 3，reviewer 会抓）
manifest 标 `q/k/v/o_proj = exact_copy` 指的是 **source tensors are copied exactly**（max_abs_diff==0）。
**不指** attention computation is functionally preserved——mixing 已从 softmax 换成 retention。

论文/设计文档措辞：
> projection parameters admit exact tensor migration despite a changed mixing operator.

**不写** "attention weights can be exactly migrated" 并暗示模型函数保持。这个区别在 §0D manifest 的 `notes` 字段和 monopaper §6.1 都要如实标注。

- **UNCHANGED_PARAMETER_CHECK（§13）**：未替换的 18 层 attention + 全部 MLP + norm 必须与 source checkpoint 逐 tensor `max_abs_diff==0`。替换层的 Q/K/V/O 也 exact_copy（mixing 变了，投影没变）。

## 14. distillation requirement

M1 迁移后 student 能力损失若大（§16/0G），需 offline teacher cache（§17）+ D1 block-output / D2 hidden / D3 logit matching（§18）。勿把"student 模仿 teacher"当"student 学会 persistence"（§19）。

## 15. risks

- 迁移毁 backbone 能力（0G 兜底：先修 migration 再训练）。
- shared-weight 下替换层影响所有 UT step（放大效应，可能不稳）。
- 内部 substrate 的 token state 与 Ouro loop state 语义混淆（§7 Amendment 5 已钉 loop boundary reset）。
- 8GB 装不下 teacher+student+cache（§34，靠 offline cache 缓解）。
- C-reset≈C-persist 时排查 state 是否真保留/被读/被 detach（§29 Result D）。
- **parallel/recurrent equivalence 不过（Amendment 7）**：自家 retention 实现若 parallel 与 recurrent 两路径输出不一致，说明实现 bug——蒸馏前 stop，先修（§49 stop condition）。
- **retention bandwidth 1536× CLR（Amendment 6）**：C-persist 若大赢，无法排除"塞了大 memory"的解释——P1 补 capacity-controlled 对照，P0 如实报告不调和。

## 16. falsification criteria

- C-persist ≤ C-reset（且 state 确实保留+被读+未误 detach）⇒ internal cross-turn persistence 在本配置下无额外 causal effect（negative result，如实记录，§37）。
- 迁移后未改层 max_abs_diff≠0 ⇒ migration 破坏 backbone，stop（§49）。
- baseline 无法复现 ⇒ 已通过（76/150），不计入。
