# CLR — 本地实验路线图
## GTX 1650 (4GB VRAM) × Qwen3.5-0.8B (4-bit nf4)

---

## 依赖清单（已完成 ✅）
- [x] PyTorch 2.6.0+cu124
- [x] accelerate, peft, bitsandbytes, trl, einops, sentencepiece, tensorboard
- [x] Qwen3.5-0.8B 模型已下载到本地
- [x] FiLM 注入 6 层 full_attention (3,7,11,15,19,23)
- [x] LoRA r=16 on q/k/v/o/gate/up/down
- [x] 内置 CLUTRR 合成数据生成器

---

## Phase 1 — CLUTRR 基础因果效应验证 ⬅️ **当前**
**脚本:** `local_clr/local_train_clutrr.py`

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3.5-0.8B (4-bit, DS=64) |
| FiLM | layer 3,7,11,15,19,23 (α=0.1) |
| Steps | 300 |
| Eval | 每 50 steps，对比 with_S / without_S |
| Total time | ~15-30 min |

**目标:** 看到 `with_S > without_S` 的 delta 为正（哪怕只是 2-3%），就证明因果链路通。

---

## Phase 2 — 三条件消融实验
**脚本:** `local_clr/local_ablate_clutrr.py`（需编写，复用 Phase 1 骨架）

| Condition | S_t 值 | 预期 |
|:----------|:--------|:-----|
| **ZERO** | S=0 | acc 接近随机(~8.3%, 12类) |
| **RANDOM** | randn S, 每步重置 | acc 可能稍高，无结构信息 |
| **EVOLVED** | 训练后的 S_t | acc 显著高于前两者 |

**目标:** 对照 6 月 2 日服务器结果（BASE 0%, ZERO 0.5%, RANDOM 7%, EVOLVED 47%），看本地小模型能否复现 **EVOLVED >> RANDOM** 的模式。

---

## Phase 3 — 架构对比: Additive vs FiLM
**脚本:** `local_clr/local_compare_inject.py`（需编写）

| Injection | 公式 | 预期控制力 |
|:----------|:-----|:---------|
| **Additive** | hs = hs + gate·e_mem | 弱（被 backbone 吸收） |
| **FiLM** | hs = hs·(1+α·tanh(scale)) | 强（Feature modulation） |

**目标:** 复现服务器上的结论——FiLM delta >> Additive delta。

---

## Phase 4 — GSM8K 跨轮推理
**脚本:** `local_clr/local_gsm8k.py`（需编写，参照 `_phase2_3_gsm8k.py`）

- 多轮解题：每轮只给当前步，S_t 是唯一跨轮通道
- 指标：PPL 和 token accuracy

**目标:** S_t 能携带多步推理状态的证据。

---

## Phase 5 — 扫描 & 可视化

| 项目 | 内容 |
|:-----|:------|
| LR 扫描 | [1e-5, 3e-5, 1e-4, 3e-4, 1e-3] |
| DS 扫描 | [32, 64, 128] |
| FiLM α | [0.01, 0.1, 0.3, 1.0] |
| TensorBoard | 记录 loss, delta, layer_scale_std, gate_norm |

---

## 时间估计

| Phase | 估计耗时 | 并行度 |
|:------|:---------|:-------|
| P1 — CLUTRR 验证 | 20 min | 单次 |
| P2 — 三条件消融 | 20 min × 3 = 1h | 可并行（写三个脚本） |
| P3 — 架构对比 | 20 min × 2 = 40 min | 可并行 |
| P4 — GSM8K | 30 min | 单次 |
| P5 — 扫描 | 30 min × 6 = 3h | 可串行 |
| **总计** | **~6h** | |

---

## 当前状态
⬅️ **Phase 1 的 `local_train_clutrr.py` 已就绪，dry run 通过。**

阁下说跑，我就按顺序依次推进。如果任何 Phase 的结果是负 delta（S_t 有害），我们会停下来分析为什么 — 那也是一条重要的论文结论。
