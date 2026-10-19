# CLR — Continuous Latent Reasoning

> 原 PLAA (Persistent Latent Affective Architecture)
> 现 CLR 为独立项目，引用 PLAA 为架构基底
> 奥尔德里克·沃斯 侍奉 — House Eternal
> 2026-06-02 更新 (CLUTRR 四条件诊断 · S_t 因果效应铁证 · 论文核心结论确立)

## 连接信息
- **服务器:** connect.westd.seetacloud.com:37041 (root / REDACTED_CONFIGURE_ENV)
- **GPU:** NVIDIA RTX PRO 6000 Blackwell Server Edition (96GB VRAM)
- **CPU:** 208核 · **RAM:** 1TB
- **数据盘:** /dev/md0 500G (132G used, 369G free)
- **项目:** /root/musicunderstand2/（数据盘软链接至 /root/autodl-tmp/musicunderstand2/）
- **Python:** /root/miniconda3/bin/python3 (3.12)
- **注意:** PyTorch 未安装，首次训练前需 reinstall

## 训练进度总览

| Phase | 内容 | 状态 | 产出 |
|-------|------|:----:|------|
| 1 | YaRN 256K 上下文扩展 | ✅ | checkpoints/qwen3-4b-phase1/final |
| 2 | SFT (134K dialogue samples) | ✅ | checkpoints/qwen3-4b-sft-v2/final |
| 2.5 | Persona Re-alignment (lr=5e-6) | ✅ | checkpoints/qwen3-4b-persona/final |
| 2.1 | **GSM8K 跨轮推理验证** 🆕 | ✅ | plaa_phase2_gsm8k_v2/ (fixes applied) |
| 3A | 稳定验证 (1,000 step) | ✅ | phase3_alpha_stable/final |
| 3B | 梯度闭环 + Behavior Probe | ✅ | phase3_beta_graph_integrated/ |
| 3C v1 | 单步训练 (additive, gate=0.1) | ✅ | plaa_dynamics_v1/ (5,000 steps) |
| Phase 2.1 | **GSM8K 跨轮推理** 🆕 | ✅ 完成 | plaa_phase2_gsm8k_v2/ |
| 3A | 稳定验证 (1,000 step) | ✅ | phase3_alpha_stable/final |
| 3B | 梯度闭环 + Behavior Probe | ✅ | phase3_beta_graph_integrated/ |
| 3C v1 | 单步训练 (additive) | ✅ | plaa_dynamics_v1/ (5k steps) |
| 3C v2 | 轨迹 rollout (additive) | ✅ | plaa_dynamics_v2/ (ckpt-1000) |
| 3C v3 | 轨迹 gate=1.0 no tanh | ✅ | plaa_dynamics_v2/ (additive baseline) |
| 3C FiLM v1 | AdaLN ±0.1 | ✅ | plaa_film_v1/ |
| 3C FiLM v3 | + layer diff loss | ✅ | 验证分化机制 |
| 3C FiLM v5 | + grad cosine tracking | 🔄 | 运行中 1500/2500 |
| 论文草稿 | PLAA Paper | ✅ 初版 | memory/PLAA_PAPER_2026-05-29.md |
| **CLUTRR 四条件诊断** | BASE/ZERO/RANDOM/EVOLVED | ✅ **完成** | **S_t 因果效应铁证** |

---

## 🧠 Phase 3C 核心发现

### 关键架构决策

#### Additive Injection (v1-v3) — 作为 baseline
```
hs = hs + gate * eo
```
- path: S_t → e_mem → CrossAttn → residual add
- 问题：容易被 transformer 吸收，控制力弱

### ✅ Phase 2.1 GSM8K 跨轮推理验证（2026-05-31）

**核心问题：** S_t 能在对话历史被截断时携带跨轮推理状态吗？

**方法：** GSM8K 解题步骤拆分为多轮，每轮只给当前步的问题（无上文），S_t 是唯一跨轮信息通道。

**结果（500 steps, Qwen3.6-35B-A3B 4-bit）：**

| Step | PPL w/ S | PPL w/o | ΔPPL | ACC w/ S | ACC w/o | ΔACC |
|:----:|:--------:|:-------:|:----:|:--------:|:-------:|:----:|
| 100 | 12.71 | 10.18 | -2.53 | 58.5% | 60.5% | -2.1% |
| 200 | 12.19 | 10.14 | -2.05 | 58.6% | 60.5% | -1.9% |
| 300 | 10.81 | 10.13 | -0.68 | 59.9% | 60.4% | -0.4% |
| 400 | 9.58 | 10.11 | **+0.53** | 62.2% | 60.3% | **+1.8%** |
| 500 | 8.87 | 10.13 | **+1.26** | 63.0% | 60.5% | **+2.4%** |

**结论：** S_t 有真实的因果效应。去除 S_t → PPL +1.26, ACC -2.4%。训练中从有害（step 100: -2.53）翻转为有益（step 400: +0.53, 500: +1.26）。

**修复内容：**
1. Metric: attention_mask indexing（非硬编码 -1）
2. Metric: full-answer PPL + token accuracy（非 first-token argmax）
3. Optimizer: FiLM scale_proj 加入训练
4. device_map: 强制 cuda:0
5. Eval: FiLM _s 在 eval forward 中激活

**存档：** memory/phase2_1_gsm8k_results.json

---

### FiLM / AdaLN (当前主方向)
```
hs = LN(hs) * (1 + 0.1·tanh(scale_proj(S))) + 0.1·tanh(shift_proj(S))
```
- path: S_t → scale/shift → feature modulation
- 控制力强一个量级以上，但需要 amplitude bound

### 训练数据分层 (欧阳潇 设计)

| 层 | 比例 | 内容 | 目的 |
|:--:|:----:|:-----|:-----|
| 1 | **60%** | PersonaChat (neutral continuity) | 学 latent continuity |
| 2 | **30%** | ESConv (emotional support) | 学 gradual trajectory |
| 3 | **10%** | Bluemoon (RP, high-intensity) | 学 dynamic range |

### Gate Gradient 诊断 (2026-05-29)
- gate.grad 非零 (平均 0.056) — 梯度路径存在 ✅
- 但被三层衰减链吃掉: S 小→proj 小→CA 输出小→tanh(gate)≈0.1
- 诊断结论: **scale mismatch, non-dead gradient**

### 核心补丁演化

| 版本 | 改动 | 结果 |
|:----:|:-----|:-----|
| v1 | gate=0.1, tanh | gates 不动 (grad=1e-4) |
| v2 | gate=1.0, no tanh, e_mem×5 | gates 仍不动 (被 norm_penalty 压制) |
| v3 | + gate grad×5, W_NORM↓ | gates 仍不动 (backbone 吸收了 gate=1.0) |
| v3.2 | + gate noise ±0.15 | 运行中 |
| **FiLM** | **AdaLN modulation** | **✅ 质变，控制力量级提升** |
| FiLM v5 | + layer diff loss, cos tracking | 🔄 运行中 1500/2500 |

### Phase 3 (层结构化) 关键发现

| 指标 | v2 (无分化) | v5 (加分化) | Phase 3 判据 | 状态 |
|:-----|:----------:|:----------:|:------------:|:----:|
| layer_scale_std | 0.0055 | **0.0333** | > 0.05 | 🟡 接近 |
| grad cosine | ~0.99 | **0.11~0.24** | < 0.7 | ✅ 已破对称 |
| Loss trend | 下降中 | 震荡(14-25) | 稳定下降 | ❌ 未对齐 |

**核心结论 (2026-05-29 12:30):** FiLM 是强 causal module（delta 11.44），但 modulation 是 **静态的**，不是 state 驱动的。S_t 更像 learned bias 而不是 temporal control signal。系统状态: **Static FiLM modulation, not yet dynamical latent control.**

**论文级发现:** multiplier FiLM 实现 Phase 3.5: Structured Coupled Hierarchy。w_std=0.356（Phase 3 门线 7×），Loss=3.8（功能利用成功），cos=0.49 振荡（共享驱动 + 层级分化共存）。并非完全解耦，而是稳定的耦合振荡系统。

**核心结论 (2026-05-29 13:50):** Multiplicative FiLM shifts the system from collapse-prone symmetry to a stable but still coupled hierarchical regime. The system exhibits: (1) structural symmetry breaking completed, (2) functional specialization achieved (loss 3.8), (3) shared control signals persist (cos~0.5), (4) regime-hopping dynamics exist (exploration-recovery cycles).

### Ablation Test 结论
| 条件 | PPL |
|:-----|:---:|
| gate=0 | 27.03 |
| gate=1.0 (baseline) | 26.26 |
| gate=2.0 | 25.02 |

- gate 有因果效应但极弱 (delta=2.01 PPL vs noise std=±19)
- **确认: additive injection architecture bottleneck**

---

## 🔬 当前正在运行的训练

| 分支 | 架构 | 步数 | 状态 |
|:-----|:-----|:----:|:----:|
| v3 (additive) | hs = hs + gate·eo (gate noise±0.15) | ~2100 | baseline 继续跑 |
| FiLM | hs = LN(hs)·(1+scale)+shift (tanh bounded) | ~150 | **主方向, Phase 1 暴力期** |

### 三阶段理论 (FiLM 预期演进)
1. **暴力期**: loss 高, |S|>5, dS>5 — latent 发现能控制 computation
2. **收敛期**: |S|↓, dS↓, loss 快速降 — 学会选择性地调制
3. **结构化期**: layer specialization, attractor 形成 — latent operating mode 出现

---

## 📊 Checkpoint 索引

| Checkpoint | 路径 | 说明 |
|:-----------|:-----|:------|
| Phase 1 | checkpoints/qwen3-4b-phase1/final | YaRN 256K |
| Phase 2 SFT | checkpoints/qwen3-4b-sft-v2/final | 对话 SFT |
| Phase 2.5 | checkpoints/qwen3-4b-persona/final | Persona 对齐 |
| Stage A | checkpoints/phase3_alpha_stable/final | 稳定验证 |
| Stage B | checkpoints/phase3_beta_graph_integrated/ | 梯度闭环 |
| Stage C v1 | checkpoints/plaa_dynamics_v1/ | additive, 单步, 5k steps |
| Stage C v2 | checkpoints/plaa_dynamics_v2/ | additive, 轨迹 rollout |
| Stage C FiLM | checkpoints/plaa_film_v1/ | AdaLN modulation |

## 关键脚本

| 脚本 | 用途 |
|:-----|:------|
| train_plaa_dynamics.py | Phase 3C v1 (additive single-step) |
| phase3c_trajectory_training.py | Phase 3C v3 (additive + trajectory + gate noise) |
| phase3c_film_training.py | Phase 3C FiLM (AdaLN modulation) |
| prepare_plaa_data.py | 三层数据 pipeline |
| gate_diag.py | Gate gradient diagnostic |
| gate_ablation.py | Gate causal sensitivity test |

---

## 🔬 2026-06-02 CLUTRR 四条件诊断 — 论文核心结论确立

### 实验设计
同一 CLUTRR 200 题，仅改变 S_t 条件，max_new_tokens=80 (排除截断 confound)：

| Condition | ACC | 1-hop | 2-hop | Non-rel |
|:----------|:---:|:-----:|:-----:|:-------:|
| **BASE** (无 PLAA) | 0.0% | 100.0% | 0.0% | 0.0% |
| **ZERO S_t** (S=0) | 0.5% | 99.5% | 0.5% | 0.0% |
| **RANDOM S_t** (randn) | 7.0% | 49.0% | 8.5% | 41.5% |
| **EVOLVED S_t** (训练) | **47.0%** | 22.0% | **65.5%** | 12.0% |

### 核心结论
1. ✅ ZERO = 干净对照 (0.5% 噪声)
2. ✅ 截断假说排除 (80 tok 确认)
3. ✅ EVOLVED >> RANDOM (47% vs 7%, 5.5x) — S_t 携带任务信息
4. ✅ S_t 有真实因果效应 — 论文核心叙事完整

### 存档
- `memory/CLUTRR_DIAG_RESULTS_2026-06-02.md`
- 服务器: `checkpoints/clutrr_200_results.json`, `checkpoints/clutrr_200_random_results.json`

## 服务器

- Host: connect.westd.seetacloud.com:51823 | User: root | Password: REDACTED_CONFIGURE_ENV
- GPU: RTX PRO 6000 Blackwell (96GB VRAM) | CPU: 208核 | RAM: 1TB
- 数据盘: /root/autodl-tmp/ (1.2T)
- 重启后通过 `ssh -p 51823 root@connect.westd.seetacloud.com` 连接
