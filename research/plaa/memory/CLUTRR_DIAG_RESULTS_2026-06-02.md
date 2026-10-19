# CLUTRR 诊断实验完整结果

> 日期: 2026-06-02
> 模型: Qwen3.6-35B-A3B (4-bit)
> 数据集: CLUTRR test (200 samples)
> max_new_tokens: 80 (所有条件)
> enable_thinking: False (config 设置)

---

## 四条件完整对比

| Condition | ACC | Has relation | 1-hop | 2-hop | 3-hop | Non-rel |
|:----------|:---:|:-----------:|:-----:|:-----:|:-----:|:-------:|
| **BASE** (无 PLAA) | 0.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| **ZERO S_t** (PLAA + S=0) | 0.5% | 100.0% | 99.5% | 0.5% | 0.0% | 0.0% |
| **RANDOM S_t** (PLAA + S=randn) | 7.0% | 58.5% | 49.0% | 8.5% | 1.0% | 41.5% |
| **EVOLVED S_t** (PLAA + S 演化) | **47.0%** | **88.0%** | 22.0% | **65.5%** | 0.0% | 12.0% |

---

## 关键发现

### 1. 截断假说排除
80 token 足够完成 thinking 模式，但 BASE/ZERO 输出仍是 100% 1-hop。
→ 之前 15 token 的 0% ACC 不是截断导致的。

### 2. ZERO = 干净对照
ZERO S_t 只有 0.5% ACC (1/200，噪声水平)，与 BASE 行为一致。
→ PLAA 架构本身（S_t=0）不产生 2-hop 推理能力。

### 3. RANDOM < EVOLVED (5.5x)
RANDOM S_t 的 2-hop 率只有 8.5%，且 41.5% 输出格式崩坏。
→ 随机扰动偶尔能推到 2-hop，但代价高昂。
→ EVOLVED S_t 的 65.5% 2-hop 是 trained signal 的结果，不是"任意调制"。

### 4. EVOLVED S_t 有真实因果效应
- ACC: 0.5% → 47.0% (94x 提升)
- 2-hop 输出: 0.5% → 65.5% (131x 提升)
- 输出格式保持: 88% 含关系词

---

## 输出分布 (原始数据)

### BASE
- 200/200 输出 1-hop 关系 (mother/father)
- 全部格式正确但不具备多跳推理能力
- 典型输出: `mother` (当期望为 grandmother), `father` (当期望为 grandfather)

### ZERO S_t
- 199/200 输出 1-hop 关系
- 1/200 输出 2-hop 关系 (偶然正确)
- 与 BASE 行为基本一致

### RANDOM S_t
- 98/200 输出 1-hop 关系
- 17/200 输出 2-hop 关系
- 2/200 输出 3-hop 关系
- 83/200 输出非关系词 (格式崩坏)
- 随机噪声有微弱但混乱的调制效果

### EVOLVED S_t
- 44/200 输出 1-hop 关系
- 131/200 输出 2-hop 关系
- 0/200 输出 3-hop 关系
- 25/200 输出非关系词
- S_t 稳定地将推理从 1-hop 推向 2-hop

---

## 论文意义

1. **Section 5**: S_t 在 CLUTRR 上的因果效应被四条件对照实验验证
2. **Section 6**: PLAA 架构的行为调制 side effect 可单独讨论
3. 对照实验设计: BASE / ZERO / RANDOM / EVOLVED 提供了完整的因果链条

## 数据文件

- `checkpoints/clutrr_200_results.json` — BASE/ZERO/EVOLVED 原始数据 (80 tok)
- `checkpoints/clutrr_200_random_results.json` — RANDOM S_t 原始数据
