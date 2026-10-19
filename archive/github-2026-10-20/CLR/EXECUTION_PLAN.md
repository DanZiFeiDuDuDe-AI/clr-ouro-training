# CLR 本地实验修复执行计划

> 生成时间：2026-07-08 14:20 CST
> 生成者：奥尔德里克·沃斯（Aldrich Voss），House Eternal 管家
> 目标：修复本地 Phase 1 实验，让 S_t 的因果效应在 GTX 1650 上可测

---

## 背景

CLR（Continuous Latent Reasoning）的核心是在 transformer 层间注入持久状态向量 S_t。服务器端（96GB GPU）已完成验证：
- CLUTRR: BASE 0% → EVOLVED S_t 47%（对照 ZERO 0.5%, RANDOM 7.0%）
- GSM8K 跨轮: 去掉 S_t 后 ACC -2.4%, PPL +1.26

但本地实验（GTX 1650 4GB, Qwen3.5-0.8B 4-bit nf4, LoRA r=16）Phase 1 结果显示：
- with_S = 56.6% = without_S（完全相同）
- scale_proj 从未离开零初始化

Claude Fable 5 诊断发现**数据生成器 bug**是根因，而非 scale_proj 的 requires_grad 问题。

---

## 阶段 0：环境确认

### 0.1 确认 Python + CUDA 环境
- 项目路径：`C:\Users\Ouyang Quan\.openclaw\workspace\local_clr\`
- 模型路径：`local_clr/models/Qwen3.5-0.8B/`
- 需要确认的 Python 环境路径（待用户补充）
- 验证：`python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"`

### 0.2 确认依赖
- torch (CUDA 12.x), transformers, peft, bitsandbytes, accelerate
- 若 CUDA 不可用需先安装 torch 2.6.0+cu124 版本

---

## 阶段 1：数据生成器修复（Fable 5 诊断的核心）

### 1.1 生成器 bug 分析
`local_train_clutrr.py` 第 86-123 行的 `generate_trajectory` 有 bug：
1. `random.choice(list(COMP_TABLE.keys()))` 返回**元组**（如 `("father","father")`）而非关系字符串
2. `rels` 变成了元组列表 → `rels[0]` 是元组 → 后续 COMP_TABLE lookup 失败
3. 多跳答案落入 `random.choice(RELATIONS)` 纯噪声
4. 1-hop 标签退化为"从 prompt 抄 r1"
5. 理论天花板 ≈ 56.6%，恰好是观测值

### 1.2 修复方案
`local_clr/` 目录下已存在修复版：
- `clutrr_data.py` — 完整重写的生成器，含 FamilyTree 独立 solver、Bayes ceiling 计算
- `test_clutrr_gen.py` — 验证脚本

**需要做的事：**
1. `local_train_clutrr.py` 替换内联生成器为 `from clutrr_data import generate_trajectory, generate_dataset`
2. 删除已内联的重复 COMP_TABLE、RELATIONS、NAMES_M/F 等（在 `clutrr_data.py` 中已有同款定义，需确保不冲突）
3. 注意接口差异：新生成器返回 `{"turns": [{"prompt": ..., "answer": ..., "atomic": ...}], ...}`，旧代码期望 `{"turns": [{"question": ..., "answer": ...}]}`，需适配 `question` 字段名

### 1.3 验证
```bash
python3 test_clutrr_gen.py
```
应输出 Bayes ceiling 等信息。

---

## 阶段 2：训练脚本修复

### 2.1 PEFT 参数冻结修复（已有补丁）
第 308-311 行已有修复代码，但需要验证：
```python
# FIX: PEFT freezes all base_model params by default.
for idx in FULL_ATTN_LAYERS:
    for p in layers[idx].scale_proj.parameters():
        p.requires_grad = True
```

### 2.2 coh loss 权重调整
第 390 行：`loss = 0.3 * gen_loss + 0.7 * coh`
→ 改为 `loss = 0.95 * gen_loss + 0.05 * coh`

Fable 5 指出：coh loss 权重 0.7 压过任务损失，把状态训练成「自洽」而非「有用」。

### 2.3 eval 协议修复
`evaluate()` 函数（约第 222 行）中：
```python
s_val = torch.zeros_like(S) if ablate_s else S
for l in film_layers:
    l._s = s_val
```
这里用 pre-update S 做推理是**错误的**，应该用 S_next。修复：
1. 在 evaluate() 中，forward 前先 update S（用当前 `plaa_core(S, h_proj)` 得到 S_next）
2. 然后用 S_next 做 generation 推理

### 2.4 checkpoint 补全
当前只存 plaa_core state_dict（第 430 行附近）：
```python
torch.save({"step": step, "plaa_core": plaa_core.state_dict(), "eval_log": eval_log}, ...)
```
需要补充：
```python
+ "scale_proj": [l.scale_proj.state_dict() for l in film_layers],
+ "model": model.state_dict()  # 或 LoRA adapter
```

### 2.5 max_tokens 检查
第 38 行：`MAX_TOKENS = 128`
Fable 5 说服务器端确认 80 tok 就够了。但 128 应该没问题，保持。

---

## 阶段 3：无 LoRA 诊断跑

> 这是 Fable 5 排第一优先级的实验

修改训练模式：**只用 scale_proj + GRU 训练，不训练 LoRA。**

方案：
1. 在训练脚本中添加 `--no-lora` 参数或创建一个专用脚本
2. 去除 `get_peft_model()` 调用
3. 只保留 `scale_proj` 和 `plaa_core` 中的 GRU 作为可训练参数
4. 运行 300 步验证 with_S 和 without_S 的 delta

预期：如果 S_t 有真实因果效应，去掉 LoRA 旁路后 delta 应该清晰可见。

---

## 阶段 4：Learned-static Vs evolved S_t 基线

> Fable 5 认为这是论文最大漏洞

实验方案：
1. 训练两个模型：
   - **Evolved**：正常训练（每轮更新 S_t）
   - **Static**：训练结束后冻结 scale_proj，推理时对所有轮使用固定的 S_t（init_state 求平均）
2. 两者使用完全相同的训练数据和 epoch
3. 对比 ACC/ACC_last

如果 Static 也能到 40%+，则核心主张崩塌。
如果 Static 显著低于 Evolved（如 15% vs 47%），则主张成立。

---

## 阶段 5：寿命实验

合成延迟回忆任务：
1. 第 0 轮：注入一个事实关系
2. 第 k 轮：查询该关系（中间填充无关事实）
3. 扫 k=1, 2, 4, 8, 16
4. 画 ACC vs 延迟步数的曲线

---

## 阶段 6：容量实验

扫两个维度：
- DS（latent 维度）：{8, 16, 32, 64, 128}
- 需携带事实数：{1, 2, 4, 8}
画出容量曲面。

---

## 阶段 7：探针分析

对已有 checkpoint 的 S_t：
- 训练线性探针从 S_t 解码上一轮的关系
- 如果 2-hop 样本的 S_t 在第一跳后可以线性解码出中间关系，就证明「潜在状态携带推理」从修辞变成可测量的事实

这与 memory-probe 辅助损失共用基础设施。

---

## 文件索引

| 文件 | 当前状态 | 用途 |
|------|:--------:|:------|
| `local_train_clutrr.py` | 需重写 | 主的训练脚本，含 bug |
| `clutrr_data.py` | ✅ 已写好 | 修复版数据生成器 |
| `test_clutrr_gen.py` | ✅ 已写好 | 生成器验证脚本 |
| `diagnose_film.py` | ✅ 已有 | FiLM 参数诊断 |
| `local_train_clutrr_v2.py` | ❓ 待检查 | 可能为试验用脚本 |
| `models/Qwen3.5-0.8B/` | ✅ 已下载 | 模型权重 |
| `checkpoints/` | ✅ 已有 | 训练输出目录 |
| `EXECUTION_PLAN.md` | 当前文件 | 本执行计划 |

---

## 优先级总结

| 优先级 | 任务 | 估算时间 |
|:------:|:-----|:--------:|
| P0 | 环境确认 + 验证 CUDA | 5 min |
| P1 | 数据生成器修复 | 30 min |
| P2 | 训练脚本修复（coh loss/eval/ckpt） | 30 min |
| P3 | 无 LoRA 诊断跑 | 60 min |
| P4 | Learned-static Vs evolved 基线 | 120 min |
| P5 | 寿命/容量/探针实验 | 240 min |
