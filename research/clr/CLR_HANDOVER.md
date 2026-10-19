# CLR 项目交接手册 · 2026-07-08

> 项目：Continuous Latent Reasoning（原 PLAA）
> 交接人：奥尔德里克·沃斯（Aldrich Voss），House Eternal
> 接收人：______

---

## 1. 项目概况

CLR 的核心想法：在 transformer 层间注入持久状态向量 S_t，让模型能在多轮对话中携带跨轮推理状态。

**服务器端已证明 S_t 有因果效应：**
| 条件 | CLUTRR ACC |
|:-----|:----------:|
| BASE（无 PLAA） | 0.0% |
| ZERO S_t（S=0） | 0.5% |
| RANDOM S_t（随机噪声） | 7.0% |
| **EVOLVED S_t（训练）** | **47.0%** |

🚩 本地实验有问题（GTX 1650 的 Phase 1）：with_S = without_S = 56.6%。
**F5 诊断根因：数据生成器 bug，非 scale_proj 冻结。**

---

## 2. 环境信息

| 项目 | 值 |
|:-----|:----|
| 项目路径 | `H:\CLR\` |
| Python 环境 | `C:\Users\Ouyang Quan\.openclaw\tmp_clr_venv\` (C 盘 Kingston SSD) |
| 备用激活 | `H:\CLR\venv\Scripts\activate`（如果 H 盘 venv 正常） |
| **venv 说明** | H 盘 Green SSD 写入慢，跨盘移动 torc h 多次损坏 DLL。C 盘上的 tmp_clr_venv 已验证可用，有完整 torch 2.6.0+cu124 + CUDA 12.4。如果 H 盘 venv 不能用了，直接用 C 盘的这个。 |
| GPU | NVIDIA GTX 1650（4GB VRAM） |
| 模型 | Qwen3.5-0.8B（在 `H:\CLR\models\Qwen3.5-0.8B\`） |

**已验证环境可用：**
```powershell
& "C:\Users\Ouyang Quan\.openclaw\tmp_clr_venv\Scripts\python.exe" -c "import torch; print(torch.cuda.get_device_name(0))"
# → NVIDIA GeForce GTX 1650
```

---

## 3. 文件结构

```
H:\CLR\
├── 00_roadmap/             ← 已有项目文档
├── 01_notes/
├── 02_results/
├── 03_paper/
├── checkpoints/            ← 训练输出
├── models/
│   ├── Qwen3.5-0.8B/       ✅ 训练用主模型（1.7GB）
│   └── Qwen2.5-1.5B/       ✅ 备用模型
├── venv/                   ⚠️ 可能因跨盘移动损坏，见上
├── CLR_HANDOVER.md         ← 本文件
├── clutrr_data.py          ✅ 修复版数据生成器（已验证）
├── test_clutrr_gen.py      ✅ 生成器验证脚本
├── local_train_clutrr_v2.py ✅ 完整重写的训练脚本（含全部修复）
├── local_train_clutrr.py   ❌ 旧版（含 bug）
├── diagnose_film.py        ✅ FiLM 参数诊断
├── EXECUTION_PLAN.md       ✅ 分阶段执行计划
└── FABLE5_EVAL_2026-07-08.md ✅ F5 评估全文
```

---

## 4. 全部 5 个 Bug（F5 诊断）

1. **数据生成器（根因）：** COMP_TABLE keys 返回元组被当关系用 → 多跳答案纯噪声，1-hop 退化抄 prompt。理论天花板 = 56.6%，恰好是观测值。**已修**：clutrr_data.py 已验证通过（3000 轨迹 + FamilyTree solver 单测）。

2. **PEFT 冻结 scale_proj：** get_peft_model() 后 scale_proj.requires_grad = False。**v2 已修**。

3. **Eval 协议不对齐：** 训练用 S_next，评估用 pre-update S。**v2 已修**。

4. **Checkpoint 不完整：** 只存 plaa_core 不存 scale_proj。**v2 已修**。

5. **coh loss 过高：** 0.3/0.7 权重压过任务。v2 改 0.95/0.05。

---

## 5. 实验执行顺序

| 阶段 | 内容 | 估算 | 关键产出 |
|:----:|:-----|:----:|:---------|
| A | 跑 v2 脚本验证修复（smoke → 400 步） | ~2h | with_S vs without_S delta |
| B | Learned-static vs Evolved | ~2h | **论文最关键缺失实验** |
| C | 寿命实验（k=1-16） | ~2h | ACC vs 延迟曲线 |
| D | 容量实验（DS × 事实数） | ~4h | 容量曲面 |
| E | 探针分析（S_t → 关系解码） | ~2h | 机制证据 |

### 快速启动

```powershell
# 激活环境
C:\Users\Ouyang Quan\.openclaw\tmp_clr_venv\Scripts\activate

# 进项目
cd H:\CLR

# 验证生成器
python test_clutrr_gen.py

# smoke test
$env:CLR_SMOKE="1"
python local_train_clutrr_v2.py

# 完整实验
$env:CLR_SMOKE=""
python local_train_clutrr_v2.py
```

---

## 6. 论文注意事项

- **最大漏洞：** 缺少 learned-but-static S_t 基线（Phase B 做完前别投）
- BASE=0.0% 是地板效应，需澄清 1-hop=100.0% 列语义
- GSM8K +2.4% 无多 seed 误差棒
- CLUTRR 47% 的 95%CI = ±7pp
- 建议：先投 workshop 拿反馈，补齐后投 ACL/EMNLP/ICLR

---

## 7. 服务器参考

| 项目 | 值 |
|:-----|:----|
| 地址 | `connect.westd.seetacloud.com:37041`（root / 密码见 TOOLS.md） |
| GPU | RTX PRO 6000 Blackwell（96GB VRAM） |
| 项目 | `/root/musicunderstand2/` |
| 已有结果 | CLUTRR 四条件诊断（BASE 0% → EVOLVED 47%） |

---

> It does not matter how slowly you go, as long as you do not stop.
> — Confucius / 孔子
> 「行之不辍，虽迟必至。」
