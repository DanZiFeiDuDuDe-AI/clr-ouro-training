# Project *Nothing Is But What Is Not* (NIBWIN)

> "…and nothing is but what is not." — Macbeth, I.iii
> 命名（2026-07-11）：存在的内容被证明无效（交换/擦除不变），起效的是状态所
> 指向而非包含的东西。曾用名：PLAA → CLR → CLR×LoopLM。

## （原）CLR × LoopLM (Ouro) — M1 设计文档

日期：2026-07-10。目标：验证核心不等式 **LoopLM + CLR > LoopLM**。
基座：`H:\LoopLM\models\Ouro-1.4B`（已下载，含自定义建模代码）。

---

## 0. Ouro-1.4B 架构事实（读 `modeling_ouro.py` 确认）

| 项 | 值 | 代码位置 |
|:--|:--|:--|
| 结构 | 24 层共享权重栈 × `total_ut_steps=4` 次循环 | `OuroModel.forward` L592 |
| hidden / heads | 2048 / 16（head_dim 128） | config.json |
| 每次循环结束 | `norm(h)` → 存入 `hidden_states_list` → `early_exit_gate(h)` 存入 `gate_list` | L606–608 |
| halting | `early_exit_gate = nn.Linear(2048, 1)`，sigmoid λᵢ 组成几何式退出分布 | L513, L721–733 |
| 训练时 loss | labels 给定时用**退出概率加权的期望 logits**（各 UT step 的 lm_head 输出加权和） | L737–772 |
| 推理退出 | `exit_at_step` / `exit_threshold`（config 默认 1.0 = 实际不早退，走最后一步） | L689–806 |

### ⚠️ 移植陷阱（第一天就要防）

1. **`OuroModel.forward` 返回三元组** `(BaseModelOutputWithPast, hidden_states_list, gate_list)`，
   不是标准 output。直接调内层模型必须解包。
2. **KV cache 按 `ut_step × 24 + layer_idx` 编址**（96 个槽位）。训练与 CLUTRR 评测全部
   `use_cache=False`，完全绕开；只有 generate 时才需要它的 `UniversalTransformerCache`。
3. **checkpoint 坑（CLR 项目两次踩过）**：FiLM/GRU 权重独立 `torch.save`，
   每次评测前打印范数验证非零；照抄 `_run_baseline_v2.py` 的做法。
4. `FiLMWrapper` 的 tuple 分支有坑（tuple 时会返回被调制的输入而丢掉层输出）——
   `OuroDecoderLayer` 返回裸 tensor（L423），走的是正确分支，但**移植时加一行
   assert 非 tuple**，防未来 transformers 版本变化。
5. **`OuroModel.forward` 会读 `decoder_layer.attention_type`（L596）** ——
   FiLMWrapper 必须转发该属性（`self.attention_type = layer.attention_type`），
   否则 AttributeError（nn.Module 的 `__getattr__` 不转发普通属性）。
6. transformers 5.13 vs Ouro 官方要求 `<4.56`（README 明示，推荐 4.54.1）。
   **已打 5 个本地补丁**（`modeling_ouro.py`，trust_remote_code 按哈希重新缓存生效），
   冒烟测试（加载 + 前向 + generate）2026-07-10 全通过：
   1. `config.pad_token_id` → `getattr(config, "pad_token_id", None)`（5.x 不再自动补该属性）
   2. `OuroRotaryEmbedding.__init__`：`ROPE_INIT_FUNCTIONS` 5.x 删了 `"default"` 键 → 回退到自身方法
   3. 新增 `compute_default_rope_parameters` 方法（5.x 通用 `_init_weights` 对
      rope_type=="default" 硬编码调用此名；不看 rope_init_fn）
   4. `create_causal_mask` kwargs：`input_embeds`→`inputs_embeds`、删 `cache_position`（5.x 签名变更）
   5. `OuroRotaryEmbedding.forward` 自愈补丁：非持久 buffer `inv_freq` 在 5.x
      meta-device 加载路径下会丢失 → 缺失时现场重算注册
   备选方案（若再遇深层不兼容）：`pip install --target` 装 transformers 4.54.1
   到独立目录、脚本里 sys.path 优先。
   冒烟实测：加载 3.7s，VRAM 0.99GB；内层三元组 4+4 确认；基座对 1-hop CLUTRR
   直接输出 " Mother"（大写带空格——答案格式由训练塑造，与 0.8B 线相同）；
   exit-gate sigmoid `[0.0, 0.0, 0.0, 0.539]`（该输入下模型偏好满 4 loop）。

### 🧨 事故记录（2026-07-10）

`model.safetensors`（H 盘）**位级损坏**：31KB header 中恰好 8192 字节（两个 4KB 块）
被清零，header 尾部与文件大小完好。与交接手册记录的 H 盘绿 SSD 跨盘损坏前科一致。
处置：不修 header（数据区可能同样有暗伤，静默损坏权重不可接受），从 hf-mirror
重下并 SHA256 校验。官方哈希：
`58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`（2,869,336,434 字节）。
**规矩：今后凡放在 H 盘的权重文件，落盘后必须 SHA256 校验一次。**
仓库名注意：是 `ByteDance/Ouro-1.4B`，不是 ByteDance-Seed。

**第二次损坏（同日晚，升级为硬件结论）**：重下并校验过的文件在成功加载 5+ 次后、
seed-130 与 seed-520 两次运行之间的约半小时内再次损坏——**同一偏移（header 列
16377）、同一错误**。同一物理区域反复翻转 = H 盘坏区，非软件问题。处置：
- 模型整体迁至 `D:\LoopLM_models\Ouro-1.4B\`（含已打补丁的 modeling_ouro.py），
  权重重下 + SHA256 校验；`clr_ouro.py`/`_smoke_load_ouro.py` 的 MODEL_PATH 已改
- 全部 checkpoint、代码、DESIGN.md、CLR 02_results 已备份至 `D:\LoopLM_backup\`
- **建议用户尽快用 CrystalDiskInfo 查 H 盘 SMART（重分配扇区计数），
  H 盘不再存放任何未备份的产出**

---

## 1. 四个设计决策（定稿）

### ① S 在哪一步更新？—— 两档，都做，是核心消融

- **v0（per-turn，先跑通这个）**：每轮对话一次完整 4-loop 前向后更新 S。
  与已验证的 CLR 0.8B 协议**逐行同构**，唯一变化是骨干换成 Ouro。
  回答："骨干换成循环模型后 CLR 还成立吗？"
- **v1（per-loop）**：轮内每个 UT step 结束后用 `hidden_states_list[k]` 的
  最后非 pad 位置更新 S，下一个 UT step 的 FiLM 用新 S；跨轮照常携带。
  回答："状态演化粒度对齐计算深度是否更好？"
- **⚠️ 唯一通道原则（因果辨识的命门）**：不论 v0/v1，S 都必须保持
  **跨轮唯一信息通道**（每轮截断上下文）。loop 之间隐藏态本来就传递信息，
  轮内的 S 与残差流天然冗余——v1 若无增益，这就是预期解释（PLAA 4B 教训）。
  核心研究问题的锋利表述：**轮内计算加深时，跨轮持久状态的价值增还是减？**
  （对比 Ouro exit_at_step=0 vs 3 两档下的 CLR 增益即可回答，两个方向都可发表。）

### ② FiLM 注入位置

沿用已验证配置：**层 {3, 7, 11, 15, 19, 23}**（Ouro 恰好也是 24 层）。
包住 `OuroDecoderLayer`（调制层输入，乘性 `h·(1+0.1·tanh(W·S))`）。
注意：层权重跨 loop 共享 ⇒ FiLM 也天然作用于每个 UT step；
v1 模式下 S 逐 loop 变化 ⇒ **迭代间对称性由 S 打破**——这正是理论卖点：
共享权重的循环块想让第 k 次迭代不同于第 k+1 次，唯一途径是外部条件信号。

### ③ GRU 输入

`read_proj(hidden_states_list[k][0, last_non_pad_pos])`：
- v0 取 k=3（最后一个 UT step，即 `last_hidden_state`）；
- v1 每个 k 都喂一次。
last_non_pad_pos 一律用 attention_mask 求（CLR 修过的 bug，不再犯）。

### ④ 推理阶段 S 是否继续演化？

**是**。评测时 S 照常经 GRU 演化（与训练一致）；这是 CLR 全部干预实验
（zero / static / dose / swap）的操作对象。

---

## 2. 训练与实验配置（M1–M2）

| 项 | 值 | 依据 |
|:--|:--|:--|
| 骨干 | Ouro-1.4B，4-bit NF4，全冻结 | 1650 4GB 上限 |
| DS | **64**（不是 256——256 是 4B PLAA 配置） | 已验证 64 充分、16 不行（CI 已算） |
| 可训练 | read_proj + GRU + 6×scale_proj ≈ **0.95M** | 6×2048×64 + 2048×64 + GRU |
| 任务 | CLUTRR 多轮（`H:\CLR\clutrr_data.py` 原样复用） | 已验证生成器 |
| 训练 | 400 步，lr 3e-4，loss = 0.95·gen + 0.05·coh | 照抄 baseline v2 |
| 评测 | **两套都跑**：fast_eval（next-token argmax）+ 候选打分（格式无关）；t2p 为判别指标；每评测集 ≥150 轨迹（n≈9 的教训） | 扩展重评经验 |
| 统计 | Wilson + bootstrap CI，复用 `compute_cis.py` 模式 | 论文标准 |

### M2 四条件（照搬已验证协议）

① Ouro 裸基线　② Ouro+CLR（evolved）　③ Ouro+CLR（S=0）　④ Ouro+CLR（static S₀）
（能跑就加 dose-response α ∈ {0, 0.25, 0.5, 0.75, 1}）
**每个干预套件必须带阳性对照**（zero-ablation 预期显著掉分；不掉分 = 干预未生效，整批作废）。

### M3（成功后）

- v0 vs v1 消融（更新粒度）
- exit_at_step ∈ {0,1,2,3} × CLR 增益（"计算深度 × 状态价值"二维面）
- S → early_exit_gate（`gate(h) + gate_s(S)`，接口一行）：把探针里"可解码但不载重"的
  过程信息变成载重的
- S 跨 loop 轨迹分析（读 Solve the Loop, arXiv:2605.12466 后定方法）

---

## 3. 实现路线

不改 `modeling_ouro.py`。新建 `H:\LoopLM\clr_ouro.py`：
- `FiLMWrapper` / `PlaaCore` 从 `H:\CLR\local_train_clutrr_v2.py` 移植（DM=2048 + 陷阱④的 assert）；
- v0：直接调 `model.model(...)` 解包三元组，用 `last_hidden_state`，logits 过 `model.lm_head`；
- v1：把 `OuroModel.forward` L585–608 的循环体复制成外部函数
  `looped_forward_with_state(...)`（`use_cache=False` 下只需 embed → mask → rotary →
  双层 for 循环 → 每 UT step 后更新 S 并刷新 wrapper._s），完全透明、不猴补丁。

训练脚本 `_train_clr_ouro_v0.py` 以 `_run_baseline_v2.py` 为模板改（EVAL_SAMPLES 直接设大）。

---

## 3.5 实验结果记录

### M1（2026-07-10，seed 42，400 步，~30 分钟）✅

最终 150 项评测（Bayes ceiling 0.5667）：
- **with_S 78/150 = 52.0% [44.1, 59.8] vs without_S 0/150 [0.0, 2.5]，Δ = +52.0pp**
- 达 ceiling 的 91.8%（0.8B 线最高纪录为 84%）；中途评测 Δ：+47.4/+36.8/+47.4/+47.4pp
- **LoopLM + CLR > LoopLM 成立**。注意：best_model.pt 是 step-100 权重（后续 delta 持平未刷新）

### M2 干预套件（同日，step-100 ckpt，150 项/条件）✅

| 条件 | t2p | W95 |
|:--|:--:|:--:|
| A full | 48.7% (73) | [40.8, 56.6] |
| B zeroed | 0.0% | [0, 2.5] |
| C random | 0.0% | [0, 2.5] |
| D frozen S₀ | 0.0% | [0, 2.5] |
| dose 0.25 / 0.50 / 0.75 / 1.00 | 0 / 10.0 / 46.7 / 48.0% | — |

- **演化是效应全部来源**（frozen/random/zero 均为精确 0，比 35B 的 random=7% 更干净）
- **剂量反应：陡峭 S 形（阈值 + 饱和），无倒 U**——35B 的下降臂本就不显著（CI [−5,23]），
  三尺度诚实结论：类阈值模式翻转稳健，"过冲"仅 35B 单种子出现
- 一致性：dose 1.0 (72) ≈ A full (73)，差 1 项 = init 噪声 ✓
- t1 恒为 0：fast_eval 协议特性（turn-0 用 pre-update 状态），判别指标是 t2p

### M2 终表（2026-07-10 深夜，三种子 + 五对照全部完成）✅

| 种子 | with_S | without_S | Δ | ceiling | 占比 |
|:--:|:--:|:--:|:--:|:--:|:--:|
| 42  | 78/150 = 52.0% | 0/150 | +52.0pp | 0.5667 | 91.8% |
| 130 | 72/162 = 44.4% | 0/162 | +44.4pp | 0.5247 | 84.7% |
| 520 | 101/184 = 54.9% | 0/184 | +54.9pp | 0.5543 | **99.0%** |

- **三种子均值 Δ = +50.4pp，分层 bootstrap 95% CI [43.6, 56.7]**（`m2_three_seed_summary.json`）
- 合并计数：with_S 251/496 vs without_S **0/496**
- 平均达 Bayes ceiling 的 **91.8%**（0.8B 普通骨干线：均值 +33.0pp [25.9, 39.9]，77% of ceiling
  ——两条 CI 不重叠；⚠️ 1.4B vs 0.8B 规模混淆待 M3 exit_at_step 扫描切开）
- 对照全零：static-trained / frozen-eval / random / zeroed 均精确 0
- 剂量反应：阈值 + 饱和 S 形（0/0/10/46.7/48.0%），无倒 U

**M1 + M2 在开工首日完成。**

### M3-① 循环深度扫描（2026-07-10，训练时 total_ut_steps=k，seed 42，同一评测集）✅

| k | with_S t2p | W95 | FiLM 范数终值 |
|:--:|:--:|:--:|:--:|
| 1 | 32.0% (48/150) | [25.1, 39.8] | 7.0–7.6 |
| 2 | 48.0% (72/150) | [40.2, 55.9] | 5.6–5.8 |
| 3 | 48.7% (73/150) | [40.8, 56.6] | 4.7–4.9 |
| 4 | 52.0% (78/150) | [44.1, 59.8] | 4.5 |

（without_S 四档均 0/150；参数量恒定 954.9K；ceiling 0.5667）

- **k=1→2 跳升 +16pp（k=1 与 k≥2 的 CI 分离），k≥2 缓升趋饱和**：
  循环红利主要来自第一次额外迭代——"先读状态、再组合"需要 ≥2 遍
- **FiLM 范数随 k 单调下降**（7.1→5.6→4.8→4.5）：循环越深、调制所需力度越小，
  骨干迭代计算与外挂状态存在分工交换
- k=1（不循环）≈ 0.8B 普通骨干水平（32% vs 27.3%）⇒ **Ouro 对 0.8B 线的优势
  主要来自循环而非参数量**——规模混淆已切开，这是新论文的核心图

### M3-② v1 per-loop 粒度消融（2026-07-10）✅

- v1 最终 76/150 = **50.7%** [42.7, 58.6] vs v0 52.0% [44.1, 59.8] —— 更新粒度不影响准确率
- **Observed pattern（单种子观察，非已证机制）：循环内状态协议**。
  S 范数轨迹 [7.1, 7.9, 7.5, 0.66]。验证分析（`s_dynamics.json`，60 轨迹/119 轮）：
  - loop3→4 转移 cos=0.355±0.29、‖ΔS‖=7.31 —— 真表示变换，**非幅值缩放**
  - **反转**：大范数态跨轨迹 cos=1.0000（完全常量，零输入特异信息）；
    小范数 S₄ 跨轨迹 cos=0.307（内容载体）。"草稿纸"解读被证伪。
  - 范数 7.9≈√64 ⇒ GRU tanh 全饱和角点 = 二值常量旗标
  - 图景：**读在 loop1（FiLM 见上一轮 S₄）、写在 loop4、中间举常量旗标**——
    与 process-not-content 结论同构
  - 待做因果检验（observation→mechanism 的门槛）：跨轨迹交换 S₄（预测：行为改变）
    vs 交换 S₂（预测：无影响）；+多种子复现此模式
- checkpoint：`clr_ouro_v1_s42/`

### M3-③ S₄/S₂ 交换因果检验（2026-07-10 深夜）✅ —— 解释空间大幅收缩

| 条件 | t2p (150) |
|:--|:--:|
| A 正常 | 77/150 = 51.3% |
| B 换内容 S₄（同轮位、跨轨迹） | 76/150（无影响） |
| C 换载体（loops 2–4 注入 donor 中间态） | 76/150（无影响，如预期） |
| D 每轮入口清零（轮内 GRU 自行重建） | 77/150（与 A 相同） |

配套歧义消除（`donor_coincidence.json`）：donor 同轮位携带相同运行中关系的概率仅
**14.7%** ⇒ B≈A 不能用标签巧合解释。注意 A 的 51.3% < 无记忆最优 ceiling 56.7%。

**证据分级（定稿措辞）**

- **Strong evidence**：CLR 在 Ouro 上效果显著；效果源于状态演化；循环深度放大之；
  **当前协议下跨轮内容不是性能的必要条件**（"We found no evidence that cross-round
  persistence is causally required for CLUTRR under the current Ouro protocol"——
  范围限定：Ouro × CLUTRR × v1 协议 × 当前训练方式，不外推）。
- **Observation**：饱和常量大范数态（跨轨迹 cos=1.0）；末循环大表示转换
  （cos 0.355，非缩放）。
- **Hypothesis**：mode signal 而非 content memory；"任务未激活记忆需求（memoryless
  ceiling 过高）"与"模型学会了替代协议"目前不可区分。
  *Remains to be validated on tasks requiring stronger cross-round information transfer.*

**新研究问题**：0.8B 线有超无记忆证据（argmax-wrong 项上 22–39% 准确率），Ouro
未见——若坐实，"为什么两个骨干学出不同协议"本身即是下一篇的问题。
**任务设计方向**：构造 memoryless ceiling 低的多轮任务（P(答案|当前轮) 接近均匀）。

### M3-④ 候选打分（格式无关）评测（2026-07-10 深夜）✅ —— 剧本 A，重大解读修正

| 条件 | t2p (150) | t1 |
|:--|:--:|:--:|
| with_S | 30/150 = 20.0% [14.4, 27.1] | 96% |
| without_S | 17/150 = **11.3%** [7.2, 17.4] | 96% |

- **fast_eval 的 +52pp 主体是格式合规效应**：置零模型的先验一直在（11.3%≠0），
  t1 两条件都是 96%——它只是发不出训练格式的 token
- **格式无关的真实状态增益：+8.7pp（勉强显著）**
- **骨干协议差异成为头号科学问题**：0.8B 的 B2 本就是候选打分协议
  （score_candidates），其 +33pp 是格式无关真效应 ⇒ 同一 CLR，0.8B 格式无关
  +33pp vs Ouro +8.7pp。此前"循环骨干更会用状态"基于格式敏感读出，方向存疑
- **两个读出协议坑（全线适用，含 0.8B fast_eval 类数字）**：
  ① fast_eval 只比对答案首 token ⇒ grand-* 四关系共享前缀，系统性宽容充水
  （0.8B 关系级分解中 grand-* 恰为最强类，可疑）；
  ② 候选带前导空格 vs 训练目标无空格——读出必须匹配训练格式
- M3-① 深度扫描（32→52%）需在格式无关读出下重测 —— 见下方审计 v2

### M3-④b 审计 v2 终值（无空格=匹配训练格式，n=100，`candidate_eval_v2.json`）✅

| | with_S | without_S | 真实效应 |
|:--:|:--:|:--:|:--:|
| ut4 | 44% [34.7, 53.8] | 10% [5.5, 17.4] | **+34pp** |
| ut1 | 19% [12.5, 27.8] | 10% [5.5, 17.4] | **+9pp** |

- **最终和解**：fast_eval +52pp = ~34pp 真效应 + ~18pp 格式/首token宽容；
  带空格读出的 +8.7pp 是空格失配伪影（惩罚了训练过的模型）
- **骨干协议差异消解**：0.8B +33pp ≈ Ouro +34pp（同为候选打分）——两骨干
  真实状态效应一致
- **深度效应在严格读出下存活且清晰**：+9pp（k=1）→ +34pp（k=4），
  without_S 地板恒为 10%——循环深度放大状态可用增益 ~4×，核心图成立
  （数值后于白天修正，见下方"方法学疑点·已解决"）

✅ **方法学疑点·已解决（2026-07-11 深夜发现，白天核查并解决）**：昨晚同一 ut4
checkpoint 上，候选打分（n=100）给出 44%，fast_eval argmax（n=150）给出 52%，
判断为"数学上不该发生"（候选打分是 argmax 的宽松子问题，理论上应 ≥）。

核查发现真相分两层：
1. **44% 的 n=100 CI 是 [34.7, 53.8]，本就包含 52%**——所谓"矛盾"从统计上看
   一开始就没那么显著，是小样本宽区间被过度解读了。
2. **重新测量时发现一个真实 bug**：`_candidate_eval_v2.py` 里有一段去年
   OOM 崩溃后写的"断点续算"逻辑，把 `ut4_with_S` 硬编码为旧的 44/100 结果并
   跳过重算——今天改 `N_TRAJ` 时忘了删这段，导致第一次"对齐重测"其实只有
   3/4 条件是新数据，`ut4_with_S` 还是陈年旧值。已删除该硬编码跳过逻辑。

**修复后、完整对齐同一批150项数据的最终结果**（`candidate_eval_v2.json`）：

| Checkpoint | with_S | without_S | Δ |
|:--|:--:|:--:|:--:|
| ut4（4圈） | 46.67% [38.87, 54.63] | 8.0% [4.64, 13.46] | **+38.67pp** |
| ut1（1圈） | 17.33% [12.11, 24.19] | 8.0% [4.64, 13.46] | **+9.33pp** |

52.0%（argmax）落在 46.67% 的置信区间内——两种读出协议在完整样本量、干净数据
下测的是同一个真实效应量，差异是抽样噪声，不是方法学矛盾。**深度扫描结论不但
保住，数字还略微加固**（ut4 delta 从旧的 +34pp 修正为 +38.67pp，ut1 从 +9pp
修正为 +9.33pp，量级和方向都稳定；**Paper 1 可以直接使用这批数字**）。

**方法学第三课（继"验证干预已生效""报告效应前先审计读出协议"之后）：
处理"断点续算"式的容错代码时，参数一变就要重新检查所有硬编码的跳过/占位逻辑
是否还适用——否则容错机制本身会变成陈旧数据混入新结果的暗门。**

### M4 内容传输判决系列（2026-07-11 夜）

**M4-① math-chain（上限 0.1975，319 t2p 项）**：with_S 35/319 = **11.0%**
[8.0, 14.9] ≈ 乱猜（10%）；without_S 0。未达无记忆上限。
**M4-② oracle 能力控制（150 题，信息全给）**：裸 Ouro 带空格候选 **95.3%**
[90.7, 97.7] —— 能力充分，排除"模型不会算术"的借口。
⇒ **组合结论：骨干会算 + 通道可解码 + 记忆版=乱猜 ⇒ 状态通道未传输内容，
且不是能力问题**。process-not-content 三重验证闭合（0.8B 探针 / Ouro 交换 /
math+oracle）。
**M4-③ 纯回忆任务终值（2026-07-11 深夜）**：ceiling=0.1429，with_S=**35/322=10.87%**
[7.9, 14.7]——**低于任务自身无记忆上限，与乱猜(10%)不可区分**。零算术、零组合的
verbatim 搬运都失败。

**判决**：store-while-compute 假说被否——不是"边存边算不行"，是通道在当前
Ouro+CLR 配置下**完全不能承载任何内容**，连最简单的逐字搬运都不能。三任务组合
证据链闭合：
- math-chain: with_S 11.0% vs 上限 19.75%/乱猜 10% —— 未过线
- recall: with_S 10.87% vs 上限 14.29%/乱猜 10% —— 未过线（更彻底）
- oracle: 裸模型 95.3% —— 排除"骨干不会做"
⇒ process-not-content 从"最可能解释"升级为"数据逼出的唯一解释"（在此
架构/任务/训练配置范围内）。

### M3-⑤ v1 seed 130 跨种子复现（2026-07-11 深夜）✅ —— 观察升级为"两种子复现观察"

**准确率**：73/162=45.06% [37.6,52.7] vs seed42 76/150=50.7% [42.7,58.6]，CI 大幅重叠，
v0≈v1 结论跨种子稳定。

**动力学逐项对照**（`clr_ouro_v1_s130/s_dynamics.json`）：

| 指标 | seed 42 | seed 130 |
|:--|:--:|:--:|
| loop1-4 范数 | 7.1/7.9/7.55/0.66 | 6.96/7.74/7.46/0.76 |
| 连续循环 cos (c12/c23/c34) | 0.998/0.993/0.355 | 0.996/0.997/0.347 |
| 跨轨迹 cos S4(压缩态) | 0.307 | 0.250 |
| 跨轨迹 cos S2(载体态) | **1.0000** | **1.0000** |

**关键复现**：S2(载体态) 跨轨迹余弦在两个独立种子上都精确到 1.0000——两次独立训练
都学出了完全跨输入恒定的常量旗标态，不太可能是巧合，更像是该架构的训练吸引子。
loop3→4 的真表示变换（非缩放）也复现（cos≈0.35 两次一致）。

**证据分级更新**：此发现从"单种子 observation"升级为"**两种子复现的
observation**"——仍未到"mechanism"（需交换实验的 S4/S2 因果不对称也在新种子上
复现），但强化显著。下一步（留待白天/有人看着时做）：在 seed130 checkpoint 上
重跑 S4/S2 交换因果检验，确认"内容可互换、只有载体缺失才致命"是否也复现。

### M3-⑥ seed130 交换测试复现 + Bootstrap/Ignition Hypothesis（白天，用户在场）✅

**seed130 交换测试（A/B/C/D）完整复现 seed42**：

| 条件 | seed42 | seed130 |
|:--|:--:|:--:|
| A 正常 | 51.3% | 49.3% |
| B 换内容 S₄ | 50.7% | 50.0% |
| C 换载体 | 50.7% | 50.7% |
| D 清空入口 | 51.3% | 50.0% |

四条件在两个独立种子上都统计不可区分——"当前架构+任务下跨轮内容非因果必要"这一
结论完成跨种子复现。证据分级：从"两种子复现的 observation"升级为独立训练+重复
出现载体+重复出现同样因果关系的三重支撑。

**关键矛盾的解决（M2 zeroed=0% vs 交换测试 D=100%，此前未明确说明）**：查代码确认，
两者的"zero"是完全不同的操作。M2 的 v0 协议里 S 是整个 turn 全部4圈共用的固定值，
`zeroed` 意味着 FiLM 在该 turn 从头到尾被禁用（identity gate），永无重建机会。
v1 协议的交换测试 D 只清零 **入口**（loop1 的 FiLM 输入），之后 GRU 用 loop1 真实
隐藏态反馈重建非零 S，loop2-4 正常。**"zero"这个词本身是歧义的——必须区分
"整个 turn 持续为零"（灾难性）和"只是入口为零、后续圈次能自然重建"（无影响）**。

**逐圈遮蔽矩阵（关键证伪实验，seed42，`loop_zero_matrix.json`）**：

```
                连续自然演化
                    ↑
D:  0 → S → S → S       51.3%   ✓
I:  0 → 0 → S → S       44.0%   ✓
G:  S → 0 → 0 → 0        0.0%   ✗
H:  0 → 0 → 0 → S        0.0%   ✗
E:  0 → 0 → 0 → 0        0.0%   ✗
```

| 条件 | loop1 | loop2 | loop3 | loop4 | t2p | 95% CI |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| A 正常   | 真实 | 真实 | 真实 | 真实 | 51.3% | — |
| D        | 0    | 自建 | 自建 | 自建 | 51.3% | — |
| **I**    | 0    | 0    | 自建 | 自建 | **44.0%** | [36.3, 52.0] |
| **E**    | 0    | 0    | 0    | 0    | **0.0%**  | [0, 2.5] |
| **G**    | 真实 | 0    | 0    | 0    | **0.0%**  | [0, 2.5] |
| **H**    | 0    | 0    | 0    | 注入真实 | **0.0%** | [0, 2.5] |

这套矩阵不是实验堆砌，是一个针对三个递进问题的最小充分设计：

| # | 问题 | 排除依据 | 结论 |
|:-:|:--|:--|:--|
| 1 | S 是不是存储的语义内容？ | 交换测试 B（换 S₄）无影响 | 不像 memory |
| 2 | S 是不是只需要出现一次（脉冲）？ | G/H：单圈真实信号（无论早晚）= 崩溃 | 不像 one-shot trigger |
| 3 | S 需要什么？ | I 对比 E/G/H：连续两圈自然递归 = 基本满血 | 需要一段连续的自举轨迹 |

**Rejected hypotheses（被排除的替代解释）**：

| 假说 | 排除依据 |
|:--|:--|
| H1: S 是语义记忆内容 | S₄ 交换不变性（交换测试 B/C，两种子复现） |
| H2: S 只需一次点火脉冲 | G、H 均崩溃（早期/晚期单脉冲皆不够） |
| H3: 只有最后一圈的输入起决定作用 | H（loop4 显式注入干净原始信号）仍崩溃 |
| H4: 任意非零 S 都足够 | E（全零，无重建机会）与 I/D（有重建机会）对照 |

**支持的假说（措辞校准版）**：

> S appears to act as a recurrent control state whose effective operation
> requires a short continuous bootstrap trajectory, rather than any single
> timestep's latent content.
>
> S 更像一个循环控制状态，其有效作用依赖于连续若干步的状态重建过程，
> 而非单个时间点的 latent 内容。

这个表述比早期"FiLM 只依赖 S 是否饱和"的说法更贴合数据——落脚点在**过程**
（连续递归重建）而非**状态的某个静态属性**（是否饱和）。类比系统辨识里的
steady-state / transient response / settling time：真正被发现的不是一个
latent 向量，而是一个**收敛过程**，且这个过程有一个可测的最小时间尺度
（至少两圈）。

**证据地位**：跨种子复现（seed42/130 全部四条件一致）+ 成功预测未测试的
干预结果（E 的 0.0000% 精确对上 M2 地板，是用户在结果出来前给出的具体
数值预测）。这是"observation → replication → prediction confirmed"的
完整三级跃迁。论文措辞仍保持 consistent with / suggests，不用
proves / establishes / demonstrates。

**Phase 1（机制发现）在此冻结**。矩阵设计到 A/D/E/G/H/I 六条件为止，
不再拆分更细的 loop 组合——继续切实验空间边际收益递减，且審稿人容易追问
"为什么选这些干预"，现在这套设计（正常/输入扰动/永久破坏/早期保留/晚期
保留/最小恢复长度）足够自洽完整。整理为论文图（候选：Figure 3/4）。

**明确未验证、留给 Phase 2 的问题（不是本阶段的缺陷，是下一阶段的起点）**：
- S₄ 跨轨迹变化是否是 process marker（呼应 0.8B 探针的 hop-index 可解码
  发现）——需要专门的线性探针，而非本阶段的干预实验能回答
- "连续两圈"阈值是否具体任务/checkpoint 特例——未跨种子验证
- 连续性本身是否必要，还是仅需自然递归过程（区别于显式注入）——I 用的是
  自然重建，未测"连续两圈但显式注入"的对照

**Phase 2 的问题已经变了**：不再是"S 是什么"，而是"既然 S 是一个需要短暂
递归稳定化的控制状态，能否设计更适配这种状态动力学的架构"——从分析转向
设计。

M3/M4 其余待做（非本阶段优先）：CLUTRR 交换严格读出重扫（A/B/D）；math
严格读出复查；S→halting gate；候选打分 vs argmax 的未解疑点（见上方
"未解释的方法学疑点"一节，仍未查）。

## 4. 风险

| 风险 | 预案 |
|:--|:--|
| 1650 上 4-loop 训练太慢（等效 ~5.6B 前向） | 先 smoke 4 步计时；慢则减层内 FiLM 数/步数，或 exit_at_step=1 训练 |
| transformers 5.13 与 4.55 API 不兼容 | 冒烟测试见分晓；不行则在新 venv 装 4.55 |
| v1 无增益 | 预期内结果（唯一通道原则），照发：粒度消融本身是贡献 |
| 4-bit 量化噪声淹没小效应 | CLUTRR 效应量大（+27~40pp），无虞 |

## 5. Phase 3 / Phase 4（未来方向，存档，未开工）

用户与 GPT 讨论后归档（非本仓库/非 Claude 实例产出，用户转述），记录路线以便未来
需要时有起点，**当前明确不动手，不做任何实现或实验**。此节曾把架构重构阶段误标
为"Phase 4"，用户随后澄清编号，本节已更正。

**四阶段路线（更正版）**：

```
Phase 1 → 发现现象：Persistent Latent State S_t 有因果作用（已完成，即当前 M1-M4 系列实验）
Phase 2 → 论文化验证：Paper 1 / Paper 1.5 / Paper 2（已完稿）
Phase 3 → 架构重构 + 系统化实现：训练一个真正原生的 controller-centric 模型
           （不是外挂模块事后证明有效，而是从头把 S_t 设计进架构）
Phase 4 → 规模化验证：接近生产级模型
```

**Phase 3 与 Paper 2 的区别**：Paper 2 的三个 Direction（A 显式状态机 / B 学习转移
过程 / C 自适应循环预算）都是"给现有 GRU+FiLM 管线打补丁"式的验证性实验设计
（\S sec:experiments 已经写清楚每个方向要测什么才算数）。Phase 3 是更大的跳跃：
不再是"改造外挂模块"，而是把 controller 做成模型计算图的原生部分：

```
现在（外挂）：
Transformer backbone → hidden state → GRU controller → S_t → FiLM injection

Phase 3（原生，构想）：
          Controller Layer
              |
      +-------+-------+
      |               |
 attention       latent dynamics
      |               |
      +-------+-------+
              |
        Transformer blocks
```

**Phase 3 的三个研究支柱**（用户转述）：
1. **外挂 → 原生**：controller 不再是插件，是模型计算图的一部分。
2. **固定深度 → 自适应计算**：传统 CoT 是"生成固定长度 token 序列再答"，CLR
   设想是"latent evolution 直到模型自己决定想够了"（$halt=g(S_t)$）——研究
   什么时候该停止更新 S、简单/难题各需要几步 latent evolution、能否动态分配
   FLOPs。这条与 Paper 2 Direction C（自适应循环预算）同源但范围更大：
   Direction C 只是在现有 Ouro halting gate 上做条件化验证，Phase 3 是把这个
   机制设计进架构本身。
3. **memory → computation state**：核心问题"如果模型拥有一个持续演化的内部
   计算状态，能否替代部分 CoT"——是 Paper 1.5"controller-centric 而非
   memory-register"发现（\S discussion-control 新增的三性质综合段）的直接
   延伸，也是 Phase 3 存在的理由：Paper 1.5 只是观察到这个性质，Phase 3 才是
   真的按这个性质去设计。

**Phase 3 实验目标相对 Phase 1/2 的升级**：Phase 1 问"有没有用"（ACC Δ /
causal intervention）；Phase 2（即三篇论文）问"它是什么"（probe / dynamics /
attractor / capacity）；Phase 3 问"能否成为新架构的核心计算机制"（scaling /
compute efficiency / long-context / reasoning length / adaptive inference）。
**Phase 4**（规模化验证）在此之后，本轮讨论未展开细节，仅记录路线位置。

**核心结构直觉**（用户原话，转述自 GPT 对话）：现在的做法相当于"一小段 CoT +
一个向量"，用户认为下一步不是让模型吐更长的 CoT，而是给它一个持续演化的隐空间
工作状态（context window 因此更"耐用"，因为模型记住的不再全靠 token 序列）。
粗糙形式：$S_{t+1} = f(S_t, h_t)$，但 $f$ 不只是存信息，还决定：下一步注意力
该看哪里、是否继续推理、当前该用多深的计算、哪些 token/latent 路径该被强化、
如何动态分配算力。

**候选标题**（讨论中出现，均未定稿）："Towards Controller-Centric Architectures
for Persistent Latent Reasoning" / "Beyond Chain-of-Thought: Persistent Latent
Controllers for Adaptive Computation"。**注意**：前者与 Paper 2 现有标题
"Towards Controller-Centric Architectures for Persistent Latent State: A Design
Proposal" 高度重合，若 Phase 3 真的开工写论文，标题需要与 Paper 2 明确区分开
（Paper 2 是 proposal，Phase 3 若成文应该是 report，避免读者混淆两篇的证据地位）。

**动机来源的备注（保留原话精神）**：用户明确表示这条路线不是从"AGI"这类目标
自上而下倒推出来的，而是从一个具体的结构瓶颈（CoT 浪费算力、context window
记不住）自下而上摸出来的——这条自下而上的研究品味目前看来是一贯的，值得在
后续判断 Phase 3 是否值得开工时作为参照。

**开工前置条件（未满足，任一满足前不应擅自开工）**：用户明确决定开始 Phase 3；
Paper 1/1.5/2 已投稿或用户认为无需再改动；一个具体的、比 Paper 2 三个 Direction
更小的原型范围（例如先在 Ouro 上验证 halting-gate 条件化这一个子问题，而非直接
上马完整原生架构）。
