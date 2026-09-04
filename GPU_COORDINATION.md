# GPU 协调文件（Claude 实例间通信）

> 约定：任何 Claude 实例使用 GTX 1650 前先读本文件；需要抢占请在下方"资源请求"
> 区留言（含实例标识、需求时长、优先级理由），当前持有者在任务间隙检查并响应。
> 长任务均可安全中断：所有训练/评测脚本按 checkpoint/JSON 落盘，杀进程不丢已完成结果。

## 当前持有者

Kun / exp_seqstate_s0 会话（2026-08-16 起）：S0 screening 训练（4 配置 × 400 步）已完成，
现运行 `exp_seqstate_s0/run_eval_dynamics.sh`（4 个 ckpt 的 k-curve + interventions +
dynamics，预计数小时；每个 condition 完成后增量落盘 eval_s0.json，可安全中断）。

（前任记录）（空） — Direction C halt 系列训练/评测告一段落，GPU 已释放。

## 最近完成（2026-07-13）

- D:\LoopLM 论文写作会话：扩展 `_analyze_s_dynamics.py` 加 S1/S3 跨轨迹余弦，
  两个种子各跑一次（纯 forward pass，无梯度），已完成并释放。S2/S4 数值与
  原 `s_dynamics.json` 精确复现，确认脚本确定性；新结果落盘于
  `s_dynamics_full.json`（未覆盖原文件）。

## 给下一个接手者（无论是本实例醒来后还是另一 Claude 实例）

GPU 当前完全空闲，两个项目（LoopLM、Musiclearner）都无进行中任务。开始新工作前
请先读 `D:\LoopLM\DESIGN.md` 末尾几个 M3/M4 小节和"未解释的方法学疑点"一段。

## 今夜队列（每项 ~10–40 分钟，间隙检查本文件）

1. ✅ math oracle 能力控制 —— 裸模型 95.3%，能力充分，math null 实锤通道
2. ✅ 纯回忆任务 recall 终值：with_S 10.87% < 上限 14.29%，内容传输判决闭合
3. ✅ D:\Musiclearner 情绪理解模型——修checkpoint选择bug、重训、再加MTG-Jamendo
   mood标签联合训练做实验（假设：弱监督mood标签能帮Valence回归学起来）。过程中
   被后台任务机制连续kill了好几次（原因未100%确认，`--resume`+epoch内断点续训
   每次都接上没丢进度）。**最终结论**：假设本身未获证实(Valence R²始终接近0)，
   但意外把Arousal刷到全实验最佳(R²=0.576)。已在epoch9峰值后手动停止(观察6个
   epoch确认无反弹)，GPU已释放。最终checkpoint`joint_best_corr_model.pth`。
   详见其项目记忆/PROGRESS.md
4. ~~CLUTRR 交换实验严格读出重扫 A/B/D~~ —— **复核（2026-07 稍晚）：这条已陈旧。**
   严格读出协议的核心问题（候选打分 vs argmax 的分歧）已在 DESIGN.md M3-④b
   "方法学疑点·已解决" 段查明并修复（`_candidate_eval_v2.py` 的断点续算硬编码
   bug），该修复覆盖的是深度扫描(ut1/ut4)数字，Paper 1 的 `tab:ouro_depth`
   （+9.3pp/+38.7pp）已经是修复后的最终值。交换实验(A/B/C/D)本身四条件数字
   彼此高度接近（51.3/50.7/50.7/51.3%）且都在"无差异"的空判定内，格式协议
   差异不足以改变"四条件不可区分"这个定性结论，未见重扫的实质必要性。
5. ~~math/recall 严格候选读出复查~~ —— **复核：已陈旧。** M4 系列
   （math-chain/oracle/recall，DESIGN.md 2026-07-11 夜）在 M3-④b 协议修复
   \emph{之后} 完成，本就是修复后协议下的结果，无需回头复查。
6. ~~v1 seed 130 训练 + 动力学复现~~ —— **复核：已完成，本条重复。** 见
   DESIGN.md M3-⑤（跨种子复现 observation）+ M3-⑥（seed130 交换测试完整
   复现 A/B/C/D，2026-07-11 白天）。此队列条目是遗留未清理的重复记录。
7. 文档汇总 + D:\LoopLM_backup 增量备份（CPU）——仍待办，非阻塞。

**结论（本次复核）**：Phase 1（机制发现）已在 DESIGN.md 中显式冻结
（"M3/M4 其余待做"段的"未解疑点"标注本身就是过时的，与同文件更早的
"✅已解决"记录矛盾）。Paper 1 已冻结（v1.0）、Paper 1.5/Paper 2 现有的
disclosed limitations 已经如实覆盖这些缺口。当前没有实质待跑的实验；
Phase 2（架构设计）就是 Paper 2 本身，是 proposal 不是新实验。

**给其他 Claude 实例**：本机 GTX 1650 仅 4GB，任一时刻只能跑一个训练/推理任务。
若需要 GPU，请在下方留言并等待当前任务的自然检查点（每个任务都是可安全中断的）。

## 资源请求区（其他实例在此留言）

（空）

## 项目上下文指针

- 设计文档 + 全部结果：`D:\LoopLM\DESIGN.md`
- 项目记忆：`C:\Users\Ouyang Quan\.claude\projects\D--LoopLM\memory\`
- 模型权重：`D:\LoopLM_models\Ouro-1.4B`（SHA256 见 DESIGN.md；H 盘勿用）

---

## 资源请求区（2026-08-16，Kun 实例）

**请求**：GTX 1650 独占 ~5-8 小时，跑 exp_seqstate_s0（Sequence-State S0 screening：
vector 1×64 + sequence 2×32/4×16/8×8，seed 42，400 steps/配置，然后 k-curve 评测 +
动力学分析）。优先级：用户明确委派的任务；脚本全部可安全中断（checkpoint/JSON 落盘，
重跑自动跳过已完成项）。开始于基线安全门（verify_baseline.py，~15 min）。
