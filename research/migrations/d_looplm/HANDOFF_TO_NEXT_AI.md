# HANDOFF_TO_NEXT_AI.md — LoopLM monopaper P0 当前交接（2026-08-25 午）

> 交接给下一位 AI/agent。先读本文件，再读 `HANDOFF.md`、`MONOPAPER_STATUS.md`、`P0_SUBSTRATE_DESIGN.md`。
> 本地工作目录：`D:\LoopLM`。远端训练机：cheems `D:\training`。
> 本文件记录的是 0H 蒸馏阶段最新状态，包含 2026-08-24 深夜 ~ 2026-08-25 凌晨完成的关键实验。

---

## 1. 项目目标

回答：**persistent computation 能否从外挂 GRU–FiLM 内化进 recurrent substrate？**
P0 在 Ouro-1.4B 上做 retention hybrid（替换 6 层 `{3,7,11,15,19,23}` 的 softmax mixing 为 linear retention，γ 唯一新增参数）。
核心实验矩阵：A/B（历史基线）、C-reset/C-persist（internal retention 跨 turn）、D-reset/D-persist（internal + external CLR）。

## 2. 阶段总览

| 阶段 | 状态 |
|---|---|
| 0A 审计 | ✅ |
| 0B 复现 | ✅ v1_s42 = 76/150, without_S=0 |
| 0C 选型 | ✅ retention + 6-layer hybrid + 6项修订 |
| 0D 迁移设计 | ✅ exact_copy ~100%，γ 唯一新参数 |
| 0E 实现 | ✅ retention_ouro.py forward 跑通，Amendment 7 通过 |
| 0F 迁移验证 | ✅ 269 tensor max_abs_diff==0，γ 96 params |
| 0G pre-distill eval | ✅ M1 受损未崩（logit cos 0.77，NLL+15.2） |
| 0H 蒸馏 | 🟡 基本完成，结论见下 |
| 0I C-reset/C-persist | ☐ 未开始（下一步） |
| 0J D 条件 | ☐ 未开始 |
| 0K 因果干预 | ☐ 未开始 |
| 0L 分析 | ☐ 未开始 |

## 3. 0H 最新结果（canonical matched_eval_r2，seed=42，N=30）

`D:/training/matched_eval_results_r2.json`（远端，待拉回本地）

| Model | candidate acc | hidden cos | logit cos | ΔNLL |
|---|---:|---:|---:|---:|
| M0 teacher | 1/30=.033 | — | — | — |
| M1 migration only | 3/30=.100 | .265 | .749 | +14.52 |
| M2_d1 γ300 old frozen | 3/30=.100 | .265 | .749 | +14.52 |
| M2_d2 norm600 γ frozen | 0/30 | .562 | .847 | +6.77 |
| M2_d3 norm1800 γ frozen | 0/30 | .607 | .886 | +5.12 |
| γ-only r2 (fp32) 600 | 3/30=.100 | .271 | .747 | +14.27 |
| γ+norm r2 (fp32) 600 | 1/30=.033 | .488 | .825 | +7.93 |

## 4. 关键结论（已经钉死，不可漂移）

1. **γ 参数化问题**：原始 `γ=sigmoid(logit)` 在 multi-scale init 下高 head 饱和，γ 梯度消失（12/16 head 不动）。r1 的 clamped_direct 又因为 γ 参数被 `.to(bf16)` 精度太低，仍不动。**r2 修复：clamped_direct 下 γ 参数保持 fp32，clamp 到 `[1e-6, 1-1e-6]`。**
2. **γ-only（fp32 可训）≈ M1**：单独 retune retention timescale 不能修复 block distillation objective。这不是 sigmoid 饱和或低精度造成的假象。
3. **post-norm adaptation 是主要修复路径**：norm-only 600 从 .749→.847，续到 1800 → .886。
4. **γ+norm r2 在 600 步、canonical seed 下未超过 norm-only 600**（.825 vs .847），加训 γ 未显示协同增益。但 γ+norm r2 训练 loss 仍在下降，所以此结论目前限于 600 步/D1+D3 objective/30 traj。
5. **多指标验收原则**：task behavior > logit distribution > final hidden similarity。hidden cosine 是诊断指标，不是 acceptance criterion。
6. **下一步分叉（用户钉死）**：
   - 若 γ+norm 能到 .90+ 且 NLL 明显降 → 冻结 M2，进入 0I。
   - 若 plateau .86~.89 → H2 minimal projection adaptation：LoRA rank 4/8，只动 converted Q/K/V/O，very low LR，不要扩大 norm 参数范围。
   - 不要无脑堆步数追 0.9；不要追 hidden cosine 0.9。

## 5. 远端 cheems 状态（截至 2026-08-25 凌晨）

- 训练机：RTX 4060 8GB，Python 3.13.15，torch 2.13.0+cu126，transformers 4.54.1，bnb 0.50.1。
- SSH：paramiko 密码认证，用户 `ai training` / 密码 `REDACTED_CONFIGURE_ENV`（详见 `CHEEMS_CONNECTION.md`）。
- 工作目录：`D:\training`（原 canonical 工作区）；另有 `D:\training_marvis`（另一个 agent 的镜像工作区，实验结果设置相同但路径不同，不要混用）。
- 远端已有 checkpoint 目录：
  - `D:\training\checkpoints\ouro_retention_m1`
  - `..._m2_d1`（γ-only old，γ 冻结）
  - `..._m2_d2`（norm600，γ 冻结）
  - `..._m2_d3`（norm1800，γ 冻结）
  - `..._m2_gammafix`（r1 γ-only，bf16 γ，已作废/诊断用）
  - `..._m2_gammafix_norm`（r1 γ+norm，bf16 γ，已作废/诊断用）
  - `..._m2_gammafix_r2`（r2 γ-only fp32，600步）
  - `..._m2_gammafix_norm_r2`（r2 γ+norm fp32，600步）
- 远端结果文件：
  - `D:/training/matched_eval_results.json`（r1，unseeded，仅参考）
  - `D:/training/matched_eval_results_r2.json`（canonical）
  - `D:/training/p0h_post_distill_eval_gammafix_r2_gamma.json`
  - `D:/training/p0h_post_distill_eval_gammafix_r2_norm.json`
  - `D:/training/checkpoints/ouro_retention_m2_gammafix_norm_r2/gamma_diagnostics.json`（per-head γ/θ/grad/boundary/timescale）

## 6. 待办（下一位 AI 接手后）

1. **先把远端权重和当前数据拉到本地**（用户 2026-08-25 明确要求）。本地执行器（PowerShell）此前反复 `PTY shell exited during startup`，拉取尚未完成。
   - 拉取内容：`D:\training\checkpoints\**`（全部 checkpoint，小文件）、`matched_eval_results_r2.json`、两个 gammafix eval json、`gamma_diagnostics.json`、`teacher_cache_manifest.json`、`weight_migration_manifest.json`。
   - 本地已有拉取脚本：`D:/LoopLM/_fetch_results.py`（可改成 r2 路径），或直接用 paramiko SFTP。
2. **检查 cheems 是否可达**：`tailscale status` / `ping cheems`。若 Tailscale 断，先等用户重连。
3. **不要关机**：用户已取消自动关机。确认本地没有 `_auto_shutdown_after_r2.py` 残留进程（当前已无）。
4. **分析 γ+norm r2 的 gamma_diagnostics.json**：per-head γ 移动量、boundary count、timescale τ。
5. **根据第 4 节分叉决定下一步**：
   - 如果用户/证据支持 γ+norm 多训几步（比如 1200）再看是否 > norm-only 1800，可继续小范围测试；
   - 否则进入 H2 LoRA（rank 4/8, Q/K/V/O only, low LR）。
6. **0I 准备**：在 M2（蒸馏后 student）上训练 C-reset/C-persist。需要 forward API 支持 `recurrent_state=None → logits, new_state`（retention_ouro 已具备底层 API，模型级 API 尚未封装）。

## 7. 本地关键文件

- 训练/评估脚本：
  - `D:/LoopLM/distill.py`（0H 第一版，γ only old）
  - `D:/LoopLM/distill_v2.py`（norm unfreeze）
  - `D:/LoopLM/distill_v3.py`（norm continuation，保存中间 ckpt）
  - `D:/LoopLM/distill_gammafix.py`（γ/γ+norm fixed，带 per-head 诊断）
  - `D:/LoopLM/post_distill_eval*.py`、`D:/LoopLM/matched_eval.py`（canonical 统一评测）
  - `D:/LoopLM/_auto_gammafix_r2.py`（r2 队列启动器，已完成）
  - `D:/LoopLM/_fetch_results.py`（远端结果拉取）
- 核心模块：`D:/LoopLM/retention_ouro.py`（**注意**：clamped_direct 模式 γ 参数保持 fp32，clamp 1e-6~1-1e-6；sigmoid_logit 模式保持旧行为兼容旧 ckpt）
- 数据：`D:/LoopLM/clutrr_data.py`、`D:/LoopLM/teacher_cache` 本地没有，teacher_cache 在远端 `D:\training\teacher_cache`（300 轨迹/608 turns，D1+D2+D3 cache）

## 8. 纪律提醒（来自用户/项目）

- 历史数字不得静默修改（§2）。
- candidate-scoring 作主结果，fast_eval 仅 debug（§30）。
- 每套干预带阳性对照。
- 实验目标不是证明 retention 更好，是判断 persistent computation 最合理的 substrate（§52）。失败也是结果。
- γ-only weakness 不是 artifact（r2 fp32 已排除 sigmoid 饱和/低精度假冻结）。
- 表述要严谨：当前能说的是“γ alone 在当前 distillation objective 下不能 substantially recover teacher function”，不能说“初始 multi-scale decay 已最优”。
