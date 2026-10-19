# HANDOFF.md — Monopaper P0 阶段交接文档

> 交接日期：2026-08-24。项目：Ouro-1.4B → Internal Recurrent Substrate 研究。
> 交接人：Claude（P0 接手实例）。下任接手者读此文档 + `MONOPAPER_STATUS.md` + 各阶段产出的 .md。

---

## 1. 项目目标（§0）

回答：**persistent computation 能否从外挂 GRU–FiLM 内化进 recurrent substrate？**

本轮从 Ouro-1.4B 起，做 A/B/C/D 实验矩阵（§7/§8），核心对比 **C-reset vs C-persist**——internal cross-turn persistence 是否产生额外 causal effect。

---

## 2. 总进度（Phase 0A–0L）

| 阶段 | 状态 | 关键产出 |
|---|---|---|
| 0A 审计 | ✅ | `OURO_BASELINE_AUDIT.md`（架构+CLR+金标准数字） |
| 0B 复现 | ✅ | cheems 4060 精确复现 v1_s42 = 76/150, without_S=0 |
| 0C 选型 | ✅ | `P0_SUBSTRATE_DESIGN.md`（retention+6层 hybrid+6项修订锁死） |
| 0D 迁移设计 | ✅ | `P0_WEIGHT_MIGRATION.md`（exact_copy ~100%，γ 唯一新参数） |
| 0E 实现 | ✅ | `retention_ouro.py`（forward 跑通，Amendment 7 diff<1e-3） |
| 0F 迁移+验证 | ✅ | M1 ckpt + `weight_migration_manifest.json`（269 tensor max_abs_diff==0） |
| 0G pre-distill eval | ✅ | M1 受损未崩（logit cos 0.77），可进 0H |
| **0H 蒸馏** | 🟡 进行中 | smoke 通过(10步 loss 0.99)，**全量300步 detached 启动未成功** |
| 0I C-reset/C-persist 训练 | ☐ | 待 0H 完成 |
| 0J D 条件 | ☐ | |
| 0K 因果干预 | ☐ | |
| 0L 分析 | ☐ | |

---

## 3. 关键技术决策（不可推翻）

### 3.1 选型（Phase 0C）
- **Linear recurrent attention（retention，保留 RoPE）**，hybrid 替换 6 层 `{3,7,11,15,19,23}`（= FiLM 注入层同集）。
- mixing：`S_n=γ_h·S_{n-1}+K_n^T·V_n; O_n=(Q_n·S_n)/sqrt(z_n)` 替换 softmax(QK^T)V。
- Q/K/V/O proj **exact_copy**（tensor 精确拷贝，非函数保持）。
- **γ 唯一新增 trainable 参数**。

### 3.2 六项修订（0D 前锁死，见 `P0_SUBSTRATE_DESIGN.md` §0a + 记忆 `p0c-retention-design-locked`）
1. multi-scale γ init：`γ_h≈1-2^{-5-h}`，非全 0.95。
2. non-parametric z_n normalization accumulator（trainable 仍只 γ）。
3. exact_copy 措辞：tensor 精确拷贝，非 attention 函数保持。
4. RoPE 跨 turn = episode-global position counter；C 条件拆 `C-reset+local pos` / `C-persist+global pos`。
5. **retention state 在每个 Ouro loop 边界 reset**（最关键）——只有 turn/episode persist 是实验轴。
6. "structurally analogous" 非 "mathematically isomorphic"；retention bandwidth ≈1536× CLR，必须报告，C-persist 大赢则 P1 补 capacity-controlled 对照。
7. 0E gate：parallel/recurrent equivalence（已通过，diff<1e-3）。

---

## 4. 环境（cheems 训练机）

- **Python 3.13.15**，`C:\Users\AI Training\AppData\Local\Programs\Python\Python313\python.exe`
- **torch 2.13.0+cu126**，CUDA 12.6，RTX 4060 8GB
- **transformers 4.54.1**（Ouro 官方推荐，干净环境无需 5.x 补丁）
- **bitsandbytes 0.50.1** + **accelerate 1.14.0**
- 工作目录 `D:\training`；权重 `D:\training\Ouro-1.4B`（model.safetensors 2.87GB，大小校验通过 + baseline 命中间接验证完整性）
- 连接：paramiko 密码认证，用户 `ai training` / 密码 `REDACTED_CONFIGURE_ENV`。Tailscale 内网，链路**不稳**（通一阵断一阵，需重试）。
- 机主常在打游戏（GPU 占用是游戏，不管它）。

### 远程执行模式（已验证可用）
- detached：PowerShell `Start-Process -FilePath D:/path.bat -WindowStyle Hidden`（**bat 启动器最稳**，见 `_run_distill.bat`；直接 ArgumentList 引号嵌套易坏）
- 前台长任务：`exec_command(..., timeout=N)` + 本地 Bash `run_in_background`
- 路径一律正斜杠；反斜杠在 paramiko→cmd 层触发"文件名语法不正确"
- env 覆盖：脚本内 `os.environ[...]` hardcode，不用 cmd `set`（反斜杠转义）

### 两处 create_causal_mask 兼容补丁
4.54.1 的 `create_causal_mask` 用 `input_embeds`（单数）+ `cache_position`，5.x 用 `inputs_embeds`（复数）。两处打动态签名适配：
1. `modeling_ouro.py` mask_kwargs（backbone 内部）
2. `clr_ouro.py` `turn_forward_v1` 的 create_causal_mask 调用

---

## 5. 文件清单

### 本地 `D:\LoopLM\`
- 设计文档：`OURO_BASELINE_AUDIT.md`、`P0_SUBSTRATE_DESIGN.md`、`P0_WEIGHT_MIGRATION.md`、`MONOPAPER_STATUS.md`、`MONOPAPER_PLAN.md`、`HANDOFF.md`（本文档）
- 代码：`clr_ouro.py`（加 env 覆盖 + 4.54.1 兼容补丁）、`retention_ouro.py`（retention 模块）、`migrate_weights.py`（0F）、`pre_distill_eval.py`（0G）、`offline_teacher_cache.py`（0H step1）、`distill.py`（0H step2）、`_remote_*.py`（冒烟/启动器）、`_run_*.bat`
- 论文：`papers/monopaper.tex`（skeleton）、`papers/PAPER1_FROZEN.md`（金标准）、`papers/paper2.tex`（reviewer debt）
- 历史：`DESIGN.md`、`GPU_COORDINATION.md`、`CHEEMS_CONNECTION.md`
- `exp_seqstate_s0/`（S0 状态模块，外挂基线，非本轮 C/D）

### cheems `D:\training\`
- 同步的全部代码 + `Ouro-1.4B/` 权重
- `checkpoints/ouro_retention_m1/best_model.pt`（M1，只存 γ+元数据）
- `checkpoints/clr_ouro_v1_s42/`（历史 baseline ckpt，0B 复现用）
- `teacher_cache/`（0H step1 产出，300 轨迹/608 turns，D1+D2+D3 cache）
- `teacher_cache_manifest.json`、`weight_migration_manifest.json`、`p0g_pre_distill_eval.json`

---

## 6. Phase 0H 当前状态（接手重点）

### smoke 通过
`distill.py` D_SMOKE=1 D_STEPS=10 跑通：10 步，loss=0.99，存了 M2。trainable 96 params（γ only），D1 block+D3 logit loss 都在反传。

### 全量 300 步启动
- detached（bat + Start-Process）**未成功**——无进程、log 空。
- **前台后台模式跑起来了**（本地 Bash run_in_background，任务 ID bjpcqadul，2026-08-24 ~21:40 启动）。确认 cheems 上 python.exe 运行中，GPU 4882MiB/41%。
- 预计 10-15 分钟完成，输出在本地 task 文件 `bjpcqadul.output`。

### 接手要做的（若蒸馏已完成）
1. 检查 `D:\training\checkpoints\ouro_retention_m2_d1\best_model.pt` + `distill_log.json` 的 loss 曲线（loss 是否下降）。
2. 0G 评测 M2（蒸馏后）——重跑 `pre_distill_eval.py`（改成加载 M2 gamma），看 logit cosine 是否从 0.77 回升。若回升说明蒸馏修复了 mixing 损失。
3. 若 M2 仍不够（logit cos <0.9 或 CLUTRR candidate-scoring 无改善），按 §1.4 解冻替换层 post-norm（8 个 RMSNorm）重蒸，或调 λ_block/λ_logit。

### 已知坑（distill.py）
- `block_outs.clear()` 必须在每次 student forward 前（hook 跨 turn 残留）。
- dataset 必须用 seed 42 + `generate(600)` replay + `generate(len(shards))` 与 cache 对齐 prompt。
- 4-bit 冻结 backbone 能传梯度到 γ（requires_grad=False 不阻断反向传播）。

---

## 7. 后续阶段（0I–0L）

### 0I：C-reset / C-persist 训练（§8/§21/§22）
- 在 M2（蒸馏后 student）上训 reset vs persist 两条件。
- reset：retention state 每 turn 开始归零（S=0,z=eps）。**注意 Ouro loop 边界也 reset**（Amendment 5）。
- persist：retention state 跨 turn 保存，episode-global position（Amendment 4）。
- forward API 要支持 `forward(..., recurrent_state=None) → logits, new_state`（retention_ouro 已有）。
- 梯度语义：turn boundary detach（与历史 CLR 对齐，§22）。
- 任务：CLUTRR，candidate-scoring 作 final protocol。

### 0J：D-reset / D-persist（§7）
- 在 internal substrate 上加 external CLR（复用 `clr_ouro.inject_film` + `PlaaCore`，D 条件 FiLM 与 retention 在同层 {3,7,11,15,19,23} 叠加）。
- 跑 reset/persist 两条件。

### 0K：因果干预电池（§24/§25/§26）
- 对 retention state：normal/zero/random/freeze/reset/persist/swap/shuffle/scale/selective reset。
- per-iteration diagnostics（real/zero/rebuilt）检验 two-step bootstrap 阈值。
- candidate-scoring final protocol。

### 0L：分析
- `P0_RESULTS.md`（raw numbers、CI、负结果、不稳定性、VRAM、compute、迁移/蒸馏保真度、causal+dynamical 对比）
- `P0_RESOURCE_REPORT.md`（idle/loaded/peak VRAM、tokens/sec、step time）

---

## 8. 纪律（§42/§52）
- 历史数字不得静默修改（§2）。
- candidate-scoring 作主结果，fast_eval 仅 debug（§30）。
- 每套干预带阳性对照。
- monopaper 实验完成前只许 we ask/we test/we hypothesize（§42）。
- 目标不是证明 retention 更好，是判断 persistent computation 最合理的 substrate（§52）。失败也是结果。

---

## 9. 记忆系统
项目记忆在 `C:\Users\Ouyang Quan\.claude\projects\D--LoopLM\memory\`，索引 `MEMORY.md`。关键记忆：
- `ouro-architecture-audit`、`ouro-layer-param-shapes`、`p0c-retention-design-locked`
- `cheems-env-and-baseline-repro`、`cheems-sync-bottleneck`
- `retention-0e-impl-verified`（含 6 个实现坑）、`p0f-migration-verified`、`p0g-pre-distill-eval`
- `state-module-vs-substrate`（外挂 StateModule ≠ internal substrate）
- `clr-plaa-merged-paper`（历史三篇论文状态）
- `user-highschool-researcher`、`feedback-fable5-delegate`

下任接手先读 `MEMORY.md` 索引，再读本文档 + `MONOPAPER_STATUS.md`。
