# PLAA — Persistent Latent Adaptive Architecture

House Eternal · 奥尔德里克·沃斯 侍奉

让 Transformer 拥有持续内部状态 S_t 的研究框架。基底架构项目。

## Papers

| Paper | Dir | Content |
|:------|:----|:--------|
| Framework | `paper1/` | 初版框架 (LaTeX + md) |
| Multiplicative FiLM | `paper2/` | 乘法调控论文 + 数据 |
| Draft history | `paper2/history/` | 5 个版本草稿 |

## Key Findings (from paper2)

1. **Structural ≠ Functional** — FiLM induces 6.6× layer variance but causal Δ≈0.04
2. **Training-time scaffold** — Modulation prevents LoRA divergence on trajectory data
3. **PEFT checkpoint failure** — `save_pretrained()` discards custom injection layers

## CLR Project

[CLR (Continuous Latent Reasoning)](../CLR/) 基于 PLAA 架构构建，验证 S_t 在认知任务中的因果效应。Phase 2.1 GSM8K 已证明 S_t 携带跨轮推理状态 (ΔACC=+2.4%)。

## 目录

| Path | Content |
|:-----|:--------|
| `checkpoints/` | Phase 1/2/2.5/3A 产出 |
| `experts/` | 专家模块 (6 files) |
| `model/` | 可部署模型 (269 MB) |
| `paper1/` | 框架论文 |
| `paper2/` | Multiplicative FiLM 论文 |
| `tokenizers/` | BPE 64K |
