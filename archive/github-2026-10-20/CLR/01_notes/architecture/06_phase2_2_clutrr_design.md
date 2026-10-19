# CLR Phase 2.2 — CLUTRR 关系推理

## 核心问题

S_t 能否编码关系图结构（而非纯数值）？这是对"持续隐状态"更本质的测试。

## 数据集：CLUTRR

CLUTRR (Compositional Language Understanding with Transformers and Recursive Reasoning)
- 家庭关系多跳推理
- 每样本：N 个关系事实 → 问目标关系
- 跳数：2-10 跳

## Turn 设计

```
Turn 1: "John is the father of Mary."        → S_t 编码 {John→father→Mary}
Turn 2: "Mary is the mother of Tom."         → S_t 编码 {..., Mary→mother→Tom}
Turn 3: "Tom is the brother of Alice."       → S_t 编码 {..., Tom→brother→Alice}
...
Turn N: "What is John to Alice?"             → 从 S_t 提取路径
```

**关键:** 每轮历史完全截断，S_t 是唯一信息通道。

## 成功判据

| 条件 | 期望准确率 |
|:-----|:---------:|
| 有 S_t | > 60% |
| 去掉 S_t (zeroed) | ≈ 随机猜测 |
| Δ | > 50% |

## 与 GSM8K 的区别

| 维度 | GSM8K (Phase 2.1) | CLUTRR (Phase 2.2) |
|:-----|:-----------------:|:------------------:|
| 推理类型 | 数字链式推理 | 关系图推理 |
| S_t 编码 | 当前中间结果 | 累积关系图结构 |
| 难度 | 2-5 步 | 2-10 跳 |
| 更根本的问题 | 能传数字吗？ | 能编码结构吗？ |
