"""
MusicUnderstand 3.0 — 多任务联合训练脚本

Step 2-3: 同时训练 Emotion + Genre + Instrument

支持多数据集联合:
  - DEAM (emotion regression)
  - GTZAN (genre classification)
  - 可选 instrument 数据（当前用合成数据）

训练模式:
  - freeze_backbone: 加载预训练 backbone，只训练各 head
  - finetune: 加载预训练 backbone + emotion head，联合微调所有参数
  - end_to_end: 从头训练全部

损失函数:
  - Emotion: MSE (6 维 V/A mean/delta/var)
  - Genre: CrossEntropy
  - Instrument: BCEWithLogitsLoss (多标签)
  - 各任务 loss 可配置权重

用法:
  python train/train_multitask.py --epochs 100 --batch-size 32 \\
      --lr 0.001 --freeze-backbone --weight-emotion 1.0 --weight-genre 1.0
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Optional

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# 确保能找到项目模块
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from model.music_understand import MusicUnderstand2
from data.deam_dataset import DEAMDataset, create_dataloaders as create_deam_loaders
from data.gtzan_dataset import GTZANDataset, create_gtzan_dataloaders
from data.fma_dataset import FMADataset, create_fma_dataloaders
from model.experts.genre_head import GENRES, NUM_GENRES
from model.experts.instrument_head import INSTRUMENTS, NUM_INSTRUMENTS
from utils.metrics import evaluate_emotion, evaluate_genre

# --- 配置 ---
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "config", "default.yaml")
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_emotion_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_weights: dict,
) -> torch.Tensor:
    """
    计算 12 维情绪损失的加权和

    Args:
        pred:   [B, T, 12] 模型预测
        target: [B, T, 12] 真实值
        mask:   [B, T]     有效帧掩码
        loss_weights: 各分量权重

    Returns:
        total_loss: 标量 emotion loss
        loss_dict:  各分量 loss 值
    """
    B, T, D = pred.shape
    mask_3d = mask.unsqueeze(-1).expand_as(pred)  # [B, T, 12]

    diff = (pred - target) ** 2
    diff = diff * mask_3d

    dim_losses = diff.sum(dim=(0, 1)) / (mask_3d.sum(dim=(0, 1)) + 1e-8)

    loss_dict = {
        "V_mean": dim_losses[0].item(),
        "V_delta": dim_losses[1].item(),
        "V_var": dim_losses[2].item(),
        "A_mean": dim_losses[3].item(),
        "A_delta": dim_losses[4].item(),
        "A_var": dim_losses[5].item(),
    }

    total_loss = (
        dim_losses[0] * loss_weights.get("valence_mean", 1.0) +
        dim_losses[3] * loss_weights.get("arousal_mean", 1.0) +
        dim_losses[1] * loss_weights.get("valence_delta", 0.5) +
        dim_losses[4] * loss_weights.get("arousal_delta", 0.5) +
        dim_losses[2] * loss_weights.get("valence_var", 0.3) +
        dim_losses[5] * loss_weights.get("arousal_var", 0.3)
    )

    return total_loss, loss_dict


def build_multitask_loss(
    emotion_pred: Optional[torch.Tensor],
    emotion_target: Optional[torch.Tensor],
    emotion_mask: Optional[torch.Tensor],
    genre_logits: Optional[torch.Tensor],
    genre_target: Optional[torch.Tensor],
    instr_logits: Optional[torch.Tensor],
    instr_target: Optional[torch.Tensor],
    loss_weights: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    """
    计算多任务损失

    Args:
        emotion_pred:   [B, T, 12] 情绪预测（可选）
        emotion_target: [B, T, 12] 情绪目标（可选）
        emotion_mask:   [B, T] 有效掩码（可选）
        genre_logits:   [B, N_genre] 体裁 logits（可选）
        genre_target:   [B] 体裁标签（可选）
        instr_logits:   [B, N_instr] 乐器 logits（可选）
        instr_target:   [B, N_instr] 多标签目标（可选）
        loss_weights:   各任务权重 dict

    Returns:
        loss_dict: {
            "total": total_loss (加权和),
            "emotion": emotion_loss_value (已加权),
            "genre": genre_loss_value (已加权),
            "instrument": instrument_loss_value (已加权),
            "emotion_raw": raw emotion loss (仅统计),
            "genre_raw": raw genre loss,
            "instrument_raw": raw instrument loss,
        }
    """
    total_loss = torch.tensor(0.0, device=DEVICE)
    # 使用 dict 记录权重的 loss 值（用于反向传播）和 raw 值（用于日志）
    weighted_losses = {}
    raw_losses = {}

    # --- Emotion Loss (MSE) ---
    if emotion_pred is not None and emotion_target is not None:
        emo_w = loss_weights.get("emotion", 1.0)
        emo_loss, _ = build_emotion_loss(
            emotion_pred, emotion_target, emotion_mask,
            {k: v for k, v in loss_weights.items()
             if k in ("valence_mean", "arousal_mean",
                      "valence_delta", "arousal_delta",
                      "valence_var", "arousal_var")}
        )
        weighted_losses["emotion"] = emo_loss * emo_w
        raw_losses["emotion"] = emo_loss.detach()

    # --- Genre Loss (CrossEntropy) ---
    if genre_logits is not None and genre_target is not None:
        genre_w = loss_weights.get("genre", 1.0)
        genre_loss = nn.functional.cross_entropy(genre_logits, genre_target)
        weighted_losses["genre"] = genre_loss * genre_w
        raw_losses["genre"] = genre_loss.detach()

    # --- Instrument Loss (BCEWithLogits) ---
    if instr_logits is not None and instr_target is not None:
        instr_w = loss_weights.get("instrument", 1.0)
        instr_loss = nn.functional.binary_cross_entropy_with_logits(
            instr_logits, instr_target.float()
        )
        weighted_losses["instrument"] = instr_loss * instr_w
        raw_losses["instrument"] = instr_loss.detach()

    # 加总
    for v in weighted_losses.values():
        total_loss = total_loss + v

    loss_dict = {
        "total": total_loss,
        "emotion": weighted_losses.get("emotion", torch.tensor(0.0, device=DEVICE)),
        "genre": weighted_losses.get("genre", torch.tensor(0.0, device=DEVICE)),
        "instrument": weighted_losses.get("instrument", torch.tensor(0.0, device=DEVICE)),
        "emotion_raw": raw_losses.get("emotion", torch.tensor(0.0, device=DEVICE)),
        "genre_raw": raw_losses.get("genre", torch.tensor(0.0, device=DEVICE)),
        "instrument_raw": raw_losses.get("instrument", torch.tensor(0.0, device=DEVICE)),
    }

    return loss_dict


def train_epoch_multitask(
    model: nn.Module,
    deam_loader: torch.utils.data.DataLoader,
    gtzan_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    loss_weights: Dict[str, float],
    device: torch.device,
    grad_clip: float = 1.0,
    tasks: list = None,
    fma_loader: torch.utils.data.DataLoader = None,
) -> Dict[str, float]:
    """
    多任务训练一个 epoch

    使用交替批次训练策略:
      1. 交替从 DEAM 和 GTZAN 取 batch
      2. 每个 batch 只计算有数据的任务 loss
      3. 累积梯度后 step

    Args:
        tasks: 启用的任务列表 ['emotion', 'genre', 'instrument']

    Returns:
        avg_losses: 各任务平均 loss
    """
    if tasks is None:
        tasks = ['emotion', 'genre', 'instrument']

    model.train()
    total_losses = {
        "total": 0.0,
        "emotion": 0.0, "genre": 0.0, "instrument": 0.0,
        "emotion_raw": 0.0, "genre_raw": 0.0, "instrument_raw": 0.0,
    }

    # 创建三个 loader 的迭代器
    deam_iter = iter(deam_loader)
    gtzan_iter = iter(gtzan_loader)
    fma_iter = iter(fma_loader) if fma_loader else None

    # 确定最大步数
    max_batches = max(len(deam_loader), len(gtzan_loader))
    n_batches = 0

    pbar = tqdm(range(max_batches), desc="Train", leave=False)
    for step in pbar:
        # --- DEAM Batch (emotion) ---
        try:
            deam_batch = next(deam_iter)
        except StopIteration:
            deam_iter = iter(deam_loader)
            deam_batch = next(deam_iter)

        mel_deam = deam_batch["mel"].to(device)          # [B, 1, 128, T]
        emotion_target = deam_batch["emotion"].to(device)  # [B, T, 12]
        emotion_mask = deam_batch["mask"].to(device)       # [B, T]

        active_experts = []
        if 'emotion' in tasks:
            active_experts.append('emotion')
        if 'genre' in tasks:
            active_experts.append('genre')
        if 'instrument' in tasks:
            active_experts.append('instrument')

        outputs = model(mel_deam, active_experts=active_experts)

        loss_dict = build_multitask_loss(
            emotion_pred=outputs.get('emotion'),
            emotion_target=emotion_target,
            emotion_mask=emotion_mask,
            genre_logits=outputs.get('genre_logits'),
            genre_target=None,      # DEAM 没有体裁标签
            instr_logits=outputs.get('instrument_logits'),
            instr_target=None,      # DEAM 没有乐器标签
            loss_weights=loss_weights,
        )

        # --- GTZAN Batch (genre) ---
        try:
            gtzan_batch = next(gtzan_iter)
        except StopIteration:
            gtzan_iter = iter(gtzan_loader)
            gtzan_batch = next(gtzan_iter)

        mel_gtzan = gtzan_batch["mel"].to(device)
        genre_target = gtzan_batch["genre"].to(device)

        outputs_g = model(mel_gtzan, active_experts=active_experts)

        loss_dict_g = build_multitask_loss(
            emotion_pred=outputs_g.get('emotion'),
            emotion_target=None,
            emotion_mask=None,
            genre_logits=outputs_g.get('genre_logits'),
            genre_target=genre_target,
            instr_logits=outputs_g.get('instrument_logits'),
            instr_target=None,      # GTZAN 没有乐器标签
            loss_weights=loss_weights,
        )

        total_loss = loss_dict["total"] + loss_dict_g["total"]

        # --- FMA Batch (额外的体裁数据) ---
        if fma_iter is not None:
            try:
                fma_batch = next(fma_iter)
            except StopIteration:
                fma_iter = iter(fma_loader)
                fma_batch = next(fma_iter)

            mel_fma = fma_batch["mel"].to(device)
            genre_fma = fma_batch["genre"].to(device)

            outputs_f = model(mel_fma, active_experts=active_experts)
            loss_dict_f = build_multitask_loss(
                emotion_pred=None, emotion_target=None, emotion_mask=None,
                genre_logits=outputs_f.get('genre_logits'),
                genre_target=genre_fma,
                instr_logits=None, instr_target=None,
                loss_weights=loss_weights,
            )
            total_loss = total_loss + loss_dict_f["total"]

        optimizer.zero_grad()
        total_loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # 统计
        step_losses = {
            "total": total_loss.item(),
            "emotion": loss_dict["emotion"].item(),
            "genre": loss_dict_g["genre"].item(),
            "emotion_raw": loss_dict["emotion_raw"].item(),
            "genre_raw": loss_dict_g["genre_raw"].item(),
        }

        for k, v in step_losses.items():
            total_losses[k] += v

        n_batches += 1
        pbar.set_postfix({
            "total": f"{step_losses['total']:.4f}",
        })

    # 平均
    avg_losses = {k: v / n_batches for k, v in total_losses.items()}
    return avg_losses


@torch.no_grad()
def validate_multitask(
    model: nn.Module,
    deam_loader: torch.utils.data.DataLoader,
    gtzan_loader: torch.utils.data.DataLoader,
    loss_weights: Dict[str, float],
    device: torch.device,
    tasks: list = None,
    fma_loader: torch.utils.data.DataLoader = None,
) -> Dict[str, float]:
    """
    多任务验证

    Returns:
        metrics: 各任务指标 dict
    """
    if tasks is None:
        tasks = ['emotion', 'genre', 'instrument']

    model.eval()

    # 收集情绪预测用于详细评估
    all_emo_preds = []
    all_emo_targets = []
    all_emo_masks = []

    # 收集体裁预测用于详细评估
    all_genre_preds = []
    all_genre_targets = []

    total_losses = {
        "total": 0.0,
        "emotion": 0.0, "genre": 0.0, "instrument": 0.0,
        "emotion_raw": 0.0, "genre_raw": 0.0, "instrument_raw": 0.0,
    }
    n_batches_emo = 0
    n_batches_genre = 0

    # --- 验证情绪 ---
    if 'emotion' in tasks:
        for batch in tqdm(deam_loader, desc=" Val[Emo]", leave=False):
            mel = batch["mel"].to(device)
            emotion = batch["emotion"].to(device)
            mask = batch["mask"].to(device)

            outputs = model(mel, active_experts=['emotion'])
            pred = outputs['emotion']

            # loss
            loss_dict = build_multitask_loss(
                emotion_pred=pred, emotion_target=emotion,
                emotion_mask=mask,
                genre_logits=None, genre_target=None,
                instr_logits=None, instr_target=None,
                loss_weights=loss_weights,
            )

            total_losses["emotion"] += loss_dict["emotion"].item()
            total_losses["emotion_raw"] += loss_dict["emotion_raw"].item()
            n_batches_emo += 1

            all_emo_preds.append(pred.cpu())
            all_emo_targets.append(emotion.cpu())
            all_emo_masks.append(mask.cpu())

    # --- 验证体裁 (GTZAN) ---
    if 'genre' in tasks:
        for batch in tqdm(gtzan_loader, desc=" Val[Gen]", leave=False):
            mel = batch["mel"].to(device)
            genre_target = batch["genre"].to(device)

            outputs = model(mel, active_experts=['genre'])
            genre_logits = outputs['genre_logits']

            loss_dict = build_multitask_loss(
                emotion_pred=None, emotion_target=None, emotion_mask=None,
                genre_logits=genre_logits, genre_target=genre_target,
                instr_logits=None, instr_target=None,
                loss_weights=loss_weights,
            )

            total_losses["genre"] += loss_dict["genre"].item()
            total_losses["genre_raw"] += loss_dict["genre_raw"].item()
            n_batches_genre += 1

            all_genre_preds.append(genre_logits.cpu())
            all_genre_targets.append(genre_target.cpu())

    # --- 验证体裁 (FMA) ---
    if 'genre' in tasks and fma_loader is not None:
        for batch in tqdm(fma_loader, desc=" Val[FMA]", leave=False):
            mel = batch["mel"].to(device)
            genre_target = batch["genre"].to(device)

            outputs = model(mel, active_experts=['genre'])
            genre_logits = outputs['genre_logits']

            loss_dict = build_multitask_loss(
                emotion_pred=None, emotion_target=None, emotion_mask=None,
                genre_logits=genre_logits, genre_target=genre_target,
                instr_logits=None, instr_target=None,
                loss_weights=loss_weights,
            )

            total_losses["genre"] += loss_dict["genre"].item()
            total_losses["genre_raw"] += loss_dict["genre_raw"].item()
            n_batches_genre += 1

            all_genre_preds.append(genre_logits.cpu())
            all_genre_targets.append(genre_target.cpu())

    # 平均 loss
    if n_batches_emo > 0:
        total_losses["emotion"] /= n_batches_emo
        total_losses["emotion_raw"] /= n_batches_emo
    if n_batches_genre > 0:
        total_losses["genre"] /= n_batches_genre
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 bel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Training Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # 各任务 loss
    ax = axes[0, 1]
    if history["train_emotion"]:
        ax.plot(history["train_emotion"], label="Train Emotion",
                color="green", alpha=0.7)
    if history["train_genre"]:
        ax.plot(history["train_genre"], label="Train Genre",
                color="orange", alpha=0.7)
    if history["val_emotion_raw"]:
        ax.plot(history["val_emotion_raw"], label="Val Emotion",
                color="green", alpha=0.7, linestyle="--")
    if history["val_genre_raw"]:
        ax.plot(history["val_genre_raw"], label="Val Genre",
                color="orange", alpha=0.7, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Task Losses")
    ax.legend()
    ax.grid(alpha=0.3)

    # 准确率
    ax = axes[1, 0]
    if history["val_accuracy"]:
        ax.plot(history["val_accuracy"], label="Genre Acc",
                color="purple", alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Genre Classification Accuracy")
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No accuracy data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Genre Accuracy")

    # Emotion R²
    ax = axes[1, 1]
    if history["val_V_mean_R2"]:
        ax.plot(history["val_V_mean_R2"], label="Valence R²",
                color="red", alpha=0.7)
    if history["val_A_mean_R2"]:
        ax.plot(history["val_A_mean_R2"], label="Arousal R²",
                color="blue", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R²")
    ax.set_title("Emotion R² Score")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[图表] 训练曲线 → {save_path}")


if __name__ == "__main__":
    main()
