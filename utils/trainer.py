"""
Shared training engine.

Provides:
  train_one_epoch   — generic single-epoch training loop
  evaluate          — generic evaluation loop
  Trainer           — wraps model + optimiser + scheduler + W&B logging
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb


# ---------------------------------------------------------------------------
# Generic one-epoch loops
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: torch.device,
    epoch: int,
    log_interval: int = 20,
) -> Dict[str, float]:
    """
    Generic training loop. loss_fn must accept (model_output, batch) and
    return a scalar loss tensor.

    Returns dict of aggregated metrics.
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0
    t0         = time.time()

    for batch_idx, batch in enumerate(loader):
        # Move all batch tensors to device
        batch = _to_device(batch, device)

        optimizer.zero_grad()
        outputs = model(batch[0])          # batch[0] is always the image
        loss, extra = loss_fn(outputs, batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                f"loss={loss.item():.4f}  ({elapsed:.1f}s)"
            )

    return {"train/loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    metric_fn: Callable,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluation loop. metric_fn(all_outputs, all_batches) → dict of metrics.
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    all_outputs, all_batches = [], []

    for batch in loader:
        batch = _to_device(batch, device)
        outputs = model(batch[0])
        loss, _ = loss_fn(outputs, batch)

        total_loss += loss.item()
        n_batches  += 1
        all_outputs.append(_detach(outputs))
        all_batches.append(_detach(batch))

    metrics = metric_fn(all_outputs, all_batches)
    metrics["val/loss"] = total_loss / max(n_batches, 1)
    return metrics


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------
class Trainer:
    """
    High-level trainer that wraps the training / eval loops and handles
    checkpointing and W&B logging.

    Args:
        model       : the nn.Module to train
        train_loader: DataLoader for training set
        val_loader  : DataLoader for validation set
        optimizer   : torch optimiser
        loss_fn     : callable(outputs, batch) → (loss, extras_dict)
        metric_fn   : callable(all_outputs, all_batches) → dict
        scheduler   : optional LR scheduler (step called after each epoch)
        device      : torch.device
        save_dir    : directory to save checkpoints
        run_name    : string prefix for saved files
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        metric_fn: Callable,
        scheduler=None,
        device: torch.device = None,
        save_dir: str = "outputs",
        run_name: str = "run",
    ):
        self.model        = model.to(device or torch.device("cpu"))
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.optimizer    = optimizer
        self.loss_fn      = loss_fn
        self.metric_fn    = metric_fn
        self.scheduler    = scheduler
        self.device       = device or torch.device("cpu")
        self.save_dir     = save_dir
        self.run_name     = run_name
        self.best_val_loss = float("inf")

        import os
        os.makedirs(save_dir, exist_ok=True)

    def fit(self, epochs: int, log_interval: int = 20):
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            train_metrics = train_one_epoch(
                self.model, self.train_loader, self.optimizer,
                self._loss_wrapper, self.device, epoch, log_interval
            )
            val_metrics = evaluate(
                self.model, self.val_loader,
                self._loss_wrapper, self.metric_fn, self.device
            )

            metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            if self.scheduler is not None:
                self.scheduler.step(val_metrics.get("val/loss", 0))
                metrics["lr"] = self.optimizer.param_groups[0]["lr"]

            # W&B log
            if wandb.run is not None:
                wandb.log(metrics, step=epoch)

            # Print summary
            print("  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                    if isinstance(v, float)))

            # Save best
            val_loss = val_metrics.get("val/loss", float("inf"))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss)

    def _loss_wrapper(self, outputs, batch):
        """Adapts outputs+batch into the loss_fn signature."""
        return self.loss_fn(outputs, batch)

    def _save_checkpoint(self, epoch: int, val_loss: float):
        import os
        path = os.path.join(self.save_dir, f"{self.run_name}_best.pt")
        torch.save({
            "epoch":      epoch,
            "val_loss":   val_loss,
            "model":      self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
        }, path)
        print(f"  Saved best checkpoint → {path}  (val_loss={val_loss:.4f})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(b, device) for b in batch)
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    return batch


def _detach(obj):
    if isinstance(obj, (list, tuple)):
        return type(obj)(_detach(o) for o in obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    return obj
