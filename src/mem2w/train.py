"""Dual-objective Mem2W training primitives and CLI.

The actual model wrapper is kept separate so these loss and gradient rules can
be tested with a tiny fake backbone before loading a 9B checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import load_config
from .data_contract import read_jsonl


def _torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Mem2W training requires torch") from exc
    return torch, F


def masked_causal_cross_entropy(logits: Any, labels: Any, *, ignore_index: int = -100) -> tuple[Any, int]:
    """Compute a single causal shift and return (mean_loss, valid_token_count)."""

    torch, F = _torch()
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("logits must be [B,L,V] and labels must be [B,L]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels sequence dimensions differ")
    if labels.shape[1] < 2:
        raise ValueError("causal loss needs at least two sequence positions")
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    valid = shifted_labels.ne(ignore_index)
    token_count = int(valid.sum().item())
    if token_count == 0:
        raise ValueError("branch has zero supervised tokens")
    loss = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=ignore_index,
        reduction="sum",
    ) / token_count
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("non-finite causal loss")
    return loss, token_count


@dataclass(frozen=True)
class DualObjectiveConfig:
    lambda_recall: float = 1.0
    warmup_fraction: float = 0.20
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.lambda_recall < 0:
            raise ValueError("lambda_recall must be non-negative")
        if not 0 < self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be strictly between 0 and 1")


def training_stage(update_index: int, total_updates: int, warmup_fraction: float) -> str:
    if update_index < 1 or total_updates < 1:
        raise ValueError("update indices are one-based and total_updates must be positive")
    return "W" if update_index <= math.floor(warmup_fraction * total_updates) else "C"


def _call_model(model: Any, batch: Mapping[str, Any], *, stop_content_grad: bool) -> Any:
    kwargs = dict(batch)
    kwargs.setdefault("use_cache", False)
    kwargs["stop_content_grad"] = stop_content_grad
    try:
        return model(**kwargs)
    except TypeError as exc:
        if "stop_content_grad" not in str(exc):
            raise
        raise TypeError(
            "Mem2W model must expose forward(..., stop_content_grad=bool); "
            "do not implement local routing by detaching memory_output"
        ) from exc


class DualObjectiveTrainer:
    """One logical update over a paired action and recall batch.

    `model` must return an object with `.logits` and route the explicit
    `stop_content_grad` flag to the memory layer. The optimizer should contain
    only the memory parameters.
    """

    def __init__(self, model: Any, optimizer: Any, config: DualObjectiveConfig, *, total_updates: int):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.total_updates = total_updates

    def train_step(self, action_batch: Mapping[str, Any], recall_batch: Mapping[str, Any], update_index: int) -> dict[str, Any]:
        torch, _ = _torch()
        stage = training_stage(update_index, self.total_updates, self.config.warmup_fraction)
        self.optimizer.zero_grad(set_to_none=True)

        action_outputs = _call_model(self.model, action_batch, stop_content_grad=False)
        action_loss, action_tokens = masked_causal_cross_entropy(action_outputs.logits, action_batch["labels"])
        action_loss.backward()

        recall_outputs = _call_model(self.model, recall_batch, stop_content_grad=(stage == "C"))
        recall_loss, recall_tokens = masked_causal_cross_entropy(recall_outputs.logits, recall_batch["labels"])
        (self.config.lambda_recall * recall_loss).backward()

        gradients = [parameter for parameter in self.model.memory_parameters() if parameter.grad is not None]
        if not gradients:
            raise RuntimeError("no memory parameter received a gradient")
        total_norm = torch.nn.utils.clip_grad_norm_(gradients, self.config.max_grad_norm)
        if not bool(torch.isfinite(total_norm).item()):
            raise FloatingPointError("non-finite memory gradient norm")
        self.optimizer.step()

        return {
            "stage": stage,
            "loss_action": float(action_loss.detach().cpu()),
            "loss_recall": float(recall_loss.detach().cpu()),
            "tokens_action": action_tokens,
            "tokens_recall": recall_tokens,
            "memory_grad_norm": float(total_norm.detach().cpu()),
        }


def build_memory_optimizer(model: Any, config: Mapping[str, Any]) -> Any:
    torch, _ = _torch()
    parameters = [parameter for parameter in model.memory_parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("memory_parameters() returned no trainable parameters")
    optimizer_cfg = config.get("optimizer", {})
    return torch.optim.AdamW(
        parameters,
        lr=float(optimizer_cfg.get("learning_rate", 1e-4)),
        betas=tuple(optimizer_cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
    )


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mem2W's dual objective")
    parser.add_argument("--config", required=True)
    parser.add_argument("--action-data", required=True)
    parser.add_argument("--recall-data", required=True)
    parser.add_argument("--dry-run", action="store_true", help="validate config/data without loading a checkpoint")
    return parser.parse_args()


def main() -> None:
    args = _cli()
    config = load_config(args.config)
    action_rows = list(read_jsonl(args.action_data))
    recall_rows = list(read_jsonl(args.recall_data))
    if not action_rows or not recall_rows:
        raise SystemExit("action and recall JSONL files must both contain at least one sample")
    types = {row.get("sample_type") for row in action_rows}
    if types != {"action"}:
        raise SystemExit(f"action data contains unexpected sample types: {sorted(types)}")
    types = {row.get("sample_type") for row in recall_rows}
    if types != {"recall"}:
        raise SystemExit(f"recall data contains unexpected sample types: {sorted(types)}")
    if args.dry_run:
        print(json.dumps({"status": "validated", "action_samples": len(action_rows), "recall_samples": len(recall_rows)}, ensure_ascii=False))
        return
    try:
        from .qwen_integration import load_mem2w_model
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("model integration is unavailable; install the project with model dependencies") from exc
    model = load_mem2w_model(config)
    optimizer = build_memory_optimizer(model, config)
    updates = int(config["training"]["total_logical_updates_pilot"])
    trainer = DualObjectiveTrainer(
        model,
        optimizer,
        DualObjectiveConfig(
            lambda_recall=float(config["loss"]["lambda_recall"]),
            warmup_fraction=float(config["schedule"]["warmup_stage_fraction"]),
            max_grad_norm=float(config["optimizer"]["max_grad_norm"]),
        ),
        total_updates=updates,
    )
    raise SystemExit(
        "model loaded successfully; dataset tokenization/batching is intentionally explicit. "
        "Use the ms-swift adapter or provide a batch collator before launching a long run."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
