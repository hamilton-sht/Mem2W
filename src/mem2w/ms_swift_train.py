"""Minimal ms-swift SFT runner that trains only Mem2W memory parameters.

This is deliberately a one-stream runner.  ``--mode action`` and
``--mode recall`` are separate jobs, each with the ordinary ms-swift causal-LM
loss.  The later W/C dual-stream schedule is intentionally outside this first
training path.

The runner uses ms-swift for model/processor loading, chat-template encoding,
dataset preparation and Hugging Face Trainer construction.  Mem2W is attached
after ms-swift's model preparation, then the original model is frozen again so
the optimizer sees only the four memory tensors.  The final artifact is a
small ``memory.safetensors`` file rather than a full model checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .checkpointing import load_memory_checkpoint, save_memory_checkpoint
from .config import MemoryConfig, load_config
from .qwen_integration import attach_memory, get_memory_module


def _require_swift() -> tuple[Any, Any, Any, Any]:
    try:
        from swift.arguments import SftArguments
        from swift.pipelines.train.sft import SwiftSft
        from swift.trainers import TrainerFactory
        from swift.utils import get_model_parameter_info
    except ImportError as exc:  # pragma: no cover - depends on training env
        raise RuntimeError(
            "Mem2W ms-swift training requires ms-swift>=4.5 and its model dependencies; "
            "activate the clin-swift/metis-swift environment first"
        ) from exc
    return SftArguments, SwiftSft, TrainerFactory, get_model_parameter_info


def _freeze_backbone_keep_memory(model: Any) -> list[str]:
    """Freeze every parameter, then expose only the attached memory module."""

    model.requires_grad_(False)
    memory = get_memory_module(model)
    for parameter in memory.parameters():
        parameter.requires_grad_(True)
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    expected = {
        name for name, parameter in model.named_parameters() if getattr(parameter, "is_memory_parameter", False)
    }
    if set(names) != expected or len(names) != 4:
        raise RuntimeError(f"memory-only parameter check failed: trainable={names}, expected={sorted(expected)}")
    return names


def _make_args(args: argparse.Namespace, config: dict[str, Any]) -> Any:
    SftArguments, _, _, _ = _require_swift()
    # Keep the runner importable and testable on login/CPU nodes.  On the
    # target Muxi GPU this evaluates to True; when CUDA is intentionally
    # hidden for a preflight, ms-swift otherwise rejects ``bf16=True`` before
    # it even gets to model loading.
    import torch

    base = config.get("base_model", {})
    model_id = args.model or base.get("model_id", "Qwen/Qwen3.5-9B")
    kwargs: dict[str, Any] = {
        "model": model_id,
        "dataset": [str(Path(args.dataset).expanduser().resolve())],
        "template": "qwen3_5",
        "tuner_type": "full",
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "max_length": args.max_length,
        "packing": False,
        "gradient_checkpointing": False,
        "remove_unused_columns": False,
        "save_strategy": "no",
        "report_to": [],
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "seed": args.seed,
        "bf16": bool(torch.cuda.is_available()),
    }
    revision = args.revision or base.get("revision")
    if revision:
        kwargs["model_revision"] = revision
    return SftArguments(**kwargs)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    SftArguments, SwiftSft, TrainerFactory, get_model_parameter_info = _require_swift()
    swift_args = _make_args(args, config)

    # SwiftSft performs the official Qwen3.5 processor/model load and template
    # setup.  We intentionally reuse those APIs instead of constructing a
    # parallel Transformers loader for this first path.
    pipeline = SwiftSft(swift_args)
    train_dataset, val_dataset = pipeline._prepare_dataset()
    pipeline.args.save_args()
    pipeline.model = pipeline.prepare_model(
        pipeline.args,
        pipeline.model,
        template=pipeline.template,
        train_dataset=train_dataset,
    )
    memory_config = MemoryConfig.from_mapping(config)
    # The 9B target is hidden_size=4096.  For a smoke run on another official
    # Qwen3.5 checkpoint, infer the text width from the loaded model instead of
    # silently constructing a shape-incompatible memory layer.
    model_config = getattr(pipeline.model, "config", None)
    text_config = getattr(model_config, "text_config", None)
    detected_hidden_size = getattr(text_config, "hidden_size", None) or getattr(model_config, "hidden_size", None)
    if detected_hidden_size is not None and int(detected_hidden_size) != memory_config.hidden_size:
        memory_config.hidden_size = int(detected_hidden_size)
    model = attach_memory(pipeline.model, config=memory_config)
    trainable_names = _freeze_backbone_keep_memory(model)
    pipeline.model = model

    if args.memory_checkpoint:
        load_memory_checkpoint(model, args.memory_checkpoint, strict=True)

    trainer_cls = TrainerFactory.get_trainer_cls(pipeline.args)
    trainer = trainer_cls(
        model=model,
        args=pipeline.args.training_args,
        template=pipeline.template,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    pipeline.train(trainer)

    metadata = {
        "format": "mem2w-memory-v0.1",
        "mode": args.mode,
        "model_id": args.model or config["base_model"]["model_id"],
        "model_revision": args.revision or config["base_model"].get("revision"),
        "insertion_index_0based": memory_config.insertion_index,
        "hidden_size": memory_config.hidden_size,
        "slots": memory_config.slots,
        "key_dim": memory_config.key_dim,
        "value_dim": memory_config.value_dim,
        "trainable_parameter_names": trainable_names,
        "ms_swift_mode": args.mode,
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "model_parameter_info": get_model_parameter_info(model),
    }
    output_dir = save_memory_checkpoint(model, args.output_dir, metadata)
    summary = {"status": "completed", "mode": args.mode, "output_dir": str(output_dir), "trainable": trainable_names}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train only Mem2W memory with ms-swift")
    parser.add_argument("--mode", choices=("action", "recall"), required=True)
    parser.add_argument("--dataset", required=True, help="one converted ms-swift JSONL file")
    parser.add_argument("--output-dir", required=True, help="directory receiving memory.safetensors")
    parser.add_argument("--memory-checkpoint", help="optional prior Mem2W directory to resume from")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_length <= 0 or args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise SystemExit("max-length, batch-size and gradient-accumulation-steps must be positive")
    run(args)


if __name__ == "__main__":  # pragma: no cover
    main()
