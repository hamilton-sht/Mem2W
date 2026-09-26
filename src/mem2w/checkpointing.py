"""Checkpoint helpers for the small, independently deployable memory module."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def model_parameter_fingerprint(model: Any) -> str:
    """Hash all original backbone tensors in a deterministic name order."""

    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if getattr(parameter, "is_memory_parameter", False):
            continue
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        # NumPy cannot represent bfloat16 on several supported versions.  Use
        # the tensor's raw bytes in that case, while retaining dtype in the
        # fingerprint so fp16/bf16 checkpoints cannot compare equal by accident.
        digest.update(str(tensor.dtype).encode("ascii"))
        try:
            raw = tensor.numpy().tobytes()
        except TypeError:
            import torch

            raw = tensor.view(torch.uint8).numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def save_memory_checkpoint(model: Any, output_dir: str | Path, metadata: Mapping[str, Any]) -> Path:
    """Save only memory parameters plus an explicit deployment manifest."""

    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("checkpoint export requires torch and safetensors") from exc

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    tensors = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if getattr(parameter, "is_memory_parameter", False)
    }
    if not tensors:
        raise ValueError("no memory parameters found for deployment checkpoint")
    save_file(tensors, str(target / "memory.safetensors"))
    manifest = dict(metadata)
    manifest["memory_parameter_names"] = sorted(tensors)
    manifest["backbone_parameter_fingerprint"] = model_parameter_fingerprint(model)
    (target / "memory_config.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def load_memory_checkpoint(model: Any, checkpoint_dir: str | Path, *, strict: bool = True) -> None:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("checkpoint loading requires safetensors") from exc
    tensors = load_file(str(Path(checkpoint_dir) / "memory.safetensors"))
    missing, unexpected = model.load_state_dict(tensors, strict=False)
    missing_memory = [name for name in missing if getattr(dict(model.named_parameters()).get(name), "is_memory_parameter", False)]
    if strict and (missing_memory or unexpected):
        raise RuntimeError(f"memory checkpoint mismatch: missing={missing_memory}, unexpected={unexpected}")
