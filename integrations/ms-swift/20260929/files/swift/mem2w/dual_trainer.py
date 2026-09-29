"""Paired action/recall Mem2W training on top of ms-swift templates.

This module intentionally stays below the public ``swift sft`` path.  The
standard SFT trainer remains a useful one-stream smoke test, while this
trainer owns the Mem2W-specific logical update:

* encode action and recall rows with the native ms-swift template;
* normalize each branch by its own supervised-token count;
* run both forwards before one optimizer step;
* in stage C, detach only the recall content path (V/W_O), while retaining
  the recall query/key path (W_Q/K).

It is a small loop rather than a Hugging Face Trainer subclass because the
paired branches do not have a one-to-one dataset row or a single ``compute_loss``
call.  Model loading, templates, collation, native tuner registration and
checkpoint serialization are still provided by ms-swift itself.
"""

from __future__ import annotations

import json
import math
import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from contextlib import nullcontext

from .integration import memory_parameters, set_stop_content_grad


@dataclass(frozen=True)
class Mem2WDualConfig:
    lambda_recall: float = 1.0
    warmup_fraction: float = 0.20
    max_grad_norm: float = 1.0
    accumulation_steps: int = 8
    loss_chunk_size: int = 0
    lazy_encode: bool = False
    activation_offload: bool = False
    sequence_parallel_size: int = 1
    stage_plan: tuple[tuple[str, int], ...] | None = None

    def __post_init__(self) -> None:
        if self.lambda_recall < 0:
            raise ValueError('lambda_recall must be non-negative')
        if not 0 < self.warmup_fraction < 1:
            raise ValueError('warmup_fraction must be strictly between 0 and 1')
        if self.max_grad_norm <= 0:
            raise ValueError('max_grad_norm must be positive')
        if self.accumulation_steps <= 0:
            raise ValueError('accumulation_steps must be positive')
        if self.loss_chunk_size < 0:
            raise ValueError('loss_chunk_size must be non-negative')
        if self.sequence_parallel_size <= 0:
            raise ValueError('sequence_parallel_size must be positive')
        if self.stage_plan is not None:
            if not self.stage_plan:
                raise ValueError('stage_plan must not be empty')
            for stage, updates in self.stage_plan:
                if stage not in {'W', 'C'}:
                    raise ValueError(f'unknown Mem2W stage: {stage!r}')
                if int(updates) <= 0:
                    raise ValueError('stage_plan update counts must be positive')


def chunked_hidden_cross_entropy(hidden, lm_head, labels, chunk_size, *, labels_are_shifted=False):
    """Exact causal CE without materializing [sequence, vocabulary] logits.

    All original context and target tokens are retained. Only the independent
    output projection/CE is chunked; attention is still full-sequence. Checkpoint
    each projection so autograd recomputes logits one chunk at a time.
    """
    if chunk_size < 1 or hidden.ndim != 3 or labels.shape != hidden.shape[:2]:
        raise ValueError('expected hidden [B,L,H], labels [B,L], and positive chunk_size')
    # With model-parallel loading, the input batch starts on the first device
    # while the final hidden state and lm_head may live on a later device.
    labels = labels.to(hidden.device)
    if labels_are_shifted:
        # Sequence-parallel input preparation rolls labels before sharding so
        # each local hidden position already has its next-token target.
        shifted_labels = labels.reshape(-1)
        selected_hidden = hidden.reshape(-1, hidden.shape[-1])
    else:
        shifted_labels = labels[:, 1:].reshape(-1)
        selected_hidden = hidden[:, :-1, :].reshape(-1, hidden.shape[-1])
    valid = shifted_labels.ne(-100)
    count = int(valid.sum().item())
    if not count:
        raise ValueError('branch has zero supervised tokens')
    selected = selected_hidden[valid]
    targets = shifted_labels[valid]

    def projected_ce(states, target):
        logits = lm_head(states).float()
        return F.cross_entropy(logits, target.to(logits.device), reduction='sum')

    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, count, chunk_size):
        chunk_loss = checkpoint(projected_ce, selected[start:start + chunk_size],
                                targets[start:start + chunk_size], use_reentrant=False)
        total = total + chunk_loss.to(total.device)
    loss = total / count
    if not bool(torch.isfinite(loss).item()):
        rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
        print({'event': 'nonfinite_chunked_loss', 'rank': rank,
               'hidden_finite': bool(torch.isfinite(hidden).all().item()),
               'hidden_max': float(hidden.detach().float().abs().nan_to_num().max().item()),
               'loss_value': float(loss.detach().float().nan_to_num().item()),
               'supervised_tokens': count}, flush=True)
        raise FloatingPointError('non-finite chunked causal loss')
    return loss, count


class _EncodedRows(Sequence):
    def __init__(self, template, rows):
        self.template, self.rows = template, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        result = self.template.encode(copy.deepcopy(dict(self.rows[index])))
        if isinstance(result, (list, tuple)):
            raise ValueError('automatic splitting is not supported; preserve full sample context')
        return result


def objective_stage(update_index: int, total_updates: int, warmup_fraction: float,
                    stage_plan: tuple[tuple[str, int], ...] | None = None) -> str:
    if update_index < 1 or total_updates < 1:
        raise ValueError('update indices are one-based and total_updates must be positive')
    if stage_plan is not None:
        consumed = 0
        for stage, updates in stage_plan:
            consumed += int(updates)
            if update_index <= consumed:
                return stage
        raise ValueError(
            f'update_index {update_index} exceeds explicit stage plan of {consumed} updates')
    return 'W' if update_index <= math.floor(warmup_fraction * total_updates) else 'C'


def masked_causal_cross_entropy(logits: torch.Tensor,
                                labels: torch.Tensor,
                                *,
                                ignore_index: int = -100,
                                labels_are_shifted: bool = False) -> tuple[torch.Tensor, int]:
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError(f'expected logits [B,L,V] and labels [B,L], got {tuple(logits.shape)} and {tuple(labels.shape)}')
    if labels.shape[1] < 2:
        raise ValueError('causal loss needs at least two sequence positions')
    labels = labels.to(logits.device)
    if labels_are_shifted:
        shifted_logits = logits.contiguous()
        shifted_labels = labels.contiguous()
    else:
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
    valid = shifted_labels.ne(ignore_index)
    token_count = int(valid.sum().item())
    if token_count == 0:
        raise ValueError('branch has zero supervised tokens')
    loss = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=ignore_index,
        reduction='sum') / token_count
    if not bool(torch.isfinite(loss).item()):
        rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
        print({'event': 'nonfinite_full_loss', 'rank': rank,
               'logits_finite': bool(torch.isfinite(logits).all().item()),
               'logits_max': float(logits.detach().float().abs().nan_to_num().max().item()),
               'loss_value': float(loss.detach().float().nan_to_num().item()),
               'supervised_tokens': token_count}, flush=True)
        raise FloatingPointError('non-finite causal loss')
    return loss, token_count


def _model_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only model inputs; labels/loss scales are consumed locally."""

    excluded = {'labels', 'loss_scale', 'channel', 'lengths', 'template_inputs', '_extra_kwargs'}
    return {key: value for key, value in batch.items() if key not in excluded}


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result = {}
    for key, value in batch.items():
        result[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return result


class Mem2WDualObjectiveTrainer:
    """Run paired action/recall updates over already loaded native ms-swift state."""

    def __init__(self,
                 model: Any,
                 template: Any,
                 action_rows: Sequence[Mapping[str, Any]],
                 recall_rows: Sequence[Mapping[str, Any]],
                 optimizer: torch.optim.Optimizer,
                 config: Mem2WDualConfig,
                 *,
                 total_updates: int,
                 scheduler: Any | None = None,
                 accumulation_steps: int | None = None,
                 device: torch.device | None = None) -> None:
        if not action_rows or not recall_rows:
            raise ValueError('action and recall datasets must both contain at least one row')
        self.model = model
        self.template = template
        self.action_rows = list(action_rows)
        self.recall_rows = list(recall_rows)
        self.optimizer = optimizer
        self.config = config
        self.total_updates = int(total_updates)
        self.scheduler = scheduler
        self.accumulation_steps = int(accumulation_steps or config.accumulation_steps)
        if self.accumulation_steps <= 0:
            raise ValueError('accumulation_steps must be positive')
        self.device = device or next(model.parameters()).device
        self.action_encoded = _EncodedRows(template, self.action_rows)
        self.recall_encoded = _EncodedRows(template, self.recall_rows)
        if not config.lazy_encode:
            self.action_encoded = list(self.action_encoded)
            self.recall_encoded = list(self.recall_encoded)

    def _batch(self, encoded: Sequence[Mapping[str, Any]], index: int) -> dict[str, Any]:
        row = encoded[index % len(encoded)]
        batch = self.template.data_collator([dict(row)])
        batch = _to_device(batch, self.device)
        # Sequence-parallel preparation is deliberately deferred until the
        # branch is about to run. The native helper stores full position ids
        # in process-global state; preparing action and recall batches up front
        # would let the later branch overwrite the earlier one's lengths.
        return batch

    def _valid_token_count(self, batch: Mapping[str, Any]) -> int:
        labels = batch.get('labels')
        if not isinstance(labels, torch.Tensor) or labels.ndim != 2 or labels.shape[1] < 2:
            raise ValueError('paired causal batches must contain labels with shape [B,L]')
        count = int(labels[:, 1:].ne(-100).sum().item())
        if count == 0 and self.config.sequence_parallel_size <= 1:
            raise ValueError('paired branch has zero supervised tokens')
        return count

    def _global_count(self, count: int) -> int:
        if not dist.is_available() or not dist.is_initialized():
            return count
        value = torch.tensor([count], dtype=torch.long, device=self.device)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return int(value.item())

    def _forward_loss(self, batch: Mapping[str, Any], *, stop_content_grad: bool) -> tuple[torch.Tensor, int]:
        set_stop_content_grad(self.model, stop_content_grad)
        if self.config.loss_chunk_size:
            if getattr(self.model.config, 'model_type', None) != 'qwen3_5':
                raise ValueError('chunked hidden loss currently supports native Qwen3.5 only')
            outputs = self.model.model(**_model_batch(batch), use_cache=False)
            if self._valid_token_count(batch) == 0:
                return outputs[0].float().sum() * 0.0, 0
            return chunked_hidden_cross_entropy(
                outputs[0], self.model.get_output_embeddings(), batch['labels'],
                self.config.loss_chunk_size,
                labels_are_shifted=self.config.sequence_parallel_size > 1)
        outputs = self.model(**_model_batch(batch), use_cache=False)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        if self._valid_token_count(batch) == 0:
            return logits.float().sum() * 0.0, 0
        return masked_causal_cross_entropy(
            logits, batch['labels'], labels_are_shifted=self.config.sequence_parallel_size > 1)

    def _backward_loss(self, batch: Mapping[str, Any], *, stop_content_grad: bool,
                       weight: float) -> tuple[float, int]:
        """Forward and backward one branch, optionally offloading saved activations.

        ``save_on_cpu`` is deliberately opt-in. It is useful for the few
        lossless rows whose full context is larger than the four-card C500
        device-parallel memory budget; it preserves every token while trading
        GPU memory for host-memory traffic. The context must include backward,
        not just forward, because autograd saves tensors during the forward.
        """
        context = (torch.autograd.graph.save_on_cpu(pin_memory=False)
                   if self.config.activation_offload else nullcontext())
        with context:
            loss, count = self._forward_loss(batch, stop_content_grad=stop_content_grad)
            (loss * weight).backward()
        return float(loss.detach().cpu()), count

    @staticmethod
    def _synchronize_gradients(parameters: Sequence[torch.nn.Parameter]) -> None:
        """Average the four memory gradients when launched under torchrun.

        The paired trainer deliberately does not wrap the frozen Qwen backbone
        in ``DistributedDataParallel``: only the attached Mem2W tensors are
        trainable, and the native hook-based model path must remain unchanged.
        Each rank therefore performs the small explicit all-reduce here.
        """
        if not dist.is_available() or not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        for parameter in parameters:
            gradient = parameter.grad
            if gradient is None:
                gradient = torch.zeros_like(parameter)
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            parameter.grad = gradient / world_size

    def train_step(self, update_index: int) -> dict[str, Any]:
        started = time.monotonic()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        stage = objective_stage(update_index, self.total_updates, self.config.warmup_fraction,
                                self.config.stage_plan)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        base_index = (update_index - 1) * self.accumulation_steps
        action_batches = [self._batch(self.action_encoded, base_index + offset)
                          for offset in range(self.accumulation_steps)]
        recall_batches = [self._batch(self.recall_encoded, base_index + offset)
                          for offset in range(self.accumulation_steps)]
        action_counts = [self._valid_token_count(batch) for batch in action_batches]
        recall_counts = [self._valid_token_count(batch) for batch in recall_batches]
        action_tokens = self._global_count(sum(action_counts))
        recall_tokens = self._global_count(sum(recall_counts))
        if action_tokens == 0 or recall_tokens == 0:
            raise ValueError('paired dataset has zero supervised tokens across sequence-parallel ranks')
        action_loss_sum = 0.0
        for batch, count in zip(action_batches, action_counts):
            if self.config.sequence_parallel_size > 1:
                from swift.sequence_parallel import sequence_parallel
                sequence_parallel.prepare_inputs(batch)
            action_loss, _ = self._backward_loss(
                batch, stop_content_grad=False, weight=count / action_tokens)
            action_loss_sum += action_loss * count
        recall_loss_sum = 0.0
        for batch, count in zip(recall_batches, recall_counts):
            if self.config.sequence_parallel_size > 1:
                from swift.sequence_parallel import sequence_parallel
                sequence_parallel.prepare_inputs(batch)
            recall_loss, _ = self._backward_loss(
                batch,
                stop_content_grad=(stage == 'C'),
                weight=self.config.lambda_recall * count / recall_tokens)
            recall_loss_sum += recall_loss * count

        parameters = memory_parameters(self.model)
        self._synchronize_gradients(parameters)
        gradients = [parameter for parameter in parameters if parameter.grad is not None]
        if not gradients:
            raise RuntimeError('no Mem2W parameter received a gradient')
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm)
        if not bool(torch.isfinite(total_norm).item()):
            raise FloatingPointError('non-finite Mem2W gradient norm')
        grad_norms = {name: float(parameter.grad.float().norm().item()) if parameter.grad is not None else None
                      for name, parameter in self.model.named_parameters() if parameter.requires_grad}
        before = [parameter.detach().clone() for parameter in parameters]
        self.optimizer.step()
        deltas = [float((parameter.detach().float() - initial.float()).norm().item())
                  for parameter, initial in zip(parameters, before)]
        if self.scheduler is not None:
            self.scheduler.step()
        return {
            'stage': stage,
            'loss_action': action_loss_sum / action_tokens,
            'loss_recall': recall_loss_sum / recall_tokens,
            'tokens_action': action_tokens,
            'tokens_recall': recall_tokens,
            'memory_grad_norm': float(total_norm.detach().cpu()),
            'learning_rate': float(self.optimizer.param_groups[0]['lr']),
            'parameter_grad_norms': grad_norms,
            'parameter_delta_norms': deltas,
            'seconds': time.monotonic() - started,
            'peak_memory_bytes': torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else 0,
        }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).expanduser().open('r', encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f'{path}:{line_no}: expected a JSON object')
            rows.append(value)
    return rows
