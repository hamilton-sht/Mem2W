"""Native ms-swift Trainer integration for paired Mem2W SFT.

The dataset is grouped into logical updates (``mem2w_accumulation_steps``
rows per branch).  Hugging Face/Swift still owns the dataloader, DDP,
autocast, backward, optimizer, scheduler, checkpoint and resume lifecycle;
this trainer only supplies the paired objective.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from contextlib import nullcontext

from transformers import Trainer as HfTrainer, TrainerCallback
from transformers.modeling_utils import unwrap_model

from swift.mem2w.dual_trainer import (chunked_hidden_cross_entropy, masked_causal_cross_entropy,
                                      _model_batch)
from swift.mem2w.integration import memory_parameters, set_stop_content_grad
from swift.tuners.mem2w import Mem2WTuner
from .seq2seq_trainer import Seq2SeqTrainer


class Mem2WDualDataset(Sequence):
    """Pair action and recall rows into one Trainer example per logical update."""

    def __init__(self, action_dataset, recall_dataset, group_size: int = 8):
        if len(action_dataset) == 0 or len(recall_dataset) == 0:
            raise ValueError('Mem2W action and recall datasets must both be non-empty')
        if group_size <= 0:
            raise ValueError('Mem2W group_size must be positive')
        self.action_dataset = action_dataset
        self.recall_dataset = recall_dataset
        self.group_size = int(group_size)

    def __len__(self):
        logical_updates = math.ceil(max(len(self.action_dataset), len(self.recall_dataset)) / self.group_size)
        # Swift's distributed batch sampler expects a length divisible by the
        # data-parallel world size.  Padding here repeats the last logical
        # pair; the modulo indexing in __getitem__ keeps the branch aligned.
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        return math.ceil(logical_updates / world_size) * world_size

    def __getitem__(self, index):
        start = int(index) * self.group_size
        action = [self.action_dataset[(start + offset) % len(self.action_dataset)]
                  for offset in range(self.group_size)]
        recall = [self.recall_dataset[(start + offset) % len(self.recall_dataset)]
                  for offset in range(self.group_size)]
        return {'action': action, 'recall': recall}


def parse_stage_plan(value: str | None) -> tuple[tuple[str, int], ...] | None:
    if value is None or not str(value).strip():
        return None
    result = []
    for item in str(value).split(','):
        stage, separator, count = item.strip().partition(':')
        if separator != ':' or stage not in {'W', 'C'}:
            raise ValueError('mem2w_stage_plan entries must use W:<updates> or C:<updates>')
        count = int(count)
        if count <= 0:
            raise ValueError('mem2w_stage_plan update counts must be positive')
        result.append((stage, count))
    return tuple(result)


def stage_for_step(step: int, total_steps: int, warmup_fraction: float,
                   stage_plan: tuple[tuple[str, int], ...] | None) -> str:
    if stage_plan is not None:
        consumed = 0
        for stage, count in stage_plan:
            consumed += count
            if step <= consumed:
                return stage
        raise ValueError(f'Mem2W step {step} exceeds explicit stage plan of {consumed} steps')
    return 'W' if step <= math.floor(warmup_fraction * total_steps) else 'C'


def _prepare_nested(value, device):
    if isinstance(value, Mapping):
        return {key: _prepare_nested(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_prepare_nested(item, device) for item in value)
    if isinstance(value, list):
        return [_prepare_nested(item, device) for item in value]
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


class _Mem2WBoundarySaveCallback(TrainerCallback):
    """Save exactly at the non-uniform W/C/W stage boundaries."""

    def __init__(self, boundaries):
        self.boundaries = frozenset(int(item) for item in boundaries)

    def on_step_end(self, args, state, control, **kwargs):
        if int(state.global_step) in self.boundaries:
            control.should_save = True
        return control


class Mem2WDualTrainer(Seq2SeqTrainer):
    """ms-swift/Transformers Trainer with the Mem2W W/C paired objective."""

    def __init__(self, *args, **kwargs):
        self._mem2w_last_metrics = {}
        self._mem2w_manual_grad_sync = False
        self._mem2w_action_metric = 0.0
        self._mem2w_recall_metric = 0.0
        super().__init__(*args, **kwargs)
        stage_plan = parse_stage_plan(getattr(self.args, 'mem2w_stage_plan', None))
        if stage_plan:
            boundaries = []
            consumed = 0
            for _, count in stage_plan:
                consumed += count
                boundaries.append(consumed)
            self.add_callback(_Mem2WBoundarySaveCallback(boundaries))

    def _get_data_collator(self, args, template):
        def collate(features):
            action_rows = [row for feature in features for row in feature['action']]
            recall_rows = [row for feature in features for row in feature['recall']]
            return {
                'action': [template.data_collator([dict(row)]) for row in action_rows],
                'recall': [template.data_collator([dict(row)]) for row in recall_rows],
            }

        return collate

    def _get_collator_with_removed_columns(self, data_collator, description=None):
        # ``action`` and ``recall`` are intentionally not model.forward fields.
        # Keep the paired envelope intact for the custom compute_loss method.
        return data_collator

    def _prepare_inputs(self, inputs):
        return _prepare_nested(inputs, self.args.device)

    @staticmethod
    def _base_model(model):
        return unwrap_model(model)

    def _global_count(self, count: int) -> int:
        if not dist.is_available() or not dist.is_initialized():
            return int(count)
        value = torch.tensor([count], dtype=torch.long, device=self.args.device)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return int(value.item())

    @staticmethod
    def _valid_count(batch) -> int:
        labels = batch.get('labels')
        if not isinstance(labels, torch.Tensor) or labels.ndim != 2 or labels.shape[1] < 2:
            raise ValueError('Mem2W paired batches must contain labels with shape [B,L]')
        count = int(labels[:, 1:].ne(-100).sum().item())
        if count <= 0:
            raise ValueError('Mem2W paired branch has zero supervised tokens')
        return count

    def _forward_branch(self, model, batch, *, stop_content_grad: bool):
        base = self._base_model(model)
        set_stop_content_grad(base, stop_content_grad)
        count = self._valid_count(batch)
        if self.args.mem2w_loss_chunk_size:
            if getattr(base.config, 'model_type', None) != 'qwen3_5':
                raise ValueError('mem2w_loss_chunk_size currently supports native Qwen3.5 only')
            # Bypass DDP only for the chunked hidden-state path.  training_step
            # performs an explicit four-parameter all-reduce afterwards.
            outputs = base.model(**_model_batch(batch), use_cache=False)
            loss, _ = chunked_hidden_cross_entropy(
                outputs[0], base.get_output_embeddings(), batch['labels'], self.args.mem2w_loss_chunk_size)
            self._mem2w_manual_grad_sync = True
            return loss, count

        outputs = model(**_model_batch(batch), use_cache=False)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        loss, _ = masked_causal_cross_entropy(logits, batch['labels'])
        return loss, count

    def _iter_loss_terms(self, model, inputs):
        """Yield one normalized branch loss at a time.

        The previous implementation summed all 16 branch graphs before
        returning from compute_loss.  With 200k-token recall rows that kept
        every activation alive until the final backward and forced
        save_on_cpu to page the whole logical update.  Yielding terms lets
        training_step backward and release each graph immediately.
        """
        action_batches = inputs['action']
        recall_batches = inputs['recall']
        step = int(self.state.global_step) + 1
        total_steps = int(self.state.max_steps or self.args.max_steps)
        stage = stage_for_step(step, total_steps, self.args.mem2w_warmup_fraction,
                               parse_stage_plan(self.args.mem2w_stage_plan))

        # W and C are true alternating stages.  W trains action rows; C
        # reconstructs recall rows.  The previous implementation computed
        # both branches in every stage, turning one logical update into 16
        # long-context forwards and also cycling the 864 recall rows through
        # the 6804-action epoch.
        active_action = stage == 'W'
        active_recall = stage == 'C'
        action_counts = [self._valid_count(batch) for batch in action_batches] if active_action else []
        recall_counts = [self._valid_count(batch) for batch in recall_batches] if active_recall else []
        action_total = self._global_count(sum(action_counts)) if active_action else 0
        recall_total = self._global_count(sum(recall_counts)) if active_recall else 0
        if (active_action and action_total <= 0) or (active_recall and recall_total <= 0):
            raise ValueError(f'Mem2W {stage} stage has zero supervised tokens')

        self._mem2w_manual_grad_sync = False
        action_metric_sum = 0.0
        if active_action:
            for batch, count in zip(action_batches, action_counts):
                loss, _ = self._forward_branch(model, batch, stop_content_grad=False)
                term = loss * (float(count) / action_total)
                action_metric_sum += float(loss.detach().item()) * count
                if dist.is_available() and dist.is_initialized():
                    term = term * dist.get_world_size()
                yield term

        recall_metric_sum = 0.0
        if active_recall:
            for batch, count in zip(recall_batches, recall_counts):
                loss, _ = self._forward_branch(model, batch, stop_content_grad=True)
                term = loss * (float(count) / recall_total) * float(self.args.mem2w_lambda_recall)
                recall_metric_sum += float(loss.detach().item()) * count
                if dist.is_available() and dist.is_initialized():
                    term = term * dist.get_world_size()
                yield term

        self._mem2w_last_metrics = {
            'mem2w/stage': stage,
            'mem2w/loss_action': action_metric_sum / max(1, sum(action_counts)),
            'mem2w/loss_recall': recall_metric_sum / max(1, sum(recall_counts)),
            'mem2w/tokens_action': action_total,
            'mem2w/tokens_recall': recall_total,
        }
 
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = None
        for term in self._iter_loss_terms(model, inputs):
            loss = term if loss is None else loss + term
        if loss is None:
            raise ValueError('Mem2W paired data produced no branch losses')
        return (loss, None) if return_outputs else loss

    def _sync_manual_memory_grads(self):
        if not self._mem2w_manual_grad_sync or not dist.is_available() or not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        for parameter in memory_parameters(self._base_model(self.model)):
            # Do not materialize a zero gradient for a detached C-stage
            # parameter: AdamW would then apply weight decay/momentum to it.
            # First agree whether any rank actually produced a gradient.
            has_grad = torch.tensor([int(parameter.grad is not None)],
                                    dtype=torch.long, device=self.args.device)
            dist.all_reduce(has_grad, op=dist.ReduceOp.SUM)
            if has_grad.item() == 0:
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)

    def training_step(self, model, inputs, *args, **kwargs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        context = (torch.autograd.graph.save_on_cpu(pin_memory=False)
                   if self.args.mem2w_activation_offload else nullcontext())
        with context:
            total = None
            with self.compute_loss_context_manager():
                for term in self._iter_loss_terms(model, inputs):
                    self.accelerator.backward(term)
                    value = term.detach()
                    total = value if total is None else total + value
        self._sync_manual_memory_grads()
        if total is None:
            raise ValueError('Mem2W paired data produced no branch losses')
        return total

    def log(self, logs, start_time=None):
        if self._mem2w_last_metrics:
            logs = {**logs, **self._mem2w_last_metrics}
        return super().log(logs, start_time=start_time)

    def _save_model(self, output_dir=None, state_dict=None):
        if getattr(self.args, 'tuner_type', None) != 'mem2w':
            return super()._save_model(output_dir, state_dict)
        output_dir = output_dir or self.args.output_dir
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        model = self._base_model(self.model)
        Mem2WTuner.save_pretrained(
            model,
            str(output),
            state_dict=state_dict or model.state_dict(),
            safe_serialization=self.args.safe_serialization,
        )
