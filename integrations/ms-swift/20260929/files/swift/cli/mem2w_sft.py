"""Native ms-swift paired Mem2W action/recall training entry point."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist

from swift.arguments import SftArguments
from swift.mem2w import get_memory_module
from swift.mem2w.dual_trainer import Mem2WDualConfig, Mem2WDualObjectiveTrainer, read_jsonl
from swift.tuners.mem2w import Mem2WTuner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train Mem2W with paired action/recall W/C objectives')
    parser.add_argument('--model', required=True)
    parser.add_argument('--action-dataset', required=True)
    parser.add_argument('--recall-dataset', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--resume-from-checkpoint')
    parser.add_argument('--template', default='qwen3_5')
    parser.add_argument('--max-length', type=int, default=4096)
    parser.add_argument('--attn-impl', default=None)
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--loss-chunk-size', type=int, default=0)
    parser.add_argument('--lazy-encode', action='store_true')
    parser.add_argument('--activation-offload', action='store_true',
                        help='offload autograd-saved activations to host RAM for very long lossless rows')
    parser.add_argument('--max-steps', type=int, default=2)
    parser.add_argument(
        '--stage-plan',
        default=None,
        help='explicit stage schedule such as W:851,C:851,W:851; overrides --max-steps',
    )
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--lambda-recall', type=float, default=1.0)
    parser.add_argument('--warmup-fraction', type=float, default=0.20)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--accumulation-steps', type=int, default=8)
    parser.add_argument('--lr-warmup-fraction', type=float, default=0.03)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mem2w-insertion-index', type=int, default=15)
    parser.add_argument('--mem2w-slots', type=int, default=512)
    parser.add_argument('--mem2w-key-dim', type=int, default=256)
    parser.add_argument('--mem2w-value-dim', type=int, default=256)
    parser.add_argument('--mem2w-rms-eps', type=float, default=1e-6)
    parser.add_argument('--mem2w-dropout', type=float, default=0.0)
    parser.add_argument('--mem2w-stop-content-grad', action='store_true')
    parser.add_argument('--mem2w-compute-chunk-size', type=int, default=2048)
    parser.add_argument('--distributed-backend', default='nccl')
    parser.add_argument('--sequence-parallel-size', type=int, default=1)
    return parser


def _parse_stage_plan(value: str | None) -> tuple[tuple[str, int], ...] | None:
    if value is None:
        return None
    plan = []
    for item in value.split(','):
        stage, separator, count = item.strip().partition(':')
        if not separator or stage not in {'W', 'C'}:
            raise ValueError('--stage-plan entries must use W:<updates> or C:<updates>')
        try:
            updates = int(count)
        except ValueError as exc:
            raise ValueError(f'invalid stage-plan update count: {count!r}') from exc
        if updates <= 0:
            raise ValueError('--stage-plan update counts must be positive')
        plan.append((stage, updates))
    if not plan:
        raise ValueError('--stage-plan must not be empty')
    return tuple(plan)


def _make_args(ns: argparse.Namespace) -> SftArguments:
    return SftArguments(
        model=ns.model,
        dataset=[str(Path(ns.action_dataset).expanduser().resolve())],
        # Keep the native SftArguments contract in sync with the paired
        # trainer. Mem2W validation requires this path even though the
        # custom loop reads/encodes the recall stream itself below.
        mem2w_recall_dataset=[str(Path(ns.recall_dataset).expanduser().resolve())],
        mem2w_lambda_recall=ns.lambda_recall,
        mem2w_stage_plan=ns.stage_plan,
        mem2w_accumulation_steps=ns.accumulation_steps,
        mem2w_loss_chunk_size=ns.loss_chunk_size,
        mem2w_activation_offload=ns.activation_offload,
        template=ns.template,
        tuner_type='mem2w',
        output_dir=str(Path(ns.output_dir).expanduser().resolve()),
        max_length=ns.max_length,
        truncation_strategy='raise',
        enable_thinking=False,
        attn_impl=ns.attn_impl,
        packing=False,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        sequence_parallel_size=ns.sequence_parallel_size,
        report_to=[],
        seed=ns.seed,
        gradient_accumulation_steps=ns.accumulation_steps,
        add_version=False,
        load_args=False,
        bf16=bool(torch.cuda.is_available()),
        mem2w_insertion_index=ns.mem2w_insertion_index,
        mem2w_slots=ns.mem2w_slots,
        mem2w_key_dim=ns.mem2w_key_dim,
        mem2w_value_dim=ns.mem2w_value_dim,
        mem2w_rms_eps=ns.mem2w_rms_eps,
        mem2w_dropout=ns.mem2w_dropout,
        mem2w_stop_content_grad=ns.mem2w_stop_content_grad,
        mem2w_compute_chunk_size=ns.mem2w_compute_chunk_size,
    )


def _distributed_setup(backend: str) -> tuple[int, int, int]:
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError('distributed Mem2W training requires CUDA/MACA devices')
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
    return rank, world_size, local_rank


def _shard_rows(rows: list[dict], rank: int, world_size: int) -> list[dict]:
    if world_size <= 1:
        return rows
    shard = rows[rank::world_size]
    # Tiny smoke fixtures may contain one row.  Replicate that row rather than
    # constructing an empty branch on ranks which are only testing collectives.
    return shard or rows[:1]


def run(ns: argparse.Namespace) -> dict:
    rank, world_size, _local_rank = _distributed_setup(ns.distributed_backend)
    is_main = rank == 0
    if is_main:
        print(json.dumps({'event': 'runtime_devices', 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                          'cuda_available': torch.cuda.is_available(),
                          'cuda_device_count': torch.cuda.device_count()}), flush=True)
    stage_plan = _parse_stage_plan(ns.stage_plan)
    if stage_plan is None and ns.max_steps < 1:
        raise ValueError('--max-steps must be positive')
    total_updates = sum(updates for _stage, updates in stage_plan) if stage_plan else ns.max_steps
    action_rows = read_jsonl(ns.action_dataset)
    recall_rows = read_jsonl(ns.recall_dataset)
    if not action_rows or not recall_rows:
        raise ValueError('action and recall JSONL files must both contain at least one row')
    if {row.get('sample_type') for row in action_rows} != {'action'}:
        raise ValueError('action dataset contains a row whose sample_type is not action')
    if {row.get('sample_type') for row in recall_rows} != {'recall'}:
        raise ValueError('recall dataset contains a row whose sample_type is not recall')
    if ns.sequence_parallel_size <= 1:
        action_rows = _shard_rows(action_rows, rank, world_size)
        recall_rows = _shard_rows(recall_rows, rank, world_size)

    args = _make_args(ns)
    model, processor = args.get_model_processor()
    if is_main:
        print(json.dumps({'event': 'model_loaded', 'first_parameter_device': str(next(model.parameters()).device),
                          'parameter_devices': sorted({str(parameter.device) for parameter in model.parameters()})}), flush=True)
    if ns.sequence_parallel_size > 1:
        if world_size != ns.sequence_parallel_size:
            raise ValueError('--sequence-parallel-size must equal WORLD_SIZE for the paired trainer')
        from swift.sequence_parallel import sequence_parallel
        sequence_parallel.prepare(ns.sequence_parallel_size, model=model, tokenizer=processor, padding_free=False)
    template = args.get_template(processor)
    template.set_mode('train')
    template.sequence_parallel_size = ns.sequence_parallel_size
    if getattr(template, 'use_model', False):
        template.model = model
    model = Mem2WTuner.prepare_model(args, model)
    insertion_index = int(getattr(model, '_mem2w_insertion_index', -1))
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if insertion_index != ns.mem2w_insertion_index:
        raise RuntimeError(
            f'Mem2W insertion mismatch: requested {ns.mem2w_insertion_index}, got {insertion_index}')
    if len(trainable) != 4 or not all(getattr(parameter, 'is_mem2w_parameter', False)
                                      for _name, parameter in trainable):
        raise RuntimeError(f'Mem2W must expose exactly four trainable tensors, got {[name for name, _ in trainable]}')
    if is_main:
        print(json.dumps({
            'event': 'mem2w_trainable_scope',
            'insertion_index_0based': insertion_index,
            'layer_number_1based': insertion_index + 1,
            'trainable': [name for name, _parameter in trainable],
        }), flush=True)
        print(json.dumps({'event': 'model_prepared', 'first_parameter_device': str(next(model.parameters()).device),
                          'parameter_devices': sorted({str(parameter.device) for parameter in model.parameters()})}), flush=True)
    if world_size > 1:
        for parameter in model.parameters():
            if getattr(parameter, 'is_mem2w_parameter', False):
                dist.broadcast(parameter.data, src=0)
    if ns.gradient_checkpointing:
        # Non-reentrant recomputation retains gradients through a frozen
        # backbone into the attached memory without requiring input gradients.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    device = next(model.parameters()).device
    if is_main:
        device_map = getattr(model, 'hf_device_map', None)
        if device_map is None:
            device_map = getattr(getattr(model, 'model', None), 'hf_device_map', None)
        print(json.dumps({'event': 'model_placement', 'first_parameter_device': str(device),
                          'parameter_devices': sorted({str(parameter.device) for parameter in model.parameters()}),
                          'device_map': device_map}, default=str), flush=True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=ns.learning_rate,
    )
    warmup_steps = max(1, math.ceil(ns.lr_warmup_fraction * total_updates))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_updates - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start_step = 0
    if ns.resume_from_checkpoint:
        checkpoint = Path(ns.resume_from_checkpoint).expanduser().resolve()
        Mem2WTuner.from_pretrained(model, str(checkpoint), is_trainable=True)
        optimizer_path = checkpoint / 'optimizer.pt'
        state_path = checkpoint / 'mem2w_dual_state.json'
        if optimizer_path.is_file():
            optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu', weights_only=True))
        scheduler_path = checkpoint / 'scheduler.pt'
        if scheduler_path.is_file():
            scheduler.load_state_dict(torch.load(scheduler_path, map_location='cpu', weights_only=True))
        if state_path.is_file():
            start_step = int(json.loads(state_path.read_text(encoding='utf-8')).get('global_step', 0))
        rng_path = checkpoint / 'rng_state.pt'
        if rng_path.is_file():
            rng_state = torch.load(rng_path, map_location='cpu', weights_only=True)
            torch.set_rng_state(rng_state['cpu'])
            if torch.cuda.is_available() and rng_state.get('cuda') is not None:
                torch.cuda.set_rng_state_all(rng_state['cuda'])

    trainer = Mem2WDualObjectiveTrainer(
        model,
        template,
        action_rows,
        recall_rows,
        optimizer,
        Mem2WDualConfig(
            lambda_recall=ns.lambda_recall,
            warmup_fraction=ns.warmup_fraction,
            max_grad_norm=ns.max_grad_norm,
            loss_chunk_size=ns.loss_chunk_size,
            lazy_encode=ns.lazy_encode,
            activation_offload=ns.activation_offload,
            sequence_parallel_size=ns.sequence_parallel_size,
            stage_plan=stage_plan,
        ),
        total_updates=total_updates,
        scheduler=scheduler,
        accumulation_steps=ns.accumulation_steps,
        device=device,
    )
    output_dir = Path(ns.output_dir).expanduser().resolve()
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_config = {**vars(ns), 'rank': rank, 'world_size': world_size}
        (output_dir / 'run_config.json').write_text(json.dumps(run_config, indent=2) + '\n', encoding='utf-8')
    logs = []
    for update_index in range(start_step + 1, total_updates + 1):
        if is_main:
            print(json.dumps({'event': 'step_start', 'global_step': update_index}), flush=True)
        metrics = trainer.train_step(update_index)
        if is_main:
            print(json.dumps({'event': 'step_completed', 'global_step': update_index, **metrics}), flush=True)
            logs.append(metrics)
            checkpoint = output_dir / f'checkpoint-{update_index}'
            Mem2WTuner.save_pretrained(model, str(checkpoint), state_dict=model.state_dict())
            torch.save(optimizer.state_dict(), checkpoint / 'optimizer.pt')
            torch.save(scheduler.state_dict(), checkpoint / 'scheduler.pt')
            torch.save(
                {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                checkpoint / 'rng_state.pt')
            (checkpoint / 'mem2w_dual_state.json').write_text(
                json.dumps({'global_step': update_index, 'total_updates': total_updates, 'stage': metrics['stage']}, indent=2)
                + '\n', encoding='utf-8')
            with (output_dir / 'training_metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'global_step': update_index, **metrics}, ensure_ascii=False) + '\n')
        if world_size > 1:
            dist.barrier()
    summary = {
        'status': 'completed',
        'global_step': max(start_step, total_updates),
        'logs': logs,
        'trainable': [name for name, parameter in model.named_parameters() if parameter.requires_grad],
        'output_dir': str(output_dir),
        'memory_shapes': {key: list(value.shape) for key, value in get_memory_module(model).state_dict().items()},
        'accumulation_steps': ns.accumulation_steps,
        'lr_warmup_fraction': ns.lr_warmup_fraction,
        'distributed_backend': ns.distributed_backend if world_size > 1 else None,
        'world_size': world_size,
        'stage_plan': stage_plan,
        'total_updates': total_updates,
    }
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'training_summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return summary


def main() -> None:
    run(build_parser().parse_args())


if __name__ == '__main__':
    main()
