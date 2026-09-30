import pytest

torch = pytest.importorskip('torch')
from torch import nn

from swift.mem2w import Mem2WConfig, attach_memory, freeze_memory_only, get_memory_module
from swift.mem2w.dual_trainer import (Mem2WDualObjectiveTrainer, parse_stage_plan,
                                      stage_and_local_update, stage_for_step)
from swift.tuner_plugin.mapping import tuners_map


class _Block(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, **kwargs):
        return (self.proj(hidden_states), 'cache')


class _TinyQwen(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Block(hidden_size) for _ in range(16)])
        self.config = type('Config', (), {'hidden_size': hidden_size})()


def test_mem2w_is_a_native_tuner_and_preserves_layer_shape():
    assert 'mem2w' in tuners_map
    model = _TinyQwen()
    attach_memory(model, Mem2WConfig(hidden_size=8, insertion_index=15, slots=4, key_dim=3, value_dim=3))
    assert len(model.model.layers) == 16
    freeze_memory_only(model)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert len(trainable) == 4
    assert {parameter.dtype for parameter in get_memory_module(model).parameters()} == {torch.float32}
    hidden = torch.randn(2, 5, 8)
    layer = model.model.layers[15]
    layer._mem2w_enabled = False
    bypass = layer(hidden)[0]
    assert torch.equal(bypass, layer.proj(hidden))
    assert get_memory_module(model).W_Q.shape == (8, 3)


def test_c_stage_clears_content_gradients_before_adamw():
    model = _TinyQwen()
    attach_memory(model, Mem2WConfig(hidden_size=8, insertion_index=15, slots=4, key_dim=3, value_dim=3))
    memory = freeze_memory_only(model)
    module = get_memory_module(memory)
    for parameter in module.parameters():
        parameter.grad = torch.ones_like(parameter)
    Mem2WDualObjectiveTrainer._clear_stage_frozen_gradients(memory, 'C')
    assert module.W_Q.grad is not None
    assert module.K.grad is not None
    assert module.V.grad is None
    assert module.W_O.grad is None


def test_shared_stage_plan_has_consistent_global_and_local_cursors():
    plan = parse_stage_plan('W:2,C:3,W:2')
    assert stage_for_step(3, 7, 0.2, plan) == 'C'
    assert stage_and_local_update(3, 7, 0.2, plan) == ('C', 1)
    assert stage_and_local_update(6, 7, 0.2, plan) == ('W', 1)
