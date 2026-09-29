import pytest

torch = pytest.importorskip('torch')
from torch import nn

from swift.mem2w import Mem2WConfig, attach_memory, freeze_memory_only, get_memory_module
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
    hidden = torch.randn(2, 5, 8)
    layer = model.model.layers[15]
    layer._mem2w_enabled = False
    bypass = layer(hidden)[0]
    assert torch.equal(bypass, layer.proj(hidden))
    assert get_memory_module(model).W_Q.shape == (8, 3)
