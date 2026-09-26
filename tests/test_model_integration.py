import pytest


torch = pytest.importorskip("torch")
from torch import nn

from mem2w.config import MemoryConfig
from mem2w.qwen_integration import attach_memory, get_memory_module, memory_parameters


class _Block(nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return (hidden_states + 0.1,)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 8, "num_hidden_layers": 16})()
        self.model = nn.Module()
        self.model.add_module("layers", nn.ModuleList([_Block() for _ in range(16)]))


def test_attach_after_block_and_bypass():
    model = _TinyModel()
    attach_memory(
        model,
        config=MemoryConfig(hidden_size=8, slots=4, key_dim=3, value_dim=5, insertion_index=15),
    )
    assert model._mem2w_insertion_index == 15
    assert type(model.model.layers[15]).__name__ == "MemoryAfterDecoderBlock"
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable == [
        "model.layers.15.memory.W_Q",
        "model.layers.15.memory.K",
        "model.layers.15.memory.V",
        "model.layers.15.memory.W_O",
    ]
    assert len(memory_parameters(model)) == 4

    hidden = torch.randn(1, 3, 8)
    wrapper = model.model.layers[15]
    wrapper.memory_enabled = False
    expected = hidden + 0.1
    actual = wrapper(hidden)[0]
    assert torch.equal(actual, expected)
    assert get_memory_module(model) is wrapper.memory
