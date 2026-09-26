import pytest


torch = pytest.importorskip("torch")

from mem2w.checkpointing import load_memory_checkpoint, save_memory_checkpoint
from mem2w.memory_layer import PersistentKVMemory


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = PersistentKVMemory(hidden_size=8, slots=4, key_dim=3, value_dim=5)


def test_memory_safetensors_roundtrip(tmp_path):
    model = Tiny()
    out = save_memory_checkpoint(model, tmp_path, {"mode": "action"})
    assert (out / "memory.safetensors").is_file()
    original = {name: value.detach().clone() for name, value in model.named_parameters() if getattr(value, "is_memory_parameter", False)}
    with torch.no_grad():
        model.memory.W_Q.zero_()
    load_memory_checkpoint(model, tmp_path)
    for name, value in model.named_parameters():
        if getattr(value, "is_memory_parameter", False):
            assert torch.equal(value, original[name])
