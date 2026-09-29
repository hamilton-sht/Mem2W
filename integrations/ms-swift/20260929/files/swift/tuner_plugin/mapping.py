# Copyright (c) ModelScope Contributors. All rights reserved.
from .dummy import DummyTuner
from .ia3 import IA3Tuner
from .lora_llm import LoRALLMTuner
from swift.tuners.mem2w import Mem2WTuner

tuners_map = {
    'ia3': IA3Tuner,
    'lora_llm': LoRALLMTuner,
    'dummy': DummyTuner,
    'mem2w': Mem2WTuner,
}
