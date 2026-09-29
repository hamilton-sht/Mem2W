# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import PeftTuner, Tuner


def __getattr__(name):
    """Load the tuner registry lazily.

    ``mapping`` imports the concrete tuners, including the native Mem2W tuner.
    The Mem2W tuner itself needs the base ``Tuner`` class, so importing the
    registry eagerly here creates a package-initialisation cycle.  Keeping the
    registry lazy preserves the public ``from swift.tuner_plugin import
    tuners_map`` API while allowing concrete tuners to import the base safely.
    """
    if name == 'tuners_map':
        from .mapping import tuners_map
        return tuners_map
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = ['PeftTuner', 'Tuner', 'tuners_map']
