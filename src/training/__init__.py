"""Training package public API.

Keep executable entry points out of this initializer so CLI modules can import
trainer submodules without triggering circular imports.
"""

from .base_pretrainer import AutoencoderPretrainer, PretrainingConfig
from .incremental_trainer import IncrementalTrainer
from .predictive_coding_trainer import PredictiveCodingTrainer
from .neurogenesis_trainer import NeurogenesisTrainer

__all__ = [
    "AutoencoderPretrainer",
    "PretrainingConfig",
    "IncrementalTrainer",
    "NeurogenesisTrainer",
]
