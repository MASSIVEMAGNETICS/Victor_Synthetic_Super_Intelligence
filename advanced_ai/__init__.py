# FILE: advanced_ai/__init__.py
# PURPOSE: Advanced AI module for Victor ecosystem

from .tensor_core import Tensor, ODEIntegrator, SGD, Adam, BLOODLINE_HASH
from .liquid_attention import LiquidAttentionHead, CfCCell, MixedMemoryCfC
from .hyperneat_substrate import ESHyperNEATSubstrate, CPPN, FitnessEvaluator
from .fractal_coordination import FMACPOrchestrator, CognitiveRiver, PulseBus
from .genesis_core import UpgradedVictor, genesis_loop

__all__ = [
    'Tensor', 'ODEIntegrator', 'SGD', 'Adam',
    'LiquidAttentionHead', 'CfCCell', 'MixedMemoryCfC',
    'ESHyperNEATSubstrate', 'CPPN', 'FitnessEvaluator',
    'FMACPOrchestrator', 'CognitiveRiver', 'PulseBus',
    'UpgradedVictor', 'genesis_loop',
    'BLOODLINE_HASH'
]

__version__ = '2.0.0-HYPERNEAT-LIQUID-FRACTAL'
