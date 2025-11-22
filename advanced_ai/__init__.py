# FILE: advanced_ai/__init__.py
# PURPOSE: Advanced AI module for Victor ecosystem

from .tensor_core import Tensor, ODEIntegrator, SGD, Adam, BLOODLINE_HASH
from .holon_omega import (
    HLHFM, HolonΩ, DNA, LiquidGate, HoloEntry, SimpleHashEmbedder,
    _unit_norm, _circ_conv, _circ_deconv, _superpose, _cos
)

__all__ = [
    'Tensor', 'ODEIntegrator', 'SGD', 'Adam',
    'HLHFM', 'HolonΩ', 'DNA', 'LiquidGate', 'HoloEntry', 'SimpleHashEmbedder',
    '_unit_norm', '_circ_conv', '_circ_deconv', '_superpose', '_cos',
    'BLOODLINE_HASH'
]

__version__ = '2.1.0-HOLON-OMEGA-HLHFM'
