"""
Graph optimization module for SparseOpt.
"""

from .base import GraphPass
from .optimizer import GraphOptimizer
from .passes.dead_node import DeadNodeEliminationPass
from .passes.reordering import NodeReorderingPass
from .passes.linear_fusion import LinearGELUFusion, LinearReLUFusion
from .passes.conv_fusion import ConvBatchNormReLUFusion
from .passes.dropout_elimination import DropoutElimination

__all__ = [
    'GraphPass',
    'GraphOptimizer',
    'DeadNodeEliminationPass',
    'NodeReorderingPass',
    'LinearGELUFusion',
    'LinearReLUFusion',
    'ConvBatchNormReLUFusion',
    'DropoutElimination'
] 