"""
scope3_gnn - Graph Neural Network framework for Scope 3 emissions mapping.

Modules:
    data        - Graph construction, datasets, and data loaders
    models      - GNN architectures (GAT, GraphSAGE, GCN, HeteroGNN)
    training    - Training loop, losses, and callbacks
    inference   - Prediction, attribution, and scenario simulation
    evaluation  - Metrics and benchmarks
    utils       - Shared utilities and helpers
"""

__version__ = "0.1.0"
__author__ = "virbahu"
__license__ = "MIT"

from scope3_gnn.data.graph_builder import SupplyChainGraphBuilder
from scope3_gnn.models.gat_mapper import GATEmissionMapper
from scope3_gnn.models.sage_mapper import SAGEEmissionMapper
from scope3_gnn.training.trainer import Scope3Trainer
from scope3_gnn.inference.predictor import EmissionPredictor

__all__ = [
    "SupplyChainGraphBuilder",
    "GATEmissionMapper",
    "SAGEEmissionMapper",
    "Scope3Trainer",
    "EmissionPredictor",
]
