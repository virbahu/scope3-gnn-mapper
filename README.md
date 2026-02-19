# 🌍 scope3-gnn-mapper

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3+-purple.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GHG Protocol](https://img.shields.io/badge/Standard-GHG%20Protocol-green.svg)](https://ghgprotocol.org/)

> **Graph Neural Network framework for mapping, attributing, and predicting Scope 3 greenhouse gas emissions across multi-tier supply chains.**

---

## 📋 Overview

**scope3-gnn-mapper** is an open-source Python framework that applies Graph Neural Networks (GNNs) to the challenge of Scope 3 supply chain emissions mapping. Traditional approaches to Scope 3 accounting suffer from data gaps, manual attribution errors, and inability to propagate uncertainty through complex multi-tier supplier networks. This framework addresses these limitations by:

- Representing supplier relationships as a **heterogeneous graph** where nodes are companies/facilities and edges are procurement flows
- Using **Graph Attention Networks (GAT)** and **GraphSAGE** to propagate emission intensities through the supply chain graph
- Enabling **uncertainty quantification** at each node via Monte Carlo Dropout or Bayesian GNN layers
- Providing **automated hotspot detection** to identify high-emission upstream tiers
- Supporting **what-if scenario analysis** for supplier switching and decarbonization roadmaps

---

## ⚙️ Installation

```bash
git clone https://github.com/virbahu/scope3-gnn-mapper.git
cd scope3-gnn-mapper
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## 🚀 Quick Start

```python
from scope3_gnn.data.graph_builder import SupplyChainGraphBuilder
from scope3_gnn.models.gat_mapper import GATEmissionMapper
from scope3_gnn.training.trainer import Scope3Trainer

builder = SupplyChainGraphBuilder(emission_factor_db="exiobase3")
graph = builder.from_csv("data/raw/suppliers.csv", "data/raw/flows.csv")

model = GATEmissionMapper(in_channels=96, hidden_channels=256, num_layers=4, heads=8)
trainer = Scope3Trainer(model=model, graph=graph, epochs=200)
trainer.fit()
```

---

## 📄 License

MIT License
