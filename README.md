# 🌍 scope3-gnn-mapper

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3+-purple.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GHG Protocol](https://img.shields.io/badge/Standard-GHG%20Protocol-green.svg)](https://ghgprotocol.org/)
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-Virbahu%20Jain-4285F4?logo=googlescholar&logoColor=white)](https://scholar.google.com/citations?user=4SN8o-QAAAAJ&hl=en)

> **Graph Neural Network framework for mapping, attributing, and predicting Scope 3 GHG emissions across multi-tier supply chains.**
>
> ---
>
> ## 📋 Overview
>
> **scope3-gnn-mapper** is an open-source Python framework that applies Graph Neural Networks (GNNs) to the Scope 3 supply chain emissions mapping challenge.
>
> Traditional Scope 3 accounting relies on spend-based estimates, supplier surveys, and static IO tables — methods that introduce systematic errors of **40–70%** and fail to capture the dynamic, multi-tier nature of modern supply chains.
>
> This framework addresses these limitations by:
>
> - Representing supplier relationships as a **heterogeneous graph** where nodes are companies/facilities and edges are procurement flows
> - - Using **Graph Attention Networks (GAT)** and **GraphSAGE** to propagate emission intensities through the supply chain graph
>   - - Enabling **uncertainty quantification** at each node via Monte Carlo Dropout or Bayesian GNN layers
>     - - Providing **automated hotspot detection** to identify high-emission upstream tiers
>       - - Supporting **what-if scenario analysis** for supplier switching and decarbonization roadmaps
>        
>         - ---
>
> ## 🖼️ System Overview
>
> ![Scope 3 Supply Chain Mapping](https://raw.githubusercontent.com/virbahu/scope3-gnn-mapper/main/docs/images/scope3_overview.png)
>
> > *Multi-tier supplier graph showing emission intensity propagation across Tier 1–4 suppliers. Node size = annual spend; node color = kgCO2e/USD intensity.*
> >
> > ---
> >
> > ## 🏗️ Architecture Diagram
> >
> > ```
> > ╔══════════════════════════════════════════════════════════════════╗
> > ║               SCOPE 3 GNN MAPPER — SYSTEM ARCHITECTURE          ║
> > ╠══════════════════════════════════════════════════════════════════╣
> > ║                                                                  ║
> > ║  [Raw Supply Chain Data]  [Emission Factor DBs]  [IoT Sensors]  ║
> > ║    CSV / ERP / REST API   EXIOBASE / ECOINVENT     MQTT/REST    ║
> > ║           │                      │                     │         ║
> > ║           └──────────────────────┴─────────────────────┘         ║
> > ║                                  │                               ║
> > ║                    ┌─────────────▼────────────┐                  ║
> > ║                    │  Graph Construction Layer │                  ║
> > ║                    │  SupplyChainGraphBuilder  │                  ║
> > ║                    │  • Node: Company/Facility │                  ║
> > ║                    │  • Edge: Procurement Flow │                  ║
> > ║                    │  • HeteroData (PyG)       │                  ║
> > ║                    └─────────────┬────────────┘                  ║
> > ║                                  │                               ║
> > ║         ┌────────────────────────┼──────────────────┐            ║
> > ║         │                        │                  │            ║
> > ║  ┌──────▼───────┐   ┌────────────▼──────────┐ ┌────▼─────────┐  ║
> > ║  │  GAT Mapper  │   │  GraphSAGE (Inductive) │ │ Bayesian GNN │  ║
> > ║  │  8 Heads×4L  │   │  Multi-hop Aggregation │ │ Uncertainty  │  ║
> > ║  └──────┬───────┘   └────────────┬──────────┘ └────┬─────────┘  ║
> > ║         └────────────────────────┴──────────────────┘            ║
> > ║                                  │                               ║
> > ║                    ┌─────────────▼────────────┐                  ║
> > ║                    │  Emission Attribution     │                  ║
> > ║                    │  • Scope 3 Cat 1–15       │                  ║
> > ║                    │  • Tier attribution       │                  ║
> > ║                    │  • Confidence intervals   │                  ║
> > ║                    └─────────────┬────────────┘                  ║
> > ║                                  │                               ║
> > ║        ┌─────────────────────────┼─────────────────────┐         ║
> > ║        │                         │                     │         ║
> > ║  ┌─────▼──────┐       ┌──────────▼────────┐   ┌────────▼──────┐  ║
> > ║  │  Hotspot   │       │ Scenario Analysis │   │  Dashboard    │  ║
> > ║  │  Detection │       │ Supplier Switching│   │  Streamlit +  │  ║
> > ║  │  & Ranking │       │ Decarbonization   │   │  FastAPI      │  ║
> > ║  └────────────┘       └───────────────────┘   └───────────────┘  ║
> > ╚══════════════════════════════════════════════════════════════════╝
> > ```
> >
> > ### GNN Emission Propagation
> >
> > ![GNN Architecture](https://mermaid.ink/img/pako:eNptkMFqwzAMhl9F-LRC-wI-bKXrYGxQWHcpPgi3SQxxLGNboCz47nNSGMsu-mT9_n9JCDNGJBv_VBQF4Zr-nEi5nKvFkx3JTkfkizGb5hKSHWJBlpADiLBjEOFuBcKtJOJkTEOANRrGqODJxcUDi6yJ3lA4qHdBh_nEsPPdRn7RLESJK5tD2KiKRdYzBpfH57vfrg?type=png)
> >
> > ---
> >
> > ## ❗ Problem Statement
> >
> > ### The Invisible 90% Problem
> >
> > Scope 3 emissions represent on average **70–90% of total corporate GHG footprint**, yet remain the most poorly measured category in enterprise sustainability reporting.
> >
> > | Challenge | Traditional Approach | GNN Solution |
> > |---|---|---|
> > | **Data Gaps** | Spend-based estimates ±40% error | Graph imputation from neighboring nodes |
> > | **Multi-Tier Blindness** | Tier 1 coverage only | N-tier propagation through graph |
> > | **Static Data** | Annual surveys (12-month lag) | Real-time IoT + procurement feeds |
> > | **Attribution Errors** | Manual category mapping | Automated GHG Protocol classification |
> > | **Uncertainty** | Point estimates only | Full posterior distributions per supplier |
> > | **Scalability** | 50–100 suppliers max | 100,000+ nodes via mini-batch training |
> >
> > > *"A supplier network is a graph. Emission factors flow through procurement edges. GNNs were built to reason over exactly this structure."*
> > >
> > > ---
> > >
> > > ## ✅ Solution Overview
> > >
> > > ### 3-Phase Emission Intelligence Pipeline
> > >
> > > **Phase 1 — Graph Construction**
> > >
> > > Raw supplier CSV and procurement flow data is ingested, normalized, and assembled into a PyTorch Geometric `HeteroData` object. Each supplier becomes a node with a 96-dimensional feature vector encoding industry codes, country of origin, revenue band, and matched emission factor database entries. Each procurement relationship becomes a directed edge with spend, volume, and commodity metadata.
> > >
> > > **Phase 2 — GNN-Based Emission Attribution**
> > >
> > > The `GATEmissionMapper` runs 4 attention layers with 8 heads each, progressively aggregating neighborhood information to estimate per-supplier emission intensities for all 15 GHG Protocol Scope 3 categories. Bayesian dropout layers provide posterior uncertainty estimates without requiring full Bayesian inference.
> > >
> > > **Phase 3 — Hotspot Detection and Scenario Analysis**
> > >
> > > An R-Value scoring algorithm ranks suppliers by reduction potential per unit effort. The Scenario Engine models supplier switching, technology adoption, and procurement changes to generate science-aligned decarbonization roadmaps with quantified abatement potential.
> > >
> > > ---
> > >
> > > ## 💻 Code, Installation & Analysis
> > >
> > > ### Prerequisites
> > >
> > > | Requirement | Version |
> > > |---|---|
> > > | Python | 3.9+ |
> > > | CUDA (optional) | 11.8+ |
> > > | RAM | 16 GB minimum (32 GB recommended) |
> > > | Disk Space | 5 GB (emission factor databases) |
> > >
> > > ### Installation
> > >
> > > ```bash
> > > # Clone the repository
> > > git clone https://github.com/virbahu/scope3-gnn-mapper.git
> > > cd scope3-gnn-mapper
> > >
> > > # Create virtual environment
> > > python -m venv .venv
> > > source .venv/bin/activate  # Linux/Mac
> > > # .venv\Scripts\activate    # Windows
> > >
> > > # Install with all dependencies
> > > pip install -e ".[dev]"
> > >
> > > # Download emission factor databases
> > > python scripts/download_emission_factors.py --databases exiobase3,ecoinvent38
> > > ```
> > >
> > > ### Quick Start
> > >
> > > ```python
> > > from scope3_gnn.data.graph_builder import SupplyChainGraphBuilder
> > > from scope3_gnn.models.gat_mapper import GATEmissionMapper
> > > from scope3_gnn.training.trainer import Scope3Trainer
> > > from scope3_gnn.analysis.hotspot import HotspotDetector
> > >
> > > # 1. Build supply chain graph
> > > builder = SupplyChainGraphBuilder(emission_factor_db="exiobase3")
> > > graph = builder.from_csv(
> > >     suppliers_path="data/raw/suppliers.csv",
> > >     flows_path="data/raw/flows.csv"
> > > )
> > > print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
> > >
> > > # 2. Train GNN model
> > > model = GATEmissionMapper(
> > >     in_channels=96, hidden_channels=256, num_layers=4, heads=8
> > > )
> > > trainer = Scope3Trainer(model=model, graph=graph, epochs=200)
> > > trainer.fit()
> > >
> > > # 3. Attribution and hotspot analysis
> > > results = model.attribute_emissions(graph)
> > > detector = HotspotDetector(top_k=20)
> > > hotspots = detector.detect(results, graph)
> > > hotspots.plot_sankey()
> > > ```
> > >
> > > ### Sample Output
> > >
> > > ```
> > > SCOPE 3 EMISSION ATTRIBUTION REPORT — 2026
> > > ──────────────────────────────────────────────────────────────
> > >  Category                    tCO2e/year    % Total   Confidence
> > > ──────────────────────────────────────────────────────────────
> > >  Cat 1: Purchased Goods       45,823        34.2%      ±6.1%
> > >  Cat 4: Upstream Transport    12,104         9.0%      ±8.3%
> > >  Cat 11: Use of Sold Prods    38,912        29.0%     ±11.2%
> > >  Cat 12: End-of-Life          14,332        10.7%      ±9.4%
> > >  Other Categories (11)        23,057        17.2%     ±14.1%
> > > ──────────────────────────────────────────────────────────────
> > >  TOTAL SCOPE 3               134,228       100.0%      ±7.8%
> > > ──────────────────────────────────────────────────────────────
> > >  Graph Coverage: 847 direct suppliers | Model Accuracy: 91.3%
> > > ```
> > >
> > > ---
> > >
> > > ## 📦 Dependencies
> > >
> > > ```toml
> > > [tool.poetry.dependencies]
> > > python = "^3.9"
> > > torch = "^2.0"
> > > torch-geometric = "^2.3"
> > > torch-scatter = "*"
> > > torch-sparse = "*"
> > > networkx = "^3.1"
> > > neo4j = "^5.0"
> > > pandas = "^2.0"
> > > numpy = "^1.24"
> > > scikit-learn = "^1.3"
> > > fastapi = "^0.104"
> > > streamlit = "^1.28"
> > > plotly = "^5.17"
> > > pydantic = "^2.0"
> > > ```
> > >
> > > ### Emission Factor Databases
> > >
> > > | Database | Coverage | Frequency |
> > > |---|---|---|
> > > | EXIOBASE 3 | 44 countries, 163 sectors | Annual |
> > > | ecoinvent 3.8 | 16,000+ processes | Annual |
> > > | GHG Protocol | 300+ factors | Quarterly |
> > > | EPA EEIO | US industry IO tables | Annual |
> > > | GLEC Framework | Transport factors | Bi-annual |
> > >
> > > ---
> > >
> > > ## 👤 Author
> > >
> > > <img src="https://avatars.githubusercontent.com/u/virbahu" width="80" align="left" style="margin-right:15px; border-radius:50%"/>

**Virbahu Jain** — Founder & CEO, [Quantisage](https://quantisage.com)

> *Building the AI Operating System for Scope 3 emissions management and supply chain decarbonization.*
>
> <br clear="left"/>

| | |
|---|---|
| 🎓 **Education** | MBA, Kellogg School of Management, Northwestern University |
| 🏭 **Experience** | 20+ years across manufacturing, life sciences, energy & public sector |
| 🌍 **Scope** | Supply chain operations on five continents |
| 📝 **Research** | Peer-reviewed publications on AI in sustainable supply chains |
| 🔬 **Patents** | IoT and AI solutions for manufacturing and logistics |

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin)](https://linkedin.com/in/virbahu)
[![GitHub](https://img.shields.io/badge/GitHub-virbahu-181717?logo=github)](https://github.com/virbahu)
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-Publications-4285F4?logo=googlescholar&logoColor=white)](https://scholar.google.com/citations?user=4SN8o-QAAAAJ&hl=en)
[![Quantisage](https://img.shields.io/badge/Company-Quantisage-00C853)](https://quantisage.com)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

![Quantisage](https://img.shields.io/badge/Quantisage-Open%20Source%20Initiative-00C853?style=for-the-badge)
![Supply Chain](https://img.shields.io/badge/AI-Supply%20Chain-blue?style=for-the-badge)
![Climate](https://img.shields.io/badge/Climate-Tech-green?style=for-the-badge)

<sub>Part of the <strong>Quantisage Open Source Initiative</strong> | AI × Supply Chain × Climate</sub>
</div>
