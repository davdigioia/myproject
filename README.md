# Spatio-Temporal Epistemic Claim Retrieval (ST-ECR)

A modular deep learning architecture for uncertainty-aware multi-asset portfolio allocation under non-stationarity.

## Overview

ST-ECR is a novel systems contribution that elevates epistemic uncertainty from a passive output to an active **computation routing signal**. The system decomposes non-stationary forecasting into three orthogonal modules:

1. **Temporal Transformer with Dynamic Graph Attention** — adapts representation to regime shifts
2. **Episodic Trajectory Memory** — stabilizes calibration without sacrificing returns
3. **Empirically Calibrated Conformal Abstention Gate** — provides selective decision-making

Epistemic uncertainty drives a tri-level routing policy:
- **Route 1 (ECR Trigger)**: Memory queries for uncertain assets only
- **Route 2 (Conformal Gate)**: Selective abstention calibrated to target abstain rate
- **Route 3 (tanh-Kelly)**: Position sizing with epistemic uncertainty discount

## Key Results

On spatial forecasting benchmark (BikeNYC-style data):
- **12.9% test loss reduction** vs. PredRNN++ baseline
- Epistemic routing shows monotone improvement across all components
- 5-arm factorial study isolates mechanistic contributions

## Repository Structure

```
ST-ECR-Github/
├── src/
│   ├── financial_st_ecr_v3.py           # Core PyTorch implementation
│   ├── train_bikenyc_paper_style.py     # Main training script
│   ├── benchmark_ablation.py            # 5-arm factorial study
│   ├── run_revolution_bikenyc.py        # Full experiment runner
│   ├── preprocess_bikenyc_h5.py         # Data preprocessing
│   ├── dataset_preparation.py           # Dataset setup
│   ├── spatio_temporal_activation.py    # Baseline model
│   ├── generate_plots.py                # Figure generation
│   └── generate_dummy_bair.py           # Synthetic data generator
├── figures/
│   ├── fig_benchmark.pdf            # Benchmark results
│   ├── fig_ablation.pdf             # 5-arm ablation study
│   └── fig_calibration.pdf          # Uncertainty calibration analysis
├── docs/
│   └── ST_ECR_PhD_Paper.tex         # Full paper (15 pages)
├── notebooks/                        # Jupyter notebooks (optional)
├── data/
│   └── preprocessed/                # Preprocessed datasets
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── .gitignore                       # Git ignore patterns
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.10+
- NumPy, Pandas, SciPy, Matplotlib

### Setup

```bash
# Clone repository
git clone https://github.com/ucesigi/ST-ECR.git
cd ST-ECR

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Replication Workflow

### 1. Prepare Data

```bash
cd src

# Generate synthetic data for quick testing
python generate_dummy_bair.py

# Or preprocess real BikeNYC dataset (H5 format required)
python preprocess_bikenyc_h5.py \
  --input NYC14_M16x8_T60_NewEnd.h5 \
  --output ../data/preprocessed/
```

### 2. Train Models

```bash
# Train full ST-ECR (paper results)
python run_revolution_bikenyc.py \
  --model st-ecr-full \
  --dataset bikenyc \
  --epochs 100 \
  --batch-size 32 \
  --save-checkpoint ../checkpoints/

# Train routed variant
python train_bikenyc_paper_style.py \
  --model st-ecr-routed \
  --epochs 100

# Train baselines
python spatio_temporal_activation.py \
  --model predrnn-pp \
  --epochs 100
```

### 3. Run 5-Arm Ablation Study

```bash
# Execute all 5 factorial arms
python benchmark_ablation.py \
  --arms static,memory_only,no_mem_online,no_conformal,online_full \
  --output ../results/

# Produces: factorial_results.json + per-arm metrics
```

### 4. Generate Figures (Reproduces Paper Figures 2-4)

```bash
# Generate all benchmark and ablation visualizations
python generate_plots.py \
  --results ../results/factorial_results.json \
  --output ../figures/

# Output:
#   fig_benchmark.pdf     → Main benchmark results (Figure 2)
#   fig_ablation.pdf      → 5-arm ablation study (Figure 3)
#   fig_calibration.pdf   → Uncertainty calibration (Figure 4)
```

### Quick Start (30 minutes, Synthetic Data)

```bash
cd src
python generate_dummy_bair.py          # 2 min
python train_bikenyc_paper_style.py --epochs 5 --batch-size 4  # 10 min
python generate_plots.py               # 1 min
# Figures now in ../figures/
```

## Usage: Core Model

```python
from financial_st_ecr_v3 import ST_ECR_Model

# Initialize
model = ST_ECR_Model(
    input_dim=4,
    hidden_dim=64,
    num_assets=16,
    num_ensemble_members=5,
    memory_capacity=512,
    target_abstain_rate=0.1
)

# Forward pass: (B, T, N, F) → outputs
X = torch.randn(32, 12, 16, 4)  # batch, time, assets, features
outputs = model(X)

# Extract predictions & routing decisions
mu = outputs['mu']                    # Point predictions
sigma_al = outputs['sigma_al']        # Aleatoric uncertainty
sigma_ep = outputs['sigma_ep']        # Epistemic uncertainty
portfolio = outputs['portfolio']      # Allocation weights p ∈ [-1,1]
mask = outputs['mask']                # Abstention mask m ∈ {0,1}
```

### Key Model Architecture

**Stage 1: Representation**
```
Input (B×T×N×F) 
  → Temporal Transformer (channel-independent)
  → Dynamic Graph Attention (regime-adaptive)
  → Ensemble Head (K members) 
  → σ_draft (epistemic uncertainty)
```

**Stage 2: Retrieval (Route 1)**
```
σ_draft > τ_p → Query episodic memory
  → Retrieve regime-similar trajectory embedding
  → Refined representation H_ref
  → Ensemble Head (refined)
  → σ_after (updated epistemic uncertainty)
```

**Stage 3: Allocation (Routes 2-3)**
```
σ_after > Ω → Abstention mask m
  ↓
  p_i = tanh(μ_i / (σ_al² + λ_ep·σ_after))·(1 - m_i)
  ↓
  Portfolio p ∈ [-1, 1], mask m ∈ {0,1}
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.10+
- NumPy, Pandas, SciPy, Matplotlib

### Setup

## Paper

The full paper is available in `docs/ST_ECR_PhD_Paper.tex`. To compile:

```bash
pdflatex -> bibtex -> pdflatex -> pdflatex
```

Or use an IDE like TeXstudio/Overleaf.

### Main Contributions

1. **Uncertainty as computation routing** — First to use epistemic uncertainty as an active control signal across memory, representation, and allocation
2. **Mechanistic ablation study** — 5-arm factorial design isolates individual component effects
3. **Resolution objective** — Trains memory module independently to reduce epistemic uncertainty
4. **Practical validation** — Empirical evidence for uncertainty-driven systems paradigm

## Key Findings from Ablation Study

| Arm | Online Head | Memory | Conformal Gate | Effect |
|-----|-------------|--------|---|---|
| `static` | ✗ | ✗ | FROZEN | Baseline |
| `memory_only` | ✗ | ✓ | FROZEN | Retrieval only |
| `no_mem_online` | ✓ | ✗ | ✓ | Head + conformal |
| `no_conformal` | ✓ | ✓ | FROZEN | Head + memory |
| `online_full` | ✓ | ✓ | ✓ | Full ST-ECR |

**Key Insights:**
- Memory is a **calibration stabilizer** — reduces ECE independently of Sharpe
- Conformal gating **resolves the conformal paradox** — improves selectivity at calibration cost
- MaxDD increases are **structural sparsification** — breadth reduction, not directional error

## Limitations & Future Work

- Conformal guarantees require exchangeability (financial data violates this)
- Single-run benchmark results; multi-seed study deferred
- Scalability bottleneck for N > 500 assets (O(N²) graph attention)
- Learned eviction policy for memory buffer (currently FIFO)

## Citation

```bibtex
@article{stecrephd2026,
  title={Spatio-Temporal Epistemic Claim Retrieval: 
         Uncertainty-Aware Multi-Asset Portfolio Allocation 
         in Non-Stationary Environments},
  author={Anonymous},
  year={2026},
  note={Submitted for review}
}
```

## License

This project is licensed under the MIT License — see LICENSE file for details.

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Last Updated:** May 10, 2026  
**Paper Status:** Submitted for review  
**Code Status:** Research-grade (not production)
