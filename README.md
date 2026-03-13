# Hybrid Weather Model: Dinosaur Dynamical Core + GNN Correction

A hybrid weather forecasting model that combines [Dinosaur](https://github.com/google-research/dinosaur)'s differentiable spectral dynamical core with a Graph Neural Network (GNN) correction module, trained end-to-end on ERA5 reanalysis data using JAX.

This codebase accompanies the thesis *"Revealing Cross-Scale Interactions in Hybrid Weather Models"* (University of Amsterdam, ETH Zurich, 2025).

## Overview

Traditional numerical weather prediction solves the primitive equations on a sphere but accumulates systematic errors over time. Neural weather models learn corrections from data but can violate physical constraints. This project bridges both approaches:

1. **Dinosaur dynamical core** — solves the hydrostatic primitive equations in spectral (spherical harmonic) space using a semi-implicit time integration scheme.
2. **GNN correction module** — a message-passing graph neural network (built with [Jraph](https://github.com/google/jraph)) that learns residual corrections to the dynamical core's tendencies at each correction step.
3. **End-to-end training** — the full hybrid model is trained through the dynamical core via JAX's automatic differentiation, using ERA5 reanalysis as ground truth.
4. **Gain experiment (cross-scale interactions)** — beyond forecast error, the repository studies correction gain as a function of wavenumber to quantify where the learned correction amplifies or damps dynamics across scales.

The model supports multiple spectral truncations (TL31, TL47, TL63, TL95, TL127) and a curriculum training strategy that progressively extends the forecast horizon from 6 hours to multiple days.

## Architecture

```
ERA5 input state (modal space)
        │
        ▼
┌─────────────────────┐
│   Digital Filter     │
│   Initialization     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐     ┌──────────────────────┐
│  Dinosaur Dycore    │────▶│  GNN Correction      │
│  (Primitive Eqs.)   │     │  (Jraph GraphNet)    │
│  Semi-implicit step │     │  Residual tendencies │
└────────┬────────────┘     └──────────┬───────────┘
         │                             │
         ▼                             ▼
┌─────────────────────────────────────────────────┐
│  Combined state = dycore + filtered correction   │
│  (exponential spectral filter applied)           │
└────────┬────────────────────────────────────────┘
         │
         ▼  (repeat for prediction_range × 6h)
    Final forecast
```

## Repository Structure

```
├── gcm.py                    # HybridModel (main model combining dycore + GNN)
├── dycore.py                 # DynamicalCore wrapper around Dinosaur
├── twod_gnn_correction.py    # 2D GNN correction module (per-level)
├── threed_gnn_correction.py  # 3D GNN correction module (cross-level)
├── experiment.py             # Training and evaluation loop
├── dataloader.py             # ERA5 data loading and batching
├── data_cacher.py            # ERA5 preprocessing and caching to Zarr
├── data_utils.py             # Data utility functions
├── utils.py                  # Training utilities (pmap, metrics, I/O)
├── weatherbench.py           # WeatherbenchX evaluation pipeline
├── requirements.txt          # Python dependencies
├── THIRD_PARTY.md            # Third-party attribution and license notes
```

## Installation

### Prerequisites

- Python >= 3.10
- CUDA-capable GPU (tested on NVIDIA A100, V100)
- conda or mamba for environment management

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/hybrid-weather-model.git
cd hybrid-weather-model

# Create and activate conda environment
conda create -n hybrid-weather python=3.10
conda activate hybrid-weather

# Install dependencies
pip install -r requirements.txt

# Install Dinosaur and NeuralGCM from upstream
pip install "git+https://github.com/google-research/dinosaur.git"
pip install "git+https://github.com/google-research/neuralgcm.git"
```

If you keep local mirrors (for development) under `google_train/`, that is fine; this repository excludes that folder from the published snapshot.

### Environment Variables

Configure these environment variables for your setup (all have sensible defaults if unset):

```bash
export WANDB_API_KEY="your-wandb-api-key"    # Required for W&B logging
export CACHE_DIR="./cache"                    # ERA5 preprocessed data cache
export RESULTS_DIR="./results"                # Model evaluation output
export SAVE_DIR="./saved_models"              # Model save directory
export CHKPT_DIR="./checkpoints"              # Orbax checkpoint directory
```

## Usage

### Data Preparation

The model uses ERA5 reanalysis data from the [WeatherBench2](https://weatherbench2.readthedocs.io/) public bucket. Data is automatically downloaded and cached on first use:

```bash
python data_cacher.py \
    --resolution TL47 \
    --start_time 2005-01-01 \
    --end_time 2018-12-31 \
    --cache_dir ./cache
```

### Training

Train the hybrid model with curriculum learning (progressively increasing forecast horizon):

```bash
# Single prediction range (6h forecast)
python experiment.py \
    --resolution TL47 \
    --start_time 2005-01-01 \
    --end_time 2018-12-31 \
    --prediction_range 1 \
    --train_length 4000 \
    --learning_rate 0.00035 \
    --correction GNNCorrection \
    --gnn_hidden_dims 176 176 \
    --num_levels 13 \
    --trajectory \
    --batch_size 1
```

### Evaluation

Run inference and save predictions as Zarr:

```bash
python experiment.py \
    --test \
    --xarray \
    --resolution TL47 \
    --prediction_range 8 \
    --test_year 2020 \
    --load_checkpoint True \
    --checkpoint_name LUKE47 \
    --load_checkpoint_name pr6
```

Evaluate against ERA5 using WeatherbenchX:

```bash
python weatherbench.py \
    --prediction_path ./results/test_2020_8_*.zarr \
    --target_path gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr \
    --time_start 2020-01-01 \
    --time_stop 2020-12-31
```

## Correction Modes

| Mode | Description |
|------|-------------|
| `GNNCorrection` | GNN learns residual corrections to the dynamical core tendencies |
| `NullCorrection` | GNN is present but with near-zero scaling (ablation baseline) |
| `FullTendency` | GNN receives full state (not just tendencies) as input |
| `TGNNCorrection` | 3D GNN with cross-level message passing |

## Supported Resolutions

| Resolution | Spectral Truncation | Grid Points | Approximate Grid Spacing |
|------------|---------------------|-------------|--------------------------|
| TL31 | 31 | 48 x 96 | ~5.6° |
| TL47 | 47 | 72 x 144 | ~3.75° |
| TL63 | 63 | 96 x 192 | ~2.8° |
| TL95 | 95 | 144 x 288 | ~1.875° |
| TL127 | 127 | 192 x 384 | ~1.4° |

## Acknowledgments

This project builds upon:

- **[Dinosaur](https://github.com/google-research/dinosaur)** — Differentiable spectral dynamical core by Google Research
- **[NeuralGCM](https://github.com/google-research/neuralgcm)** — Neural General Circulation Models by Google Research
- **[WeatherBench2](https://github.com/google-research/weatherbench2)** — Evaluation framework for weather forecasting

## License

This project is released under the [MIT License](LICENSE) for original code in this repository.

Third-party dependencies (including Dinosaur and NeuralGCM) remain under their own licenses (Apache-2.0). See `THIRD_PARTY.md` for attribution details.

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{szarvas2025hybrid,
  title={Revealing Cross-Scale Interactions in Hybrid Weather Models},
  author={Szarvas, Balint},
  school={ETH Zurich},
  year={2025}
}
```
