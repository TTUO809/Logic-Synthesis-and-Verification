# PA 2 — Supervised Learning for EDA Logic Gate Representation Learning through GNN

Using [DeepGate2](https://github.com/zshi0616/DeepGate2) for graph neural network-based circuit analysis.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [DeepGate2 Modifications](#deepgate2-modifications)
- [Assignment 1 — Hidden Dimension Sweep](#assignment-1--hidden-dimension-sweep)
- [Assignment 2 — Ablation Study](#assignment-2--ablation-study)
- [Technical Details](TECHNICAL.md)

---

## Project Structure

```
PA_2/
├── run_assignment1.sh        # Assignment 1 training script
├── run_assignment2.sh        # Assignment 2 training script
├── SLforEDA_v0329.pdf        # Course slides
├── report.md                 # Assignment report
└── 5_epochs/                 # Results from 5-epoch runs
    ├── assignment1_master.log    # Raw stdout from assignment 1 run
    ├── assignment2_master.log    # Raw stdout from assignment 2 run
    └── results/
        ├── assignment1/          # Per-dim training logs and summaries
        └── assignment2/          # Per-experiment training logs and summaries
```

---

## Environment Setup

```bash
conda create -n deepgate2 python=3.8
conda activate deepgate2

# Clone the modified DeepGate2 (fork with PA2 changes, NOT the original)
# DeepGate2 must be placed at ~/DeepGate2 (scripts reference $HOME/DeepGate2)
git clone -b pa2-modifications https://github.com/TTUO809/DeepGate2.git ~/DeepGate2
cd ~/DeepGate2

# Install PyTorch with CUDA 11.1 first (before requirements.txt)
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

pip install torch-scatter==2.0.8 torch-sparse==0.6.12 \
    torch-cluster==1.5.9 torch-spline-conv==1.2.1 \
    -f https://data.pyg.org/whl/torch-1.9.0+cu111.html

# requirements.txt already has PyG packages commented out
pip install -r requirements.txt
```

**Verify the environment:**

```bash
python ~/LSV/PA_2/test/env_final_check.py
```

**Prepare the dataset:**

```bash
cd ~/DeepGate2/dataset
tar -jxvf rawaig.tar.bz2
cd ..
python ./src/prepare_dataset.py --exp_id train --aig_folder ./dataset/rawaig
```

---

## DeepGate2 Modifications

All modifications to DeepGate2 are tracked in the [`pa2-modifications`](https://github.com/TTUO809/DeepGate2/tree/pa2-modifications) branch of the fork. You can view the full diff against the original at:

**[TTUO809/DeepGate2: main → pa2-modifications](https://github.com/TTUO809/DeepGate2/compare/main...pa2-modifications)**

| File | Change |
|---|---|
| `src/config.py` | Added `--no_func` and `--homo_pi_init` CLI flags for PA2 ablations |
| `src/models/mlpgate.py` | Pass `homo_pi_init` flag through to `generate_hs_init` |
| `src/trains/mlpgnn_trainer.py` | Skip TT loss computation when `--no_func`; adjust weight normalization |
| `src/utils/utils.py` | `generate_hs_init` supports homogeneous PI initialization mode |
| `src/main.py` | Enable `pin_memory` and `persistent_workers` for DataLoader |
| `requirements.txt` | Comment out PyG packages installed separately via pip |
| `.gitignore` | Ignore `dataset/rawaig/` and `*.npz` |

---

## Assignment 1 — Hidden Dimension Sweep

Train DeepGate2 with `dim_hidden` ∈ {32, 64, 128} and compare signal-probability loss and TT-distance accuracy.

```bash
bash PA_2/run_assignment1.sh
```

Results are saved to `PA_2/5_epochs/results/assignment1/`.

### Results

| dim\_hidden | Val LProb (↓) | Val ACC final (↑) | Val ACC best (↑) |
|---|---|---|---|
| 32 | 0.058797 | 0.850625 | 0.850625 |
| 64 | 0.030562 | 0.851290 | 0.851290 |
| **128** | 0.032277 | **0.857742** | **0.857742** |

dim=64 achieves the lowest signal-probability loss; dim=128 achieves the highest TT-distance accuracy.

---

## Assignment 2 — Ablation Study

Ablate two design choices of DeepGate2 to verify their contribution.

```bash
bash PA_2/run_assignment2.sh
```

Results are saved to `PA_2/5_epochs/results/assignment2/`.

### Experiments

| Experiment | Flag | What is ablated |
|---|---|---|
| `baseline` | — | Full model (orthogonal PI init + pairwise TT loss) |
| `no_tt_loss` | `--no_func` | Remove pairwise TT difference loss |
| `homo_pi_init` | `--homo_pi_init` | Replace orthogonal PI init with homogeneous (all-ones) init |

### Results

| Experiment | Val LProb (↓) | Val LFunc (↓) | Val ACC (↑) |
|---|---|---|---|
| baseline | 0.031710 | 0.139047 | 0.848710 |
| no\_tt\_loss | 0.029340 | 0.000000 | **0.722903** ↓ |
| homo\_pi\_init | 0.031384 | 0.136813 | 0.832258 ↓ |

- **no\_tt\_loss**: ACC drops from 0.849 → 0.723 (−14.8 pp), confirming the pairwise TT loss is critical for functional similarity learning.
- **homo\_pi\_init**: ACC drops from 0.849 → 0.832 (−1.6 pp), showing orthogonal PI initialization provides a meaningful but smaller improvement.

---


