# Logic Synthesis and Verification — PA2

Course assignments using [DeepGate2](https://github.com/zshi0616/DeepGate2) for graph neural network-based circuit analysis.

---

## Environment Setup

```bash
cd LSV/
conda create -n deepgate2 python=3.8
conda activate deepgate2

git clone https://github.com/zshi0616/DeepGate2.git
cd DeepGate2/

# Install PyTorch with CUDA 11.1 first (before requirements.txt)
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

pip install torch-scatter==2.0.8 torch-sparse==0.6.12 \
    torch-cluster==1.5.9 torch-spline-conv==1.2.1 \
    -f https://data.pyg.org/whl/torch-1.9.0+cu111.html

# Comment out packages already installed to avoid version conflicts
sed -i 's/^torch==/#torch==/g'                   requirements.txt
sed -i 's/^torchvision==/#torchvision==/g'       requirements.txt
sed -i 's/^torch-scatter==/#torch-scatter==/g'   requirements.txt
sed -i 's/^torch-sparse==/#torch-sparse==/g'     requirements.txt
sed -i 's/^torch-cluster==/#torch-cluster==/g'   requirements.txt
sed -i 's/^torch-spline-conv==/#torch-spline-conv==/g' requirements.txt

pip install -r requirements.txt
```

**Verify the environment:**

```bash
cd ../test/
python env_final_check.py
```

**Prepare the dataset:**

```bash
cd ../DeepGate2/dataset
tar -jxvf rawaig.tar.bz2
cd ..
python ./src/prepare_dataset.py --exp_id train --aig_folder ./dataset/rawaig
```

---

## Applying Patches to DeepGate2

This repo includes `PA2/deepgate2_changes.patch` with all modifications made to DeepGate2 for the assignments.

```bash
cd DeepGate2/
git apply ../PA2/deepgate2_changes.patch
```

### Summary of changes

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
bash PA2/run_assignment1.sh
```

Results are saved to `PA2/results/assignment1/`.

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
bash PA2/run_assignment2.sh
```

Results are saved to `PA2/results/assignment2/`.

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

## Repository Structure

```
PA2/
├── run_assignment1.sh        # Assignment 1 training script
├── run_assignment2.sh        # Assignment 2 training script
├── deepgate2_changes.patch   # All modifications to DeepGate2
├── SLforEDA_v0329.pdf        # Course slides
├── assignment1_master.log    # Raw stdout from assignment 1 run
├── assignment2_master.log    # Raw stdout from assignment 2 run
└── results/
    ├── assignment1/          # Per-dim training logs and summaries
    └── assignment2/          # Per-experiment training logs and summaries
```
