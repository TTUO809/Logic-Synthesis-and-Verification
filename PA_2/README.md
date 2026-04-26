# PA 2 — Supervised Learning for EDA Logic Gate Representation Learning through GNN

Using [DeepGate2](https://github.com/zshi0616/DeepGate2) for graph neural network-based circuit analysis.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [DeepGate2 Modifications](#deepgate2-modifications)
  - [Assignment 2 — Ablation hooks](#assignment-2--ablation-hooks)
  - [Assignment 3 — Switching-probability prediction](#assignment-3--switching-probability-prediction)
  - [Performance / Housekeeping](#performance--housekeeping)
  - [Summary table (all changes, with provenance)](#summary-table-all-changes-with-provenance)
- [Assignment 1 — Hidden Dimension Sweep](#assignment-1--hidden-dimension-sweep)
- [Assignment 2 — Ablation Study](#assignment-2--ablation-study)

---

## Project Structure

```
PA_2/
├── run_assignment1.sh        # Assignment 1 training script (EPOCHS/LOG_BASE env-overridable)
├── run_assignment2.sh        # Assignment 2 training script (EPOCHS/LOG_BASE env-overridable)
├── run_assignment3.sh        # Assignment 3 training script (EPOCHS/LOG_BASE env-overridable)
├── run_all_10epochs.sh       # Wrapper: run all three sequentially at 10 epochs
├── deepgate2.patch           # Full diff of all DeepGate2 modifications vs upstream main
├── report.md                 # Assignment report (with 10-epoch results)
├── README.md                 # This file
├── SLforEDA_v0329.pdf        # Course slides (2024-03-29 version)
├── test/                     
│   └── env_final_check.py     # Verify environment setup (PyTorch, PyG, CUDA) before running experiments
└── log/
    ├── assignment_all_master.log          # Combined master log (all three assignments)
    ├── 10_epochs/                         # Definitive 10-epoch results
    │   ├── run_all.log
    │   ├── assignment1_master.log
    │   ├── assignment2_master.log
    │   ├── assignment3_master.log
    │   └── results/
    │       ├── assignment1/               # dim{32,64,128}_train.log , dim{32,64,128}_summary.txt , comparison.txt
    │       ├── assignment2/               # {baseline,no_tt_loss,homo_pi_init}_train.log / summary.txt , comparison.txt
    │       └── assignment3/               # {prob_only,with_trans}_train.log / summary.txt / eval.log , comparison.txt
    └── 5_epochs_assignment3/              # 5-epoch reference logs for Assignment 3 comparisons
        ├── assignment3_master.log
        └── results/
            └── assignment3/               # {prob_only,with_trans}_train.log / summary.txt / eval.log , prepare_dataset_markov.log , comparison.txt
```

---

## Environment Setup

> `git clone https://github.com/TTUO809/Logic-Synthesis-and-Verification.git`
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

All modifications to DeepGate2 are tracked in the [`pa2-modifications`](https://github.com/TTUO809/DeepGate2/tree/pa2-modifications) branch of the fork. Full diff against the original:

**[TTUO809/DeepGate2: main → pa2-modifications](https://github.com/TTUO809/DeepGate2/compare/main...pa2-modifications)**

Changes are grouped below by what motivated them: Assignment 2 ablations, Assignment 3 switching-probability extension, and performance / housekeeping. Each row lists the file, the actual change, and the experiment item it serves.

### Assignment 2 — Ablation hooks

Adds the two CLI ablation switches used by [`run_assignment2.sh`](run_assignment2.sh) (`--no_func`, `--homo_pi_init`).

| File | Change | For which item |
|---|---|---|
| `src/config.py` | Add `--no_func` flag (disable pairwise TT loss) | `no_tt_loss` experiment |
| `src/config.py` | Add `--homo_pi_init` flag (homogeneous PI init) | `homo_pi_init` experiment |
| `src/utils/utils.py` | `generate_hs_init(..., homo_pi_init=False)` — when true, all PI nodes share $v = \frac{1}{\sqrt{d}} \mathbb{1} = \left[ \frac{1}{\sqrt{d}}, \frac{1}{\sqrt{d}}, \dots, \frac{1}{\sqrt{d}} \right]^T$; otherwise keep orthogonal init | `homo_pi_init` experiment |
| `src/models/mlpgate.py` | Forward `args.homo_pi_init` into `generate_hs_init` | `homo_pi_init` experiment |
| `src/trains/mlpgnn_trainer.py` | When `--no_func`: zero out `LFunc` and drop `Func_weight` from the normalizer (`max(total_w, 1.0)`) so the loss scale stays comparable across ablations | `no_tt_loss` experiment |

### Assignment 3 — Switching-probability prediction

Adds a new per-node transition-probability task (Markov-stimulus labels, new readout head, new loss term, new evaluation script). Driven by [`run_assignment3.sh`](run_assignment3.sh).

| File | Change | For which item |
|---|---|---|
| `src/config.py` | Add `--Trans_weight` (transition-loss weight) | `prob_only` vs `with_trans` weighting |
| `src/config.py` | Add `--no_trans` flag (disable transition loss) | Optional ablation symmetric to `--no_func` |
| `src/config.py` | Add `--label_file` CLI arg; remove the hard-coded `args.label_file = "labels.npz"` | Allow `labels_markov.npz` alongside legacy `labels.npz` |
| `src/prepare_dataset.py` | Add `--markov`, `--flip_prob`, `--label_out`; compute `trans_prob` label as fraction of consecutive-pattern flips; write to `args.label_out` | Phase 1 of assignment 3 (label generation) |
| `src/utils/circuit_utils.py` | New `simulator_truth_table_markov(...)` — Markov PI stream with per-cycle flip probability, ordered patterns so transitions are well-defined | Phase 1 of assignment 3 |
| `src/datasets/load_data.py` | `parse_pyg_mlpgate(..., trans_y=None)` — attach `graph.trans_prob`; fall back to analytic `2p(1-p)` when label is absent so legacy `labels.npz` still loads | Backward-compat for assignment 1/2 datasets |
| `src/datasets/mlpgate_dataset.py` | Read optional `trans_prob` from labels; include label-file stem in cache dir name so Markov labels don't collide with i.i.d. cache | Phase 2 of assignment 3 (training) |
| `src/models/mlpgate.py` | Add `readout_trans` MLP (sigmoid head); model now returns 5-tuple `(hs, hf, prob, trans, is_rc)` | Phase 2 of assignment 3 |
| `src/trains/mlpgnn_trainer.py` | Unpack 5-tuple; add `LTrans = L1(trans, batch.trans_prob)`; weight by `Trans_weight` (zeroed when `--no_trans`); add `LTrans` to `loss_states` so it appears in logs | Phase 2 of assignment 3 |
| `src/test_trans.py` *(new)* | Standalone eval: per-node L1 of model trans vs target, L1 of analytic `2p(1-p)` baseline, per-gate-type (PI/AND/NOT) breakdown | Phase 3 of assignment 3 (evaluation) |
| `src/get_emb_aig.py`, `src/get_emb_bench.py`, `src/test_acc_bin.py` | Accept either 4-tuple or 5-tuple from `model.run(...)` | Keep legacy embedding/eval scripts working after the new readout head |

### Performance / Housekeeping

Not required by the assignment spec — purely throughput or repo hygiene.

| File | Change | Reason |
|---|---|---|
| `src/main.py` | `DataLoader(..., pin_memory=True, persistent_workers=(num_workers>0))` for both train and val loaders | **Performance**: avoid re-spawning worker processes each epoch and speed up host→GPU copies |
| `src/main.py` | `CUDA_LAUNCH_BLOCKING=0` (was `'1'`) | **Performance**: re-enable async CUDA dispatch (the `'1'` was a debug leftover that serialized every kernel launch) |
| `requirements.txt` | Comment out PyG packages (`torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv`) | These need a CUDA-matched wheel index and must be installed before `requirements.txt` (see [Environment Setup](#environment-setup)) |
| `.gitignore` | Ignore `dataset/rawaig/` and `*.npz` | Avoid committing the multi-GB raw AIG dataset and generated label/graph npz files |

### Summary table (all changes, with provenance)

| File | Change | Category | Motivated by |
|---|---|---|---|
| `src/config.py` | `--no_func` flag | Assignment 2 | `no_tt_loss` ablation |
| `src/config.py` | `--homo_pi_init` flag | Assignment 2 | `homo_pi_init` ablation |
| `src/config.py` | `--Trans_weight` | Assignment 3 | Weight the new transition loss |
| `src/config.py` | `--no_trans` flag | Assignment 3 | Optional disable of transition loss |
| `src/config.py` | `--label_file` CLI arg (replaces hard-coded) | Assignment 3 | Switch between `labels.npz` and `labels_markov.npz` |
| `src/utils/utils.py` | `generate_hs_init` homogeneous mode | Assignment 2 | `homo_pi_init` ablation |
| `src/utils/circuit_utils.py` | `simulator_truth_table_markov` | Assignment 3 | Markov stimulus for transition labels |
| `src/prepare_dataset.py` | `--markov` / `--flip_prob` / `--label_out`, emit `trans_prob` | Assignment 3 | Phase 1 label generation |
| `src/datasets/load_data.py` | `trans_y` param; analytic `2p(1-p)` fallback | Assignment 3 | Attach `graph.trans_prob`, keep legacy datasets working |
| `src/datasets/mlpgate_dataset.py` | Read `trans_prob`; per-label-file cache dir | Assignment 3 | Avoid Markov ↔ i.i.d. cache collisions |
| `src/models/mlpgate.py` | Forward `homo_pi_init` to init | Assignment 2 | `homo_pi_init` ablation |
| `src/models/mlpgate.py` | `readout_trans` head; return 5-tuple | Assignment 3 | New transition-prob output |
| `src/trains/mlpgnn_trainer.py` | Skip `LFunc` + renormalize when `--no_func` | Assignment 2 | `no_tt_loss` ablation |
| `src/trains/mlpgnn_trainer.py` | `LTrans` loss + `Trans_weight` integration; expose in `loss_states` | Assignment 3 | Train and log transition objective |
| `src/test_trans.py` *(new)* | Per-node L1 + analytic-baseline gap + per-gate breakdown | Assignment 3 | Phase 3 evaluation |
| `src/get_emb_aig.py`, `src/get_emb_bench.py`, `src/test_acc_bin.py` | Accept 4- or 5-tuple model output | Assignment 3 | Compatibility shim after readout-head change |
| `src/main.py` | `pin_memory=True`, `persistent_workers` | **Performance** | Faster H2D copies, avoid worker respawn each epoch |
| `src/main.py` | `CUDA_LAUNCH_BLOCKING=0` | **Performance** | Re-enable async CUDA dispatch (debug leftover) |
| `requirements.txt` | Comment out PyG packages | Housekeeping | Installed separately with matching CUDA wheel index |
| `.gitignore` | Ignore `dataset/rawaig/`, `*.npz` | Housekeeping | Keep large data files out of git |

---

## Assignment 1 — Hidden Dimension Sweep

Train DeepGate2 with `dim_hidden` ∈ {32, 64, 128} and compare signal-probability loss and TT-distance accuracy.

```bash
# single run (default 5 epochs):
bash PA_2/run_assignment1.sh

# 10-epoch run via wrapper:
bash PA_2/run_all_10epochs.sh
```

Results are saved to `PA_2/log/10_epochs/results/assignment1/`.

### Results (10 epochs)

| dim\_hidden | Val LProb (↓) | Val ACC final (↑) | Val ACC best (↑) |
|:-----------:|:-------------:|:-----------------:|:----------------:|
| 32  | 0.035522 | 0.867097 | 0.867097 |
| 64  | 0.024337 | 0.910968 | 0.910968 |
| **128** | **0.023696** | **0.911935** | **0.911935** |

Larger `dim_hidden` yields strictly lower LProb and strictly higher ACC.
The gain from 32→64 (+4.4 pp ACC) is larger than 64→128 (+0.1 pp), indicating capacity saturation for the TT-ranking task above dim=64.
LProb continues to improve modestly (64: 0.0243 → 128: 0.0237), confirming dim=128 is still the best overall configuration.

---

## Assignment 2 — Ablation Study

Ablate two design choices of DeepGate2 to verify their contribution.

```bash
bash PA_2/run_assignment2.sh
```

Results are saved to `PA_2/log/10_epochs/results/assignment2/`.

### Experiments

| Experiment | Flag | What is ablated |
|---|---|---|
| `baseline` | — | Full model (orthogonal PI init + pairwise TT loss) |
| `no_tt_loss` | `--no_func` | Remove pairwise TT difference loss |
| `homo_pi_init` | `--homo_pi_init` | Replace orthogonal PI init with homogeneous (all-ones) init |

### Results (10 epochs)

| Experiment | Val LProb (↓) | Val LFunc (↓) | Val ACC (↑) | Δ ACC vs baseline |
|---|:---:|:---:|:---:|:---:|
| **baseline** | 0.024151 | 0.125278 | **0.910645** | — |
| no\_tt\_loss | 0.024276 | 0.000000 | 0.753871 | **−15.68 pp** |
| homo\_pi\_init | 0.025419 | 0.129545 | 0.863226 | −4.74 pp |

- **no\_tt\_loss**: 
  - ACC gap widens with more training (−15.7 pp at 10 ep vs −8.1 pp at 5 ep). 
  - With no functional supervision, the model converges to a different optimum that cannot distinguish functionally similar circuits.
  - LProb is nearly identical to baseline (0.0243 vs 0.0242), confirming the two objectives compete for gradient budget.
- **homo\_pi\_init**:
  - ACC recovers to 0.863 at 10 ep (vs 0.817 at 5 ep), but remains 4.7 pp below baseline, confirming that orthogonal PI initialization provides a persistent advantage for structural discrimination.

> Val LRC is computed on dummy RC pairs (`--no_rc`); values are not meaningful and excluded from analysis.

---

## Assignment 3 — Switching-Probability Prediction

```bash
bash PA_2/run_assignment3.sh
```

Results are saved to `PA_2/log/10_epochs/results/assignment3/`.

### Results (10 epochs)

#### Training metrics (final epoch)

| Experiment | Val LProb (↓) | Val LFunc (↓) | Val LTrans (↓) | Val ACC (↑) |
|---|:---:|:---:|:---:|:---:|
| prob\_only (Trans\_weight=0) | 0.037693 | 0.134224 | 0.461819 | 0.845806 |
| **with\_trans** (Trans\_weight=2) | 0.046856 | 0.136056 | **0.026568** | 0.831935 |
| Δ (with\_trans − prob\_only) | +0.009 | +0.002 | **−0.435** | −0.014 |

#### Evaluation (test\_trans.py — 9 756 circuits, 3.4 M nodes)

| Experiment | L1(prob) | L1 analytic 2p(1−p) | L1(trans) model | Gap (analytic−model) |
|---|:---:|:---:|:---:|:---:|
| prob\_only | 0.036752 | 0.249507 | 0.462986 | **−0.213** (model worse) |
| **with\_trans** | 0.045719 | 0.255825 | **0.025580** | **+0.230** (model beats baseline) |

#### Per-gate-type L1(trans) — with\_trans checkpoint

| Gate type | N nodes | L1(trans) |
|:---|---:|:---:|
| PI  |   533,440 | 0.025038 |
| AND | 1,487,449 | 0.024455 |
| NOT | 1,391,852 | 0.026990 |

At 10 epochs, `with_trans` achieves LTrans=0.0257 (vs 0.0517 at 5 ep, a **50% further reduction**).
The analytic gap improves from +0.205 to **+0.230**. All three gate types converge to near-equal L1 (~0.025), indicating the model has learned a structurally uniform representation of switching probability.

---

