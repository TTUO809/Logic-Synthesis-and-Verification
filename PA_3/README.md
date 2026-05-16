# PA 3 — RL for EDA: Library Tuning for Technology Mapping

> **Course:** Logic Synthesis and Verification (LSV)  
> **Due:** 2026/6/1  
> **Paper:** [MapTune (ICCAD 2024)](https://dl.acm.org/doi/10.1145/3676536.3676762)  
> **MapTune Repo:** https://github.com/Yu-Maryland/MapTune

---

## Environment

| Component | Version / Location |
|---|---|
| OS | Ubuntu 20.04, Linux 5.4.0 |
| Python | 3.10.20 (via conda `LSV_PA3`) |
| PyTorch | 2.12.0+cpu |
| gymnasium | 1.3.0 |
| ABC | UC Berkeley ABC 1.01, built from source at `~/abc/` |
| MapTune | `~/MapTune/` (cloned from Yu-Maryland/MapTune, commit `a2df87a`) |

---

## Setup (Reproduce from scratch)

### 1. Clone MapTune

```bash
git clone https://github.com/Yu-Maryland/MapTune.git ~/MapTune
```

### 2. Create conda environment

```bash
conda create -n LSV_PA3 python=3.10 -y
conda activate LSV_PA3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install gymnasium
```

### 3. Build ABC from source

```bash
bash LSV/PA_3/setup_abc.sh
source ~/.bashrc
```

`setup_abc.sh` clones [berkeley-abc/abc](https://github.com/berkeley-abc/abc) to `~/abc/`, compiles it, and appends `~/abc` to `PATH` in `~/.bashrc`.

Verify:

```bash
abc -q "quit"   # should exit silently with no error
```

### 4. Download missing benchmarks

The PA3 required designs `s13207` and `c2670` are not included in MapTune's repo. Download them manually:

```bash
# c2670 — ISCAS 85
curl -o ~/MapTune/benchmarks/c2670.bench \
     https://www.pld.ttu.ee/~maksim/benchmarks/iscas85/bench/c2670.bench

# s13207 — ISCAS 89
curl -o ~/MapTune/benchmarks/s13207.bench \
     https://www.pld.ttu.ee/~maksim/benchmarks/iscas89/bench/s13207.bench
```

> **Note:** `b20.bench` is not available. Use `b20_1.bench` (already in `MapTune/benchmarks/`) as a substitute throughout all assignments.

---

## Directory Structure

```
~/
├── MapTune/                    ← MapTune (unmodified upstream)
│   ├── benchmarks/
│   │   ├── s13207.bench        ← downloaded from ISCAS 89
│   │   ├── c2670.bench         ← downloaded from ISCAS 85
│   │   ├── b20_1.bench         ← substitute for b20
│   │   └── ...
│   ├── nan45.genlib
│   ├── sky130.genlib
│   ├── 7nm.genlib
│   ├── batched_MAB_UCB.py
│   ├── batched_MAB_EP.py
│   ├── batched_DQN.py
│   ├── batched_DDQN.py
│   └── abc.rc
├── abc/                        ← ABC built from source
│   └── abc
└── LSV/PA_3/
    ├── README.md                ← this file
    ├── setup_abc.sh             ← ABC build & PATH setup script
    ├── run_a1_baseline.py       ← A1: full-library baseline (all 5 libs × 3 designs)
    ├── run_maptune_baseline.sh  ← A3: run MapTune on all lib×design×budget combos
    ├── src/
    │   ├── code.py              ← A3: custom optimization method
    │   └── readme.txt
    └── Materials/
        ├── LSV_PA3_Spec.md
        └── Paper/
            └── MapTune_Paper_Note.md
```

---

## Assignments

### Assignment 1 — Baseline (30%)

Run full-library technology mapping with ABC for all 5 libraries × 3 designs automatically:

```bash
conda activate LSV_PA3
python ~/LSV/PA_3/run_a1_baseline.py
```

The script uses the same ABC pipeline as MapTune's internal baseline:
```
read <genlib>; read <design>; map -a; write <tmp.blif>;
read <lib>; read -m <tmp.blif>; ps; topo; upsize; dnsize; stime;
```
ABC path is resolved automatically (`~/abc/abc` fallback if not in PATH).

**Deliverable:** `report.pdf` with terminal screenshot + the table below. No plots needed.

#### Baseline Results (obtained 2026-05-16, mapper: `map -a`)

| Library | Design | Delay (ps) | Area |
|---|---|---:|---:|
| ASAP7 7nm (161 cells) | s13207 | 241.14 | 1796.02 |
| ASAP7 7nm (161 cells) | c2670 | 130.92 | 399.61 |
| ASAP7 7nm (161 cells) | b20_1 | 821.18 | 4720.89 |
| GF180 ff125C (151 cells) | s13207 | 15253.21 | 30199.37 |
| GF180 ff125C (151 cells) | c2670 | 8609.96 | 6901.71 |
| GF180 ff125C (151 cells) | b20_1 | 41607.07 | 98757.66 |
| GF180 tt025C (151 cells) | s13207 | 6396.32 | 30359.62 |
| GF180 tt025C (151 cells) | c2670 | 3563.00 | 6932.44 |
| GF180 tt025C (151 cells) | b20_1 | 17324.59 | 99328.41 |
| NAN45 (94 cells) | s13207 | 731.66 | 1909.35 |
| NAN45 (94 cells) | c2670 | 392.54 | 478.00 |
| NAN45 (94 cells) | b20_1 | 2129.22 | 6335.06 |
| SKY130 (343 cells) | s13207 | 2957.89 | 10130.97 |
| SKY130 (343 cells) | c2670 | 1608.88 | 2253.41 |
| SKY130 (343 cells) | b20_1 | 8077.29 | 32595.01 |

> These D\_Base / A\_Base values are reused in A3 for normalized ADP = (D/D\_Base)×(A/A\_Base).

---

### Assignment 2 — Random Sampling (30%)

Validate that partial libraries can outperform the full library using ASAP7 (`7nm.genlib`, 161 cells):

```bash
conda activate LSV_PA3
cd ~/MapTune

# Sampling 1: ~75-100 cells → use num_sampled_gate = 88 (midpoint, excl. pre-selected INV/BUF)
python batched_MAB_EP.py 88 benchmarks/s13207.bench 7nm.genlib

# Sampling 2: ~100-125 cells → num_sampled_gate = 112
python batched_MAB_EP.py 112 benchmarks/s13207.bench 7nm.genlib

# Sampling 3: ~125-150 cells → num_sampled_gate = 137
python batched_MAB_EP.py 137 benchmarks/s13207.bench 7nm.genlib
```

Repeat for all 3 designs. Plot Area-Delay scatter plots (one per design).

Deliverable: `report.pdf` with scatter plots.

---

### Assignment 3 — Beat MapTune (40%)

**Goal:** Find a cell subset with strictly lower ADP than MapTune's best result.

#### Step 1: Get MapTune baseline

```bash
conda activate LSV_PA3
bash ~/LSV/PA_3/run_maptune_baseline.sh
```

Budget levels per library:

| Library | Total Cells | 30% | 50% | 70% |
|---|---|---|---|---|
| `nan45.genlib` | 94 | 28 | 47 | 66 |
| `sky130.genlib` | 343 | 103 | 171 | 240 |

#### Step 2: Run custom method

```bash
conda activate LSV_PA3
python ~/LSV/PA_3/src/code.py \
    --design ~/MapTune/benchmarks/s13207.bench \
    --lib ~/MapTune/nan45.genlib \
    --budget 47 \
    --adp_baseline <ADP_MapTune>
```

Your method must satisfy: `|S_new| ≤ |S_MapTune|` and `ADP_new < ADP_MapTune`.

Deliverable: source in `src/`, `report.pdf` with comparison table.

---

## Notes

- Always `cd ~/MapTune` before running MapTune scripts — `abc.rc` and `gen_newlibs/`, `temp_blifs/` paths are relative
- `abc.history` is ABC's command history file; it appears in whichever directory ABC is invoked from — safe to ignore or `.gitignore`
- ABC's `num_sampled_gate` parameter excludes pre-selected INV/BUF gates (~10–20 gates per library)
