# LSV PA3 — RL for EDA: Library Tuning for Technology Mapping

> **Due:** 2026/6/1 before class starts  
> **Repo:** https://github.com/Yu-Maryland/MapTune  
> **Paper:** MapTune (ICCAD 2024) — M. Liu et al.

---

## Background

**Technology mapping** converts RTL-level circuit descriptions into technology-specific gate-level netlists. The choice of technology library (which gates are available) critically affects **Power, Performance, and Area (PPA)**.

**Key counter-intuitive finding (MapTune):** Using a *partial subset* of cells from the full library can **significantly improve** ADP compared to using all cells.

---

## MapTune — RL Formulation

### Action Space
- Each action $a^i$ = selecting cell $i$ from the library
- Cardinality = $N$ (total number of cells in library)

### State
$$S = [S_0, S_1, \ldots, S_{N-1}], \quad S_i \in \{0, 1\}$$
- $S_i = 1$ means cell $i$ is selected
- **Complete state:** $\sum_{i=0}^{N-1} S_i = n$ (exactly $n$ cells selected)

### Reward
$$\mathcal{R}_S = -ADP_S = -\left(\frac{D_S}{D_{Base}} \cdot \frac{A_S}{A_{Base}}\right)$$

- $D_S$, $A_S$: Delay and Area from mapping with selected subset at state $S$
- $D_{Base}$, $A_{Base}$: Baseline metrics using the **full** library
- Goal: **minimize ADP** → maximize reward

### Probability Vector
$$P = [P_{a^0}, P_{a^1}, \ldots, P_{a^{N-1}}]$$
- $P_{a^i}$ = likelihood of selecting cell $i$ in the sampled library
- Updated by RL to maximize expected reward

### MapTune Variants
| Variant | Description |
|---|---|
| MapTune-DQN | Uses DQN to update probability |
| MapTune-DDQN | Uses Double DQN |
| MapTune-ε | ε-greedy exploration |
| MapTune-UCB | UCB-based exploration |

---

## Libraries

| `.genlib` | # of Gates |
|---|---|
| `7nm.genlib` (ASAP7) | 161 |
| `gf180mcu_ff_125C.genlib` | 151 |
| `gf180mcu_tt_025C.genlib` | 151 |
| `nan45.genlib` | 94 |
| `sky130.genlib` | 343 |

---

## Benchmarks

Used across assignments: **s13207**, **c2670**, **b20**

> **Availability note (2026-05-16):**
> - `c2670.bench` — **not in MapTune repo**, download from [ISCAS 85](https://www.pld.ttu.ee/~maksim/benchmarks/iscas85/bench/c2670.bench) and place in `MapTune/benchmarks/`
> - `s13207.bench` — **not in MapTune repo**, download from [ISCAS 89](https://www.pld.ttu.ee/~maksim/benchmarks/iscas89/bench/s13207.bench) and place in `MapTune/benchmarks/`
> - `b20.bench` — **not available**; use `b20_1.bench` (already in `MapTune/benchmarks/`) as the substitute

---

## Assignments

### Assignment 1 — Environment Setup & Baseline (30%)

**Goal:** Install MapTune, run full-library baseline mapping for every library × every design.

**Steps:**
1. Clone and install MapTune:
   ```bash
   git clone https://github.com/Yu-Maryland/MapTune.git
   cd MapTune
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install gymnasium
   export PATH=/your/path/to/ABC:${PATH}
   ```
2. For **every library** × **every design**, run full-cell-set technology mapping
3. Record baseline **Area** and **Delay**

**ABC Command Tip:**
```
read <library>.lib; map; topo; upsize; dnsize; stime
```

**Deliverable:** `report.pdf` with screenshots of execution results + Area-Delay table (no source code needed)

---

### Assignment 2 — Random Sampling Replication (30%)

**Goal:** Validate that a partial library can outperform the full library.

**Steps:**
1. Use `ASAP7.genlib` (161 cells)
2. Randomly sample at **three size levels**:
   - Sampling 1: 75–100 cells
   - Sampling 2: 100–125 cells
   - Sampling 3: 125–150 cells
3. Run technology mapping for each sample on benchmarks: **s13207**, **c2670**, **b20**
4. Plot **Area-Delay scatter plot** (similar to MapTune Figure 1), one plot per benchmark, showing all 3 sampling levels + baseline

**Deliverable:** `report.pdf` with scatter plots + brief analysis (no source code needed)

---

### Assignment 3 — Outperform MapTune (40%) ← Main Coding Task

**Goal:** Develop an **automated optimization method** to find a cell subset with better ADP than MapTune's baseline, under fixed budget constraints.

#### MapTune Baseline Setup
- **Benchmarks:** s13207, c2670, b20
- **Libraries:** `nan45.genlib`, `sky130.genlib`
- **Fixed budgets:** 30%, 50%, 70% of total library cardinality
- **Baseline:** Run MapTune at each budget → find subset with minimum ADP → this is $S_{MapTune}$, $ADP_{MapTune}$

#### Your Method Must Satisfy
$$|S_{new}| \leq |S_{MapTune}| \quad \text{and} \quad ADP_{new} < ADP_{MapTune}$$

- $|S_{MapTune}|$ and $ADP_{MapTune}$ **can be given as inputs** to your method

#### Deliverable
```
M11407407_pa3/
├── report.pdf
└── src/
    ├── code.py   # (.cpp or .c also OK)
    └── readme.txt  # how to run
```

`report.pdf` must include:
1. Screenshots of program execution results
2. Comparison table: your method vs. MapTune baseline (Area, Delay, ADP)
3. PA3 conclusion
4. Observations and findings

---

## Grading Summary

| Assignment | Weight | Code Required |
|---|---|---|
| A1: Baseline replication | 30% | No |
| A2: Random sampling + scatter plot | 30% | No |
| A3: Automated method to beat MapTune | 40% | **Yes** |

---

## Key Concepts Glossary

| Term | Definition |
|---|---|
| ADP | Area-Delay Product = $A \times D$ (lower is better) |
| Technology Mapping | Converting logic network to gate-level netlist using a cell library |
| `.genlib` | Standard cell library format used by ABC |
| ABC | Academic logic synthesis tool (used for mapping) |
| MapTune | RL-based framework for cell subset selection to improve ADP |
| Partial Library | A subset of cells from the full library used for mapping |
| Complete State | A state where exactly $n$ cells have been selected |
