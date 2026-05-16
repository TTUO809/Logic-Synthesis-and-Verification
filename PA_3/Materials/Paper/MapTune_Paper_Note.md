# MapTune — Full Paper Note

> **Title:** MapTune: Advancing ASIC Technology Mapping via Reinforcement Learning Guided Library Tuning  
> **Venue:** ICCAD 2024  
> **Authors:** Mingju Liu, Daniel Robinson, Yingjie Li, Cunxi Yu (UMD + MIT)  
> **DOI:** https://dl.acm.org/doi/10.1145/3676536.3676762  
> **Repo:** https://github.com/Yu-Maryland/MapTune

---

## 1. Problem Statement

Technology mapping converts RTL-level circuit descriptions into gate-level netlists using a **technology library** (a set of standard cells). Traditional practice uses the **full library**, which causes two problems:

1. Large search space → heuristic mappers make suboptimal choices
2. Cell diversity introduces conflicting trade-offs that the mapper cannot resolve well

**Key counter-intuitive finding:** A **partially selected subset** of cells from the full library can significantly *improve* ADP (Area-Delay Product) compared to using all cells.

The intuition: academic and industrial mappers are heavily heuristic-based. As library size grows, algorithm complexity explodes and heuristics degrade. Removing "distracting" cells helps the mapper focus.

---

## 2. Motivation: Case Study with ASAP7

Random sampling from ASAP7 (161 cells) at three size ranges, then running ABC mapping (`map; topo; upsize; dnsize; stime`) on three benchmarks:

| Sampling Range | Label |
|---|---|
| Full 161 cells | Baseline (red star) |
| 75–100 cells | Sampling 1 |
| 100–125 cells | Sampling 2 |
| 125–150 cells | Sampling 3 |

**Three observations from the case study:**

**Obs 1 — Partial libraries have significant PPA impact.**
For `s13207`: ~30% of sampled results outperform baseline in area, ~60% outperform in delay. The baseline is *not* a lower bound.

**Obs 2 — Smaller sampling = wider QoR distribution.**
For `c2670`: smaller sampling range (75–100) shows a wider spread of results — more potential for improvement, but also more variance. Larger sampling sizes cluster closer to baseline.

**Obs 3 — Partial library doesn't always win.**
For `b20`: the baseline falls near the lower-left Pareto frontier of sampled results — partial library doesn't consistently help *every* design. Also, very small subsets risk **mapping failure** (not enough cells to implement the design).

---

## 3. Notation Table

| Symbol | Meaning |
|---|---|
| $\mathcal{L}$ | Set of all cell variants in the full library |
| $\mathcal{N}$ | Total number of cells ($|\mathcal{L}|$) |
| $\mathcal{A}$ | Action space |
| $a^i$ | Action: select cell $i$ |
| $\mathcal{S}$ | State: binary vector of selected cells |
| $ADP_\mathcal{S}$ | Normalized Area-Delay Product at state $\mathcal{S}$ |
| $p_{a^i}$ | Probability that selecting cell $i$ maximizes reward |

---

## 4. RL Formulation

### 4.1 Action Space
$$\mathcal{A} = \{a^0, a^1, \ldots, a^{\mathcal{N}-1}\}, \quad |\mathcal{A}| = \mathcal{N}$$
Taking action $a^i$ means setting $S_i = 1$ (selecting cell $i$).

### 4.2 State
$$\mathcal{S} = [S_0, S_1, \ldots, S_{\mathcal{N}-1}], \quad S_i \in \{0, 1\}$$
**Complete state:** $\sum_{i=0}^{\mathcal{N}-1} S_i = n$ (exactly $n$ cells selected → ready to map).

### 4.3 Reward
$$\mathcal{R}_\mathcal{S} = -ADP_\mathcal{S} = -\left(\frac{D_\mathcal{S}}{D_\text{Base}} \cdot \frac{A_\mathcal{S}}{A_\text{Base}}\right)$$

- $D_\mathcal{S}$, $A_\mathcal{S}$: Delay (ps) and Area (μm²) from ABC mapping with partial library at state $\mathcal{S}$
- $D_\text{Base}$, $A_\text{Base}$: Baseline from mapping with full library
- Reward only assigned at **complete state** (after all $n$ cells selected)
- Maximizing reward = minimizing normalized ADP

### 4.4 Probability Vector
$$\boldsymbol{p} = [p_{a^0}, p_{a^1}, \ldots, p_{a^{\mathcal{N}-1}}]$$
Updated each iteration. $p_{a^i}$ = estimated likelihood that including cell $i$ leads to lower ADP.

---

## 5. Algorithms

### 5.1 MapTune-MAB (Multi-Armed Bandit)

Each cell = one arm. Each iteration: select $n$ arms → form partial library → run ABC → get reward → update $p$.

**MapTune-ε (ε-greedy):**
$$a^i = \begin{cases} \arg\max_{a^i \in \mathcal{A}} p_{a^i} & \text{with prob. } 1-\varepsilon \\ \text{random} & \text{with prob. } \varepsilon \end{cases}$$

**MapTune-UCB (Upper Confidence Bound):**
$$a^i = \arg\max_{a^i \in \mathcal{A}} \left( p_{a^i} + c\sqrt{\frac{\log(t)}{n_{a^i}}} \right)$$
where $t$ = current iteration, $n_{a^i}$ = times action $a^i$ taken, $c$ = exploration coefficient.

**Probability update (both MAB methods):**
$$p_{a^i}(t+1) = \frac{p_{a^i}(t) \cdot n_{a^i}(t) + \mathcal{R}_\mathcal{S}(t)}{n_{a^i}(t)}$$
Running average of reward for each cell — updated only when cell $i$ was part of the selected subset at iteration $t$.

### 5.2 MapTune-Q (Q-Learning)

Same MDP environment, but uses neural networks to approximate Q-values instead of direct probability tracking.

**MapTune-DQN:**
- Input: state $\mathcal{S}$ and action $a^i$ → output: Q-value $Q(\mathcal{S}, a^i)$
- Use $p_{a^i} = Q(\mathcal{S}, a^i)$ (Q-value as probability proxy)
- Bellman target:
$$p_{a^i}^\text{tar} = \mathcal{R} + \gamma \max_{a^i} p_{a^i}$$
- Loss: $L_\theta = \text{MSE}(p_{a^i},\ p_{a^i}^\text{tar})$

**MapTune-DDQN:**
- Adds a **target network** $Q_\text{target}$ (periodically synced, not online-trained) to reduce overestimation bias
- Target Q-value:
$$p_{a^i}^\text{tar} = \mathcal{R} + \gamma \cdot Q_\text{target}\!\left(\mathcal{S},\ \arg\max_{a^i} p_{a^i}\right)$$
- Soft target update:
$$\theta_\text{target} \leftarrow \tau \theta_\text{online} + (1-\tau)\theta_\text{target}$$

---

## 6. System Pipeline

```
Full Library (N cells)
        │
        ▼
  Initialize p = uniform [1/N, ..., 1/N]
        │
   ┌────▼──────────────────────────────────┐
   │  Sample n cells using p               │
   │  → Write partial .genlib             │
   │  → ABC: read lib; map; topo;          │
   │         upsize; dnsize; stime         │
   │  → Parse D_S, A_S from output         │
   │  → reward = -(D_S/D_base)*(A_S/A_base)│
   │  → Update p via MAB or Q-Learning     │
   └────────────┬──────────────────────────┘
                │  repeat until timeout (1 hr)
                ▼
     Best partial library found
```

**ABC command sequence:**
```
read <library>.genlib
map          # delay-driven mapper (default)
# or: map -a  # area-driven mapper
topo
upsize
dnsize
stime        # get timing (Delay)
print_stats  # get Area
```

**Batch size:** 10 (run 10 episodes per update step)  
**Timeout:** 1 hour per experiment  
**Sampling size ranges used in paper experiments:**

| Library | Sampling range | Step |
|---|---|---|
| ASAP7 (161 cells) | 45–135 | 10 |
| NAN45 (94 cells) | 35–75 | 10 |
| SKY130 (343 cells) | 220–310 | 10 |
| GF180 (151 cells) | 40–130 | 10 |

---

## 7. Libraries & Benchmarks

### Libraries

| Alias | File | Technology | # Gates |
|---|---|---|---|
| ASAP7 | `7nm.genlib` | 7 nm | 161 |
| NAN45 | `nan45.genlib` | 45 nm | 94 |
| SKY130 | `sky130.genlib` | 130 nm | 343 |
| GF180 | `gf180mcu_ff_125C.genlib` | 180 nm | 151 |

### Benchmark Suites
ISCAS 85, ISCAS 89, ITC/ISCAS 99, VTR 8.0, EPFL

**PA3 designs:** `s13207`, `c2670`, `b20`

---

## 8. Experimental Results

### 8.1 Research Questions & Findings

**RQ1: How effective is MapTune in optimizing ADP?**
- All 4 MapTune variants converge stably across 9 designs on ASAP7 within 1200s
- Example: `bar` → all methods achieve ≥15% ADP reduction within 300s; MapTune-UCB achieves >20% in the first 15s
- **MAB methods slightly outperform Q-methods** in general convergence speed and lowest achievable ADP
- Example: `c880` — MapTune-ε converges to lowest ADP in 160s; MapTune-DDQN ends up ~5% higher; MapTune-DQN reaches similar ADP but takes 5× longer

**RQ2: Is MapTune adaptive to different technologies?**
- Effective across all 4 libraries (7nm, 45nm, 130nm, 180nm)
- Minimum: 4% ADP reduction (`b14` on GF180 with delay-driven mapper)
- Maximum: `s838a` averaged **36% ADP reduction** across all libraries

**RQ3: Effective across different mappers?**
- Both `map` (delay-driven) and `map -a` (area-driven) show consistent improvement
- `s35932`: 40.12% ADP reduction (delay-driven) vs 40.00% (area-driven)

### 8.2 Delay-Driven Mapper — Average Results (20 designs)

| Library | Avg Delay Δ (Best MAB) | Avg Delay Δ (Best QL) | Avg Area Δ (Best MAB) | Avg Area Δ (Best QL) |
|---|---|---|---|---|
| ASAP7 | **−23.15%** | −21.04% | −3.42% | **−3.89%** |
| NAN45 | **−22.06%** | −20.29% | 1.76% | 1.76% |
| SKY130 | **−22.51%** | −20.34% | **−2.68%** | −2.49% |
| GF180 | **−22.64%** | −22.14% | **1.19%** | 1.42% |

### 8.3 Area-Driven Mapper — Average Results (20 designs)

| Library | Avg Delay Δ (Best MAB) | Avg Delay Δ (Best QL) | Avg Area Δ (Best MAB) | Avg Area Δ (Best QL) |
|---|---|---|---|---|
| ASAP7 | **−22.27%** | −22.95% | **−2.19%** | −1.54% |
| NAN45 | **−21.52%** | −20.85% | **0.27%** | 1.32% |
| SKY130 | **−23.28%** | −22.45% | **−0.68%** | −0.10% |
| GF180 | −22.34% | **−22.70%** | **3.04%** | 3.85% |

### 8.4 Pareto Trade-off Insight

- MapTune primarily **optimizes delay** (~21–23% avg. reduction)
- Area trade-off is small: <2% penalty or simultaneous improvement in most cases
- Best delay case: `multiplier` on SKY130 (delay-driven) → **39.96% delay ↓** with only 4.03% area ↑
- Best area case: `sqrt` on GF180 (area-driven) → 1.25% delay ↑ for **14.90% area ↓**
- Overall: **average ADP improvement of 22.54%** across all settings

---

## 9. PA3 Implementation Guide

### 9.1 Budget Calculation for PA3

| Library | Total Cells | 30% | 50% | 70% |
|---|---|---|---|---|
| `nan45.genlib` | 94 | 28 | 47 | 66 |
| `sky130.genlib` | 343 | 103 | 172 | 240 |

### 9.2 What "MapTune Baseline" Means

Run MapTune (any variant) at each fixed budget → find the subset achieving minimum ADP → record $|S_\text{MapTune}|$ and $ADP_\text{MapTune}$.

Your method must satisfy:
$$|S_\text{new}| \leq |S_\text{MapTune}|, \quad ADP_\text{new} < ADP_\text{MapTune}$$

### 9.3 Strategies to Beat MapTune

| Strategy | Idea | Difficulty |
|---|---|---|
| Better RL (Dueling DQN, PER) | Reuse Lab 5 code | Low |
| Simulated Annealing | Start from MapTune's best subset, swap cells, accept worse with prob $e^{-\Delta/T}$ | Low |
| Greedy + Local Search | Rank cells by marginal ADP gain, build greedily, then swap pairs | Medium |
| Genetic Algorithm | Binary encoding, crossover + mutation | Medium |
| Bayesian Optimization | GP model of ADP(S), EI acquisition | High |

**Recommendation:** Start with Simulated Annealing — it requires very few ABC calls per iteration and is easy to implement correctly.

### 9.4 ABC Interaction Skeleton (Python)

```python
import subprocess
import re

def run_abc_mapping(design_path, lib_path, area_driven=False):
    """Run ABC mapping and return (delay_ps, area_um2)."""
    mapper = "map -a" if area_driven else "map"
    script = f"""
read_library {lib_path}
read {design_path}
{mapper}
topo
upsize
dnsize
stime
print_stats
"""
    result = subprocess.run(
        ["abc", "-c", script],
        capture_output=True, text=True, timeout=120
    )
    return parse_abc_output(result.stdout)

def parse_abc_output(stdout):
    delay = float(re.search(r'Delay\s*=\s*([\d.]+)', stdout).group(1))
    area  = float(re.search(r'Area\s*=\s*([\d.]+)', stdout).group(1))
    return delay, area

def compute_normalized_adp(delay, area, delay_base, area_base):
    return (delay / delay_base) * (area / area_base)
```

### 9.5 RL Environment Skeleton

```python
import numpy as np

class LibraryTuningEnv:
    def __init__(self, N, budget, design_path, lib_path, D_base, A_base):
        self.N = N              # total cells in library
        self.budget = budget    # n cells to select per episode
        self.design_path = design_path
        self.lib_path = lib_path
        self.D_base = D_base
        self.A_base = A_base
        self.reset()

    def reset(self):
        self.state = np.zeros(self.N, dtype=np.float32)
        self.selected = []
        return self.state.copy()

    def step(self, action):
        assert self.state[action] == 0, "Cell already selected"
        self.state[action] = 1
        self.selected.append(action)
        done = len(self.selected) == self.budget
        reward = 0.0
        if done:
            partial_lib = write_partial_genlib(self.lib_path, self.selected)
            delay, area = run_abc_mapping(self.design_path, partial_lib)
            adp = compute_normalized_adp(delay, area, self.D_base, self.A_base)
            reward = -adp
        return self.state.copy(), reward, done, {}
```

### 9.6 Mapping Lab 5 DQN → PA3

| Lab 5 (CartPole/Pong) | PA3 MapTune |
|---|---|
| State = game pixels / pole angle | State = binary selection vector $\in \{0,1\}^\mathcal{N}$ |
| Action = discrete game moves | Action = cell index to add (0 to $\mathcal{N}-1$) |
| Reward = per-step game score | Reward = $-ADP_\mathcal{S}$ only at episode end |
| Episode ends on game over | Episode ends when $n$ cells selected |
| ε-greedy / UCB exploration | Same — mask already-selected cells |
| Replay buffer + target network | Directly reusable |
| DDQN, PER, multi-step return | Directly reusable |

Your Lab 5 agent code is **directly reusable**. Just wrap the `LibraryTuningEnv` above as a `gym.Env` interface and plug in.

---

## 10. Conclusions

- Partial library tuning is valid and effective for technology mapping
- MapTune achieves **22.54% average ADP improvement** across 5 benchmark suites, 4 libraries, 2 mappers
- MAB methods slightly outperform Q-Learning in convergence speed and final ADP
- MapTune primarily optimizes **delay** (~22% avg.) with modest area trade-offs
- Future work: cross-design library exploration + integration with automatic library generation tools
