# PA2 - Supervised Learning for EDA Logic Gate Representation Learning through GNN

---

- Name: Yue-Lin Tu
- Student ID: B11107122

---

## Table of Contents

- [Assignment 1：Hidden layer dimensions](#assignment-1-hidden-layer-dimensions)
  - [Experiment setup](#experiment-setup)
  - [Per-epoch validation metrics](#per-epoch-validation-metrics)
  - [Final results comparison (validation set)](#final-results-comparison-validation-set) 
  - [Observations and analysis](#observations-and-analysis) 
- [Assignment 2：Ablation study](#assignment-2-ablation-study)
  - [Experiment setup](#experiment-setup-1)
  - [Per-epoch validation metrics](#per-epoch-validation-metrics-1)
  - [Final results comparison (validation set)](#final-results-comparison-validation-set-1)
  - [Observations and analysis](#observations-and-analysis-1)
- [Assignment 3：Switching Probability Prediction](#assignment-3-switching-probability-prediction)
  - [Problem Definition](#problem-definition)
  - [Methodology](#methodology)
  - [Results](#results)
  - [Discussion and Conclusion](#discussion-and-conclusion)
- [Summary](#summary)

---
## Assignment 1：Hidden layer dimensions

### Experiment setup

| 項目 | 設定 |
|---|---|
| 模型架構 | DeepGate2（MLP+GNN，`--arch mlpgnn`） |
| 測試維度 | `dim_hidden` ∈ {32, 64, 128} |
| 訓練輪數 | **10 epochs** |
| 損失函數 | Regression: L1、Classification: BCE |
| 損失權重 | LProb×3、LRC×1、LFunc×2 |
| 其他超參 | `batch_size=32`、`num_workers=4`、`num_rounds=1` |
| 備註 | 訓練資料缺少 RC 標籤，全部加入 `--no_rc`（LRC 為 dummy pair，不納入分析） |

**Evaluation metrics:**
- **Val LProb**（訊號概率 L1 損失，越低越好）
- **Val ACC**（TT 距離排名準確率，越高越好）

---

### Per-epoch validation metrics

#### dim_hidden = 32（10 epochs）

| Epoch | Val LProb ↓ | Val ACC ↑ |
|------:|------------:|----------:|
| 1 | 0.132608 | 0.819677 |
| 2 | 0.081799 | 0.793548 |
| 3 | 0.077942 | 0.779677 |
| 4 | 0.067498 | 0.817742 |
| **5** | **0.050534** | **0.828710** |
| 6 | 0.043799 | 0.816129 |
| 7 | 0.042671 | 0.834194 |
| 8 | 0.044585 | 0.845484 |
| 9 | 0.038039 | 0.858387 |
| **10** | **0.035522** | **0.867097** |

#### dim_hidden = 64（10 epochs）

| Epoch | Val LProb ↓ | Val ACC ↑ |
|------:|------------:|----------:|
| 1 | 0.079139 | 0.800323 |
| 2 | 0.058357 | 0.796129 |
| 3 | 0.045008 | 0.803871 |
| 4 | 0.038454 | 0.820968 |
| **5** | **0.038891** | **0.851613** |
| 6 | 0.032651 | 0.853226 |
| 7 | 0.029560 | 0.877097 |
| 8 | 0.029870 | 0.901935 |
| 9 | 0.028979 | 0.907097 |
| **10** | **0.024337** | **0.910968** |

#### dim_hidden = 128（10 epochs）

| Epoch | Val LProb ↓ | Val ACC ↑ |
|------:|------------:|----------:|
| 1 | 0.060308 | 0.782258 |
| 2 | 0.049645 | 0.810000 |
| 3 | 0.038586 | 0.809677 |
| 4 | 0.037925 | 0.837097 |
| **5** | **0.027839** | **0.853871** |
| 6 | 0.026890 | 0.854194 |
| 7 | 0.027924 | 0.873226 |
| 8 | 0.029710 | 0.907742 |
| 9 | 0.027055 | 0.908387 |
| **10** | **0.023696** | **0.911935** |

---

### Final results comparison (validation set)

| dim_hidden | Val LProb (Signal Probability) ↓ | Val ACC (Accuracy of TT distance prediction) ↑ | Best ACC ↑ | Epochs |
|:----------:|:---:|:---:|:---:|:------:|
| 32 | 0.035522 | 0.867097 | 0.867097 | 10 |
| 64 | 0.024337 | 0.910968 | 0.910968 | 10 |
| **128** | **0.023696** | **0.911935** | **0.911935** | 10 |

> Val LRC 使用 dummy RC pair（`--no_rc`），不具可比較性，略去不分析。

---

### Observations and analysis
**1. 模型 Hidden Dimension 與 LProb 損失明顯負相關，趨勢在 10 epochs 後更顯著**

dim=128 在 10 epochs 後達到最低 Val LProb（0.023696），dim=64 次之（0.024337），dim=32 最高（0.035522）。與 5 epochs 相比，三組的 LProb 均進一步下降（dim=128: 0.028→0.024，dim=64: 0.039→0.024，dim=32: 0.051→0.036），且趨勢方向完全一致，更多 epochs 只是放大了已有的規律。

**2. dim=32 收斂速度明顯較慢，10 epochs 仍未追上 dim=64 的 5 epoch 水準**

dim=32 在 10 epochs 的最終 Val LProb（0.035522）只略低於 dim=64 在 epoch 5 的 LProb（0.038891）——事實上 dim=32 需要 epoch 9（0.038039）才能接近 dim=64 的 epoch 5 水準，顯示 32 維的模型容量是真正的瓶頸，而非訓練時間不夠。

**3. ACC 在 10 epochs 後分化更清晰**

| 維度 | 5 ep ACC | 10 ep ACC | 增量 |
|:---:|:---:|:---:|:---:|
| 32 | 0.828710 | 0.867097 | +3.84 pp |
| 64 | 0.852258 | 0.910968 | +5.87 pp |
| 128 | 0.855161 | 0.911935 | +5.68 pp |

dim=64 在 10 epochs 的增益（+5.87 pp）最大，顯示該維度的學習效率最高；dim=128 與 dim=64 的 ACC 差距從 5 ep 的 0.29 pp 進一步收窄至 10 ep 的 0.10 pp，呈現容量飽和的典型特徵。

**4. 10 epochs 後 dim=64 與 dim=128 在兩項指標上均趨近飽和**

dim=128 的最終 LProb（0.023696）比 dim=64（0.024337）僅低約 2.6%（絕對差 0.000641），ACC 也僅領先 0.10 pp（0.911935 vs 0.910968）——兩者的差距都微乎其微。值得注意的是，5 epochs 時 LProb 的差距仍有 28%（0.028 vs 0.039），但 10 epochs 後收窄至 2.6%，說明 dim=64 只是在前期收斂較慢，充分訓練後兩個維度的最終精度趨於一致。ACC 的差距同樣從 5ep 的 0.29 pp 收窄至 0.10 pp。兩個指標同步飽和，確認真正的容量效益分水嶺在 32→64，64→128 的邊際收益在 10 epochs 後可忽略不計。

---

## Assignment 2：Ablation study

### Experiment setup

| 項目 | 設定 |
|---|---|
| 模型架構 | DeepGate2（MLP+GNN，`--arch mlpgnn`，`dim_hidden=64`） |
| 訓練輪數 | **10 epochs** |
| 損失函數 | Regression: L1、Classification: BCE |
| 損失權重 | LProb×3、LRC×1、LFunc×2 |
| 其他超參 | `batch_size=32`、`num_workers=4`、`num_rounds=1` |
| 備註 | 訓練資料缺少 RC 標籤，全部加入 `--no_rc`（LRC 為 dummy pair，不納入分析） |

**Ablation experiments:**

| 實驗名稱 | 額外參數 | 消融目標 |
|---|---|---|
| baseline | —— | 完整 DeepGate2（正交 PI 初始化 + pairwise TT loss） |
| no_tt_loss | `--no_func` | 移除 pairwise TT loss（LFunc 項） |
| homo_pi_init | `--homo_pi_init` | 改用同質 PI 初始化（取代正交初始化） |

**Evaluation metrics:**
- **Val LProb**（訊號概率 L1 損失，越低越好）
- **Val LFunc**（功能相似度 L1 損失，越低越好；no_tt_loss 為 0）
- **Val ACC**（TT 距離排名準確率，越高越好）

---

### Per-epoch validation metrics

#### baseline

| Epoch | Val LProb ↓ | Val LFunc ↓ | Val ACC ↑ |
|------:|------------:|------------:|----------:|
| 1 | 0.077982 | 0.236567 | 0.800000 |
| 2 | 0.060947 | 0.176555 | 0.795484 |
| 3 | 0.044657 | 0.161152 | 0.803226 |
| 4 | 0.043406 | 0.141598 | 0.821290 |
| **5** | **0.038333** | **0.136440** | **0.852258** |
| 6 | 0.031449 | 0.133411 | 0.851613 |
| 7 | 0.031116 | 0.130045 | 0.877419 |
| 8 | 0.030653 | 0.127147 | 0.902258 |
| 9 | 0.028626 | 0.126300 | 0.909032 |
| **10** | **0.024151** | **0.125278** | **0.910645** |

#### no_tt_loss（Remove TT loss）

| Epoch | Val LProb ↓ | Val LFunc ↓ | Val ACC ↑ |
|------:|------------:|------------:|----------:|
| 1 | 0.073908 | 0.000000 | 0.748710 |
| 2 | 0.051057 | 0.000000 | 0.748065 |
| 3 | 0.043072 | 0.000000 | 0.751613 |
| 4 | 0.043192 | 0.000000 | 0.760968 |
| **5** | **0.033604** | **0.000000** | **0.767419** |
| 6 | 0.032325 | 0.000000 | 0.748710 |
| 7 | 0.030695 | 0.000000 | 0.752581 |
| 8 | 0.031274 | 0.000000 | 0.754839 |
| 9 | 0.027370 | 0.000000 | 0.762903 |
| **10** | **0.024276** | **0.000000** | **0.753871** |

#### homo_pi_init（Homogeneous PI initialization）

| Epoch | Val LProb ↓ | Val LFunc ↓ | Val ACC ↑ |
|------:|------------:|------------:|----------:|
| 1 | 0.083960 | 0.237051 | 0.772581 |
| 2 | 0.063943 | 0.178981 | 0.770968 |
| 3 | 0.041481 | 0.159105 | 0.788065 |
| 4 | 0.039617 | 0.139563 | 0.794516 |
| **5** | **0.037968** | **0.135254** | **0.817742** |
| 6 | 0.034774 | 0.133014 | 0.810968 |
| 7 | 0.035054 | 0.131868 | 0.820323 |
| 8 | 0.034206 | 0.131125 | 0.833871 |
| 9 | 0.031511 | 0.130547 | 0.846452 |
| **10** | **0.025419** | **0.129545** | **0.863226** |

---

### Final results comparison (validation set)

| Experiment | Val LProb ↓ | Val LFunc ↓ | Val ACC ↑ | ACC Change |
|---|:---:|:---:|:---:|:---:|
| **baseline** | 0.024151 | 0.125278 | **0.910645** | — |
| no_tt_loss | 0.024276 | 0.000000 | 0.753871 | **−15.68 pp** |
| homo_pi_init | 0.025419 | 0.129545 | 0.863226 | −4.74 pp |

> Val LRC 使用 dummy RC pair（`--no_rc`），不具可比較性，略去不分析。

---

### Observations and analysis

**1. Pairwise TT loss 是 ACC 的關鍵來源，10 epochs 後差距進一步擴大**

移除 TT loss（no_tt_loss）後，Val ACC 從 0.910645 下降至 0.753871，絕對值下降 **15.68 個百分點**（5 epochs 時為 8.48 pp）。差距隨訓練輪數擴大的現象顯示：兩個模型正在向不同的局部最優收斂——baseline 的 GNN 在 TT loss 監督下持續學習功能區分性，而 no_tt_loss 缺乏明確的功能監督訊號，其 ACC 在 epoch 5 到 10 之間甚至出現震盪（0.767→0.749→0.753→0.763→0.754），表明模型已接近其容量上限但在次最優解附近振盪。

**2. 移除 TT loss 後，Val LProb 收斂至與 baseline 幾乎相同的水準**

no_tt_loss 的最終 Val LProb（0.024276）與 baseline（0.024151）幾乎相同（差距不足 0.001）。這一現象在 10 epochs 後尤其清楚：兩個模型在訊號概率預測任務上趨向相同收斂點，但由於 no_tt_loss 缺乏功能相似度的監督，其 hidden state 雖能預測 signal probability，卻無法建立有效的功能距離空間。這再次確認 LProb 和 ACC 衡量的是不同層次的學習質量。

**3. no_tt_loss 的 ACC 在 epoch 5 後停滯並出現週期性退化**

從逐 epoch 曲線可見，no_tt_loss 在 epoch 5 達到局部高點（0.767419）後，後續 5 個 epoch 的 ACC 在 0.749–0.763 之間反覆震盪，最終停在 0.753871，比 epoch 5 還低 1.36 pp。這說明在沒有 TT loss 的情況下，模型對 ACC 任務的學習能力在前期已耗盡，後續訓練反而可能對 hidden state 造成干擾。

**4. Orthogonal PI initialization 的效益在 10 epochs 後更加穩固且差距擴大**

homo_pi_init 的 Val ACC（0.863226）比 baseline 下降 **4.74 pp**（5 epochs 時為 3.45 pp）。儘管 homo_pi_init 的 ACC 隨訓練持續改善（5ep: 0.817→10ep: 0.863），baseline 的改善速度更快（5ep: 0.851→10ep: 0.911），導致差距從 3.45 pp 擴大至 4.74 pp。這意味著正交初始化的好處不僅是在訓練初期提供更好的起點，更重要的是它為 GNN 的長期學習提供了持續的結構優勢。

**5. homo_pi_init 的 LProb 在 10 epochs 後仍略差於 baseline（0.0254 vs 0.0242）**

5 epochs 時 LProb 幾乎無差異（0.03797 vs 0.03833，相差 1.0%），但 10 epochs 後差距拉開（homo: 0.025419，baseline: 0.024151，相差 5.2%）。這顯示正交 PI 初始化對訊號概率預測（LProb）的影響是一個緩慢累積的過程：較好的初始化使 GNN 能更精確地傳遞電路結構資訊，從而在更多輪次後顯現出更低的概率預測誤差。

**6. RC loss 為何不納入分析**

DeepGate2 公開資料集（rawaig.tar.bz2）的 `prepare_dataset.py` 並未生成 `rc_pair_index` 與 `is_rc` 標籤，且官方訓練腳本 `run/stage1_train.sh`、`run/stage2_train.sh` 均使用 `--no_rc`。本實驗沿用此設定，所有實驗的 LRC 為對 dummy pair `[[0,1]]` 的 BCE，數值不具任務意義。雖然作業敘述提到 orthogonal PI init 應「affects reconvergence analysis」，但在公開可重現條件下無法直接量測 RC accuracy；本實驗以 **ACC（TT distance）** 作為 hidden state 區分力的代理指標，此指標已展示 homo_pi_init 帶來的 −4.74 pp 退化，間接驗證了 PI 區分性對下游結構任務的重要性。

---

## Assignment 3：Switching Probability Prediction

### Problem Definition

DeepGate2 原本的 supervised 任務是預測每個節點的**訊號概率** $p = \Pr[\text{node}=1]$（在隨機輸入向量下取 1 的比例）。然而在動態功耗分析中，主導項是**切換功耗** $P_\text{dyn} = \alpha \cdot C \cdot V^2 \cdot f$，其中 $\alpha$ 為**切換機率**（switching/transition probability）：相鄰時刻節點值翻轉的機率。

Signal probability 與 switching probability 並不等價：兩個訊號同樣 50% 為 1 的時間，可能一個快速切換（每 cycle 翻一次），另一個整段保持為 1 後再整段為 0（極少切換）。前者切換功耗大，後者切換功耗小。

**新增任務**：在 DeepGate2 的多任務目標中加入第四個監督項——預測每個節點的 switching probability $\alpha = \Pr[v(t) \neq v(t-1)]$。

### Methodology

> **路徑與程式碼說明**
> 本作業在 DeepGate2 上做了「資料管線 → 模型 → 訓練 → 評估」四層的修改，具體檔案、診斷與差異請參考：
> - 修改總表：[README.md § Modifications to DeepGate2 → Assignment 3](README.md#deepgate2-modifications)
> - 與 upstream main 的完整 diff：[deepgate2.patch](deepgate2.patch)
> - 跑實驗的腳本：[run_assignment3.sh](run_assignment3.sh)（Phase 1=label gen，Phase 2=train×2，Phase 3=eval，Phase 4=compare）

#### 1. Stimulus model：Markov input stream

純 i.i.d. (independent and identically distributed) 均勻隨機輸入下，每個節點的切換機率恆等於 $2p(1-p)$（封閉式），與訊號概率為一一對應的解析函數，使預測任務退化為 signal probability 的後處理。為使任務真正非平凡，採用 **Markov 刺激**：每個 cycle 每個 PI 以機率 $q$（本實驗取 $q = 0.1$，模擬實際電路較低的輸入翻轉率）獨立翻轉。此時內部節點的切換機率由電路拓撲決定，無法以 $p$ 解析推得。

具體流程（實作於 [`src/utils/circuit_utils.py`](DeepGate2/src/utils/circuit_utils.py) 的 `simulator_truth_table_markov`，由 [`src/prepare_dataset.py`](DeepGate2/src/prepare_dataset.py) 在 `--markov` flag 下呼叫）：
1. 第一個 pattern：均勻隨機向量（初始化）。
2. 每個後續 cycle：根據前一個 pattern，每位 PI 以 $q = 0.1$ 機率翻轉。
3. 對每個 pattern 跑 logic simulation，讓邏輯值傳播到所有內部節點。
4. 記錄每個節點的時間序列 $v(0), v(1), \ldots, v(T-1)$。
5. 跑 $T = 15000$ 個 pattern，對每個節點計算經驗切換率：$\alpha = \dfrac{|\{t : v(t) \neq v(t-1)\}|}{T-1}$。
6. 將 $\alpha$ 連同 prob、tt_pair_index、tt_dis 等寫入 `labels_markov.npz`（key=`trans_prob`），與既有 `labels.npz` 並存。
> 這個 $\alpha$ 就是 ground truth label，模型要學的是「從電路結構預測這個 $\alpha$」。

#### 2. Model architecture modification

在 [`src/models/mlpgate.py`](DeepGate2/src/models/mlpgate.py) 加入新的 readout head：
```python
self.readout_trans = MLP(dim_hidden, dim_mlp, 1, num_layer=3, sigmoid=True)
```
與 `readout_prob` 平行接在功能 hidden state $h_f$ 上，輸出經 sigmoid 限定在 $[0,1]$。模型 forward 從 4-tuple `(hs, hf, prob, is_rc)` 改為 5-tuple `(hs, hf, prob, trans, is_rc)`。為向後相容，[`src/get_emb_aig.py`](DeepGate2/src/get_emb_aig.py) / [`src/get_emb_bench.py`](DeepGate2/src/get_emb_bench.py) / [`src/test_acc_bin.py`](DeepGate2/src/test_acc_bin.py) 都加入了「同時接受 4-tuple 與 5-tuple」的解包 shim。

#### 3. Loss Function

在 [`src/trains/mlpgnn_trainer.py`](DeepGate2/src/trains/mlpgnn_trainer.py) 新增 $\mathcal{L}_\text{Trans} = \|\text{trans}_\text{pred} - \alpha_\text{target}\|_1$，並以 `--Trans_weight`（在 [`src/config.py`](DeepGate2/src/config.py) 註冊）加入加權總和。同時加入 `--no_trans` 旗標可在 forward 時將 LTrans 強制歸零：

$$\mathcal{L} = \frac{w_P \mathcal{L}_\text{Prob} + w_R \mathcal{L}_\text{RC} + w_F \mathcal{L}_\text{Func} + w_T \mathcal{L}_\text{Trans}}{w_P + w_R + w_F + w_T}$$

LTrans 與既有三個 loss 共用同一個 `loss_states` dict，因此會自動進入 epoch 紀錄與 validation log，無需修改 trainer 的 logging 路徑。

#### 4. Data pipeline plumbing

為讓「Markov 標籤檔」與「原始 i.i.d. 標籤檔」可共存切換、且 PyG cache 不會互相污染：
- [`src/config.py`](DeepGate2/src/config.py)：新增 `--label_file` CLI 參數，取代原本硬編碼的 `args.label_file = "labels.npz"`。
- [`src/datasets/load_data.py`](DeepGate2/src/datasets/load_data.py)：`parse_pyg_mlpgate` 新增 `trans_y` 參數，並在缺 `trans_prob` 欄位時退回 analytic $2p(1-p)$，讓舊資料集仍可訓練。
- [`src/datasets/mlpgate_dataset.py`](DeepGate2/src/datasets/mlpgate_dataset.py)：讀 `labels[cir_name]['trans_prob']`，且 PyG cache 目錄按 `args.label_file` 命名，避免 Markov ↔ i.i.d. 之間的快取衝突。

**5. Experiment setup**

| 項目 | 設定 |
|---|---|
| 模型架構 | DeepGate2（MLP+GNN，`--arch mlpgnn`，`dim_hidden=64`） |
| 訓練輪數 | **10 epochs**（並另保留 5-epoch checkpoint 作為對照） |
| 損失函數 | Regression: L1、Classification: BCE |
| 損失權重 | LProb×3、LRC×1、LFunc×2、**LTrans×{0, 2}** |
| 其他超參 | `batch_size=32`、`num_workers=4`、`num_rounds=1` |
| 備註 | 訓練資料缺少 RC 標籤，全部加入 `--no_rc`（LRC 為 dummy pair，不納入分析） |
| **刺激模型** | **Markov，$q = 0.1$（每 PI 每 cycle 翻轉機率）** |
| **Patterns 數** | **15000（有序序列**） |
| **標籤檔** | **`labels_markov.npz`（兩組實驗共用以隔離 stimulus 變因）** |

**Ablation experiments:**

| 實驗名稱 | `--Trans_weight` | 用途 |
|---|---|---|
| prob_only | 0 | 對照組：trans head 存在但無梯度（loss 不入總和），驗證 head 在無監督下是否能自發收斂到合理值 |
| with_trans | 2 | 主要組：trans head 受監督學習，與 LProb（×3）、LFunc（×2）共同優化 |

兩組共用同一個 `labels_markov.npz`，因此 `prob` 標籤、`tt_dis` 標籤、模型結構、訓練資料完全相同，**唯一差異就是 `--Trans_weight`**——這保證 LTrans 改善只能歸因於監督訊號本身，而非資料或架構差異。

#### 6. Evaluation metrics

訓練後在驗證集上由 [`src/test_trans.py`](DeepGate2/src/test_trans.py)（PA3 新增的獨立 eval 腳本）計算（9756 個 circuits、3.4M 個節點）：

- $\text{L1}_\text{model} = \overline{|\hat{\alpha} - \alpha|}$：模型 trans head 預測誤差。
- $\text{L1}_\text{analytic} = \overline{|2\hat{p}(1-\hat{p}) - \alpha|}$：以**模型自己預測的 $\hat{p}$** 套封閉式公式所得到的 baseline 誤差。注意分母用 $\hat{p}$ 而非 ground-truth $p$，是為了在「模型可獲得的資訊」下比較——量化 GNN 從拓撲結構學到多少**超越 closed-form 後處理**的時序資訊。
- $\text{Gap} = \text{L1}_\text{analytic} - \text{L1}_\text{model}$：正值表示模型擊敗 closed-form baseline。
- 按 gate type 分組（PI / AND / NOT）的 $\text{L1}_\text{model}$，分析模型在不同節點型態的表現（PI 為 stimulus 直接決定，AND/NOT 為 GNN 必須從拓撲推導）。

---

### Results

#### 1. Per-epoch validation metrics
##### (1) prob_only（Trans_weight = 0, Control group）

| Epoch | Val LProb ↓ | Val LFunc ↓ | Val LTrans ↓ | Val ACC ↑ |
|------:|------------:|------------:|-------------:|----------:|
| 1 | 0.162537 | 0.274931 | 0.460540 | 0.805161 |
| 2 | 0.087474 | 0.196303 | 0.460856 | 0.781935 |
| 3 | 0.076255 | 0.164912 | 0.457577 | 0.810968 |
| 4 | 0.061112 | 0.149451 | 0.463093 | 0.808710 |
| **5** | **0.056711** | **0.142600** | **0.459056** | **0.813548** |
| 6 | 0.050922 | 0.139739 | 0.461642 | 0.826452 |
| 7 | 0.049572 | 0.138624 | 0.460257 | 0.822258 |
| 8 | 0.041304 | 0.137027 | 0.459669 | 0.829355 |
| 9 | 0.040071 | 0.135200 | 0.460051 | 0.842903 |
| **10** | **0.037693** | **0.134224** | **0.461819** | **0.845806** |

##### (2) with_trans（Trans_weight = 2, Main group）

| Epoch | Val LProb ↓ | Val LFunc ↓ | Val LTrans ↓ | Val ACC ↑ |
|------:|------------:|------------:|-------------:|----------:|
| 1 | 0.157799 | 0.276094 | 0.307408 | 0.802903 |
| 2 | 0.088887 | 0.198818 | 0.244997 | 0.780000 |
| 3 | 0.079079 | 0.173149 | 0.131525 | 0.803226 |
| 4 | 0.065626 | 0.158822 | 0.077650 | 0.810000 |
| **5** | **0.053630** | **0.148922** | **0.051425** | **0.804194** |
| 6 | 0.054728 | 0.142740 | 0.038093 | 0.816129 |
| 7 | 0.048744 | 0.140262 | 0.030977 | 0.816774 |
| 8 | 0.047768 | 0.138441 | 0.026847 | 0.812903 |
| 9 | 0.045867 | 0.136608 | 0.026071 | 0.824194 |
| **10** | **0.046856** | **0.136056** | **0.026568** | **0.831935** |

#### 2. Final results comparison (validation set)

| 實驗 | Val LProb ↓ | Val LFunc ↓ | Val LTrans ↓ | Val ACC ↑ |
|---|:---:|:---:|:---:|:---:|
| prob_only | 0.037693 | 0.134224 | 0.461819 | 0.845806 |
| **with_trans** | 0.046856 | 0.136056 | **0.026568** | 0.831935 |
| Δ (with_trans − prob_only) | +0.009 | +0.002 | **−0.435** | −0.014 |

#### 3. Switching probability prediction performance
(test_trans.py, full dataset 9756 circuits, 3.4M nodes)

| 實驗 | L1(prob) | L1 analytic 2p(1−p) | L1(trans) 模型 | Gap (analytic−model) |
|---|---------:|--------------------:|---------------:|---------------------:|
| prob_only | 0.036752 | 0.249507 | 0.462986 | **−0.213**（模型輸 baseline） |
| **with_trans** | 0.045719 | 0.255825 | **0.025580** | **+0.230**（模型擊敗 baseline） |

##### By Gate Type

| Gate type | N nodes | prob_only L1 | with_trans L1 |
|:---|---------:|-------------:|--------------:|
| PI  |   533,440 | 0.587132 | **0.025038** |
| AND | 1,487,449 | 0.438152 | **0.024455** |
| NOT | 1,391,852 | 0.441946 | **0.026990** |

#### 4. Reference: 5-epoch checkpoint results

**Final results comparison (validation set)**

| 實驗 | Val LProb ↓ | Val LFunc ↓ | Val LTrans ↓ | Val ACC ↑ |
|---|:---:|:---:|:---:|:---:|
| prob_only (5ep)  | 0.051210 | 0.143175 | 0.460010 | 0.813226 |
| with_trans (5ep) | 0.054054 | 0.150468 | **0.051691** | 0.803548 |
| Δ (with_trans − prob_only) | +0.002844 | +0.007293 | **−0.408319** | −0.009678 |

**Switching probability prediction performance**

| 實驗 | L1(prob) | L1 analytic 2p(1−p) | L1(trans) 模型 | Gap (analytic−model) |
|---|---------:|--------------------:|---------------:|---------------------:|
| prob_only (5ep)  | 0.049467 | 0.248469 | 0.461578 | **−0.213**（模型輸 baseline） |
| with_trans (5ep) | 0.053545 | 0.256738 | **0.051427** | **+0.205**（模型擊敗 baseline） |

**By gate type (5 epochs):**

| Gate type | N nodes | prob_only L1 | with_trans L1 |
|:---|---------:|-------------:|--------------:|
| PI  |   533,440 | 0.604731 | **0.032974** |
| AND | 1,487,449 | 0.425856 | **0.047775** |
| NOT | 1,391,852 | 0.444890 | **0.062403** |

**Quick 5ep → 10ep summary（with_trans）：**
- LTrans 0.0514 → 0.0266（−48%）。
- Gap +0.205 → +0.230（+12%）。
- PI L1 0.033 → 0.025、AND 0.048 → 0.024、NOT 0.062 → 0.027（gate type 間從不均衡走向均衡）。

---

### Discussion and Conclusion

**1. 切換機率預測任務可學習性已被充分證明，10 epochs 後 LTrans 再降 50%**

with_trans 的最終 Val LTrans（0.0266）比 prob_only（0.4618）低 **約 17 倍**，比 5 epochs 時的 LTrans（0.0514）再降低約 48%。
prob_only 因 `Trans_weight=0`，trans head 從未獲得梯度更新，sigmoid 輸出在 10 個 epoch 的訓練中始終停留在 ~0.46（標準差極小，如實值：epoch 1 的 0.4605 到 epoch 10 的 0.4618 幾乎不變），與 Markov 刺激的目標均值 ~0.1 構成固定誤差，是完全靜態的 null prediction。
> with_trans 的 LTrans 從 epoch 1 的 0.307 在前 5 epoch 快速下降至 0.051，後 5 epoch 放緩並穩定在 0.026–0.027，顯示已接近收斂。

**2. GNN 學到了超越封閉式 $2p(1-p)$ 的時序結構，Gap 進一步提升**

封閉式 baseline $2\hat{p}(1-\hat{p})$ 在驗證集上的 L1 為 0.2558，遠高於 with_trans 模型的 0.0256——**模型的誤差僅為 baseline 的 1/10**（Gap = +0.230，比 5 epochs 時的 +0.205 再提升 12%）。
> 這進一步確認 GNN 從電路拓撲（深度、扇入、reconvergence）學到了預測時序相關性的能力，且學習效果隨訓練持續提升。

**3. 10 epochs 後三種 gate type 的 L1 趨於均衡，與 5 epochs 不同**

在 5 epochs 時，with_trans 的 PI L1（0.033）明顯低於 AND（0.048）和 NOT（0.062），反映 PI 節點的學習更容易（目標近似常數）。
但到 10 epochs，三類節點的 L1 幾乎相同：PI=0.025，AND=0.024，NOT=0.027——AND 甚至略優於 PI。
> 這表明初期的節點類型差距源於**收斂速度不同**，而非 AND/NOT 存在根本性的學習困難：AND/NOT 的切換機率依賴電路拓撲的多層傳播，需要更多訓練步才能捕捉，但 GNN 的表達能力足以在充分訓練後以相近精度學習所有節點類型的切換行為。

**4. 多任務代價在 10 epochs 後略微增加但仍然可接受**

加入 LTrans 後（以最終 epoch 比較, prob_only vs. with_trans）：
- LProb 從 0.037693 升至 0.046856（+24.3%，比 5 epochs 的 +5.6% 大）。
- LFunc 從 0.134224 升至 0.136056（+1.4%）。
- ACC 從 0.845806 降至 0.831935（−1.6%）。
> LProb 的相對損失比 5 epochs 時大，原因是在更多 epochs 後，baseline（prob_only）的 prob head 有更多機會優化（5ep→10ep 改善 26%），而 with_trans 的優化預算被 LTrans 分走（同期僅改善 13%）。
然而，考慮到 LTrans 下降了 17 倍（−94%），多任務代價仍遠小於收益。

**5. Markov 刺激是讓任務非平凡的關鍵設計**

若用 i.i.d. 均勻隨機刺激（$q=0.5$），則理論上每個節點的 trans = $2p(1-p)$，模型學習此映射等價於學習 prob 後做 closed-form 後處理，新增 head 無實質意義。
> 本實驗以 $q=0.1$ 的 Markov 流產生有時序相關性的標籤，使內部節點的切換率成為拓撲相關量，新任務才有獨立資訊量可學。10 epochs 結果中 Gap +0.230 的持續成長也佐證了這一設計的正確性。

---

## Summary

三個作業的結果共同指向 DeepGate2 在電路表示學習上的三個核心問題：

**表示容量（A1）**：dim=32 明顯容量不足，10 epochs 仍比 dim=64/128 落後約 4.4 pp ACC。容量效益的分水嶺在 32→64；64→128 的邊際收益在充分訓練後趨近於零（ACC 差距縮至 0.1 pp）。

**訓練目標設計（A2）**：Pairwise TT loss 是功能區分性的核心來源，移除後 ACC 下降 15.68 pp，且差距隨訓練輪數持續擴大。Orthogonal PI 初始化提供額外提升（+4.74 pp），兩者對 ACC 的貢獻量級差異懸殊（15.68 pp vs 4.74 pp）。LProb 與 ACC 衡量的是不同層次的學習質量——no_tt_loss 移除 TT 監督後 ACC 崩塌，但 LProb 幾乎不受影響。

**任務延伸（A3）**：在 Markov 刺激下，switching probability 是一個真正可學習的獨立任務，GNN 的預測誤差（L1=0.026）僅為封閉式 baseline $2p(1-p)$（L1=0.256）的 1/10，且多任務代價（LProb +24%、ACC −1.6%）遠小於新任務的收益（LTrans −94%）。
