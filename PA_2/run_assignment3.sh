#!/bin/bash
# ==============================================================================
# Assignment 3: Switching Probability Prediction (DeepGate2 extension)
#   New per-node task: predict transition (switching) probability under a
#   Markov input stream where each PI flips with probability `flip_prob` per
#   cycle. The new readout head + LTrans loss are wired as a co-objective with
#   the existing prob/RC/Func losses.
#
#   Phases:
#     1) Generate Markov-stimulus labels (prepare_dataset.py --markov).
#     2) Train two configs:
#          - prob_only   : --Trans_weight 0 (no transition supervision; baseline)
#          - with_trans  : --Trans_weight 2 (transition co-objective enabled)
#        Both use the Markov labels file so the prob target is consistent.
#     3) Evaluate each checkpoint with src/test_trans.py:
#          - L1 of model trans pred
#          - L1 of analytic 2*p*(1-p) baseline (p = predicted prob)
#          - Per-gate-type breakdown (PI / AND / NOT)
#     4) Cross-experiment comparison.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # 腳本所在目錄（自動偵測）

NUM_PROC=1  # 設置並行進程數為 1
GPUS=0      # 指定使用的 GPU 設備號為 0
DEEPGATE_ROOT="${HOME}/DeepGate2"               # 設置 DeepGate2 的根目錄
DEEPGATE_SRC="${DEEPGATE_ROOT}/src"             # 設置 DeepGate2 源代碼的路徑
EXP_BASE="${DEEPGATE_ROOT}/exp/prob"            # 設置實驗基礎目錄路徑
LOG_BASE="${LOG_BASE:-${SCRIPT_DIR}/results/assignment3}"    # 設置日誌保存的基礎目錄（可由環境變數覆寫）
EPOCHS="${EPOCHS:-5}"    # 設置訓練的輪數，預設 5（可由環境變數覆寫）
DIM=64      # 設置隱層維度為 64
FLIP_PROB=0.1   # 設置 PI 翻轉概率為 0.1
LABEL_FILE="labels_markov.npz"
LABEL_PATH="${DEEPGATE_ROOT}/data/train/${LABEL_FILE}"

mkdir -p "$LOG_BASE"    # 創建日誌輸出目錄（若目錄不存在則創建）

# ── Helpers ─────────────────────────────────────────────────────────────────
# 定義查找最新日誌文件的函數
latest_log() {
    # 在指定實驗目錄中查找 log.txt 文件，排序後返回最新的一個
    find "${EXP_BASE}/$1" -name "log.txt" 2>/dev/null | sort | tail -1
}

# 定義從日誌文件中提取最大輪次數的函數
max_epoch_in_log() {
    # 從日誌文件中提取所有 epoch 數字，按數值排序後返回最大值
    grep -o "epoch: [0-9]*" "$1" 2>/dev/null | grep -o "[0-9]*" | sort -n | tail -1
}

# 定義解析 epoch 行的 AWK 函數
    # 解析包含 "epoch:" 的行，提取訓練指標和驗證指標
    # 輸出格式為制表符分隔的：輪次、驗證概率損失、驗證重連接損失、驗證函數損失、驗證精度
parse_epoch_line() {
    awk '
    {
        match($0, /epoch: ([0-9]+)/, e); epoch = e[1]                                       # 從輸入行中提取 epoch 號
        n = split($0, lp, "LProb "); val_lp = (n >= 3) ? sprintf("%.6f", lp[3]+0) : "N/A"   # 提取 LProb 值，若不存在則標記為 "N/A"
        n = split($0, lr, "LRC ");   val_lr = (n >= 3) ? sprintf("%.6f", lr[3]+0) : "N/A"   # 提取 LRC 值，若不存在則標記為 "N/A"
        n = split($0, lf, "LFunc "); val_lf = (n >= 3) ? sprintf("%.6f", lf[3]+0) : "N/A"   # 提取 LFunc 值，若不存在則標記為 "N/A"
        n = split($0, lt, "LTrans "); val_lt = (n >= 3) ? sprintf("%.6f", lt[3]+0)  : "N/A" # 提取 LTrans 值，若不存在則標記為 "N/A"
        match($0, /ACC ([0-9.]+)/, a); acc = (a[1] != "") ? sprintf("%.6f", a[1]+0) : "N/A" # 從輸入行中提取精度（ ACC ）值
        printf "%d\t%s\t%s\t%s\t%s\t%s\n", epoch, val_lp, val_lr, val_lf, val_lt, acc       # 以制表符分隔格式打印提取的指標
    }'
}

# 定義生成實驗摘要的函數
summarize() {
    local name=$1                                   # 獲取實驗名稱參數
    local exp_id="PA2_assignment3/${name}"          # 構建實驗 ID
    local out="${LOG_BASE}/${name}_summary.txt"     # 定義輸出摘要文件的路徑
    local log                                       # 聲明日誌文件變數
    log=$(latest_log "$exp_id")                     # 獲取該實驗的最新日誌文件路徑

    {
        echo "======================================================================"
        echo "  SUMMARY  experiment = ${name}"      # 打印【實驗名稱】摘要標題
        echo "  Source log : ${log:-<not found>}"   # 打印源日誌文件路徑，若未找到則顯示 "<not found>"
        echo "  Generated  : $(date)"               # 打印生成時間
        echo "======================================================================"
        echo ""

        # 檢查日誌文件是否存在
        if [ -z "$log" ]; then
            echo "  [WARNING] No log file found."
            return
        fi

        echo "  Per-epoch validation metrics:"  # 打印每輪次驗證指標的標題
        # 打印列名和指標說明（↓ 表示越低越好，↑ 表示越高越好）
        echo "  Epoch | Val LProb (↓) | Val LRC (↓) | Val LFunc (↓) | Val LTrans (↓) | Val ACC (↑)"
        echo "  ------|---------------|-------------|---------------|----------------|------------"
        
        # 從日誌文件中提取包含 "epoch:" 的行並解析
        grep "epoch:" "$log" | parse_epoch_line | \
            # 格式化打印每一行的指標數據
            awk -F'\t' '{ printf "  %5d | %13s | %11s | %13s | %14s | %s\n", $1, $2, $3, $4, $5, $6 }'

        echo ""

        echo "  === Final Epoch ==="    # 打印最終輪次標題
        local last_line # 聲明最後一行變數
        last_line=$(grep "epoch:" "$log" | tail -1) # 提取日誌文件中最後一個包含 "epoch:" 的行
        
        # 檢查是否成功獲取最後一行
        if [ -n "$last_line" ]; then
            # 解析最後一行的數據
            echo "$last_line" | parse_epoch_line | \
                # 按制表符分隔符分割欄位
                awk -F'\t' '{
                    printf "  Final epoch : %d\n", $1   # 打印最終輪次號
                    printf "  Val LProb   : %s\n", $2   # 打印驗證概率損失
                    printf "  Val LRC     : %s\n", $3   # 打印驗證重連接損失
                    printf "  Val LFunc   : %s\n", $4   # 打印驗證函數損失
                    printf "  Val LTrans  : %s\n", $5   # 打印驗證轉移損失
                    printf "  Val ACC     : %s\n", $6   # 打印驗證精度
                }'
        fi
        echo ""
    } | tee "$out"  # 將輸出同時顯示到屏幕和保存到文件
}

# ── Phase 1: generate Markov labels (idempotent) ────────────────────────────

# 切換到 DeepGate2 根目錄，若失敗則輸出錯誤並退出
cd "$DEEPGATE_ROOT" || { echo "ERROR: Cannot cd to $DEEPGATE_ROOT"; exit 1; }

echo ""
echo "======================================================================"
echo "  PHASE 1: Generate Markov-stimulus labels (flip_prob=${FLIP_PROB})"  # 打印 Phase 1 標題和翻轉概率
echo "  Output : ${LABEL_PATH}"                                             # 打印標籤文件的預期輸出路徑
echo "======================================================================"

# 檢查標籤文件是否已存在，若存在則跳過生成步驟，否則運行 prepare_dataset.py 生成標籤
if [ -f "$LABEL_PATH" ]; then
    # 若標籤文件已存在則打印跳過信息
    echo "  [SKIP] ${LABEL_PATH} already exists."
else
    PREP_LOG="${LOG_BASE}/prepare_dataset_markov.log"           # 定義標籤生成過程的日誌文件路徑
    echo "  [Run] python3 src/prepare_dataset.py --markov ..."  # 打印運行標籤生成腳本的提示信息

    # 運行 prepare_dataset.py 生成 Markov 標籤，並將輸出同時保存到日誌文件
    # 指定實驗 ID 為 "train"
    # 指定原始 AIG 文件所在的資料夾為 "./dataset/rawaig"
    # 啟用 Markov 標籤生成模式
    # 設置 PI 翻轉概率為 ${FLIP_PROB}
    # 指定生成的標籤文件輸出路徑為 ${LABEL_FILE}，將標準輸出和錯誤輸出同時保存到日誌文件
    python3 src/prepare_dataset.py \
        --exp_id train \
        --aig_folder ./dataset/rawaig \
        --markov \
        --flip_prob "${FLIP_PROB}" \
        --label_out "${LABEL_FILE}" 2>&1 | tee "${PREP_LOG}"

    # 檢查標籤文件是否成功生成，若未生成則輸出錯誤並退出
    if [ ! -f "$LABEL_PATH" ]; then
        # 若標籤文件未生成則打印錯誤信息並退出
        echo "  [ERROR] Label generation failed: ${LABEL_PATH} not produced."
        exit 1
    fi
fi

# ── Phase 2: train two configs ─────────────────────────────────────────────

# 切換到 DeepGate2 根目錄，若失敗則輸出錯誤並退出
cd "$DEEPGATE_SRC" || { echo "ERROR: Cannot cd to $DEEPGATE_SRC"; exit 1; }

# 定義實驗配置列表：prob_only（不使用轉移損失）和 with_trans（使用轉移損失）
EXP_LIST=("prob_only" "with_trans")

# 定義轉移損失權重的函數，根據實驗名稱返回對應的權重值
trans_weight() {
    case "$1" in
        prob_only)  echo "0" ;;
        with_trans) echo "2" ;;
        *)          echo "0" ;;
    esac
}

# 定義基礎端口號和端口索引，用於分配不同實驗的通信端口，避免衝突
BASE_PORT=29500
PORT_IDX=0

# 遍歷實驗列表中的每個實驗名稱
for NAME in "${EXP_LIST[@]}"; do
    EXP_ID="PA2_assignment3/${NAME}"            # 構建實驗 ID
    TRAIN_LOG="${LOG_BASE}/${NAME}_train.log"   # 定義訓練日誌文件路徑
    TW=$(trans_weight "$NAME")                  # 根據實驗名稱獲取轉移損失權重
    RDZV_PORT=$((BASE_PORT + PORT_IDX))         # 計算當前實驗使用的通信端口號(RDZV= rendezvous)，確保每個實驗使用不同的端口以避免衝突
    PORT_IDX=$((PORT_IDX + 1))                  # 更新端口索引以供下一個實驗使用

    echo ""
    echo "======================================================================"
    echo "  PHASE 2 experiment = ${NAME}  (Trans_weight=${TW})  |  $(date)" # 打印 Phase 2 的實驗名稱、轉移損失權重和當前時間
    echo "======================================================================"

    LATEST=$(latest_log "$EXP_ID")  # 查找該實驗的最新日誌文件
    # 檢查是否找到日誌文件
    if [ -n "$LATEST" ]; then
        MAX_EP=$(max_epoch_in_log "$LATEST")    # 從日誌文件中提取最大訓練輪次
        # 檢查訓練是否已完成（最大輪次 >= 設定輪次）
        if [ -n "$MAX_EP" ] && [ "$MAX_EP" -ge "$EPOCHS" ] 2>/dev/null; then
            # 打印跳過信息並提取指標
            echo "  [SKIP] ${NAME} already complete (epoch ${MAX_EP} >= ${EPOCHS}). Extracting metrics."
            summarize "$NAME"   # 生成實驗摘要
            continue            # 跳過本次迴圈，繼續下一個實驗
        fi
    fi

    # 打印訓練開始信息
    echo "  [Train] ${NAME}: ${EPOCHS} epochs, dim=${DIM}, Trans_weight=${TW}, port=${RDZV_PORT}"

    # 使用 PyTorch 分散式執行命令，運行主程序進行概率預測任務。【指定 master_port 避免多實驗間的通信衝突。】
    # 指定實驗 ID
    # 指定訓練數據目錄
    # 【指定使用 Markov 標籤文件】
    # 指定隱層維度
    # 指定正則化損失函數為 L1，分類損失函數為二元交叉熵
    # 指定網絡架構為 MLP+GNN
    # 設置概率損失權重為 3、重連接損失權重為 1、函數損失權重為 2
    # 【設置轉移損失權重為 ${TW}（根據實驗配置決定是否啟用轉移損失）】
    # 設置迭代輪數為 1
    # 指定 GPU 設備、批量大小為 32、數據加載工作進程數為 4
    # 指定訓練輪次
    # 禁用重連接訓練，將標準輸出和錯誤輸出同時保存到日誌文件
    python3 -m torch.distributed.run --nproc_per_node=$NUM_PROC --master_port="${RDZV_PORT}" ./main.py prob \
        --exp_id    "${EXP_ID}"                              \
        --data_dir  ../data/train                            \
        --label_file "${LABEL_FILE}"                         \
        --dim_hidden "${DIM}"                                \
        --reg_loss l1 --cls_loss bce                         \
        --arch mlpgnn                                        \
        --Prob_weight 3 --RC_weight 1 --Func_weight 2        \
        --Trans_weight "${TW}"                               \
        --num_rounds 1                                       \
        --gpus "${GPUS}" --batch_size 32 --num_workers 4     \
        --num_epochs "${EPOCHS}"                             \
        --no_rc 2>&1 | tee "${TRAIN_LOG}"

    echo "  [Train] Complete for ${NAME}"   # 打印訓練完成信息
    summarize "$NAME"   # 生成該實驗的摘要
done

# ── Phase 3: evaluate (per-gate-type + analytic baseline) ──────────────────
echo ""
echo "======================================================================"
echo "  PHASE 3: Evaluate transition-prob predictions on the validation split"
echo "======================================================================"

# 遍歷實驗列表中的每個實驗名稱
for NAME in "${EXP_LIST[@]}"; do
    EXP_ID="PA2_assignment3/${NAME}"            # 構建實驗 ID
    EVAL_LOG="${LOG_BASE}/${NAME}_eval.log"     # 定義評估日誌文件路徑
    CKPT="${EXP_BASE}/${EXP_ID}/model_last.pth" # 定義模型檢查點文件路徑

     # 檢查模型檢查點文件是否存在，若不存在則跳過評估步驟
    if [ ! -f "$CKPT" ]; then
        # 若檢查點文件未找到則打印跳過信息
        echo "  [SKIP] ${NAME}: checkpoint ${CKPT} not found"
        continue
    fi

    # 若 eval log 已存在、比 checkpoint 新、且包含完成標記 "Gap (analytic"，
    # 則直接複用既有結果（與 Phase 1/2 的 idempotent 行為一致）
    if [ -f "$EVAL_LOG" ] && [ "$EVAL_LOG" -nt "$CKPT" ] && grep -q "Gap (analytic" "$EVAL_LOG"; then
        echo ""
        echo "  [SKIP] ${NAME}: eval log already exists and is newer than checkpoint."
        echo "         (${EVAL_LOG})"
        echo "  ---- existing log ----"
        cat "$EVAL_LOG"
        echo "  ---- end of log ----"
        continue
    fi

    echo ""

    # 打印評估開始信息
    echo "  [Eval] ${NAME}  (ckpt: ${CKPT})"

    # 使用 test_trans.py 腳本評估模型的轉移概率預測性能，並將輸出同時保存到日誌文件
    # 指定實驗 ID
    # 指定訓練數據目錄
    # 【指定標籤文件路徑】
    # 指定隱層維度
    # 指定網絡架構為 MLP+GNN
    # 設置迭代輪數為 1
    # 指定 GPU 設備、批量大小為 1、數據加載工作進程數為 0
    # 禁用重連接訓練
    # 【指定加載模型檢查點文件】，將標準輸出和錯誤輸出同時保存到日誌文件
    python3 ./test_trans.py prob \
        --exp_id "${EXP_ID}"                                \
        --data_dir ../data/train                            \
        --label_file "${LABEL_FILE}"                        \
        --dim_hidden "${DIM}"                               \
        --arch mlpgnn                                       \
        --num_rounds 1                                      \
        --gpus "${GPUS}" --batch_size 1 --num_workers 0     \
        --no_rc                                             \
        --load_model model_last.pth 2>&1 | tee "${EVAL_LOG}"
done

# ── Phase 4: cross-experiment comparison ───────────────────────────────────
echo ""
echo "======================================================================"
echo "  CROSS-EXPERIMENT COMPARISON"    # 打印跨實驗比較標題
echo "  $(date)"                        # 打印當前時間
echo "======================================================================"

COMPARE="${LOG_BASE}/comparison.txt"    # 定義對比結果輸出文件路徑
{
    echo "experiment   | Val LProb (↓) | Val LRC (↓) | Val LFunc (↓) | Val LTrans (↓) | Val ACC (↑)"
    echo "-------------|---------------|-------------|---------------|----------------|------------"
    
    # 遍歷實驗列表中的每個實驗名稱
    for NAME in "${EXP_LIST[@]}"; do
        log=$(latest_log "PA2_assignment3/${NAME}") # 獲取該實驗的最新日誌文件
        # 檢查日誌文件是否存在
        if [ -n "$log" ]; then
            last_line=$(grep "epoch:" "$log" | tail -1) # 提取日誌文件中最後一個包含 "epoch:" 的行
            
            # 檢查是否成功提取最後一行，若成功則解析並格式化輸出指標數據，否則輸出 N/A
            if [ -n "$last_line" ]; then
                echo "$last_line" | parse_epoch_line | \
                    awk -F'\t' -v n="$NAME" '{
                        printf "%-12s | %13s | %11s | %13s | %14s | %s\n", n, $2, $3, $4, $5, $6
                    }'
            else
                # 若日誌為空則打印 N/A
                printf "%-12s | %13s | %11s | %13s | %14s | %s\n" "$NAME" "N/A" "N/A" "N/A" "N/A" "N/A"
            fi
        else
            # 若日誌文件不存在則打印 N/A
            printf "%-12s | %13s | %11s | %13s | %14s | %s\n" "$NAME" "N/A" "N/A" "N/A" "N/A" "N/A"
        fi
    done
    echo ""
    echo "Eval summaries (model L1 vs analytic 2p(1-p) baseline) are in:"   # 打印評估摘要所在位置
    # 列出每個實驗的評估日誌文件路徑
    for NAME in "${EXP_LIST[@]}"; do
        echo "  ${LOG_BASE}/${NAME}_eval.log"
    done
    echo ""
    echo "Interpretation:"  # 打印結果解讀提示
    # 說明：prob_only 是不使用轉移損失的基線，with_trans 是啟用轉移損失的實驗配置。
    echo "  - with_trans should improve LTrans materially over prob_only (which has Trans_weight=0)."
    # 說明：在 test_trans.py 的輸出中，正的 "Gap (analytic - model)" 表示 GNN 學習到了超越封閉形式 2p(1-p) 基線的結構。
    echo "  - In test_trans.py output, a positive 'Gap (analytic - model)' means the GNN learned"
    echo "    structure beyond the closed-form 2p(1-p) baseline."
    # 說明：PI 節點的轉移概率約等於翻轉概率（FLIP_PROB）是由標籤生成過程決定的，因此預期在 PI 節點上 L1 會較小。
    echo "  - PI nodes have trans_prob ≈ flip_prob = ${FLIP_PROB} by construction; expect small L1 there."
} | tee "$COMPARE"  # 將輸出同時顯示到屏幕和保存到文件

echo ""
echo "Results saved to: ${LOG_BASE}/"   # 打印結果保存的目錄
echo "  prepare_dataset_markov.log   — label generation log"                        # 說明：標籤生成過程的日誌文件
echo "  {name}_train.log             — full training stdout per experiment"         # 說明：每個實驗的訓練日誌文件
echo "  {name}_summary.txt           — epoch-by-epoch metrics"                      # 說明：每個實驗的摘要文件，包含每輪次的指標數據
echo "  {name}_eval.log              — test_trans.py output (L1 + per-gate-type)"   # 說明：每個實驗的評估日誌文件
echo "  comparison.txt               — cross-experiment comparison table"           # 說明：跨實驗比較的總結表格文件
