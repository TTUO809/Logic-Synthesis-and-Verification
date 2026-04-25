#!/bin/bash
# 使用 bash shell 解釋器
# ==============================================================================
# Assignment 1: Hidden Layer Dimension Sweep (32, 64, 128)
# Metrics logged per dim:
#   - Signal Probability: Val LProb (L1 loss, lower = better)
#   - TT Distance Accuracy: ACC (ranking acc, higher = better)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # 腳本所在目錄（自動偵測）

NUM_PROC=1  # 設置並行進程數為 1
GPUS=0      # 指定使用的 GPU 設備號為 0
DEEPGATE_SRC="${HOME}/DeepGate2/src"            # 設置 DeepGate2 源代碼的路徑
EXP_BASE="${HOME}/DeepGate2/exp/prob"           # 設置實驗基礎目錄路徑
LOG_BASE="${SCRIPT_DIR}/results/assignment1"    # 設置日誌保存的基礎目錄
EPOCHS=5    # 設置訓練的輪數為 5

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
        match($0, /ACC ([0-9.]+)/, a); acc = (a[1] != "") ? sprintf("%.6f", a[1]+0) : "N/A" # 從輸入行中提取精度（ ACC ）值
        printf "%d\t%s\t%s\n", epoch, val_lp, acc   # 以制表符分隔格式打印提取的指標
    }'
}

# 定義生成實驗摘要的函數
summarize() {
    local dim=$1                                    # 獲取隱層維度參數
    local exp_id="PA2_assignment1/dim${dim}"        # 構建實驗 ID
    local out="${LOG_BASE}/dim${dim}_summary.txt"   # 定義輸出摘要文件的路徑
    local log                                       # 聲明日誌文件變數
    log=$(latest_log "$exp_id")                     # 找到最新的 log.txt 路徑

    {
        echo "======================================================================"
        echo "  SUMMARY  dim_hidden = ${dim}"       # 打印隱層維度摘要標題
        echo "  Source log : ${log:-<not found>}"   # 打印源日誌文件路徑，若未找到則顯示 "<not found>"
        echo "  Generated  : $(date)"               # 打印生成時間
        echo "======================================================================"
        echo ""

        # 檢查日誌文件是否存在
        if [ -z "$log" ]; then
            echo "  [WARNING] No log file found."
            return
        fi

        echo "  Metrics (validation split):"        # 打印驗證指標標題
        # 打印列名和指標說明（↓ 表示越低越好，↑ 表示越高越好）
        echo "  Epoch | Val LProb (Signal Prob Loss ↓) | Val ACC (TT Dist ACC ↑)"
        echo "  ------|--------------------------------|------------------------"

        # 從日誌文件中提取包含 "epoch:" 的行並解析
        grep "epoch:" "$log" | parse_epoch_line | \
            # 格式化打印每一行的指標數據
            awk -F'\t' '{ printf "  %5d | %30s | %s\n", $1, $2, $3}'

        echo ""

        echo "  === Best / Final Results ==="   # 打印最佳和最終結果標題
        local last_line # 聲明最後一行變數
        last_line=$(grep "epoch:" "$log" | tail -1) # 提取日誌文件中最後一個包含 "epoch:" 的行
        best_acc=$(grep "epoch:" "$log" | grep -oP "ACC \K[0-9.]+" | sort -n | tail -1)     # 從日誌中提取所有精度值並找到最高的

        # 檢查是否成功獲取最後一行
        if [ -n "$last_line" ]; then
            # 解析最後一行的數據
            echo "$last_line" | parse_epoch_line | \
                # 按制表符分隔符分割欄位
                awk -F'\t' -v best_acc="$best_acc" '{
                    printf "  Final epoch                       : %d\n", $1         # 打印最終輪次號
                    printf "  Val LProb (Signal Prob Val Loss)  : %s\n", $2         # 打印驗證概率損失
                    printf "  Val ACC (TT Dist ACC) - final     : %s\n", $3         # 打印驗證精度
                    printf "  Val ACC (TT Dist ACC) - best      : %s\n", best_acc   # 打印驗證精度
                }'
        fi
        echo ""
    } | tee "$out"  # 將輸出同時顯示到屏幕和保存到文件
}

# ── 實驗定義部分 ──────────────────────────────────────────────────
DIM_LIST=(32 64 128)    # 定義隱層維度列表：32、64、128

# ── Main loop ────────────────────────────────────────────────────────────────

# 切換到 DeepGate2 源代碼目錄，若失敗則輸出錯誤並退出
cd "$DEEPGATE_SRC" || { echo "ERROR: Cannot cd to $DEEPGATE_SRC"; exit 1; } 

# 遍歷隱層維度列表中的每個維度
for DIM in "${DIM_LIST[@]}"; do
    EXP_ID="PA2_assignment1/dim${DIM}"          # 構建實驗 ID
    TRAIN_LOG="${LOG_BASE}/dim${DIM}_train.log" # 定義訓練日誌文件路徑
    CKPT="${EXP_BASE}/${EXP_ID}/model_last.pth" # 定義模型檢查點文件路徑

    echo ""
    echo "======================================================================"
    echo "  dim_hidden = ${DIM}  |  $(date)"    # 打印隱層維度和當前時間
    echo "======================================================================"

    LATEST=$(latest_log "$EXP_ID")  # 查找該實驗的最新日誌文件
    # 檢查是否找到日誌文件
    if [ -n "$LATEST" ]; then
        MAX_EP=$(max_epoch_in_log "$LATEST")    # 從日誌文件中提取最大訓練輪次
        # 檢查訓練是否已完成（最大輪次 >= 設定輪次）
        if [ -n "$MAX_EP" ] && [ "$MAX_EP" -ge "$EPOCHS" ] 2>/dev/null; then
            # 打印跳過信息並提取指標
            echo "  [SKIP] dim${DIM} already complete (epoch ${MAX_EP} ≥ ${EPOCHS}). Extracting metrics."
            summarize "$DIM"    # 生成實驗摘要
            continue            # 跳過本次迴圈，繼續下一個維度
        fi
    fi

    # 打印訓練開始信息
    echo "  [Train]  Training for dim${DIM} (${EPOCHS} epochs)..."

    # 使用 PyTorch 分散式執行命令，運行主程序進行概率預測任務
    # 指定實驗 ID
    # 指定訓練數據目錄
    # 指定隱層維度
    # 指定正則化損失函數為 L1，分類損失函數為二元交叉熵
    # 指定網絡架構為 MLP+GNN
    # 設置概率損失權重為 3、重連接損失權重為 1、函數損失權重為 2
    # 設置迭代輪數為 1
    # 指定 GPU 設備、批量大小為 32、數據加載工作進程數為 4
    # 指定訓練輪次
    # 禁用重連接訓練，將標準輸出和錯誤輸出同時保存到日誌文件
    python3 -m torch.distributed.run --nproc_per_node=$NUM_PROC ./main.py prob \
        --exp_id    "${EXP_ID}"                          \
        --data_dir  ../data/train                        \
        --dim_hidden "${DIM}"                            \
        --reg_loss l1 --cls_loss bce                     \
        --arch mlpgnn                                    \
        --Prob_weight 3 --RC_weight 1 --Func_weight 2    \
        --num_rounds 1                                   \
        --gpus "${GPUS}" --batch_size 32 --num_workers 4 \
        --num_epochs "${EPOCHS}"                         \
        --no_rc 2>&1 | tee "${TRAIN_LOG}"
    echo "  [Train] Complete for dim${DIM}" # 打印訓練完成信息

    summarize "$DIM"    # 生成該維度的摘要
done

# ── Cross-dim comparison ────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  CROSS-DIM COMPARISON"           # 打印跨維度比較標題
echo "  $(date)"                        # 打印當前時間
echo "======================================================================"

COMPARE="${LOG_BASE}/comparison.txt"    # 定義對比結果輸出文件路徑
{
    echo "dim_hidden | Val LProb (Signal Prob ↓) | TT Dist Val ACC final (↑) | TT Dist Val ACC best (↑)"
    echo "-----------|---------------------------|---------------------------|-------------------------"
    
    # 遍歷隱層維度列表中的每個維度
    for DIM in "${DIM_LIST[@]}"; do
        log=$(latest_log "PA2_assignment1/dim${DIM}") # 獲取該維度的最新日誌文件
        # 檢查日誌文件是否存在
        if [ -n "$log" ]; then
            last_line=$(grep "epoch:" "$log" | tail -1) # 提取日誌文件中最後一個包含 "epoch:" 的行
            best_acc=$(grep "epoch:" "$log" | grep -oP "ACC \K[0-9.]+" | sort -n | tail -1)     # 從日誌中提取所有精度值並找到最高的

            # 檢查是否成功提取
            if [ -n "$last_line" ]; then
                # 解析 epoch 行
                echo "$last_line" | parse_epoch_line | \
                    # 格式化打印實驗名稱和各項指標
                    awk -F'\t' -v dim="$DIM" -v best_acc="$best_acc" '{
                        printf "%10d | %25.6f | %25s | %s\n", dim, $2, $3, best_acc
                    }'
            else
                # 若日誌為空則打印 N/A
                printf "%10d | %25s | %25s | %s\n" "$DIM" "N/A" "N/A" "N/A"
            fi
        else
            # 若日誌文件不存在則打印 N/A
            printf "%10d | %25s | %25s | %s\n" "$DIM" "N/A" "N/A" "N/A"
        fi
    done

} | tee "$COMPARE"  # 將輸出同時顯示到屏幕和保存到文件

echo ""
echo "Results saved to: ${LOG_BASE}/"   # 打印結果保存位置
echo "  dim{N}_train.log    — full training stdout per dim"     # 說明：每個維度的完整訓練標準輸出日誌
echo "  dim{N}_summary.txt  — epoch-by-epoch metrics per dim"   # 說明：每個維度的逐個 epoch 指標摘要
echo "  comparison.txt      — cross-dim comparison table"       # 說明：跨維度對比表格