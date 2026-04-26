#!/bin/bash
# 使用 bash shell 解釋器
# ==============================================================================
# Assignment 2: Ablation Study (DeepGate2)
#   - baseline      : DeepGate2 with orthogonal PI init + pairwise TT loss
#   - no_tt_loss    : --no_func           (ablate pairwise TT loss)
#   - homo_pi_init  : --homo_pi_init      (ablate orthogonal PI initialization)
#
# Note: --no_rc is applied to ALL experiments because the training data
#       (labels.npz) does not contain rc_pair_index / is_rc fields.
#       Val LRC is computed on a dummy pair and is not meaningful here.
#
# Metrics logged per experiment:
#   - Val LProb (signal probability L1 loss, lower = better)
#   - Val LRC   (reconvergence BCE loss — dummy data, not comparable)
#   - Val LFunc (functional similarity L1 loss, lower = better; 0 if no_tt_loss)
#   - Val ACC   (TT distance ranking accuracy, higher = better)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # 腳本所在目錄（自動偵測）

NUM_PROC=1  # 設置並行進程數為 1
GPUS=0      # 指定使用的 GPU 設備號為 0
DEEPGATE_ROOT="${HOME}/DeepGate2"               # 設置 DeepGate2 的根目錄
DEEPGATE_SRC="${DEEPGATE_ROOT}/src"             # 設置 DeepGate2 源代碼的路徑
EXP_BASE="${DEEPGATE_ROOT}/exp/prob"            # 設置實驗基礎目錄路徑
LOG_BASE="${LOG_BASE:-${SCRIPT_DIR}/results/assignment2}"    # 設置日誌保存的基礎目錄（可由環境變數覆寫）
EPOCHS="${EPOCHS:-5}"    # 設置訓練的輪數，預設 5（可由環境變數覆寫）
DIM=64      # 設置隱層維度為 64

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
        match($0, /ACC ([0-9.]+)/, a); acc = (a[1] != "") ? sprintf("%.6f", a[1]+0) : "N/A" # 從輸入行中提取精度（ ACC ）值
        printf "%d\t%s\t%s\t%s\t%s\n", epoch, val_lp, val_lr, val_lf, acc       # 以制表符分隔格式打印提取的指標
    }'
}

# 定義生成實驗摘要的函數
summarize() {
    local name=$1                                   # 獲取實驗名稱參數
    local exp_id="PA2_assignment2/${name}"          # 構建實驗 ID
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
        echo "  Epoch | Val LProb (↓) | Val LRC (↓) | Val LFunc (↓) | Val ACC (↑)"
        echo "  ------|---------------|-------------|---------------|------------"
        
        # 從日誌文件中提取包含 "epoch:" 的行並解析
        grep "epoch:" "$log" | parse_epoch_line | \
            # 格式化打印每一行的指標數據
            awk -F'\t' '{ printf "  %5d | %13s | %11s | %13s | %s\n", $1, $2, $3, $4, $5 }' 

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
                    printf "  Val ACC     : %s\n", $5   # 打印驗證精度
                }'
        fi
        echo ""
    } | tee "$out"  # 將輸出同時顯示到屏幕和保存到文件
}

# ── 實驗定義部分 ──────────────────────────────────────────────────
# 定義實驗名稱列表：baseline（基線）、no_tt_loss（不使用 TT loss）、homo_pi_init（同質 PI 初始化）
EXP_LIST=("baseline" "no_tt_loss" "homo_pi_init")

# 定義根據實驗名稱返回額外命令行參數的函數
extra_flags() {
    # 檢查傳入的實驗名稱
    case "$1" in
        baseline)     echo "" ;;                # baseline 實驗不添加額外參數
        no_tt_loss)   echo "--no_func" ;;       # no_tt_loss 實驗添加 --no_func 參數以禁用函數相似度損失
        homo_pi_init) echo "--homo_pi_init" ;;  # homo_pi_init 實驗添加 --homo_pi_init 參數以使用同質 PI 初始化
        *)            echo "" ;;                # 其他情況返回空字符串
    esac
}

# ── Main loop ───────────────────────────────────────────────────────────────

# 切換到 DeepGate2 源代碼目錄，若失敗則輸出錯誤並退出
cd "$DEEPGATE_SRC" || { echo "ERROR: Cannot cd to $DEEPGATE_SRC"; exit 1; } 

# 遍歷實驗列表中的每個實驗名稱
for NAME in "${EXP_LIST[@]}"; do
    EXP_ID="PA2_assignment2/${NAME}"            # 構建實驗 ID
    TRAIN_LOG="${LOG_BASE}/${NAME}_train.log"   # 定義訓練日誌文件路徑
    EXTRA=$(extra_flags "$NAME")                # 獲取該實驗對應的額外命令行參數

    echo ""
    echo "======================================================================"
    echo "  experiment = ${NAME}  (extra: ${EXTRA:-<none>})  |  $(date)"    # 打印實驗名稱、額外參數和當前時間
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
    echo "  [Train] Training ${NAME} for ${EPOCHS} epochs (dim=${DIM})..."

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
    # 禁用重連接訓練，【添加額外參數】，將標準輸出和錯誤輸出同時保存到日誌文件
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
        --no_rc ${EXTRA} 2>&1 | tee "${TRAIN_LOG}"

    echo "  [Train] Complete for ${NAME}"   # 打印訓練完成信息
    summarize "$NAME"   # 生成該實驗的摘要
done

# ── Cross-experiment comparison ─────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  CROSS-EXPERIMENT COMPARISON"    # 打印跨實驗比較標題
echo "  $(date)"                        # 打印當前時間
echo "======================================================================"

COMPARE="${LOG_BASE}/comparison.txt"    # 定義對比結果輸出文件路徑
{
    echo "experiment    | Val LProb (↓) | Val LRC (↓) | Val LFunc (↓) | Val ACC (↑)"
    echo "--------------|---------------|-------------|---------------|------------"
    
    # 遍歷實驗列表中的每個實驗名稱
    for NAME in "${EXP_LIST[@]}"; do
        log=$(latest_log "PA2_assignment2/${NAME}") # 獲取該實驗的最新日誌文件
        # 檢查日誌文件是否存在
        if [ -n "$log" ]; then
            last_line=$(grep "epoch:" "$log" | tail -1) # 提取日誌文件中最後一個包含 "epoch:" 的行
            
            # 檢查是否成功提取最後一行，若成功則解析並格式化輸出指標數據，否則輸出 N/A
            if [ -n "$last_line" ]; then
                echo "$last_line" | parse_epoch_line | \
                    awk -F'\t' -v n="$NAME" '{
                        printf "%-13s | %13s | %11s | %13s | %s\n", n, $2, $3, $4, $5 
                    }'
            else
                # 若日誌為空則打印 N/A
                printf "%-13s | %13s | %11s | %13s | %s\n" "$NAME" "N/A" "N/A" "N/A" "N/A"
            fi
        else
            # 若日誌文件不存在則打印 N/A
            printf "%-13s | %13s | %11s | %13s | %s\n" "$NAME" "N/A" "N/A" "N/A" "N/A"
        fi
    done
    echo ""
    echo "Interpretation:"  # 打印結果解讀提示
    # 說明：no_tt_loss 相比 baseline，精度應該下降（因為 TT 損失被移除）
    echo "  - no_tt_loss   vs baseline : ACC and LFunc should WORSEN (TT loss removed)"
    # 說明：homo_pi_init 相比 baseline，ACC 和 LProb 應該下降（PI 初始化均質化降低區分度）
    echo "  - homo_pi_init vs baseline : ACC and LProb should WORSEN (PI init degraded)"
    echo "  * LRC is computed on dummy RC pairs (--no_rc); values are not meaningful."
} | tee "$COMPARE"  # 將輸出同時顯示到屏幕和保存到文件

echo ""
echo "Results saved to: ${LOG_BASE}/"   # 打印結果保存位置
echo "  {name}_train.log    — full training stdout per experiment"      # 說明：每個實驗的完整訓練標準輸出日誌
echo "  {name}_summary.txt  — epoch-by-epoch metrics per experiment"    # 說明：每個實驗的逐個 epoch 指標摘要
echo "  comparison.txt      — cross-experiment comparison table"        # 說明：跨實驗比較的總結表格文件