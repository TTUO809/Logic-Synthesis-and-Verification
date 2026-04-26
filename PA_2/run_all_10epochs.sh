#!/bin/bash
# ==============================================================================
# Run all three assignments sequentially with 10 epochs.
#
# Outputs:
#   ./${EPOCHS_OVERRIDE}_epochs/results/assignment{1,2,3}/   — per-assignment logs & summaries
#   ./${EPOCHS_OVERRIDE}_epochs/assignment{1,2,3}_master.log — full stdout per assignment
#   ./${EPOCHS_OVERRIDE}_epochs/run_all.log                  — combined master log
#
# DeepGate2 EXP dirs (~/DeepGate2/exp/prob/PA2_assignmentN/...) will be created
# ==============================================================================

set -u
# 不使用 set -e —— 即使其中一個作業失敗，仍想跑完其餘的

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EPOCHS_OVERRIDE=10
OUT_ROOT="${SCRIPT_DIR}/${EPOCHS_OVERRIDE}_epochs"
RESULTS_ROOT="${OUT_ROOT}/results"
COMBINED_LOG="${OUT_ROOT}/run_all.log"

mkdir -p "${RESULTS_ROOT}"

run_one() {
    local idx=$1                                          # 1 / 2 / 3
    local script="${SCRIPT_DIR}/run_assignment${idx}.sh"
    local master_log="${OUT_ROOT}/assignment${idx}_master.log"
    local log_base="${RESULTS_ROOT}/assignment${idx}"

    echo ""
    echo "######################################################################"
    echo "#  Assignment ${idx}  |  EPOCHS=${EPOCHS_OVERRIDE}  |  $(date)"
    echo "#  script   : ${script}"
    echo "#  log_base : ${log_base}"
    echo "#  master   : ${master_log}"
    echo "######################################################################"

    # 確保子 script 可執行（如果不是，嘗試 chmod +x，但不強求）
    if [ ! -x "$script" ]; then
        chmod +x "$script" 2>/dev/null || true
    fi

    # 把 EPOCHS 和 LOG_BASE 透過環境變數傳給子 script
    local start_ts=$(date +%s)
    EPOCHS="${EPOCHS_OVERRIDE}" LOG_BASE="${log_base}" bash "${script}" 2>&1 | tee "${master_log}"
    local rc=${PIPESTATUS[0]}   # 捕捉 bash script 的 return code，而不是 tee 的 return code
    local end_ts=$(date +%s)
    local dur=$((end_ts - start_ts))

    echo ""
    echo "[run_all] Assignment ${idx} exited with rc=${rc}, duration=${dur}s"
    return $rc
}

{
    echo "======================================================================"
    echo "  RUN_ALL  |  EPOCHS=${EPOCHS_OVERRIDE}  |  start: $(date)"
    echo "  Output root  : ${OUT_ROOT}"
    echo "======================================================================"

    overall_start=$(date +%s)
    declare -A RC   # 儲存每個 assignment 的 return code

    for IDX in 1 2 3; do
        run_one "$IDX"  # 執行 assignment N 的腳本
        RC[$IDX]=$?     # 儲存 return code
    done

    overall_end=$(date +%s)
    overall_dur=$((overall_end - overall_start))

    echo ""
    echo "======================================================================"
    echo "  RUN_ALL SUMMARY  |  end: $(date)  |  total: ${overall_dur}s"
    echo "======================================================================"
    # 列出每個 assignment 的 return code
    for IDX in 1 2 3; do
        echo "  Assignment ${IDX}: rc=${RC[$IDX]}"
    done
    echo ""
    # 提示使用者查看 per-assignment 的 master log 和 comparison.txt 等摘要文件
    echo "Per-assignment outputs:"
    for IDX in 1 2 3; do
        echo "  - ${RESULTS_ROOT}/assignment${IDX}/  (comparison.txt, *_summary.txt, *_train.log)"
        echo "  - ${OUT_ROOT}/assignment${IDX}_master.log"
    done
} 2>&1 | tee "${COMBINED_LOG}"
