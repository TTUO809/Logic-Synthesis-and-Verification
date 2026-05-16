#!/usr/bin/env bash
# Run MapTune baseline for PA3 Assignment 3
# Usage: bash run_maptune_baseline.sh
# Requires: conda activate LSV_PA3 && ABC in PATH

MAPTUNE_DIR="$HOME/MapTune"
DESIGNS=("benchmarks/s13207.bench" "benchmarks/c2670.bench" "benchmarks/b20_1.bench")
LIBS=("nan45.genlib" "sky130.genlib")

# Budget = floor(total_cells * ratio)
# nan45: 94 cells  → 30%=28, 50%=47, 70%=66
# sky130: 343 cells → 30%=103, 50%=171, 70%=240
declare -A BUDGETS
BUDGETS["nan45.genlib"]="28 47 66"
BUDGETS["sky130.genlib"]="103 171 240"

cd "$MAPTUNE_DIR"

for lib in "${LIBS[@]}"; do
    for design in "${DESIGNS[@]}"; do
        design_name=$(basename "$design" .bench)
        for budget in ${BUDGETS[$lib]}; do
            echo "=== MapTune-UCB | lib=$lib | design=$design_name | budget=$budget ==="
            python batched_MAB_UCB.py "$budget" "$design" "$lib" \
                2>&1 | tee "$HOME/LSV/PA_3/log_${lib%.*}_${design_name}_n${budget}.txt"
        done
    done
done
