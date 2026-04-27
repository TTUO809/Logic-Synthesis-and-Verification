"""
gen_charts.py  —  parse experiment log files and generate all charts.

Usage:
    python3 test/gen_charts.py [results_dir] [--out OUT_DIR]

    results_dir : path to the results/ directory
                  (default: <this script>/../log/10_epochs/results)
    --out       : where to write PNG files
                  (default: results_dir/../charts)

Examples:
    python3 test/gen_charts.py
    python3 test/gen_charts.py log/5_epochs_assignment3/results
    python3 test/gen_charts.py log/10_epochs/results --out my_charts/
"""

import re
import sys
import os
import argparse
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # 使用無顯示介面的後端，適合在伺服器環境產生圖片檔
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ── font ──────────────────────────────────────────────────────────────────────
_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_FONT_PATH):
    try:
        fm.fontManager.addfont(_FONT_PATH)  # 添加自定義字體到 matplotlib 的字體管理器
    except AttributeError:
        pass
    _fp = fm.FontProperties(fname=_FONT_PATH)   # 創建一個 FontProperties 對象，指定字體文件路徑為 _FONT_PATH
    plt.rcParams["font.family"] = _fp.get_name()    # 設置 matplotlib 的默認字體為剛剛創建的 FontProperties 對象的字體名稱
else:
    _fp = fm.FontProperties()   # 使用默認字體，可能不支持中文
    print("Warning: font file not found, using default font (may not support Chinese characters).", file=sys.stderr)

# ── log parsers ───────────────────────────────────────────────────────────────

def _read(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f: 
        return [ln.rstrip("\n") for ln in f] # 讀取文件並去除行尾的換行符，返回行列表


def parse_pipe_table(lines: List[str]) -> List[dict]:
    """
    解析以 '|' 分隔的表格（用於 *_summary.txt 與 comparison.txt ）。
    找到第一個含 '|' 的標頭列，跳過分隔列，將每列以 dict 形式回傳。
    """
    # 找出第一個同時包含 '|' 與英文字母的列當作標頭列
    header_idx = next(
        (i for i, ln in enumerate(lines) if "|" in ln and re.search(r"[A-Za-z]", ln)),
        None,
    )
    if header_idx is None:
        return []

    # 將標頭列以 '|' 切開並去除空白，得到欄位名稱列表
    headers = [c.strip() for c in lines[header_idx].split("|") if c.strip()]
    rows = []
    for ln in lines[header_idx + 1 :]:
        if "|" not in ln:
            continue
        cells = [c.strip() for c in ln.split("|") if c.strip()]
        # 跳過僅由破折號或冒號組成的分隔列
        if all(re.fullmatch(r"[-: ]+", c) for c in cells):
            continue
        if len(cells) != len(headers):
            continue  # 欄位數不符合標頭時略過該列
        rows.append(dict(zip(headers, cells)))  # 將欄位與數值配對成 dict
    return rows


def col_floats(rows: List[dict], key: str) -> List[float]:
    """
    依據 key 子字串（不區分大小寫）匹配欄位名稱，將該欄資料轉成 float 列表。
    """
    if not rows:
        return []
    # 在第一列的 keys 中尋找包含 key 子字串的欄位名稱
    matched = next((k for k in rows[0] if key.lower() in k.lower()), None)
    if matched is None:
        return []
    result = []
    for r in rows:
        try:
            result.append(float(r[matched]))
        except ValueError:
            result.append(float("nan"))  # 無法轉成浮點數時填入 NaN
    return result


def parse_summary(path: str) -> List[dict]:
    """
    解析 *_summary.txt 檔案，並以 dict 列表回傳每個 epoch 的紀錄。
    """
    return parse_pipe_table(_read(path))


def parse_comparison(path: str) -> List[dict]:
    """
    解析 comparison.txt 檔案，略過表格後方的說明文字列。
    """
    rows = parse_pipe_table(_read(path))
    # 過濾掉第一欄以 '-' 開頭的列（通常是註解或說明）
    return [r for r in rows if not r[list(r.keys())[0]].startswith("-")]


def parse_eval_log(path: str) -> dict:
    """
    解析 *_eval.log 檔案，擷取 L1 誤差、誤差差距，以及各 gate type 的明細。
    Returns {
        'l1_prob': float,
        'l1_analytic': float,
        'l1_model': float,
        'gap': float,
        'gate_rows': List[dict],   # [{type, n_nodes, l1_trans}, ...]
    }
    """
    lines = _read(path)
    result: dict = {}

    # 逐行掃描，使用正則表達式擷取四個關鍵指標
    for ln in lines:
        m = re.search(r"L1\(prob_pred.*?\)\s*=\s*([\d.]+)", ln)
        if m:
            result["l1_prob"] = float(m.group(1))  # 預測機率的 L1 誤差
        m = re.search(r"L1\(2p\(1-p\).*?\)\s*=\s*([\d.]+)", ln)
        if m:
            result["l1_analytic"] = float(m.group(1))  # 解析公式 2p(1-p) 的 L1 誤差
        m = re.search(r"L1\(trans_pred.*?\)\s*=\s*([\d.]+)", ln)
        if m:
            result["l1_model"] = float(m.group(1))  # 模型預測 transition 的 L1 誤差
        m = re.search(r"Gap \(analytic - model\)\s*=\s*([+-]?[\d.]+)", ln)
        if m:
            result["gap"] = float(m.group(1))  # 解析值與模型值的差距

    # 找到 "Per-gate-type" 區段並解析其後的表格
    gate_start = next(
        (i for i, ln in enumerate(lines) if "per-gate-type" in ln.lower()), None
    )
    if gate_start is not None:
        result["gate_rows"] = parse_pipe_table(lines[gate_start:])

    return result


# ── chart helpers ─────────────────────────────────────────────────────────────

def _save(fig, path: str):
    # 統一儲存圖表的輔助函式：自動排版、寫檔、關閉並印出相對路徑
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)  # 釋放記憶體，避免大量圖表時佔用過高
    print("  saved:", os.path.relpath(path))


def line_chart(fpath, title, ylabel, series: Dict[str, List[float]], colors=None):
    # 繪製多條曲線的折線圖，X 軸為 epoch
    n = max(len(v) for v in series.values())  # 取最長的序列長度作為 epoch 數
    epochs = list(range(1, n + 1))
    colors = colors or plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for (label, vals), c in zip(series.items(), colors):
        ax.plot(epochs[: len(vals)], vals, marker="o", label=label, color=c)
    ax.set_xlabel("Epoch", fontproperties=_fp)
    ax.set_ylabel(ylabel, fontproperties=_fp)
    ax.set_title(title, fontproperties=_fp)
    ax.legend(prop=_fp)
    ax.set_xticks(epochs)
    ax.grid(axis="y", alpha=0.3)  # 加上 Y 軸方向的淡格線方便比對
    _save(fig, fpath)


def grouped_bar(fpath, title, data: Dict[str, List[float]], labels: List[str],
                ylabel="", colors=None, annotate=True):
    """
    data = {series_name: [val_per_label, ...]}, labels = x-axis labels.
    """
    x = np.arange(len(labels))
    n = len(data)
    w = 0.7 / n  # 依系列數動態決定每根柱子的寬度
    colors = colors or plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.8), 4))
    for i, (name, vals) in enumerate(data.items()):
        offset = (i - (n - 1) / 2) * w  # 計算每組柱子的位置偏移，使其對齊中央
        bars = ax.bar(x + offset, vals, w, label=name, color=colors[i % len(colors)])
        if annotate:
            # 在柱子上方標註對應數值
            for xi, v in zip(x + offset, vals):
                ax.text(xi, v * 1.005, f"{v:.4f}", ha="center", fontsize=7, rotation=40)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontproperties=_fp)
    ax.set_ylabel(ylabel, fontproperties=_fp)
    ax.set_title(title, fontproperties=_fp)
    if n > 1:
        ax.legend(prop=_fp)  # 多個系列時才顯示圖例
    ax.grid(axis="y", alpha=0.3)
    _save(fig, fpath)


# ── assignment plotters ───────────────────────────────────────────────────────

# 各 assignment 圖表使用的色票，固定色彩可確保多張圖之間視覺一致
COLORS_A1 = ["#e15759", "#4e79a7", "#59a14f"]
COLORS_A2 = ["#4e79a7", "#e15759", "#f28e2b"]
COLORS_A3 = ["#e15759", "#4e79a7"]


def plot_assignment1(res_dir: str, out: str):
    # Assignment 1：比較不同 hidden dimension (32 / 64 / 128) 的訓練結果
    print("[A1] parsing summary logs...")
    epoch_data: Dict[str, dict] = {}    # key: dim=32/64/128, value: {"lprob": [...], "acc": [...]}
    for dim in ["32", "64", "128"]:
        # 讀取每個 dim 設定的 summary 檔，擷取 LProb 與 ACC 兩個指標
        path = os.path.join(res_dir, "assignment1", f"dim{dim}_summary.txt")
        rows = parse_summary(path)
        epoch_data[f"dim={dim}"] = {
            "lprob": col_floats(rows, "LProb"),
            "acc":   col_floats(rows, "ACC"),
        }

    # 繪製 LProb 與 ACC 隨 epoch 變化的折線圖
    line_chart(f"{out}/a1_lprob.png", "A1: Val LProb over Epochs", "Val LProb ↓",
               {k: v["lprob"] for k, v in epoch_data.items()}, colors=COLORS_A1)
    line_chart(f"{out}/a1_acc.png",   "A1: Val ACC over Epochs",   "Val ACC ↑",
               {k: v["acc"]   for k, v in epoch_data.items()}, colors=COLORS_A1)

    print("[A1] parsing comparison...")
    # 讀取最終比較表，繪製不同 dim 的最終結果柱狀圖
    rows = parse_comparison(os.path.join(res_dir, "assignment1", "comparison.txt"))
    dim_labels = [r[list(r.keys())[0]] for r in rows]  # 第一欄通常是 dim 名稱
    lprob_vals = col_floats(rows, "LProb")
    acc_vals   = col_floats(rows, "ACC")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))  # 並排顯示 LProb 與 ACC
    for ax, vals, title, ylabel in [
        (axes[0], lprob_vals, "Final Val LProb", "Val LProb ↓"),
        (axes[1], acc_vals,   "Final Val ACC",   "Val ACC ↑"),
    ]:
        ax.bar(dim_labels, vals, color=COLORS_A1[: len(vals)])
        ax.set_xlabel("dim_hidden", fontproperties=_fp)
        ax.set_ylabel(ylabel, fontproperties=_fp)
        ax.set_title(title, fontproperties=_fp)
        if vals:
            # 縮小 Y 軸範圍以放大數值差異，使比較更明顯
            ax.set_ylim(min(vals) * 0.97, max(vals) * 1.03)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.005, f"{v:.4f}", ha="center", fontsize=9)  # 柱頂標註數值
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("A1: Final Results Comparison", fontproperties=_fp)
    _save(fig, f"{out}/a1_final.png")


def plot_assignment2(res_dir: str, out: str):
    # Assignment 2：比較三種設定 (baseline / 無 TT loss / 均勻 pi 初始化) 的影響
    exps = ["baseline", "no_tt_loss", "homo_pi_init"]

    print("[A2] parsing summary logs...")
    epoch_data: Dict[str, dict] = {}
    for exp in exps:
        # 同時擷取 LProb、LFunc、ACC 三個指標
        path = os.path.join(res_dir, "assignment2", f"{exp}_summary.txt")
        rows = parse_summary(path)
        epoch_data[exp] = {
            "lprob": col_floats(rows, "LProb"),
            "lfunc": col_floats(rows, "LFunc"),
            "acc":   col_floats(rows, "ACC"),
        }

    # 為每個指標各畫一張折線圖
    for metric, ylabel, fname in [
        ("lprob", "Val LProb ↓", "a2_lprob.png"),
        ("lfunc", "Val LFunc ↓", "a2_lfunc.png"),
        ("acc",   "Val ACC ↑",   "a2_acc.png"),
    ]:
        line_chart(f"{out}/{fname}", f"A2: Val {metric.upper()} over Epochs", ylabel,
                   {k: v[metric] for k, v in epoch_data.items()}, colors=COLORS_A2)

    print("[A2] parsing comparison...")
    # 最終比較圖：以柱狀圖並排呈現三個指標
    rows = parse_comparison(os.path.join(res_dir, "assignment2", "comparison.txt"))
    exp_labels = [r[list(r.keys())[0]] for r in rows]  # 第一欄為實驗名稱

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))  # 三張子圖並排
    for ax, key, ylabel in zip(
        axes,
        ["LProb", "LFunc", "ACC"],
        ["Val LProb ↓", "Val LFunc ↓", "Val ACC ↑"],
    ):
        vals = col_floats(rows, key)
        x = np.arange(len(exp_labels))
        ax.bar(x, vals, color=COLORS_A2[: len(vals)])
        ax.set_xticks(x)
        # 實驗名稱稍微旋轉以避免重疊
        ax.set_xticklabels(exp_labels, fontproperties=_fp, fontsize=8, rotation=8)
        ax.set_ylabel(ylabel, fontproperties=_fp)
        ax.set_title(f"Final {key}", fontproperties=_fp)
        for xi, v in enumerate(vals):
            ax.text(xi, v * 1.005, f"{v:.4f}", ha="center", fontsize=8)  # 標註數值
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("A2: Final Results Comparison", fontproperties=_fp)
    _save(fig, f"{out}/a2_final.png")


def plot_assignment3(res_dir: str, out: str):
    # Assignment 3：比較僅學機率 (prob_only) 與額外學 transition (with_trans) 的差異
    exps = ["prob_only", "with_trans"]

    print("[A3] parsing summary logs...")
    epoch_data: Dict[str, dict] = {}
    for exp in exps:
        # 取得四個訓練指標：LProb / LFunc / LTrans / ACC
        path = os.path.join(res_dir, "assignment3", f"{exp}_summary.txt")
        rows = parse_summary(path)
        epoch_data[exp] = {
            "lprob":  col_floats(rows, "LProb"),
            "lfunc":  col_floats(rows, "LFunc"),
            "ltrans": col_floats(rows, "LTrans"),
            "acc":    col_floats(rows, "ACC"),
        }

    # 對三個重點指標分別繪製折線圖
    for metric, ylabel, fname in [
        ("ltrans", "Val LTrans ↓", "a3_ltrans.png"),
        ("lprob",  "Val LProb ↓",  "a3_lprob.png"),
        ("acc",    "Val ACC ↑",    "a3_acc.png"),
    ]:
        line_chart(f"{out}/{fname}", f"A3: Val {metric.upper()} over Epochs", ylabel,
                   {k: v[metric] for k, v in epoch_data.items()}, colors=COLORS_A3)

    print("[A3] parsing eval logs...")
    # 解析 eval 階段的 log，取出 L1 誤差與 per-gate-type 統計
    eval_data = {
        exp: parse_eval_log(
            os.path.join(res_dir, "assignment3", f"{exp}_eval.log")
        )
        for exp in exps
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 左右兩張子圖

    # left: L1 comparison (prob / analytic / model)
    x = np.arange(len(exps))
    w = 0.25  # 每根柱子的寬度（每組三根）
    # 收集所有 L1 數值以決定 Y 軸上限（保留 25% 餘裕讓標註不被截斷）
    all_bar_vals = [eval_data[e].get(k, 0) for e in exps
                    for k in ["l1_prob", "l1_analytic", "l1_model"]]
    ymax0 = max(v for v in all_bar_vals if not np.isnan(v)) * 1.25
    offset0 = ymax0 * 0.02  # 數值標註離柱頂的固定距離
    for i, (key, lbl, col) in enumerate(zip(
        ["l1_prob", "l1_analytic", "l1_model"],
        ["L1(prob)", "L1 analytic 2p(1−p)", "L1(trans) model"],
        ["#76b7b2", "#f28e2b", "#59a14f"],
    )):
        vals = [eval_data[e].get(key, float("nan")) for e in exps]
        axes[0].bar(x + i * w, vals, w, label=lbl, color=col)
        for xi, v in zip(x + i * w, vals):
            axes[0].text(xi, v + offset0, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_xticks(x + w)  # 將 x 軸刻度置中於三組柱子中央
    axes[0].set_xticklabels(exps, fontproperties=_fp)
    axes[0].set_ylabel("L1 Error", fontproperties=_fp)
    axes[0].set_title("Switching Probability Performance", fontproperties=_fp)
    axes[0].legend(prop=_fp, fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, ymax0)

    # right: by gate type
    # collect gate rows from with_trans and prob_only
    gate_types = [r.get("type", r[list(r.keys())[0]])
                  for r in eval_data["with_trans"].get("gate_rows", [])]
    po_vals = [float(r.get("L1 trans", r[list(r.keys())[-1]]))
               for r in eval_data["prob_only"].get("gate_rows", [])]
    wt_vals = [float(r.get("L1 trans", r[list(r.keys())[-1]]))
               for r in eval_data["with_trans"].get("gate_rows", [])]
    if gate_types:
        xg = np.arange(len(gate_types))
        # 兩組柱子分別位於刻度的左右側 (±0.2)
        axes[1].bar(xg - 0.2, po_vals, 0.4, label="prob_only",  color=COLORS_A3[0])
        axes[1].bar(xg + 0.2, wt_vals, 0.4, label="with_trans", color=COLORS_A3[1])
        axes[1].set_xticks(xg)
        axes[1].set_xticklabels(gate_types, fontproperties=_fp)
        axes[1].set_ylabel("L1(trans) model", fontproperties=_fp)
        axes[1].set_title("L1 by Gate Type", fontproperties=_fp)
        axes[1].legend(prop=_fp)
        axes[1].grid(axis="y", alpha=0.3)
        ymax1 = max(po_vals + wt_vals) * 1.25  # 同樣保留餘裕讓標註不被截斷
        axes[1].set_ylim(0, ymax1)
        offset1 = ymax1 * 0.02
        # prob_only 數值通常較大，使用 3 位小數即可清楚比較
        for xi, v in zip(xg - 0.2, po_vals):
            axes[1].text(xi, v + offset1, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        # with_trans 數值較小，使用 4 位小數以保留精度
        for xi, v in zip(xg + 0.2, wt_vals):
            axes[1].text(xi, v + offset1, f"{v:.4f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("A3: Switching Probability Evaluation", fontproperties=_fp)
    _save(fig, f"{out}/a3_switching.png")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    # 以 script 所在位置為基準，計算預設的 results 目錄路徑
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_results = os.path.normpath(
        os.path.join(script_dir, "..", "log", "10_epochs", "results")
    )

    # 設定命令列參數解析器
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # 使用腳本頂部的 docstring 作為說明文字，並保留格式
    parser.add_argument("results_dir", nargs="?", default=default_results,
                        help="path to results/ directory (default: %(default)s)")   # nargs="?" 代表該參數為可選，若不提供則使用 default_results
    parser.add_argument("--out", default=None,
                        help="output directory for charts (default: results_dir/../charts)")
    args = parser.parse_args()

    # 將輸入路徑轉為絕對路徑，並建立輸出目錄
    res = os.path.abspath(args.results_dir)
    out = os.path.abspath(args.out) if args.out else os.path.join(res, "..", "charts")
    os.makedirs(out, exist_ok=True)

    print(f"results dir : {res}")
    print(f"charts out  : {out}\n")

    # 檢查每個 assignment 子目錄是否存在，僅處理實際存在的部分
    a1_ok = os.path.isdir(os.path.join(res, "assignment1"))
    a2_ok = os.path.isdir(os.path.join(res, "assignment2"))
    a3_ok = os.path.isdir(os.path.join(res, "assignment3"))

    if a1_ok:
        plot_assignment1(res, out)
    else:
        print("[A1] skipped (no assignment1/ dir found)")

    if a2_ok:
        plot_assignment2(res, out)
    else:
        print("[A2] skipped (no assignment2/ dir found)")

    if a3_ok:
        plot_assignment3(res, out)
    else:
        print("[A3] skipped (no assignment3/ dir found)")

    # 列出輸出目錄中所有產生的圖表，方便確認結果
    charts = sorted(os.listdir(out))
    print(f"\nDone — {len(charts)} charts in {out}/")
    for c in charts:
        print(f"  {c}")


if __name__ == "__main__":
    main()
