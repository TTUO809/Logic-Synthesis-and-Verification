#!/usr/bin/env python3
"""
Assignment 1: Full-library baseline mapping for all 5 libraries × 3 designs.

Run from anywhere:
    conda activate LSV_PA3
    python ~/LSV/PA_3/run_a1_baseline.py

ABC command used (same as MapTune internal baseline):
    read <genlib>; read <design>; map -a; write <blif>;
    read <lib>; read -m <blif>; ps; topo; upsize; dnsize; stime;
"""

import subprocess
import re
import os
import sys
import shutil

MAPTUNE_DIR = os.path.expanduser("~/MapTune")

# Find ABC: check PATH first, then ~/abc/abc
_ABC = shutil.which("abc") or os.path.expanduser("~/abc/abc")
if not os.path.isfile(_ABC):
    sys.exit("ERROR: abc not found. Run setup_abc.sh and source ~/.bashrc")

LIBRARIES = [
    "7nm.genlib",
    "gf180mcu_ff_125C.genlib",
    "gf180mcu_tt_025C.genlib",
    "nan45.genlib",
    "sky130.genlib",
]

DESIGNS = [
    "benchmarks/s13207.bench",
    "benchmarks/c2670.bench",
    "benchmarks/b20_1.bench",
]

LIB_LABEL = {
    "7nm.genlib":              "ASAP7 (7nm,  161)",
    "gf180mcu_ff_125C.genlib": "GF180 ff125C (151)",
    "gf180mcu_tt_025C.genlib": "GF180 tt025C (151)",
    "nan45.genlib":            "NAN45  (94) ",
    "sky130.genlib":           "SKY130 (343)",
}

DESIGN_LABEL = {
    "benchmarks/s13207.bench": "s13207",
    "benchmarks/c2670.bench":  "c2670 ",
    "benchmarks/b20_1.bench":  "b20_1 ",
}


def run_baseline(genlib: str, design: str):
    """Run full-library mapping and return (delay_ps, area)."""
    lib      = genlib[:-7] + ".lib"
    d_stem   = os.path.basename(design).replace(".bench", "")
    l_stem   = genlib.replace(".genlib", "")
    tmp_blif = f"temp_blifs/a1_{l_stem}_{d_stem}.blif"

    abc_cmd = (
        f"read {genlib}; read {design}; map -a; "
        f"write {tmp_blif}; "
        f"read {lib}; read -m {tmp_blif}; "
        f"ps; topo; upsize; dnsize; stime;"
    )

    try:
        raw = subprocess.check_output(
            [_ABC, "-c", abc_cmd],
            cwd=MAPTUNE_DIR,
            stderr=subprocess.STDOUT,
            timeout=180,
        )
        out = raw.decode(errors="ignore")

        m_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", out)
        m_a = re.search(r"Area\s*=\s*([\d.]+)", out)

        if m_d and m_a:
            return float(m_d.group(1)), float(m_a.group(1)), out
        else:
            return None, None, out

    except subprocess.TimeoutExpired:
        return "TIMEOUT", "TIMEOUT", ""
    except subprocess.CalledProcessError as e:
        return "ERROR", "ERROR", e.output.decode(errors="ignore")


def main():
    results = {}

    for lib in LIBRARIES:
        for design in DESIGNS:
            tag = f"{LIB_LABEL[lib]} × {DESIGN_LABEL[design]}"
            print(f"[RUN] {tag}", flush=True)

            delay, area, raw_out = run_baseline(lib, design)

            if isinstance(delay, float):
                print(f"      Delay = {delay:.2f} ps   Area = {area:.2f}\n")
            else:
                print(f"      FAILED: {delay}")
                print("--- ABC output (last 10 lines) ---")
                for line in raw_out.splitlines()[-10:]:
                    print("  ", line)
                print()

            results[(lib, design)] = (delay, area)

    # ── Print summary table ──────────────────────────────────────────────────
    W = 88
    print("\n" + "=" * W)
    print("  ASSIGNMENT 1 — BASELINE RESULTS")
    print("  Mapper: map -a (area-driven)   Full library (all cells)")
    print("=" * W)
    print(f"  {'Library':<24} {'Design':<9} {'Delay (ps)':>12} {'Area':>12} {'ADP (D×A)':>14}")
    print("-" * W)

    for lib in LIBRARIES:
        for design in DESIGNS:
            d, a = results[(lib, design)]
            ll   = LIB_LABEL[lib]
            dl   = DESIGN_LABEL[design]
            if isinstance(d, float):
                adp = d * a
                print(f"  {ll:<24} {dl:<9} {d:>12.2f} {a:>12.2f} {adp:>14.2f}")
            else:
                print(f"  {ll:<24} {dl:<9} {str(d):>12} {str(a):>12} {'N/A':>14}")
        print()

    print("=" * W)
    print("Note: ADP shown is unnormalized (D_ps × Area). "
          "Normalized ADP = (D/D_base)×(A/A_base) is used in A3.")


if __name__ == "__main__":
    main()
