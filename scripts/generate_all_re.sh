#!/usr/bin/env bash
# Generate NS_fine_Re*_T128_part*.npy for all Re values needed for Table 8.
#
# Re=500 (part0, part1, part2) already available:
#   python download_data.py --name NS-Re500Part0 --outdir $OUTDIR
#   python download_data.py --name NS-Re500Part1 --outdir $OUTDIR
#   python download_data.py --name NS-Re500Part2 --outdir $OUTDIR
#
# Re=100 part0 already available:
#   python download_data.py --name NS-Re100Part0 --outdir $OUTDIR
#
# This script generates everything else:
#   Re=100 parts 1, 2
#   Re=200, 250, 300, 350, 400 parts 0, 1, 2
#
# Usage (3 GPUs):
#   OUTDIR=/path/to/data/ns GPU0=0 GPU1=1 GPU2=2 bash scripts/generate_all_re.sh
#
# Each run: ~100 trajectories × 128 time steps at 128×128, Re-dependent dt.
# Estimated wall time per run: 20–60 min on TITAN X (higher Re = smaller dt = longer).

set -euo pipefail

OUTDIR="${OUTDIR:-data/ns}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"

X_RES=128
T=100
T_RES=128
BURNIN=100

generate() {
    local re=$1
    local part=$2
    local gpu=$3
    local seed=$((part * 100 + ${re%.*}))   # deterministic, unique seed per (re, part)
    echo "Launching Re=${re} part${part} on cuda:${gpu} seed=${seed}"
    CUDA_VISIBLE_DEVICES=${gpu} python -m src.data.generate_ns \
        --device cuda \
        --re "${re}" \
        --x_res "${X_RES}" \
        --x_sub 1 \
        --T "${T}" \
        --t_res "${T_RES}" \
        --burnin "${BURNIN}" \
        --batchsize 1 \
        --part "${part}" \
        --outdir "${OUTDIR}" \
        --seed "${seed}"
}

# ---------- Round 1: Re=100 parts 1,2 + Re=200 part 0 ----------
generate 100  1 ${GPU0} &
generate 100  2 ${GPU1} &
generate 200  0 ${GPU2} &
wait

# ---------- Round 2: Re=200 parts 1,2 + Re=250 part 0 ----------
generate 200  1 ${GPU0} &
generate 200  2 ${GPU1} &
generate 250  0 ${GPU2} &
wait

# ---------- Round 3 ----------
generate 250  1 ${GPU0} &
generate 250  2 ${GPU1} &
generate 300  0 ${GPU2} &
wait

# ---------- Round 4 ----------
generate 300  1 ${GPU0} &
generate 300  2 ${GPU1} &
generate 350  0 ${GPU2} &
wait

# ---------- Round 5 ----------
generate 350  1 ${GPU0} &
generate 350  2 ${GPU1} &
generate 400  0 ${GPU2} &
wait

# ---------- Round 6 ----------
generate 400  1 ${GPU0} &
generate 400  2 ${GPU1} &
wait

echo "Done. Files in ${OUTDIR}:"
ls -lh "${OUTDIR}"/NS_fine_Re*.npy 2>/dev/null || echo "(outdir not found locally)"
