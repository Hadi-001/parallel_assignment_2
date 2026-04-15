#!/bin/bash
set -e

echo "============================================"
echo "  Parallel RL Assignment - Full Run Script"
echo "============================================"
echo ""

# Number of independent runs to average (default: 10)
RUNS=${1:-10}
GOAL_MODE=${2:-edge}
export GOAL_MODE

# Build
echo ">>> Building all executables..."
make clean
make all
echo ""

# Create output directories
mkdir -p output results

# Thread/process counts to test
THREAD_COUNTS="2 4 8 16 32"

echo ">>> Averaging over $RUNS independent runs per configuration"
echo ">>> Goal mode: $GOAL_MODE (edge|center)"
echo ""

# ---- Sequential ----
echo "============================================"
echo "  Step 1: Sequential Version"
echo "============================================"
./sequential "$RUNS"
echo ""

# ---- OpenMP ----
echo "============================================"
echo "  Step 2: OpenMP Version"
echo "============================================"
for T in $THREAD_COUNTS; do
    echo "--- OpenMP with $T threads ---"
    ./openmp_rl "$T" "$RUNS"
    echo ""
done

# ---- MPI ----
echo "============================================"
echo "  Step 3: MPI Version"
echo "============================================"
for P in $THREAD_COUNTS; do
    echo "--- MPI with $P processes ---"
    mpirun --allow-run-as-root --oversubscribe -np "$P" ./mpi_rl "$RUNS"
    echo ""
done

# ---- Plots ----
echo "============================================"
echo "  Step 4: Generating Plots"
echo "============================================"
python3 plot.py
echo ""

echo "============================================"
echo "  All done!"
echo "  CSV data   -> output/"
echo "  Plots      -> results/"
echo "  Averaged runs per setup: $RUNS"
echo "  Goal mode: $GOAL_MODE"
echo "============================================"
