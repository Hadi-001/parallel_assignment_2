# Parallel RL Assignment

## How to run

```bash
bash run_all.sh 300
```

Optional goal mode argument:

```bash
bash run_all.sh 300 edge
bash run_all.sh 300 center
```

This builds all binaries and runs:
- Sequential
- OpenMP (`2, 4, 8, 16, 32` threads)
- MPI (`2, 4, 8, 16, 32` processes)

with averaging over `300` runs (change `300` to any value you want).
- `edge` goal mode: goal near the end of the grid (default).
- `center` goal mode: goal moved to the center of the grid.

Generated files:
- `output/` -> CSV data
- `results/` -> plots