# Parallel RL Assignment

## How to run

```bash
bash run_all.sh 300
```

This builds all binaries and runs:
- Sequential
- OpenMP (`2, 4, 8, 16, 32` threads)
- MPI (`2, 4, 8, 16, 32` processes)

with averaging over `300` runs (change `300` to any value you want).

Generated files:
- `output/` -> CSV data
- `results/` -> plots