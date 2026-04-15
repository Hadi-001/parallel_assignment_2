#!/usr/bin/env python3
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

OUTPUT_DIR = "output"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

GRID_SIZES = [4, 8, 16]
THREAD_COUNTS = [2, 4, 8, 16, 32]

ACTION_NAMES = ["Up", "Down", "Left", "Right"]


def csv_path(name):
    return os.path.join(OUTPUT_DIR, name)


def load_rewards(filename):
    episodes, rewards = [], []
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row['episode']))
            rewards.append(float(row['reward']))
    return np.array(episodes), np.array(rewards)


def load_policy(filename):
    states, opt_actions, opt_pct = [], [], []
    avg_rewards = []
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            states.append(int(row['state']))
            opt_actions.append(int(row['optimal_action']))
            opt_pct.append(float(row['opt_action_pct']))
            ar = [float(row[f'avg_reward_{a}']) for a in ACTION_NAMES]
            avg_rewards.append(ar)
    return states, opt_actions, opt_pct, avg_rewards


def load_timing(filename):
    with open(filename) as f:
        reader = csv.DictReader(f)
        row = next(reader)
        return int(row['grid_size']), int(row['episodes']), float(row['time_seconds'])


def running_average(data, window=50):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def plot_avg_reward_per_episode():
    """Plot 1: Average reward per episode vs number of episodes (all methods)."""
    for gs in GRID_SIZES:
        fig, ax = plt.subplots(figsize=(10, 6))

        files = {}
        seq_file = csv_path(f"sequential_{gs}x{gs}_rewards.csv")
        if os.path.exists(seq_file):
            files['Sequential'] = seq_file

        for tc in THREAD_COUNTS:
            omp_file = csv_path(f"openmp_{tc}t_{gs}x{gs}_rewards.csv")
            if os.path.exists(omp_file):
                files[f'OpenMP ({tc}t)'] = omp_file

        for tc in THREAD_COUNTS:
            mpi_file = csv_path(f"mpi_{tc}p_{gs}x{gs}_rewards.csv")
            if os.path.exists(mpi_file):
                files[f'MPI ({tc}p)'] = mpi_file

        colors = plt.cm.tab10(np.linspace(0, 1, len(files)))

        for (label, fname), color in zip(files.items(), colors):
            eps, rews = load_rewards(fname)
            window = min(50, len(rews) // 5) if len(rews) > 10 else 1
            if window > 1:
                avg = running_average(rews, window)
                ax.plot(range(window, len(rews) + 1), avg, label=label, color=color, linewidth=1.5)
            else:
                ax.plot(eps, rews, label=label, color=color, linewidth=1.5)

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Average Reward (rolling window)', fontsize=12)
        ax.set_title(f'Average Reward per Episode - {gs}x{gs} Grid', fontsize=14)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/avg_reward_{gs}x{gs}.png', dpi=150)
        plt.close()
        print(f"  Saved {RESULTS_DIR}/avg_reward_{gs}x{gs}.png")


def plot_optimal_action_distribution():
    """Plot 2: Optimal action percentage per state (bar chart)."""
    for gs in GRID_SIZES:
        methods = {}
        seq_file = csv_path(f"sequential_{gs}x{gs}_policy.csv")
        if os.path.exists(seq_file):
            methods['Sequential'] = seq_file

        best_omp = None
        for tc in THREAD_COUNTS:
            f = csv_path(f"openmp_{tc}t_{gs}x{gs}_policy.csv")
            if os.path.exists(f):
                best_omp = (f'OpenMP ({tc}t)', f)
        if best_omp:
            methods[best_omp[0]] = best_omp[1]

        best_mpi = None
        for tc in THREAD_COUNTS:
            f = csv_path(f"mpi_{tc}p_{gs}x{gs}_policy.csv")
            if os.path.exists(f):
                best_mpi = (f'MPI ({tc}p)', f)
        if best_mpi:
            methods[best_mpi[0]] = best_mpi[1]

        if not methods:
            continue

        fig, axes = plt.subplots(len(methods), 1, figsize=(max(12, gs * 0.8), 5 * len(methods)))
        if len(methods) == 1:
            axes = [axes]

        for ax, (label, fname) in zip(axes, methods.items()):
            states, _, opt_pct, _ = load_policy(fname)
            colors = ['#2ecc71' if p > 0.5 else '#e74c3c' if p < 0.3 else '#f39c12'
                       for p in opt_pct]
            ax.bar(states, [p * 100 for p in opt_pct], color=colors, edgecolor='none', width=0.8)
            ax.set_xlabel('State', fontsize=11)
            ax.set_ylabel('Optimal Action %', fontsize=11)
            ax.set_title(f'{label} - {gs}x{gs} Grid', fontsize=12)
            ax.set_ylim(0, 105)
            ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.2, axis='y')

        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/opt_action_dist_{gs}x{gs}.png', dpi=150)
        plt.close()
        print(f"  Saved {RESULTS_DIR}/opt_action_dist_{gs}x{gs}.png")


def plot_execution_time_comparison():
    """Plot 3: Execution time comparison across methods and thread counts."""
    for gs in GRID_SIZES:
        labels, times = [], []

        seq_file = csv_path(f"sequential_{gs}x{gs}_timing.csv")
        if os.path.exists(seq_file):
            _, _, t = load_timing(seq_file)
            labels.append('Sequential')
            times.append(t)

        for tc in THREAD_COUNTS:
            omp_file = csv_path(f"openmp_{tc}t_{gs}x{gs}_timing.csv")
            if os.path.exists(omp_file):
                _, _, t = load_timing(omp_file)
                labels.append(f'OMP-{tc}t')
                times.append(t)

        for tc in THREAD_COUNTS:
            mpi_file = csv_path(f"mpi_{tc}p_{gs}x{gs}_timing.csv")
            if os.path.exists(mpi_file):
                _, _, t = load_timing(mpi_file)
                labels.append(f'MPI-{tc}p')
                times.append(t)

        if not labels:
            continue

        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 6))
        colors = []
        for l in labels:
            if 'Sequential' in l:
                colors.append('#3498db')
            elif 'OMP' in l:
                colors.append('#2ecc71')
            else:
                colors.append('#e74c3c')

        bars = ax.bar(labels, times, color=colors, edgecolor='white', width=0.6)
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.001,
                    f'{t:.4f}s', ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax.set_title(f'Execution Time Comparison - {gs}x{gs} Grid', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/exec_time_{gs}x{gs}.png', dpi=150)
        plt.close()
        print(f"  Saved {RESULTS_DIR}/exec_time_{gs}x{gs}.png")


def plot_speedup():
    """Plot 4: Speedup graph (sequential time / parallel time)."""
    fig, axes = plt.subplots(1, len(GRID_SIZES), figsize=(6 * len(GRID_SIZES), 5))
    if len(GRID_SIZES) == 1:
        axes = [axes]

    for ax, gs in zip(axes, GRID_SIZES):
        seq_file = csv_path(f"sequential_{gs}x{gs}_timing.csv")
        if not os.path.exists(seq_file):
            continue
        _, _, seq_time = load_timing(seq_file)

        omp_threads, omp_speedups = [], []
        mpi_procs, mpi_speedups = [], []

        for tc in THREAD_COUNTS:
            omp_file = csv_path(f"openmp_{tc}t_{gs}x{gs}_timing.csv")
            if os.path.exists(omp_file):
                _, _, t = load_timing(omp_file)
                omp_threads.append(tc)
                omp_speedups.append(seq_time / t if t > 0 else 0)

            mpi_file = csv_path(f"mpi_{tc}p_{gs}x{gs}_timing.csv")
            if os.path.exists(mpi_file):
                _, _, t = load_timing(mpi_file)
                mpi_procs.append(tc)
                mpi_speedups.append(seq_time / t if t > 0 else 0)

        all_x = sorted(set(omp_threads + mpi_procs))
        if all_x:
            ax.plot(all_x, all_x, 'k--', alpha=0.4, label='Ideal (linear)')

        if omp_threads:
            ax.plot(omp_threads, omp_speedups, 'go-', linewidth=2, markersize=8, label='OpenMP')
        if mpi_procs:
            ax.plot(mpi_procs, mpi_speedups, 'rs-', linewidth=2, markersize=8, label='MPI')

        ax.set_xlabel('Number of Threads/Processes', fontsize=11)
        ax.set_ylabel('Speedup', fontsize=11)
        ax.set_title(f'Speedup - {gs}x{gs} Grid', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/speedup.png', dpi=150)
    plt.close()
    print(f"  Saved {RESULTS_DIR}/speedup.png")


def plot_policy_grids():
    """Plot 5: Visual grid showing policy arrows for each method (small grids only)."""
    arrow_map = {0: '\u2191', 1: '\u2193', 2: '\u2190', 3: '\u2192'}
    configs = {
        4:  {'goal': 15, 'trap': 12, 'obstacles': {2, 5}},
        8:  {'goal': 63, 'trap': 56, 'obstacles': {10, 21, 42, 53}},
        16: {'goal': 255, 'trap': 240, 'obstacles': {18, 37, 74, 93, 122, 157, 198, 221}},
    }

    for gs in [4, 8]:
        methods = {}
        seq_file = csv_path(f"sequential_{gs}x{gs}_policy.csv")
        if os.path.exists(seq_file):
            methods['Sequential'] = seq_file

        for tc in THREAD_COUNTS:
            f = csv_path(f"openmp_{tc}t_{gs}x{gs}_policy.csv")
            if os.path.exists(f):
                methods[f'OpenMP ({tc}t)'] = f
                break

        for tc in THREAD_COUNTS:
            f = csv_path(f"mpi_{tc}p_{gs}x{gs}_policy.csv")
            if os.path.exists(f):
                methods[f'MPI ({tc}p)'] = f
                break

        if not methods:
            continue

        cfg = configs[gs]

        fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5))
        if len(methods) == 1:
            axes = [axes]

        for ax, (label, fname) in zip(axes, methods.items()):
            states, opt_actions, opt_pct, _ = load_policy(fname)
            grid = np.zeros((gs, gs))
            for s in range(gs * gs):
                r, c = s // gs, s % gs
                grid[r][c] = opt_pct[s]

            ax.imshow(grid, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')

            for s in range(gs * gs):
                r, c = s // gs, s % gs
                if s == cfg['goal']:
                    ax.text(c, r, 'G', ha='center', va='center', fontsize=14,
                            fontweight='bold', color='blue')
                elif s == cfg['trap']:
                    ax.text(c, r, 'T', ha='center', va='center', fontsize=14,
                            fontweight='bold', color='red')
                elif s in cfg['obstacles']:
                    ax.text(c, r, 'X', ha='center', va='center', fontsize=14,
                            fontweight='bold', color='black')
                else:
                    ax.text(c, r, arrow_map[opt_actions[s]], ha='center', va='center',
                            fontsize=16)

            ax.set_title(f'{label}', fontsize=12)
            ax.set_xticks(range(gs))
            ax.set_yticks(range(gs))
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Policy Visualization - {gs}x{gs} Grid', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/policy_grid_{gs}x{gs}.png', dpi=150)
        plt.close()
        print(f"  Saved {RESULTS_DIR}/policy_grid_{gs}x{gs}.png")


def generate_summary_table():
    """Generate a summary CSV with all timing results."""
    rows = []
    for gs in GRID_SIZES:
        seq_file = csv_path(f"sequential_{gs}x{gs}_timing.csv")
        seq_time = None
        if os.path.exists(seq_file):
            _, _, seq_time = load_timing(seq_file)
            rows.append({'grid': f'{gs}x{gs}', 'method': 'Sequential',
                         'threads': 1, 'time': seq_time, 'speedup': 1.0})

        for tc in THREAD_COUNTS:
            omp_file = csv_path(f"openmp_{tc}t_{gs}x{gs}_timing.csv")
            if os.path.exists(omp_file):
                _, _, t = load_timing(omp_file)
                sp = seq_time / t if seq_time and t > 0 else 0
                rows.append({'grid': f'{gs}x{gs}', 'method': 'OpenMP',
                             'threads': tc, 'time': t, 'speedup': sp})

            mpi_file = csv_path(f"mpi_{tc}p_{gs}x{gs}_timing.csv")
            if os.path.exists(mpi_file):
                _, _, t = load_timing(mpi_file)
                sp = seq_time / t if seq_time and t > 0 else 0
                rows.append({'grid': f'{gs}x{gs}', 'method': 'MPI',
                             'threads': tc, 'time': t, 'speedup': sp})

    if rows:
        with open(f'{RESULTS_DIR}/summary.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['grid', 'method', 'threads', 'time', 'speedup'])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved {RESULTS_DIR}/summary.csv")


def main():
    print("Generating plots...\n")

    print("1. Average Reward per Episode:")
    plot_avg_reward_per_episode()

    print("\n2. Optimal Action Distribution:")
    plot_optimal_action_distribution()

    print("\n3. Execution Time Comparison:")
    plot_execution_time_comparison()

    print("\n4. Speedup Analysis:")
    plot_speedup()

    print("\n5. Policy Grid Visualization:")
    plot_policy_grids()

    print("\n6. Summary Table:")
    generate_summary_table()

    print("\nAll plots saved to results/ directory.")


if __name__ == '__main__':
    main()
