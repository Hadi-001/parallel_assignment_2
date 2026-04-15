#include "gridworld.h"
#include <mpi.h>

#define RUN_SEED_STRIDE 100000U

static void run_mpi(int grid_size, unsigned int base_seed, int num_runs, int rank, int nprocs) {
    GridConfig cfg;
    get_grid_config(grid_size, &cfg);
    int N = grid_size * grid_size;
    int table_size = N * NUM_ACTIONS;

    int eps_per_proc = NUM_EPISODES / nprocs;
    int start_ep = rank * eps_per_proc;
    int end_ep   = (rank == nprocs - 1) ? NUM_EPISODES : start_ep + eps_per_proc;
    int local_eps = end_ep - start_ep;

    double *agg_global_reward_sum = NULL;
    int    *agg_global_count      = NULL;
    double *avg_ep_rewards        = NULL;

    if (rank == 0) {
        agg_global_reward_sum = calloc(table_size, sizeof(double));
        agg_global_count      = calloc(table_size, sizeof(int));
        avg_ep_rewards        = calloc(NUM_EPISODES, sizeof(double));
    }

    int *recv_counts = malloc(nprocs * sizeof(int));
    int *displs      = malloc(nprocs * sizeof(int));
    for (int i = 0; i < nprocs; i++) {
        int s = i * eps_per_proc;
        int e = (i == nprocs - 1) ? NUM_EPISODES : s + eps_per_proc;
        recv_counts[i] = e - s;
        displs[i] = s;
    }

    double total_elapsed_local = 0.0;

    for (int run = 0; run < num_runs; run++) {
        double *local_reward_sum = calloc(table_size, sizeof(double));
        int    *local_count      = calloc(table_size, sizeof(int));
        double *local_ep_rewards = malloc(local_eps * sizeof(double));

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        for (int ep = start_ep; ep < end_ep; ep++) {
            RNG rng;
            rng_seed(&rng, base_seed + (unsigned int)(run * RUN_SEED_STRIDE + ep));
            local_ep_rewards[ep - start_ep] = run_episode(&cfg, local_reward_sum, local_count, &rng);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        total_elapsed_local += (MPI_Wtime() - t0);

        double *global_reward_sum = NULL;
        int    *global_count      = NULL;
        double *all_ep_rewards    = NULL;

        if (rank == 0) {
            global_reward_sum = calloc(table_size, sizeof(double));
            global_count      = calloc(table_size, sizeof(int));
            all_ep_rewards    = malloc(NUM_EPISODES * sizeof(double));
        }

        MPI_Reduce(local_reward_sum, global_reward_sum, table_size,
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_count, global_count, table_size,
                   MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        MPI_Gatherv(local_ep_rewards, local_eps, MPI_DOUBLE,
                    all_ep_rewards, recv_counts, displs,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            for (int i = 0; i < table_size; i++) {
                agg_global_reward_sum[i] += global_reward_sum[i];
                agg_global_count[i] += global_count[i];
            }
            for (int ep = 0; ep < NUM_EPISODES; ep++)
                avg_ep_rewards[ep] += all_ep_rewards[ep];

            free(global_reward_sum);
            free(global_count);
            free(all_ep_rewards);
        }

        free(local_reward_sum);
        free(local_count);
        free(local_ep_rewards);
    }

    double total_elapsed_root = 0.0;
    MPI_Reduce(&total_elapsed_local, &total_elapsed_root, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int ep = 0; ep < NUM_EPISODES; ep++)
            avg_ep_rewards[ep] /= num_runs;

        int    *optimal = malloc(N * sizeof(int));
        double *avg_rew = malloc(table_size * sizeof(double));
        derive_policy(agg_global_reward_sum, agg_global_count, N, optimal, avg_rew);

        double *opt_pct = malloc(N * sizeof(double));
        compute_opt_action_pct(optimal, agg_global_count, N, opt_pct);

        char prefix[64];
        sprintf(prefix, "mpi_%dp", nprocs);
        save_results(prefix, &cfg, avg_ep_rewards, NUM_EPISODES,
                     optimal, opt_pct, avg_rew, total_elapsed_root / num_runs);
        print_grid(&cfg, optimal);

        free(optimal);
        free(avg_rew);
        free(opt_pct);
        free(agg_global_reward_sum);
        free(agg_global_count);
        free(avg_ep_rewards);
    }

    free(recv_counts);
    free(displs);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int grid_sizes[] = {4, 8, 16};
    unsigned int base_seed = 42;
    int num_runs = 10;

    if (argc > 1) {
        num_runs = atoi(argv[1]);
        if (num_runs < 1) num_runs = 1;
    }

    if (rank == 0)
        printf("=== MPI Reinforcement Learning (%d processes, %d averaged runs) ===\n\n",
               nprocs, num_runs);

    for (int i = 0; i < 3; i++)
        run_mpi(grid_sizes[i], base_seed, num_runs, rank, nprocs);

    MPI_Finalize();
    return 0;
}
