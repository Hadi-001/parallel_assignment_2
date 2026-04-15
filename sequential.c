#include "gridworld.h"

#define RUN_SEED_STRIDE 100000U

static void run_sequential(int grid_size, unsigned int base_seed, int num_runs) {
    GridConfig cfg;
    get_grid_config(grid_size, &cfg);
    int N = grid_size * grid_size;
    int table_size = N * NUM_ACTIONS;

    double *agg_reward_sum = calloc(table_size, sizeof(double));
    int    *agg_count      = calloc(table_size, sizeof(int));
    double *avg_ep_rewards = calloc(NUM_EPISODES, sizeof(double));

    double total_time = 0.0;

    for (int run = 0; run < num_runs; run++) {
        double *run_reward_sum = calloc(table_size, sizeof(double));
        int    *run_count      = calloc(table_size, sizeof(int));

        double t0 = get_time();
        for (int ep = 0; ep < NUM_EPISODES; ep++) {
            RNG rng;
            rng_seed(&rng, base_seed + (unsigned int)(run * RUN_SEED_STRIDE + ep));
            avg_ep_rewards[ep] += run_episode(&cfg, run_reward_sum, run_count, &rng);
        }
        total_time += (get_time() - t0);

        for (int i = 0; i < table_size; i++) {
            agg_reward_sum[i] += run_reward_sum[i];
            agg_count[i] += run_count[i];
        }

        free(run_reward_sum);
        free(run_count);
    }

    for (int ep = 0; ep < NUM_EPISODES; ep++)
        avg_ep_rewards[ep] /= num_runs;

    int    *optimal = malloc(N * sizeof(int));
    double *avg_rew = malloc(table_size * sizeof(double));
    derive_policy(agg_reward_sum, agg_count, N, optimal, avg_rew);

    double *opt_pct = malloc(N * sizeof(double));
    compute_opt_action_pct(optimal, agg_count, N, opt_pct);

    save_results("sequential", &cfg, avg_ep_rewards, NUM_EPISODES,
                 optimal, opt_pct, avg_rew, total_time / num_runs);
    print_grid(&cfg, optimal);

    free(agg_reward_sum);
    free(agg_count);
    free(avg_ep_rewards);
    free(optimal);
    free(avg_rew);
    free(opt_pct);
}

int main(int argc, char *argv[]) {
    int grid_sizes[] = {4, 8, 16};
    unsigned int base_seed = 42;
    int num_runs = 10;

    if (argc > 1) {
        num_runs = atoi(argv[1]);
        if (num_runs < 1) num_runs = 1;
    }

    printf("=== Sequential Reinforcement Learning (%d averaged runs) ===\n\n", num_runs);

    for (int i = 0; i < 3; i++)
        run_sequential(grid_sizes[i], base_seed, num_runs);

    return 0;
}
