#ifndef GRIDWORLD_H
#define GRIDWORLD_H

#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define NUM_ACTIONS   4
#define ACTION_UP     0
#define ACTION_DOWN   1
#define ACTION_LEFT   2
#define ACTION_RIGHT  3

#define EPSILON       0.1
#define MAX_STEPS     50
#define NUM_EPISODES  1000

static const char *ACTION_NAMES[] = {"Up", "Down", "Left", "Right"};

/* ------------------------------------------------------------------ */
/*  Thread-safe RNG (xorshift32)                                       */
/* ------------------------------------------------------------------ */

typedef struct {
    unsigned int state;
} RNG;

static inline void rng_seed(RNG *rng, unsigned int seed) {
    rng->state = seed ? seed : 1;
}

static inline unsigned int rng_next(RNG *rng) {
    unsigned int x = rng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng->state = x;
    return x;
}

/* uniform int in [0, max) */
static inline int rng_int(RNG *rng, int max) {
    return (int)(rng_next(rng) % (unsigned int)max);
}

/* uniform double in [lo, hi) */
static inline double rng_double(RNG *rng, double lo, double hi) {
    double t = (rng_next(rng) & 0x7FFFFFFF) / (double)0x7FFFFFFF;
    return lo + t * (hi - lo);
}

/* ------------------------------------------------------------------ */
/*  Grid configuration                                                 */
/* ------------------------------------------------------------------ */

#define MAX_OBSTACLES 16

typedef struct {
    int size;
    int num_obstacles;
    int obstacles[MAX_OBSTACLES];
    int goal;
    int trap;
} GridConfig;

static inline void get_grid_config(int grid_size, GridConfig *cfg) {
    int N = grid_size * grid_size;
    const char *goal_mode = getenv("GOAL_MODE");
    int center_is_obstacle = 0;
    int trap_is_obstacle = 0;
    cfg->size = grid_size;

    if (grid_size == 4) {
        cfg->num_obstacles = 2;
        cfg->obstacles[0] = 2;  cfg->obstacles[1] = 5;
        cfg->goal = 15;
        cfg->trap = 12;
    } else if (grid_size == 8) {
        cfg->num_obstacles = 4;
        cfg->obstacles[0] = 10; cfg->obstacles[1] = 21;
        cfg->obstacles[2] = 42; cfg->obstacles[3] = 53;
        cfg->goal = 63;
        cfg->trap = 56;
    } else if (grid_size == 16) {
        cfg->num_obstacles = 8;
        cfg->obstacles[0] = 18;  cfg->obstacles[1] = 37;
        cfg->obstacles[2] = 74;  cfg->obstacles[3] = 93;
        cfg->obstacles[4] = 122; cfg->obstacles[5] = 157;
        cfg->obstacles[6] = 198; cfg->obstacles[7] = 221;
        cfg->goal = 255;
        cfg->trap = 240;
    } else {
        cfg->num_obstacles = 0;
        cfg->goal = N - 1;
        cfg->trap = N - grid_size;
    }

    if (goal_mode != NULL && strcmp(goal_mode, "center") == 0) {
        int center = (grid_size / 2) * grid_size + (grid_size / 2);
        for (int i = 0; i < cfg->num_obstacles; i++) {
            if (cfg->obstacles[i] == center) {
                center_is_obstacle = 1;
                break;
            }
        }
        if (!center_is_obstacle) {
            cfg->goal = center;
        }

        for (int i = 0; i < cfg->num_obstacles; i++) {
            if (cfg->obstacles[i] == cfg->trap) {
                trap_is_obstacle = 1;
                break;
            }
        }
        if (cfg->trap == cfg->goal || trap_is_obstacle) {
            cfg->trap = N - grid_size; /* bottom-left fallback */
        }
        trap_is_obstacle = 0;
        for (int i = 0; i < cfg->num_obstacles; i++) {
            if (cfg->obstacles[i] == cfg->trap) {
                trap_is_obstacle = 1;
                break;
            }
        }
        if (cfg->trap == cfg->goal || trap_is_obstacle) {
            for (int s = 0; s < N; s++) {
                int s_is_obstacle = 0;
                for (int i = 0; i < cfg->num_obstacles; i++) {
                    if (cfg->obstacles[i] == s) {
                        s_is_obstacle = 1;
                        break;
                    }
                }
                if (s != cfg->goal && !s_is_obstacle) {
                    cfg->trap = s;
                    break;
                }
            }
        }
    }
}

static inline int is_obstacle(const GridConfig *cfg, int state) {
    for (int i = 0; i < cfg->num_obstacles; i++)
        if (cfg->obstacles[i] == state) return 1;
    return 0;
}

static inline int is_terminal(const GridConfig *cfg, int state) {
    return state == cfg->goal || state == cfg->trap;
}

/* ------------------------------------------------------------------ */
/*  Environment dynamics                                               */
/* ------------------------------------------------------------------ */

static inline int grid_step(const GridConfig *cfg, int state, int action) {
    int r = state / cfg->size;
    int c = state % cfg->size;
    int nr = r, nc = c;

    switch (action) {
        case ACTION_UP:    nr = r - 1; break;
        case ACTION_DOWN:  nr = r + 1; break;
        case ACTION_LEFT:  nc = c - 1; break;
        case ACTION_RIGHT: nc = c + 1; break;
    }

    if (nr < 0 || nr >= cfg->size || nc < 0 || nc >= cfg->size)
        return state;

    int next = nr * cfg->size + nc;
    if (is_obstacle(cfg, next))
        return state;

    return next;
}

static inline double get_reward(const GridConfig *cfg, int next_state, RNG *rng) {
    if (next_state == cfg->goal) return 1.0;
    if (next_state == cfg->trap) return -1.0;
    return rng_double(rng, -0.1, -0.05);
}

/* ------------------------------------------------------------------ */
/*  Epsilon-greedy action selection                                    */
/* ------------------------------------------------------------------ */

static inline int choose_action(const double *reward_sum, const int *count,
                                int state, int N, RNG *rng) {
    if (rng_double(rng, 0.0, 1.0) < EPSILON)
        return rng_int(rng, NUM_ACTIONS);

    double best_avg = -1e18;
    int best_action = 0;
    for (int a = 0; a < NUM_ACTIONS; a++) {
        int idx = state * NUM_ACTIONS + a;
        double avg = (count[idx] > 0) ? reward_sum[idx] / count[idx] : 0.0;
        if (avg > best_avg) {
            best_avg = avg;
            best_action = a;
        }
    }
    return best_action;
}

/* ------------------------------------------------------------------ */
/*  Run a single episode                                               */
/* ------------------------------------------------------------------ */

static inline double run_episode(const GridConfig *cfg,
                                 double *reward_sum, int *count,
                                 RNG *rng) {
    int N = cfg->size * cfg->size;
    double total_reward = 0.0;

    /* build list of valid start states */
    int valid[N];
    int num_valid = 0;
    for (int s = 0; s < N; s++)
        if (!is_obstacle(cfg, s) && !is_terminal(cfg, s))
            valid[num_valid++] = s;

    int state = valid[rng_int(rng, num_valid)];

    for (int t = 0; t < MAX_STEPS; t++) {
        int action = choose_action(reward_sum, count, state, N, rng);
        int next = grid_step(cfg, state, action);
        double reward = get_reward(cfg, next, rng);

        int idx = state * NUM_ACTIONS + action;
        count[idx] += 1;
        reward_sum[idx] += reward;
        total_reward += reward;

        state = next;
        if (is_terminal(cfg, state))
            break;
    }
    return total_reward;
}

/* ------------------------------------------------------------------ */
/*  Derive optimal policy                                              */
/* ------------------------------------------------------------------ */

static inline void derive_policy(const double *reward_sum, const int *count,
                                 int N, int *optimal_actions,
                                 double *avg_reward) {
    for (int s = 0; s < N; s++) {
        double best_avg = -1e18;
        int best_action = 0;
        for (int a = 0; a < NUM_ACTIONS; a++) {
            int idx = s * NUM_ACTIONS + a;
            avg_reward[idx] = (count[idx] > 0)
                              ? reward_sum[idx] / count[idx]
                              : 0.0;
            if (avg_reward[idx] > best_avg) {
                best_avg = avg_reward[idx];
                best_action = a;
            }
        }
        optimal_actions[s] = best_action;
    }
}

static inline void compute_opt_action_pct(const int *optimal_actions,
                                          const int *count, int N,
                                          double *opt_pct) {
    for (int s = 0; s < N; s++) {
        int total = 0;
        for (int a = 0; a < NUM_ACTIONS; a++)
            total += count[s * NUM_ACTIONS + a];
        if (total > 0)
            opt_pct[s] = (double)count[s * NUM_ACTIONS + optimal_actions[s]] / total;
        else
            opt_pct[s] = 0.0;
    }
}

/* ------------------------------------------------------------------ */
/*  Save results to CSV files in output/                               */
/* ------------------------------------------------------------------ */

static inline void save_results(const char *prefix,
                                const GridConfig *cfg,
                                const double *episode_rewards,
                                int num_episodes,
                                const int *optimal_actions,
                                const double *opt_pct,
                                const double *avg_reward,
                                double elapsed) {
    int N = cfg->size * cfg->size;
    char filename[256];

    /* rewards CSV */
    sprintf(filename, "output/%s_%dx%d_rewards.csv", prefix, cfg->size, cfg->size);
    FILE *f = fopen(filename, "w");
    fprintf(f, "episode,reward\n");
    for (int i = 0; i < num_episodes; i++)
        fprintf(f, "%d,%.6f\n", i + 1, episode_rewards[i]);
    fclose(f);

    /* policy CSV */
    sprintf(filename, "output/%s_%dx%d_policy.csv", prefix, cfg->size, cfg->size);
    f = fopen(filename, "w");
    fprintf(f, "state,optimal_action,action_name");
    for (int a = 0; a < NUM_ACTIONS; a++)
        fprintf(f, ",avg_reward_%s", ACTION_NAMES[a]);
    fprintf(f, ",opt_action_pct\n");
    for (int s = 0; s < N; s++) {
        fprintf(f, "%d,%d,%s", s, optimal_actions[s], ACTION_NAMES[optimal_actions[s]]);
        for (int a = 0; a < NUM_ACTIONS; a++)
            fprintf(f, ",%.6f", avg_reward[s * NUM_ACTIONS + a]);
        fprintf(f, ",%.6f\n", opt_pct[s]);
    }
    fclose(f);

    /* timing CSV */
    sprintf(filename, "output/%s_%dx%d_timing.csv", prefix, cfg->size, cfg->size);
    f = fopen(filename, "w");
    fprintf(f, "grid_size,episodes,time_seconds\n");
    fprintf(f, "%d,%d,%.6f\n", cfg->size, num_episodes, elapsed);
    fclose(f);

    /* console summary */
    double avg = 0.0;
    for (int i = 0; i < num_episodes; i++) avg += episode_rewards[i];
    avg /= num_episodes;
    printf("[%s_%dx%d] Time: %.4fs | Avg reward: %.4f\n",
           prefix, cfg->size, cfg->size, elapsed, avg);
}

/* ------------------------------------------------------------------ */
/*  Print grid policy to console                                       */
/* ------------------------------------------------------------------ */

static inline void print_grid(const GridConfig *cfg, const int *optimal_actions) {
    const char *arrows[] = {"^", "v", "<", ">"};
    printf("Grid %dx%d Policy:\n", cfg->size, cfg->size);
    for (int r = 0; r < cfg->size; r++) {
        for (int c = 0; c < cfg->size; c++) {
            int s = r * cfg->size + c;
            if (s == cfg->goal)          printf(" G ");
            else if (s == cfg->trap)     printf(" T ");
            else if (is_obstacle(cfg, s)) printf(" X ");
            else                         printf(" %s ", arrows[optimal_actions[s]]);
        }
        printf("\n");
    }
    printf("\n");
}

/* ------------------------------------------------------------------ */
/*  High-resolution timer                                              */
/* ------------------------------------------------------------------ */

static inline double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#endif /* GRIDWORLD_H */
