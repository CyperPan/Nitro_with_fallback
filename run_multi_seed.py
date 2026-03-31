"""
Multi-seed experiment runner comparing 4 fallback strategies:
  - no_fallback:   Original Nitro, no rollback
  - fallback_A:    Fixed window (5 rounds) + rollback
  - fallback_B:    Fixed window (5 rounds) + conservative only
  - fallback_Adam: Adaptive window/threshold/decay (Adam-style moments) + rollback

Usage:
    python run_multi_seed.py [--seeds 5] [--mode all|no_fallback|fallback_A|fallback_B|fallback_Adam]
"""

import numpy as np
import collections
import logging
import copy
import time
import csv
import os
import sys
import ray
from env import Environment
import config
import utils


# Fallback config
FB_WINDOW = 5          # observation window after boost
FB_GAMMA = 0.9         # discount factor for reward window
FB_DECAY_PENALTY = 0.9 # multiply decay_factor by this per rollback


def compute_discounted_reward(rewards, gamma):
    """Compute discounted cumulative reward: G = sum(gamma^t * r_t)."""
    G = 0.0
    for t, r in enumerate(rewards):
        G += (gamma ** t) * r
    return G


def compute_baseline_expected(baseline_reward, window, gamma):
    """Expected discounted reward if baseline_reward held constant for window rounds."""
    return sum((gamma ** t) * baseline_reward for t in range(window))


def run_one_experiment(scheduler_name, algo_name, env_name, seed, fallback_mode):
    """
    Run a single 50-round experiment.
    fallback_mode: "no_fallback" | "fallback_A" | "fallback_B"
    """
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    import random
    random.seed(seed)

    env = Environment(
        scheduler_name=scheduler_name,
        algo_name=algo_name,
        env_name=env_name,
        target_reward=config.envs[env_name]["max_reward"],
        budget=config.envs[env_name]["budget"],
        stop_min_round=config.stop_min_round,
        stop_max_round=config.stop_max_round,
        stop_num_results=config.stop_num_results,
        stop_cv=config.stop_cv,
        stop_grace_period=config.stop_grace_period,
        is_serverless=True,
        is_local_simulation=True,
    )

    state, mask, info = env.reset()

    csv_rows = []
    round_id = 1
    action = {
        "num_rollout_workers": config.num_rollout_workers_serverful,
        "num_envs_per_worker": config.num_envs_per_worker_serverful,
    }
    hessian_history = {"ratio": collections.deque(maxlen=config.Nitro_sliding_window)}

    # Fallback state
    rollback_count = 0
    pending_boost = None  # for fixed-window strategies (A, B)
    prev_reward = None

    # Adaptive fallback (Adam-style)
    from adaptive_fallback import AdaptiveFallback
    afb = AdaptiveFallback() if fallback_mode == "fallback_Adam" else None

    round_done = False
    while round_done is False:
        is_boosted = action["num_rollout_workers"] > config.num_rollout_workers_serverful
        use_fixed_fb = fallback_mode in ("fallback_A", "fallback_B")
        use_adam_fb = fallback_mode == "fallback_Adam"

        # === Fixed-window fallback (A/B): save checkpoint at boost start ===
        if use_fixed_fb and is_boosted and pending_boost is None:
            pending_boost = {
                "boost_round": round_id,
                "pre_boost_state": copy.deepcopy(env.get_policy_state()),
                "baseline_reward": csv_rows[-1]["eval_reward_mean"] if csv_rows else 0.0,
                "window_rewards": [],
            }
            print(f"[FALLBACK-{fallback_mode}] Round {round_id}: boost started, "
                  f"observing for {FB_WINDOW} rounds")

        # === Adam fallback: save checkpoint at boost start ===
        if use_adam_fb and afb.should_start_observation(is_boosted):
            baseline_r = csv_rows[-1]["eval_reward_mean"] if csv_rows else 0.0
            pre_state = copy.deepcopy(env.get_policy_state())
            win, gam = afb.start_observation(round_id, pre_state, baseline_r)
            status = afb.get_status()
            print(f"[ADAM-FB] Round {round_id}: boost started, "
                  f"adaptive window={win}, gamma={gam:.3f}, "
                  f"threshold={status['threshold']:.2f}, decay={status['decay']:.3f}")

        # Train one round
        next_state, next_mask, reward, done, info = env.step(
            round_id=round_id,
            action=action,
        )

        rollback_triggered = False

        # === Fixed-window fallback: evaluate ===
        if use_fixed_fb and pending_boost is not None:
            pending_boost["window_rewards"].append(info["eval_reward_mean"])

            if len(pending_boost["window_rewards"]) >= FB_WINDOW:
                G_actual = compute_discounted_reward(pending_boost["window_rewards"], FB_GAMMA)
                G_baseline = compute_baseline_expected(
                    pending_boost["baseline_reward"], FB_WINDOW, FB_GAMMA
                )

                if G_actual < G_baseline:
                    rollback_count += 1
                    boost_r = pending_boost["boost_round"]

                    if fallback_mode == "fallback_A":
                        env.set_policy_state(pending_boost["pre_boost_state"])
                        rollback_triggered = True
                        print(f"[FALLBACK-A] Round {round_id}: ROLLBACK to Round {boost_r}! "
                              f"G_actual={G_actual:.1f} < G_baseline={G_baseline:.1f} "
                              f"(total rollbacks: {rollback_count})")
                    else:
                        rollback_triggered = True
                        print(f"[FALLBACK-B] Round {round_id}: boost from Round {boost_r} "
                              f"underperformed. G_actual={G_actual:.1f} < G_baseline={G_baseline:.1f} "
                              f"(total: {rollback_count})")
                else:
                    print(f"[FALLBACK-{fallback_mode}] Round {round_id}: boost OK. "
                          f"G_actual={G_actual:.1f} >= G_baseline={G_baseline:.1f}")

                pending_boost = None

        # === Adam fallback: update moments + evaluate ===
        if use_adam_fb:
            afb.on_round_end(info["eval_reward_mean"], prev_reward)

            if afb.is_observing():
                result = afb.observe(info["eval_reward_mean"])
                if result is not None:  # evaluation complete
                    if not result["success"]:
                        # Rollback
                        env.set_policy_state(result["pre_boost_state"])
                        rollback_triggered = True
                        rollback_count = afb.rollback_count
                        print(f"[ADAM-FB] Round {round_id}: ROLLBACK to Round {result['boost_round']}! "
                              f"G_actual={result['G_actual']:.1f} < G_baseline={result['G_baseline']:.1f} "
                              f"(window={result['window_used']}, gamma={result['gamma_used']:.3f}, "
                              f"success_rate={result['success_rate']:.2f})")
                    else:
                        rollback_count = afb.rollback_count
                        print(f"[ADAM-FB] Round {round_id}: boost OK. "
                              f"G={result['G_actual']:.1f} >= baseline={result['G_baseline']:.1f} "
                              f"(success_rate={result['success_rate']:.2f})")

        prev_reward = info["eval_reward_mean"]

        # Hessian detection
        hessian_eigen_cv, hessian_eigen_ratio = utils.eval_hessian(
            env=env,
            estimate_batch=info["estimate_batch"],
        )
        gns = utils.eval_gns(env=env, estimate_batch=info["estimate_batch"])

        # Boost score calculation
        decay_factor = config.Nitro_decay_factor ** round_id

        if use_fixed_fb and rollback_count > 0:
            effective_decay = decay_factor * (FB_DECAY_PENALTY ** rollback_count)
        elif use_adam_fb:
            effective_decay = decay_factor * (afb.adaptive_decay ** afb.rollback_count)
        else:
            effective_decay = decay_factor

        R = hessian_eigen_ratio * effective_decay
        hessian_history["ratio"].append(R)
        if len(hessian_history["ratio"]) > 1:
            R_max = np.max(hessian_history["ratio"])
            R_min = np.min(hessian_history["ratio"])
            if abs(R_max - R_min) < 1e-10:
                boost_score = 1.0
            else:
                boost_score = (R - R_min) / (R_max - R_min)
        else:
            boost_score = 1.0

        # Apply Adam-style boost score modifier
        if use_adam_fb:
            boost_score = afb.get_boost_score_modifier(boost_score)

        # Force conservative after rollback
        if (use_fixed_fb or use_adam_fb) and rollback_triggered:
            boost_score = 0.0

        # Don't start new boost during observation window
        if use_fixed_fb and pending_boost is not None:
            boost_score = min(boost_score, 0.0)
        if use_adam_fb and afb.is_observing():
            boost_score = min(boost_score, 0.0)

        num_rollout_workers_min = int(np.clip(
            config.num_rollout_workers_max * effective_decay ** round_id,
            config.num_rollout_workers_min,
            config.num_rollout_workers_max,
        ))
        action["num_rollout_workers"] = int(np.clip(
            config.num_rollout_workers_max * boost_score,
            num_rollout_workers_min,
            config.num_rollout_workers_max,
        ))

        hessian_history["ratio"].append(hessian_eigen_ratio)

        row = {
            "round_id": round_id,
            "eval_reward_mean": info["eval_reward_mean"],
            "eval_reward_max": info["eval_reward_max"],
            "eval_reward_min": info["eval_reward_min"],
            "num_rollout_workers": action["num_rollout_workers"],
            "boost_score": boost_score,
            "hessian_eigen_ratio": hessian_eigen_ratio,
            "rollback": rollback_triggered,
            "rollback_count": rollback_count,
            "duration": info["duration"],
            "cost": info["cost"],
        }
        # Add Adam-specific fields
        if use_adam_fb:
            status = afb.get_status()
            row["afb_threshold"] = status["threshold"]
            row["afb_window"] = status["window"]
            row["afb_decay"] = status["decay"]
            row["afb_gamma"] = status["gamma"]
            row["afb_reward_std"] = status["reward_std"]
            row["afb_success_rate"] = status["success_rate"]

        csv_rows.append(row)

        log_extra = ""
        if use_adam_fb:
            s = afb.get_status()
            log_extra = f", w={s['window']}, thr={s['threshold']:.1f}, sr={s['success_rate']:.2f}"

        print(f"  [seed={seed}, {fallback_mode}] "
              f"R{round_id}: reward={info['eval_reward_mean']:.1f}, "
              f"workers={action['num_rollout_workers']}, "
              f"rb={'YES' if rollback_triggered else 'no'}{log_extra}")

        if done:
            env.stop_trainer()
            round_done = True

        state = next_state
        mask = next_mask
        round_id += 1

    return csv_rows


def save_results(all_results, output_dir="logs/multi_seed"):
    """Save results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    for key, runs in all_results.items():
        for seed, rows in runs.items():
            fname = os.path.join(output_dir, f"{key}_seed{seed}.csv")
            with open(fname, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    # Summary CSV per mode
    for key, runs in all_results.items():
        seeds = sorted(runs.keys())
        max_rounds = max(len(runs[s]) for s in seeds)

        fname = os.path.join(output_dir, f"{key}_summary.csv")
        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round_id", "mean_reward", "std_reward", "min_reward", "max_reward", "n_seeds"])
            for r in range(max_rounds):
                rewards = [runs[s][r]["eval_reward_mean"] for s in seeds if r < len(runs[s])]
                writer.writerow([
                    r + 1,
                    np.mean(rewards),
                    np.std(rewards),
                    np.min(rewards),
                    np.max(rewards),
                    len(rewards),
                ])

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--mode", choices=["all", "no_fallback", "fallback_A", "fallback_B", "fallback_Adam"], default="all")
    args = parser.parse_args()

    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    ray.init(log_to_driver=False, configure_logging=True, logging_level=logging.ERROR)

    algo_name = config.algos[0]
    env_name = list(config.envs.keys())[0]
    seeds = list(range(args.seeds))

    modes = {
        "no_fallback": "no_fallback",
        "fallback_A": "fallback_A",
        "fallback_B": "fallback_B",
        "fallback_Adam": "fallback_Adam",
    }
    if args.mode != "all":
        modes = {args.mode: args.mode}

    all_results = {}

    for mode_label, fb_mode in modes.items():
        key = f"Nitro_{mode_label}~{env_name}~{algo_name}"
        all_results[key] = {}

        print(f"\n{'='*60}")
        print(f"  Running: {mode_label.upper()} ({len(seeds)} seeds)")
        print(f"  Window={FB_WINDOW}, gamma={FB_GAMMA}")
        print(f"{'='*60}")

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            rows = run_one_experiment(
                scheduler_name=f"Nitro_{mode_label}",
                algo_name=algo_name,
                env_name=env_name,
                seed=seed,
                fallback_mode=fb_mode,
            )
            all_results[key][seed] = rows

    save_results(all_results)
    ray.shutdown()
    print("\nAll experiments complete!")
