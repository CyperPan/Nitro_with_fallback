"""
Nitro Local Simulation - runs Nitro with local process actors instead of AWS Lambda.
Uses Redis for learner-actor communication, same as the original serverless version.
All scheduling logic (Hessian detection, boost score, dynamic scaling) is identical.

Usage:
    # Start Redis first:
    redis-server --port 6379 --requirepass Nitro --daemonize yes

    # Run:
    python Nitro_local.py
"""

import numpy as np
import collections
import logging
import ray
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from env import Environment
import config
import utils
import time


def Nitro_local(
    scheduler_name,
    algo_name,
    env_name,
):
    # Set up environment with local simulation mode
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

    # Start training
    state, mask, info = env.reset()

    csv_round = [
        [
            "round_id",
            "duration",
            "lambda_duration_max",
            "num_rollout_workers",
            "num_envs_per_worker",
            "episodes_this_iter",
            "learner_time",
            "actor_time",
            "eval_reward_max",
            "eval_reward_mean",
            "eval_reward_min",
            "learner_loss",
            "cost",
            "hessian_eigen_ratio",
            "boost_score",
            "gns",
            "rollback",
        ]
    ]

    round_id = 1

    action = {}
    action["num_rollout_workers"] = config.num_rollout_workers_serverful
    action["num_envs_per_worker"] = config.num_envs_per_worker_serverful

    hessian_history = {}
    hessian_history['ratio'] = collections.deque(maxlen=config.Nitro_sliding_window)

    # Fallback state
    import copy
    prev_eval_reward_mean = None
    rollback_count = 0
    is_boosted = False

    round_done = False
    while round_done is False:
        # Determine if this round is boosted (more actors than baseline)
        is_boosted = action["num_rollout_workers"] > config.num_rollout_workers_serverful

        # Save policy weights before boost for potential rollback
        if is_boosted:
            pre_boost_policy_state = copy.deepcopy(env.get_policy_state())
            print(f"[FALLBACK] Round {round_id}: boosted with {action['num_rollout_workers']} actors, saving checkpoint for rollback")

        next_state, next_mask, reward, done, info = env.step(
            round_id=round_id,
            action=action
        )

        # Fallback: check if boost hurt reward, rollback if so
        rollback_triggered = False
        if is_boosted and prev_eval_reward_mean is not None:
            reward_change = info["eval_reward_mean"] - prev_eval_reward_mean
            if reward_change < -config.ft_rollback_threshold:
                # Boost made things worse — rollback
                env.set_policy_state(pre_boost_policy_state)
                rollback_count += 1
                rollback_triggered = True
                print(f"[FALLBACK] Round {round_id}: ROLLBACK! reward dropped {reward_change:.2f} "
                      f"(threshold: -{config.ft_rollback_threshold}), restoring pre-boost weights "
                      f"(total rollbacks: {rollback_count})")
            else:
                print(f"[FALLBACK] Round {round_id}: boost OK, reward change: {reward_change:+.2f}")

        prev_eval_reward_mean = info["eval_reward_mean"]

        detect_start_time = time.time()
        # Evaluate Hessian eigenvalues
        hessian_eigen_cv, hessian_eigen_ratio = utils.eval_hessian(
            env=env,
            estimate_batch=info['estimate_batch'],
        )
        detect_end_time = time.time()
        print("")
        print("detect overhead: {}".format(detect_end_time - detect_start_time))
        print("")

        # Evaluate gns
        gns = utils.eval_gns(
            env=env,
            estimate_batch=info['estimate_batch']
        )

        # Boost by detecting convexity
        save_checkpoint = False
        decay_factor = config.Nitro_decay_factor**round_id

        # Adaptive decay: increase decay if rollbacks are frequent
        if rollback_count > 0:
            adaptive_decay = decay_factor * (config.ft_rollback_decay_penalty ** rollback_count)
        else:
            adaptive_decay = decay_factor

        R = hessian_eigen_ratio * adaptive_decay
        hessian_history['ratio'].append(R)
        if len(hessian_history['ratio']) > 1:
            # Min-max normalization
            R_max = np.max(hessian_history['ratio'])
            R_min = np.min(hessian_history['ratio'])
            if abs(R_max - R_min) < 1e-10:
                boost_score = 1.0
            else:
                boost_score = (R - R_min) / (R_max - R_min)
        else:
            boost_score = 1.0

        # If last boost was rolled back, force conservative scaling for next round
        if rollback_triggered:
            boost_score = 0.0
            print(f"[FALLBACK] Forcing conservative scaling for next round (boost_score=0)")

        num_rollout_workers_min = int(np.clip(
            config.num_rollout_workers_max * adaptive_decay**round_id,
            config.num_rollout_workers_min,
            config.num_rollout_workers_max,
        ))

        # Scale actors proportional to boost score
        action["num_rollout_workers"] = int(np.clip(
            config.num_rollout_workers_max * boost_score,
            num_rollout_workers_min,
            config.num_rollout_workers_max,
        ))

        hessian_history['ratio'].append(hessian_eigen_ratio)

        csv_round.append(
            [
                round_id,
                info["duration"],
                info["lambda_duration_max"],
                action["num_rollout_workers"],
                action["num_envs_per_worker"],
                info["episodes_this_iter"],
                info["learner_time"],
                info["actor_time"],
                info["eval_reward_max"],
                info["eval_reward_mean"],
                info["eval_reward_min"],
                info["learner_loss"],
                info["cost"],
                hessian_eigen_ratio,
                boost_score,
                gns,
                rollback_triggered,
            ]
        )

        print("")
        print("******************")
        print("  LOCAL SIMULATION")
        print("******************")
        print("")
        print("Running {}, algo {}, env {}".format(scheduler_name, algo_name, env_name))
        print("round_id: {}".format(info["round_id"]))
        print("duration: {}".format(info["duration"]))
        print("action: {}".format(action))
        print("eval_reward_mean: {}".format(info["eval_reward_mean"]))
        print("hessian_eigen_ratio: {}".format(hessian_eigen_ratio))
        print("boost_score: {}".format(boost_score))
        print("gns: {}".format(gns))
        print("cost: {}".format(info["cost"]))
        print("rollback: {}".format(rollback_triggered))
        print("total_rollbacks: {}".format(rollback_count))

        if done:
            utils.export_csv(
                scheduler_name=scheduler_name,
                env_name=env_name,
                algo_name=algo_name,
                csv_name="traj",
                csv_file=csv_round
            )

            env.stop_trainer()
            round_done = True

        state = next_state
        mask = next_mask
        round_id = round_id + 1


if __name__ == '__main__':
    scheduler_name = "Nitro_local"

    # Use 'spawn' so actor subprocesses don't inherit parent's Ray state
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # already set

    print("")
    print("**********")
    print("  Nitro Local Simulation")
    print("  (no AWS Lambda required)")
    print("**********")
    print("")
    ray.init(
        log_to_driver=False,
        configure_logging=True,
        logging_level=logging.ERROR
    )

    for algo_name in config.algos:
        for env_name in config.envs.keys():
            Nitro_local(
                scheduler_name=scheduler_name,
                algo_name=algo_name,
                env_name=env_name,
            )

    ray.shutdown()
    print("")
    print("**********")
    print("  Done!")
    print("**********")
