import numpy as np
import collections
import logging
import ray
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from env import Environment
import config
import utils


def serverful_baseline(
    scheduler_name,
    is_serverless,
    algo_name,
    env_name,
):
    # Set up environment and load checkpoint
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
        is_serverless=is_serverless,
    )

    # Start training
    state, mask, info = env.reset()

    csv_round = [
        [
            "round_id",
            "duration",
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
            "gns",
        ]
    ]

    round_id = 1

    action = {}
    action["num_rollout_workers"] = config.num_rollout_workers_serverful
    action["num_envs_per_worker"] = config.num_envs_per_worker_serverful

    hessian_history = {}
    hessian_history['ratio'] = collections.deque(maxlen=config.Nitro_sliding_window)

    round_done = False
    while round_done is False:
        next_state, next_mask, reward, done, info = env.step(
            round_id=round_id,
            action=action
        )

        # Evaluate Hessian eigenvalues
        hessian_eigen_cv, hessian_eigen_ratio = utils.eval_hessian(
            env=env,
            estimate_batch=info['estimate_batch'],
        )
        hessian_z_score = 0

        # Evaluate gns
        gns = utils.eval_gns(
            env=env,
            estimate_batch=info['estimate_batch']
        )

        save_checkpoint = False
        if algo_name == "ppo":
            # if len(hessian_history['ratio']) > 0:
            #     scale_ratio = np.min(hessian_history['ratio']) / hessian_eigen_ratio
            #     if scale_ratio >= config.Nitro_delta_ratio:
            #         save_checkpoint = True
            #         print("")
            #         print("Save checkpoint for round {}!".format(round_id))
            #         print("")
            save_checkpoint = True

        # Save checkpoints if boost
        if save_checkpoint:
            ckpt_path = "{}/{}~{}~{}~{}".format(config.ckpt_path, scheduler_name, env_name, algo_name, round_id)
            env.save(ckpt_path)
            pickle_path = "{}/{}~{}~{}~{}.pkl".format(config.ckpt_path, scheduler_name, env_name, algo_name, round_id)
            utils.pickle_save(info['estimate_batch'], pickle_path)
            json_data = {
                "round_id": round_id,
                "hessian_eigen_ratio": hessian_eigen_ratio,
            }
            json_path = pickle_path = "{}/{}~{}~{}~{}.json".format(config.ckpt_path, scheduler_name, env_name, algo_name, round_id)
            utils.json_save(json_data, json_path)
                
        hessian_history['ratio'].append(hessian_eigen_ratio)

        csv_round.append(
            [
                round_id,
                info["duration"],
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
                gns,
            ]
        )

        print("")
        print("******************")
        print("******************")
        print("******************")
        print("")
        print("Running {}, algo {}, env {}".format(scheduler_name, algo_name, env_name))
        print("round_id: {}".format(info["round_id"]))
        print("duration: {}".format(info["duration"]))
        print("action: {}".format(action))
        print("eval_reward_mean: {}".format(info["eval_reward_mean"]))
        print("hessian_eigen_cv: {}".format(hessian_eigen_cv))
        print("hessian_eigen_ratio: {}".format(hessian_eigen_ratio))
        print("hessian_z_score: {}".format(hessian_z_score))
        print("gns: {}".format(gns))
        print("cost: {}".format(info["cost"]))

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
    scheduler_name = "serverful_baseline"
    is_serverless = False

    print("")
    print("**********")
    print("**********")
    print("**********")
    print("")
    ray.init(
        log_to_driver=False,
        configure_logging=True,
        logging_level=logging.ERROR
    )

    for algo_name in config.algos:
        for env_name in config.envs.keys():
            serverful_baseline(
                scheduler_name=scheduler_name,
                is_serverless=is_serverless,
                algo_name=algo_name,
                env_name=env_name,
            )

    ray.shutdown()
    print("")
    print("**********")
    print("**********")
    print("**********")