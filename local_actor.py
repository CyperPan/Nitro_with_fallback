"""
Local process-based actor that simulates AWS Lambda serverless actors.
Uses lightweight PyTorch inference instead of full Ray RolloutWorker.
Communicates with the learner via Redis, same as the original Lambda actors.
"""

import time
import uuid
import pickle
import numpy as np
import redis
import gymnasium as gym
import torch

import config

import warnings
warnings.filterwarnings("ignore")


def local_actor_run(
    redis_host,
    redis_port,
    redis_password,
    algo_name,
    env_name,
    num_envs_per_worker,
    rollout_fragment_length,
    simulate_cold_start=False,
):
    """
    Runs a single actor in a local process, mimicking Lambda handler behavior.
    Uses direct PyTorch policy inference instead of Ray RolloutWorker.
    """
    start_time = time.time()

    # Simulate Lambda cold start delay
    if simulate_cold_start and config.local_cold_start_delay > 0:
        time.sleep(config.local_cold_start_delay)

    # Generate a unique request ID (replaces aws_request_id)
    request_id = str(uuid.uuid4())

    # Connect to Redis
    pool = redis.ConnectionPool(
        host=redis_host,
        port=redis_port,
        password=redis_password,
    )
    redis_client = redis.Redis(connection_pool=pool)

    # Wait for model weights to be available
    for _ in range(60):
        if redis_client.exists("model_weights"):
            break
        time.sleep(1)

    # We need Ray for SampleBatch and RolloutWorker - but init a minimal local instance
    import ray
    from ray.rllib.evaluation.rollout_worker import RolloutWorker
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy

    # Initialize a minimal Ray instance in this subprocess (clean due to 'spawn')
    ray.init(
        num_cpus=1,
        num_gpus=0,
        include_dashboard=False,
        logging_level="ERROR",
    )

    try:
        # Init environment
        env = gym.make(env_name)

        # Init sampler config
        sampler_config = PPOConfig()
        sampler_config = (
            sampler_config
            .framework(framework=config.framework)
            .environment(
                env=env_name,
                observation_space=env.observation_space,
                action_space=env.action_space,
            )
            .rollouts(
                rollout_fragment_length=rollout_fragment_length,
                num_rollout_workers=0,
                num_envs_per_worker=num_envs_per_worker,
                batch_mode="truncate_episodes",
            )
            .training(
                train_batch_size=rollout_fragment_length,
            )
            .debugging(
                log_level="ERROR",
            )
        )

        # Init worker
        worker = RolloutWorker(
            env_creator=lambda _: gym.make(env_name),
            config=sampler_config,
            default_policy_class=PPOTorchPolicy,
        )

        # Fetch model weights from Redis and set
        model_weights = pickle.loads(redis_client.get("model_weights"))
        worker.get_policy().set_weights(model_weights)

        # Sample trajectories
        sample_batch = worker.sample()

        # Record time
        end_time = time.time()
        lambda_duration = end_time - start_time

        # Store results in Redis (same keys as original Lambda)
        redis_client.hset("sample_batch", request_id, pickle.dumps(sample_batch))
        redis_client.hset("lambda_duration", request_id, lambda_duration)

    finally:
        ray.shutdown()
