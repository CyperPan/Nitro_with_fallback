import pickle
import redis
import gymnasium as gym
import ray
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.appo.appo_torch_policy import APPOTorchPolicy
from ray.rllib.algorithms.impala.impala import ImpalaConfig
from ray.rllib.algorithms.impala.impala_torch_policy import ImpalaTorchPolicy
import config

import warnings
warnings.filterwarnings("ignore")


class ServerlessActor:
    '''
    Actor packaged as serverless function
    '''
    def __init__(
        self,
        redis_host,
        redis_port,
        redis_password,
        algo_name,
        env_name,
        num_envs_per_worker,
        rollout_fragment_length,
    ):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_password = redis_password
        self.algo_name = algo_name
        self.env_name = env_name
        self.rollout_fragment_length = rollout_fragment_length

        # Init environment explicitly
        env = gym.make(env_name)

        # Init sampler config
        if self.algo_name == "ppo":
            sampler_config = PPOConfig()
        elif self.algo_name == "appo":
            sampler_config = APPOConfig()
        elif self.algo_name == "impala":
            sampler_config = ImpalaConfig()
        
        sampler_config = (
            sampler_config
            .framework(framework=config.framework)
            .environment(
                env=env_name,
                observation_space=env.observation_space,
                action_space=env.action_space,
                # disable_env_checking=True,
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
                logger_config={"type": ray.tune.logger.NoopLogger},
                log_sys_usage=False
            ) # Disable all loggings to save time
        )

        # Init worker
        if self.algo_name == "ppo":
            self.worker = RolloutWorker(
                env_creator=lambda _ : env,
                config=sampler_config,
                default_policy_class=PPOTorchPolicy,
            )
        elif self.algo_name == "appo":
            self.worker = RolloutWorker(
                env_creator=lambda _ : env,
                config=sampler_config,
                default_policy_class=APPOTorchPolicy,
            )
        else:
            self.worker = RolloutWorker(
                env_creator=lambda _ : env,
                config=sampler_config,
                default_policy_class=ImpalaTorchPolicy,
            )

    def init_redis_client(self):
        self.pool = redis.ConnectionPool(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_password,
        )
        self.redis_client = redis.Redis(connection_pool=self.pool)

    def redis_hset_sample_batch(self, name, batch_id, batch):
        self.redis_client.hset(name, batch_id, pickle.dumps(batch))
    
    def redis_hset_lambda_duration(self, name, batch_id, lambda_duration):
        self.redis_client.hset(name, batch_id, lambda_duration)
        
    def sample(self):
        return self.worker.sample()
    
    def redis_get_model_weights(self):
        return pickle.loads(self.redis_client.get("model_weights"))

    def set_model_weights(self, model_weights):
        self.worker.get_policy().set_weights(model_weights)
