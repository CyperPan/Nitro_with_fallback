import socket
import torch.nn as nn

# Ray cluster
# For local simulation on HPC: reduce max workers to match available CPUs
# Original values: serverful=16, min=8, max=64
num_rollout_workers_serverful = 1
num_rollout_workers_min = 1
num_rollout_workers_max = 4
num_envs_per_worker_serverful = 1
num_envs_per_worker_min = 1
num_envs_per_worker_max = 1
num_cpus_for_local_worker = 1
num_gpus_for_local_worker = 1
num_cpus_per_worker = 1
num_gpus_per_worker = 0
evaluation_num_workers = 2
min_time_s_per_iteration = None
_enable_new_api_stack = False
learner_queue_timeout = 114514

# Redis
redis_host = "127.0.0.1"
redis_port = 6379
redis_password = "Nitro"

# Local simulation (set to True to use local processes instead of AWS Lambda)
local_simulation = True
local_simulate_cold_start = False  # Set True to add artificial cold start delay
local_cold_start_delay = 5.0  # Simulated Lambda cold start in seconds

# Training
max_exp = 1
framework = "torch"
envs = {
    "Hopper-v4": {
        "is_env_discrete": False,
        "min_reward": 0,
        "max_reward": 600,
        "budget": float("inf"),
        "rollout_fragment_length": 512,
        "eval_convex_ratio": [1, 6, 8],
        "eval_boost_efficiency": [1, 11, 28, 39, 49],
    },
    # "Humanoid-v3": {
    #     "is_env_discrete": False,
    #     "min_reward": 0,
    #     "max_reward": 1000,
    #     "budget": float("inf"),
    #     "rollout_fragment_length": 512,
    #     "eval_convex_ratio": [5, 8, 9],
    #     "eval_boost_efficiency": [8, 20, 25, 40, 46],
    # },
    # "Walker2d-v3": {
    #     "is_env_discrete": False,
    #     "min_reward": 0,
    #     "max_reward": 1000,
    #     "budget": float("inf"),
    #     "rollout_fragment_length": 512,
    #     "eval_convex_ratio": [2, 4, 9],
    #     "eval_boost_efficiency": [4, 19, 22, 35, 50],
    # },
    # "GravitarNoFrameskip-v4": {
    #     "is_env_discrete": True,
    #     "min_reward": 0,
    #     "max_reward": 300,
    #     "budget": float("inf"),
    #     "rollout_fragment_length": 16,
    #     "eval_convex_ratio": [4, 7, 9],
    #     "eval_boost_efficiency": [7, 17, 24, 39, 45],
    # },
    # "SpaceInvadersNoFrameskip-v4": {
    #     "is_env_discrete": True,
    #     "min_reward": 0,
    #     "max_reward": 1200,
    #     "budget": float("inf"),
    #     "rollout_fragment_length": 16,
    #     "eval_convex_ratio": [1, 4, 8],
    #     "eval_boost_efficiency": [1, 20, 26, 40, 48],
    # },
    # "QbertNoFrameskip-v4": {
    #     "is_env_discrete": True,
    #     "min_reward": 0,
    #     "max_reward": 300,
    #     "budget": float("inf"),
    #     "rollout_fragment_length": 16,
    #     "eval_convex_ratio": [1, 3, 6],
    #     "eval_boost_efficiency": [1, 17, 22, 39, 46],
    # },
}
algos = [
    "ppo",
    # "appo",
    # "impala",
]

# MinionsRL deprecated
max_episode_train = 100
max_episode_eval = 1
learning_rate = 0.001
state_dim = 8
action_dim = num_rollout_workers_max
hidden_dims = [64, 64]
activation = nn.Tanh()
discount_factor = 0.99
ppo_clip = 0.2
ppo_epoch = 1
value_loss_coef = 0.5
entropy_coef = 0.01
beta = 0.1
model_save_path = "pth/"
reward_window_size = 3

# Critical period
recover_round = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
actor_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Stop criteria
stop_num_results = 5
stop_cv = 0.0001
stop_grace_period = stop_num_results
stop_min_round = 1
stop_max_round = 50

# PPO to be trained
clip_param = 0.2
num_sgd_iter = 32
vf_clip_param = 100
entropy_coeff = 0.01
kl_coeff = 0

# Evaluate
evaluation_interval = 1

# PAC
delta = 5e-2
epsilon = 1e-6
alpha_min = 1e-4
num_prev_round = 1
verify_baselines = [1, 10, 20, 30, 40, 50]
verify_loop_time = 1

# Vector envs
actors_list = [64, 32, 16, 8, 4, 2, 1]
envs_list = [1, 2, 4, 8, 16, 32, 64]

# GNS window
gns_ewma_alpha = 0.9
gns_boost_score = 1.0

# Nitro
Nitro_cv = 20
Nitro_sliding_window = 6
Nitro_decay_factor = 0.96
Nitro_boost_eval_time = 6
Nitro_boost_candidates = [
    # [16, 4], 
    # [16, 8], 
    # [16, 12], 
    [16, 16], 
]
Nitro_ckpt_boost_score = 0

# Reward surfaces
grid_size = 15
log_path = "./logs"
ckpt_path = "./ckpt"
plot_path = "./plot"

# Pricing units
serverless_learner_per_s = (3.0600 + 17.92 / 30 / 24) / 60 / 60 # vm + ip + disk
serverless_actor_per_s = 0.0000000167 * 1000  # mem + network
server_learner_per_s = (3.0600 + 17.92 / 30 / 24) / 60 / 60 # vm + ip + disk
server_actor_per_s = (0.68 + 17.92 / 30 / 24) / 60 / 60 # vm + ip + disk
# server_actor_per_s = (1.53 + 17.92 / 30 / 24) / 60 / 60 # vm + ip + disk
server_startup_time = 60 # second

# Fallback / Fault tolerance
ft_rollback_threshold = 10.0       # rollback if reward drops more than this after boost
ft_rollback_decay_penalty = 0.9    # multiply decay_factor by this per rollback (makes future boosts more conservative)
