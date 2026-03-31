import ray
from ray.rllib.utils.numpy import convert_to_numpy
import torch
import numpy as np
import csv
import config
import os
import pickle
import json
from pyhessian.hessian import hessian
from statsmodels.stats.weightstats import DescrStatsW
import copy
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def remove_outliers(x, m=2):
    data = np.array(x)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0
    return data[s<m].tolist()

def scale(x, src, dst):
    """
    Scale the given value from the scale of src to the scale of dst.
    """
    return ((x - src[0]) / (src[1]-src[0])) * (dst[1]-dst[0]) + dst[0]

def cv(x):
    return abs(np.std(x) / np.mean(x))

def z_score(l, x):
    return (x - np.mean(l)) / np.std(l)

def fuse(grad_list):
    return torch.concat([torch.reshape(grad, (-1,)) for grad in grad_list], -1)

def process_logp_ratio(logp_ratio):
    processed_logp_ratio = []
    for ratio in logp_ratio.tolist():
        ratio_abs = np.abs(ratio - 1)
        processed_logp_ratio.append(ratio_abs)
    # processed_logp_ratio = remove_outliers(processed_logp_ratio)
    
    logp_ratio_min = np.min(processed_logp_ratio)
    logp_ratio_mean = np.mean(processed_logp_ratio)
    logp_ratio_max = np.max(processed_logp_ratio)

    return logp_ratio_min, logp_ratio_mean, logp_ratio_max

# def pac_m(delta, epsilon, prev_m_list, prev_j_list, bound):
#     if len(prev_m_list) == 0 or len(prev_j_list) == 0:
#         m = None
#     else:
#         if np.var(prev_j_list) == 0:
#             alpha = config.alpha_min
#         else:
#             alpha = np.sqrt(2 / (np.var(prev_j_list) * len(prev_j_list)))
#         prev_m_sum = 0
#         prev_m_z_sum = 0
#         j_max = max(prev_j_list)
#         for i, (prev_m, prev_j) in enumerate(zip(prev_m_list, prev_j_list)):
#             prev_m_sum = prev_m_sum + prev_m
#             prev_m_z_sum = prev_m_z_sum + prev_m * np.power(prev_j * (i+2) * (1+bound), 2)

#         m = -(2 * (np.log(delta) + alpha * epsilon * prev_m_sum) - np.power(alpha, 2) * prev_m_z_sum) / (np.power(alpha, 2) * np.power(j_max * (1+bound), 2) - 2 * alpha * epsilon)

#         print("alpha: {}".format(alpha))
#         print("prev_m_sum: {}".format(prev_m_sum))
#         print("prev_m_z_sum: {}".format(prev_m_z_sum))

#     return m

def pac_m(delta, epsilon, prev_j_list, j_k_sum, beta, alpha_scaling):
    if len(prev_j_list) == 0 or j_k_sum is None:
        m = None
    else:
        if np.var(prev_j_list) == 0:
            alpha = config.alpha_min
        else:
            alpha = np.sqrt(2 / ((np.var(prev_j_list))*len(prev_j_list)))
        alpha = alpha * alpha_scaling
        j_k = j_k_sum / len(prev_j_list)
        m = abs((2 * np.log(delta)) / (np.power(alpha * j_k * (1-beta), 2) - 2 * alpha * (epsilon*j_k)))

        print("alpha: {}".format(alpha))
        print("j_k: {}".format(j_k))
        print("m: {}".format(m))

    return m

def compute_hessian(model):
    model.eval()
    hessian = torch.func.hessian(model)(
        tuple([_.view(-1).cpu().detach() for _ in model.parameters()])
    )
    model.train()
    return hessian

#
# Reward surfaces
#

def pickle_save(
    data,
    file_path
):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def pickle_load(
    file_path
):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def json_save(
    data,
    file_path
):
    with open(file_path, "w") as f:
        json.dump(data, f)

def json_load(
    file_path
):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def estimate_hessian_eigens(
    model, 
    criterion,
    dist_class,
    estimate_batch,
    device,
    top_n,
):
    hes = hessian(
        model=model,
        criterion=criterion,
        dist_class=dist_class,
        estimate_batch=estimate_batch,
        device=device,
    )
    eigenvalues, eigenvectors = hes.eigenvalues(top_n=top_n)
    return eigenvalues, eigenvectors

def estimate_hessian_density(
    model, 
    criterion,
    dist_class,
    estimate_batch,
    device,
):
    hes = hessian(
        model=model,
        criterion=criterion,
        dist_class=dist_class,
        estimate_batch=estimate_batch,
        device=device,
    )
    eigen_list_full, weight_list_full = hes.density()

    def density_generate(
        eigenvalues,
        weights,
        num_bins=10000,
        sigma_squared=1e-5,
        overhead=0.01
    ):
        eigenvalues = np.array(eigenvalues)
        weights = np.array(weights)

        lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
        lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead

        grids = np.linspace(lambda_min, lambda_max, num=num_bins)
        sigma = sigma_squared * max(1, (lambda_max - lambda_min))

        num_runs = eigenvalues.shape[0]
        density_output = np.zeros((num_runs, num_bins))

        for i in range(num_runs):
            for j in range(num_bins):
                x = grids[j]
                tmp_result = gaussian(eigenvalues[i, :], x, sigma)
                density_output[i, j] = np.sum(tmp_result * weights[i, :])
        density = np.mean(density_output, axis=0)
        normalization = np.sum(density) * (grids[1] - grids[0])
        density = density / normalization
        return density, grids
    
    def gaussian(x, x0, sigma_squared):
        return np.exp(-(x0 - x)**2 / (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)
    
    density, grids = density_generate(eigen_list_full, weight_list_full)
    return grids, density

def generate_offset_list(grid_size):
    offset_list = []
    for i in range(grid_size):
        for j in range(grid_size):
            x = i - grid_size // 2
            y = j - grid_size // 2
            offset_1 = x / (grid_size // 2)
            offset_2 = y / (grid_size // 2)
            offset_list.append([offset_1, offset_2])

    return offset_list

def eval_hessian(
    env,
    estimate_batch,
):
    # Deep copy the policy first
    policy_state_cp = copy.deepcopy(env.get_policy_state())

    # Compute eigenvalue density of Hessian
    eigenvalue_list, density_list = estimate_hessian_density(
        model=env.get_policy().model, 
        criterion=env.get_policy().loss,
        dist_class=env.get_policy().dist_class,
        estimate_batch=estimate_batch,
        device='cuda',
    )
    eigenvalue_list = np.real(eigenvalue_list)
    density_list = np.real(density_list)
    dsw = DescrStatsW(eigenvalue_list, density_list)
    hessian_eigen_cv = abs(dsw.std / dsw.mean)
    hessian_eigen_ratio = - max(eigenvalue_list) / min(eigenvalue_list)

    # Reload policy state
    env.set_policy_state(policy_state_cp)

    return hessian_eigen_cv, hessian_eigen_ratio

def eval_perturbation(
    round_id,
    env,
    grid_size,
    estimate_batch,
):
    # Deep copy the policy first
    policy_state_cp = copy.deepcopy(env.get_policy_state())

    # Compute directions using Hessian top 2 eigenvectors
    _, eigenvectors = estimate_hessian_eigens(
        model=env.get_policy().model, 
        criterion=env.get_policy().loss,
        dist_class=env.get_policy().dist_class,
        estimate_batch=estimate_batch,
        device='cuda',
        top_n=2,
    )

    # Compute offsets
    offset_list = generate_offset_list(grid_size)

    # Reload policy state
    env.set_policy_state(policy_state_cp)

    # Evaluate each perturbation
    csv_reward_surfaces = [
        [
            "offset_1", 
            "offset_2", 
            "episode_reward_max", 
            "episode_reward_min", 
            "episode_reward_mean", 
        ]
    ]
    
    for (job_id, offsets) in enumerate(offset_list):
        csv_row = []
        direction_1 = eigenvectors[0]
        direction_2 = eigenvectors[1]
        offset_1 = offsets[0]
        offset_2 = offsets[1]

        csv_row.append(offset_1)
        csv_row.append(offset_2)

        # Perturb the model
        model = env.get_policy().model
        for m, d_1, d_2 in zip(model.parameters(), direction_1, direction_2):
            m.data = m.data + offset_1 * d_1 + offset_2 * d_2

        # Evaluate the model
        model.eval()
        eval_results = env.trainer.evaluate()
        csv_row.append(eval_results['evaluation']['episode_reward_max'])
        csv_row.append(eval_results['evaluation']['episode_reward_min'])
        csv_row.append(eval_results['evaluation']['episode_reward_mean'])
        csv_reward_surfaces.append(csv_row)
        
        print("")
        print("Round {}, job {} finished: {}".format(round_id, job_id, csv_row))

        # Reload policy state
        env.set_policy_state(policy_state_cp)

    return csv_reward_surfaces

#
# Compute gradient noise scale (gns)
#

def eval_gns(    
    env,
    estimate_batch,
):
    # Deep copy the policy first
    policy_state_cp = copy.deepcopy(env.get_policy_state())

    policy = env.trainer.get_policy()
    policy.model.eval()
    
    if isinstance(estimate_batch, MultiAgentBatch):
        estimate_batch = estimate_batch.as_sample_batch()
    local_batches = estimate_batch.timeslices(num_slices=env.trainer.config.num_rollout_workers)
    # print(local_batches)
    # print(len(local_batches))
    B_big = estimate_batch.env_steps()
    B_small = local_batches[0].env_steps()

    G_big, _ = policy.compute_gradients(estimate_batch)
    G_sq_big = torch.square(torch.norm(fuse(G_big)))

    G_sq_small_list = []
    for batch in local_batches:
        local_grads, _ = policy.compute_gradients(batch)
        local_grads_sq_norm = torch.square(torch.norm(fuse(local_grads)))
        G_sq_small_list.append(local_grads_sq_norm)

    G_sq_small = torch.mean(torch.stack(G_sq_small_list))

    # Reference: https://arxiv.org/abs/1812.06162
    if B_big == B_small:
        G_biased = 0
        S_biased = 0
        gns = 0
    else:
        G_biased = convert_to_numpy(1 / (B_big - B_small) * (B_big * G_sq_big - B_small * G_sq_small))
        S_biased = convert_to_numpy(1 / (1 / B_small - 1 / B_big) * (G_sq_small - G_sq_big))
        gns = S_biased / G_biased

    # Reload policy state
    env.set_policy_state(policy_state_cp)

    return gns

#
# CSV
# 

def export_csv(
    scheduler_name,
    env_name, 
    algo_name, 
    csv_name,
    csv_file
):
    with open(
        "logs/{}~{}~{}~{}.csv".format(
            scheduler_name,
            env_name, 
            algo_name, 
            csv_name,
        ), 
        "w", 
        newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerows(csv_file)
