import os
import shutil
from time import time

import gym
import random
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from env_train import create_env
from ppo_single_obj_agent import Agent
from lux.utils import draw

device = th.device("cuda" if th.cuda.is_available() else "cpu")

# Training Config
learning_rate = 2.5e-4
num_envs = 32  # N
num_steps = 200  # M
total_timesteps = 1_00_000
# 2200 * 60 * 10
anneal_lr = True
use_gae = True
gamma = 0.99
gae_lambda = 0.95
update_epochs = 4
num_minibatches = 4
clip_coef = 0.2
norm_adv = True
clip_vloss = True
ent_coef = 0.01
vf_coef = 0.5
max_grad_norm = 0.5
target_kl = None

# TRY NOT TO MODIFY: seeding
seed = 42
random.seed(seed)
np.random.seed(seed)
th.manual_seed(seed)

envs = gym.vector.SyncVectorEnv([
    create_env(
        num_factories=1,
        num_units=1,
        with_opponent=False
    ) for _ in range(num_envs)
])
agent = Agent(envs, is_cuda=True).to(device)
optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

for env_id, env in enumerate(envs.envs):
    env.set_agent(agent, env_id)

reset_start = time()
observations = envs.reset()
print(f'Reset: {time() - reset_start: .2f} secs')

# Initialize State
global_step = 0
cumsum_dones = 0
cumsum_rewards = 0
batch_size = int(num_envs * num_steps)
minibatch_size = int(batch_size // num_minibatches)
num_updates = total_timesteps // batch_size
time_stats = dict(
    turn=[],
    feed=[],
)

# Logging to Tensorboard
experiment = "lux2_ppo_find_ice"
if os.path.exists(f"./logs/{experiment}"):
    shutil.rmtree(f"./logs/{experiment}")
writer = SummaryWriter(f"logs/{experiment}")

# update = 0
start_time = time()
for update in range(num_updates):

    # =============================================================================
    #   Rollout
    # =============================================================================
    step_observation = []
    step_actions = []
    step_dones = []
    step_logprobs = []
    step_rewards = []
    step_values = []
    next_done = np.zeros(num_envs)
    observations = envs.reset()  # Limit turn = num_steps

    for t in range(num_steps):
        global_step += num_envs
        turn_start = time()
        step_observation.append(observations)
        step_dones.append(next_done)

        feed_start = time()
        with th.no_grad():
            actions, logprobs, entropies, values = agent.get_action_and_value(agent.transform(observations))
        time_stats['feed'].append(time() - feed_start)
        step_actions.append([act['player_0'] for act in actions])
        step_logprobs.append(logprobs)
        step_values.append(values.flatten())

        observations, rewards, dones, infos = envs.step(actions)
        step_rewards.append(rewards)
        time_stats['turn'].append(time() - turn_start)

    step_dones = th.tensor(np.asarray(step_dones)).to(device)
    step_rewards = th.tensor(np.asarray(step_rewards)).to(device)
    step_values = th.stack(step_values)
    # =============================================================================
    #
    # =============================================================================

    # Annealing the rate if instructed to do so.
    if anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

    with th.no_grad():
        next_value = agent.get_value(agent.transform(observations)).reshape(1, -1)
        if use_gae:
            advantages = th.zeros_like(step_rewards).to(device)
            last_gae_lam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_non_terminal = 1.0 - th.tensor(next_done).to(device)
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - step_dones[t + 1]
                    next_values = step_values[t + 1]
                delta = step_rewards[t] + gamma * next_values * next_non_terminal - step_values[t]
                advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            returns = advantages + step_values
        else:
            returns = th.zeros_like(rewards).to(device)
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_non_terminal = 1.0 - next_done
                    next_return = next_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + gamma * next_non_terminal * next_return
            advantages = returns - values

    batch_observation = step_observation
    batch_logprobs = step_logprobs
    batch_actions = []
    for step in range(num_steps):
        batch_actions.append(dict())
        for k in step_actions[0][0].keys():
            batch_actions[step][k] = []

        for k in batch_actions[step].keys():
            for eid in range(num_envs):
                batch_actions[step][k].append(step_actions[step][eid][k])
            batch_actions[step][k] = th.stack(batch_actions[step][k])
    # batch_observation = dict()
    # for k in step_observation[0].keys():
    #     batch_observation[k] = []
    # for idx in range(num_steps):
    #     for k in batch_observation.keys():
    #         batch_observation[k].append(step_observation[idx][k])
    # for k in batch_observation.keys():
    #     batch_observation[k] = np.stack(batch_observation[k]).reshape((-1,) + batch_observation[k][0].shape[1:])
    #
    # batch_logprobs = dict()
    # for k in step_logprobs[0].keys():
    #     batch_logprobs[k] = []
    # for idx in range(num_steps):
    #     for k in batch_logprobs.keys():
    #         batch_logprobs[k].append(step_logprobs[idx][k])
    # for k in batch_logprobs.keys():
    #     batch_logprobs[k] = th.stack(batch_logprobs[k]).reshape(-1, batch_logprobs[k][0].shape[-1])
    #
    # batch_actions = dict()
    # for k in step_actions[0][0].keys():
    #     batch_actions[k] = []
    # for i in range(len(step_actions)):
    #     for j in range(len(step_actions[0])):
    #         for k in batch_actions.keys():
    #             batch_actions[k].append(step_actions[i][j][k])
    # for k in batch_actions.keys():
    #     batch_actions[k] = th.stack(batch_actions[k])

    # Valid for minibatch
    batch_advantages = advantages
    batch_returns = returns
    batch_values = step_values

    # Optimizing the policy and value network
    clip_fracs = []
    # epoch = 0
    # batch_idx = 0

    def sum_dict(d: dict):
        return th.stack([d[k] for k in d.keys()]).sum(0)

    for epoch in range(update_epochs):
        agent.reset_units_data()
        for batch_idx in range(num_steps):
            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                agent.transform(batch_observation[batch_idx]),
                batch_actions[batch_idx]
            )
            logratio = sum_dict(newlogprob) - sum_dict(batch_logprobs[batch_idx])
            ratio = logratio.exp()

            with th.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clip_fracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

            mb_advantages = batch_advantages[batch_idx]
            if norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * th.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = th.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if clip_vloss:
                v_loss_unclipped = (newvalue - batch_returns[batch_idx]) ** 2
                v_clipped = batch_values[batch_idx] + th.clamp(
                    newvalue - batch_values[batch_idx],
                    -clip_coef,
                    clip_coef,
                    )
                v_loss_clipped = (v_clipped - batch_returns[batch_idx]) ** 2
                v_loss_max = th.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - batch_returns[batch_idx]) ** 2).mean()

            entropy_loss = sum_dict(entropy).mean()
            loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

        if target_kl is not None:
            if approx_kl > target_kl:
                break

    y_pred, y_true = batch_values.cpu().numpy(), batch_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    cumsum_dones += dones.sum().item()
    cumsum_rewards += rewards.sum().item()

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clip_fracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    writer.add_scalar("charts/SPS", int(global_step / (time() - start_time)), global_step)
    writer.add_scalar("charts/cumulative_reward", cumsum_rewards, global_step)
    writer.add_scalar("charts/done_games", cumsum_dones, global_step)
    print(f"Step: {global_step}/{total_timesteps} Step/Sec: {int(global_step / (time() - start_time))} Reward: {rewards.sum().item()}")

# ============================================================================================
#    Logging Timers
all_envs_stats = dict()
for stat in ["observation", "action", "step"]:
    all_envs_stats[stat] = []
    for env in envs.envs:
        all_envs_stats[stat] += env.env.time_stats[stat]

print(f'Turn                   : {np.mean(time_stats["turn"])*1000:4.0f} ms ({1/np.mean(time_stats["turn"]):8.2f}/s) - Total {np.sum(time_stats["turn"]):6.2f} secs')
print(f'Networks - Feed        : {np.mean(time_stats["feed"])*1000:4.0f} ms ({1/np.mean(time_stats["feed"]):8.2f}/s) - Total {np.sum(time_stats["feed"]):6.2f} secs')
print(f'Networks - Transform   : {np.mean(agent.time_stats["transform"])*1000:4.0f} ms ({1/np.mean(agent.time_stats["transform"]):8.2f}/s) x {len(agent.time_stats["transform"]):4d} times - Total {np.sum(agent.time_stats["transform"]):6.2f} secs')
print(f'Networks - Feed Action : {np.mean(agent.time_stats["feed_action"])*1000:4.0f} ms ({1/np.mean(agent.time_stats["feed_action"]):8.2f}/s) x {len(agent.time_stats["feed_action"]):4d} times - Total {np.sum(agent.time_stats["feed_action"]):6.2f} secs')
print(f'Networks - Feed Map 1  : {np.mean(agent.time_stats["action_map_1"])*1000:4.0f} ms ({1/np.mean(agent.time_stats["action_map_1"]):8.2f}/s) x {len(agent.time_stats["action_map_1"]):4d} times - Total {np.sum(agent.time_stats["action_map_1"]):6.2f} secs')
print(f'Networks - Feed Map 2  : {np.mean(agent.time_stats["action_map_2"])*1000:4.0f} ms ({1/np.mean(agent.time_stats["action_map_2"]):8.2f}/s) x {len(agent.time_stats["action_map_2"]):4d} times - Total {np.sum(agent.time_stats["action_map_2"]):6.2f} secs')
print(f'Networks - Feed Map 3  : {np.mean(agent.time_stats["action_map_3"])*1000:4.0f} ms ({1/np.mean(agent.time_stats["action_map_3"]):8.2f}/s) x {len(agent.time_stats["action_map_3"]):4d} times - Total {np.sum(agent.time_stats["action_map_3"]):6.2f} secs')
print(f'Networks - Feed Unit 1 : {np.mean(agent.time_stats["action_unit_1"])*1000:4.0f} ms ({1/np.mean(agent.time_stats["action_unit_1"]):8.2f}/s) x {len(agent.time_stats["action_unit_1"]):4d} times - Total {np.sum(agent.time_stats["action_unit_1"]):6.2f} secs')
print(f'Networks - Feed Unit 2 : {np.mean(agent.time_stats["action_unit_2"])*1000:4.0f} ms ({1/np.mean(agent.time_stats["action_unit_2"]):8.2f}/s) x {len(agent.time_stats["action_unit_2"]):4d} times - Total {np.sum(agent.time_stats["action_unit_2"]):6.2f} secs')
print(f'Networks - Feed Unit 3 : {np.mean(agent.time_stats["action_unit_3"])*1000:4.0f} ms ({1/np.mean(agent.time_stats["action_unit_3"]):8.2f}/s) x {len(agent.time_stats["action_unit_3"]):4d} times - Total {np.sum(agent.time_stats["action_unit_3"]):6.2f} secs')
print(f'Networks - Feed Unit 4 : {np.mean(agent.time_stats["action_unit_4"])*1000:4.0f} ms ({1/np.mean(agent.time_stats["action_unit_4"]):8.2f}/s) x {len(agent.time_stats["action_unit_4"]):4d} times - Total {np.sum(agent.time_stats["action_unit_4"]):6.2f} secs')
print(f'Networks - Find Factory: {np.mean(agent.time_stats["find_factory"])*1000:4.0f} ms ({1/np.mean(agent.time_stats["find_factory"]):8.2f}/s) x {len(agent.time_stats["find_factory"]):4d} times - Total {np.sum(agent.time_stats["find_factory"]):6.2f} secs')
print(f'Networks - Find Enemy  : {np.mean(agent.time_stats["find_enemy"])*1000:4.0f} ms ({1/np.mean(agent.time_stats["find_enemy"]):8.2f}/s) x {len(agent.time_stats["find_enemy"]):4d} times - Total {np.sum(agent.time_stats["find_enemy"]):6.2f} secs')
print(f'Networks - Find Res.   : {np.mean(agent.time_stats["find_resource"])*1000:4.0f} ms ({1/np.mean(agent.time_stats["find_resource"]):8.2f}/s) x {len(agent.time_stats["find_resource"]):4d} times - Total {np.sum(agent.time_stats["find_resource"]):6.2f} secs')
print(f'Networks - Feed Value  : {np.mean(agent.time_stats["feed_value"])*1000:4.0f} ms ({1/np.mean(agent.time_stats["feed_value"]):8.2f}/s) x {len(agent.time_stats["feed_value"]):4d} times - Total {np.sum(agent.time_stats["feed_value"]):6.2f} secs')
print(f'Networks - Cuda Tensor : {np.mean(agent.time_stats["tensor_cuda"])*1000:4.0f} ms ({1/np.mean(agent.time_stats["tensor_cuda"]):8.2f}/s) x {len(agent.time_stats["tensor_cuda"]):4d} times - Total {np.sum(agent.time_stats["tensor_cuda"]):6.2f} secs')
print(f'ENV. - T.Observation   : {np.mean(all_envs_stats["observation"])*1000:4.0f} ms ({1/np.mean(all_envs_stats["observation"]):8.2f}/s) x {len(all_envs_stats["observation"]):4d} times - Total {np.sum(all_envs_stats["observation"]):6.2f} secs')
print(f'ENV. - T.Action        : {np.mean(all_envs_stats["action"])*1000:4.0f} ms ({1/np.mean(all_envs_stats["action"]):8.2f}/s) x {len(all_envs_stats["action"]):4d} times - Total {np.sum(all_envs_stats["action"]):6.2f} secs')
print(f'ENV. - Step            : {np.mean(all_envs_stats["step"])*1000:4.0f} ms ({1/np.mean(all_envs_stats["step"]):8.2f}/s) x {len(all_envs_stats["step"]):4d} times - Total {np.sum(all_envs_stats["step"]):6.2f} secs')

# for env_id, env in enumerate(envs.envs):
#     draw(env.lux_env)

# sum([np.prod(p.shape) for p in agent.parameters(True)])

