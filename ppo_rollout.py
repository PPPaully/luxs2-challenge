from time import time

import gym
import numpy as np
import torch as th

from wrappers import create_env
from ppo_agent import Agent
from lux.utils import draw

num_envs = 4
device = th.device("cuda" if th.cuda.is_available() else "cpu")

envs = gym.vector.SyncVectorEnv([
    create_env(with_opponent=False) for _ in range(num_envs)
])
agent = Agent(envs, is_cuda=True).to(device)

for env_id, env in enumerate(envs.envs):
    env.set_agent(agent, env_id)

reset_start = time()
observations = envs.reset()
print(f'Reset: {time() - reset_start: .2f} secs')

step_observation = []
step_actions = []
step_dones = []
step_logprobs = []
step_rewards = []
step_values = []
next_done = np.zeros(num_envs)

time_stats = dict(
    turn=[],
    feed=[],
)
num_steps = 200
for t in range(num_steps):
    turn_start = time()
    step_observation.append(observations)
    step_dones.append(next_done)

    feed_start = time()
    with th.no_grad():
        actions, logprobs, entropies, values = agent.get_action_and_value(agent.transform(observations))
    time_stats['feed'].append(time() - feed_start)
    step_actions.append(actions)
    step_logprobs.append(logprobs)
    step_values.append(values)

    observations, rewards, dones, infos = envs.step(actions)
    step_rewards.append(rewards)
    time_stats['turn'].append(time() - turn_start)

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

for env_id, env in enumerate(envs.envs):
    draw(env.lux_env)

# sum([np.prod(p.shape) for p in agent.parameters(True)])

