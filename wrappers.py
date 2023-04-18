import copy

import gym
from luxai_s2 import LuxAI_S2
from luxai_s2.state import StatsStateDict

from lux.kit import obs_to_game_state
from parallel_env import VectorWrapper


class SkipSetupWrapper(gym.Wrapper):
    def __init__(self, env: VectorWrapper):
        super().__init__(env)
        self.env = env
        self.prev_step_metrics = None

    def set_agent(self, agent, env_id=None):
        self.agent = agent
        agent.handshake(id(self), env_id)  # Support parallel reset

    def reset(self, **kwargs):
        # self = env
        player = 'player_0'
        opponent = 'player_1'
        obs = self.env.lux_env.reset(**kwargs)

        step = 0
        while self.env.lux_env.state.real_env_steps < 0:
            actions = dict()
            actions[self.agent.player] = self.agent.early_setup(step, obs[self.agent.player], env_id=id(self))
            actions[self.agent.opponent] = self.agent.early_setup(step, obs[self.agent.opponent], as_opponent=True, env_id=id(self))
            # print(actions)
            obs, _, _, _ = self.env.lux_env.step(actions)
            step += 1

        # Update last_game_state for VectorWrapper
        self.env.last_game_state = obs_to_game_state(self.lux_env.env_steps, self.lux_env.env_cfg, obs[player])
        self.env.last_opponent_game_state = obs_to_game_state(self.lux_env.env_steps, self.lux_env.env_cfg, obs[opponent])

        return self.env.observation(obs)

    def step(self, action):
        '''
            Return single ('player_0') reward only
        '''
        player = 'player_0'
        observation, rewards, dones, _ = self.env.step(action)

        infos = dict()
        stats: StatsStateDict = self.env.lux_env.state.stats[player]
        metrics = dict()
        metrics["ice_dug"] = (stats["generation"]["ice"]["HEAVY"] + stats["generation"]["ice"]["LIGHT"])
        metrics["water_produced"] = stats["generation"]["water"]
        metrics["action_queue_updates_success"] = stats["action_queue_updates_success"]
        metrics["action_queue_updates_total"] = stats["action_queue_updates_total"]

        infos["metrics"] = metrics
        infos["stats"] = stats

        reward = 0
        if self.prev_step_metrics is not None:
            # we check how much ice and water is produced and reward the agent for generating both
            ice_dug_this_step = metrics["ice_dug"] - self.prev_step_metrics["ice_dug"]
            water_produced_this_step = (
                    metrics["water_produced"] - self.prev_step_metrics["water_produced"]
            )
            # we reward water production more as it is the most important resource for survival
            reward = ice_dug_this_step / 100 + water_produced_this_step
        rewards[player] = reward if not dones[player] else rewards[player]

        self.prev_step_metrics = copy.deepcopy(metrics)

        return observation, rewards[player], dones[player], infos


def create_env():
    def thunk() -> VectorWrapper:
        env = LuxAI_S2(collect_stats=True)
        env = VectorWrapper(env)
        env = SkipSetupWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def test():
    from ppo_agent import Agent
    from lux.utils import draw
    env = create_env()()

    agent = Agent(env)
    env.set_agent(agent)

    obs = env.reset()
    draw(env.lux_env)

