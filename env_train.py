import gym

from luxai_s2.env import LuxAI_S2


class TrainWrapper(gym.Wrapper):
    def __init__(self, env: LuxAI_S2, num_factories=1, num_units=1):
        super().__init__(env)

        self.env_id = f'{id(self)}'
        self.lux_env = env
        self.players = ['player_0', 'player_1']
        self.agents = dict()
        self.keep_alive = dict()
        for p in self.players:
            self.agents[p] = None
            self.keep_alive[p] = True
        self.num_factories = num_factories
        self.num_units = num_units
        self.last_observation = None

    def get_config(self):
        return self.lux_env.env_cfg

    def set_agent(self, agent, as_player, keep_alive=False):
        self.agents[as_player] = agent
        self.keep_alive[as_player] = keep_alive
        agent.register(self.env_id + '_' + as_player, self, as_player)

    def reset(self, **kwargs):
        # self = env
        self.lux_env.env_cfg.MIN_FACTORIES = self.num_factories
        self.lux_env.env_cfg.MAX_FACTORIES = self.num_factories
        self.last_observation = self.lux_env.reset()

        step = 0
        while self.lux_env.state.real_env_steps < 0:
            actions = dict()
            for player in self.players:
                if self.agents[player] is not None:
                    actions[player] = self.agents[player].early_setup(step, self.last_observation[player], env_id=self.env_id + '_' + player)
                else:
                    actions[player] = dict()
            # print(actions)
            self.last_observation, _, _, _ = self.lux_env.step(actions)
            step += 1

        return self.last_observation

    def step(self, action):
        for player, keep_alive in self.keep_alive.items():
            if keep_alive:
                factories = self.lux_env.state.factories[player]
                for k in factories.keys():
                    factory = factories[k]
                    factory.cargo.water = 10000

        self.last_observation, lux_rewards, dones, infos = self.lux_env.step(action)

        rewards = dict()
        for player in self.players:
            rewards[player] = self.agents[player].get_reward(
                self.last_observation[player],
                self.lux_env.state.stats[player],
                env_id=self.env_id + '_' + player
            ) if not dones[player] else lux_rewards[player]

        return self.last_observation, rewards, dones, infos

    def next_step(self):
        actions = dict()
        for player in self.players:
            if self.agents[player] is not None:
                actions[player] = self.agents[player].act(
                    self.lux_env.state.real_env_steps,
                    self.last_observation[player],
                    env_id=self.env_id + '_' + player
                )
        return self.step(actions)

    def test_speed(self):
        from time import time
        step_time = []
        time_stats = dict(
            num_steps=0,
        )
        reset_start = time()
        self.reset()
        reset_time = time() - reset_start
        time_stats['reset_time'] = f'{reset_time * 1000:.0f} ms'

        while True:
            step_start = time()
            _, _, dones, _ = self.next_step()
            step_time.append(time() - step_start)
            time_stats['num_steps'] += 1
            if dones['player_0']:
                break

        time_stats['fps'] = f'{time_stats["num_steps"] / sum(step_time):.2f}'
        time_stats['total'] = f'{reset_time + sum(step_time):.2f} secs'

        return time_stats


def create_env(num_factories=1, num_units=1):
    def thunk() -> TrainWrapper:
        env = LuxAI_S2(collect_stats=True)
        env = TrainWrapper(env, num_factories, num_units)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk
