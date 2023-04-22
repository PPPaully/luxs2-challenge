import gym
import numpy as np
from time import time

from luxai_s2.env import LuxAI_S2
from lux.kit import obs_to_game_state, Factory, Unit, GameState


class DIRECTIONS:
    CENTER = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


def bfs_copy(positions, ref, target):
    sum_lichen = 0
    while len(positions) > 0:
        x, y = positions.pop(0)
        for xd, yd in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if 0 <= x + xd < ref.shape[0] and 0 <= y + yd < ref.shape[1]:
                if ref[x + xd, y + yd] > 0 and target[x + xd, y + yd] == 0:
                    target[x + xd, y + yd] = ref[x + xd, y + yd]
                    sum_lichen += target[x + xd, y + yd]
                    positions.append((x + xd, y + yd))
    return sum_lichen


class VectorWrapper(gym.Wrapper):
    def __init__(self, env: LuxAI_S2):
        super().__init__(env)

        self.lux_env = env
        self.lux_action_space = env.action_space
        self.lux_observation_space = env.observation_space
        self.factories_set = set()
        self.opponent_factories_set = set()
        self.units_set = set()
        self.opponent_units_set = set()
        self.last_game_state: GameState = None
        self.last_opponent_game_state: GameState = None

        self.action_space = gym.spaces.Dict(dict(
            factories=gym.spaces.Box(low=0, high=3, shape=(10, 1), dtype=int),
            units=gym.spaces.Box(low=0, high=1000, shape=(100, 6), dtype=int),
        ))

        self.time_stats = dict(
            observation=[],
            action=[],
            step=[],
        )

        max_map_size = env.env_cfg.map_size - 1
        max_factories = 10
        max_units = 100

        fact_low = np.asarray([[0] * 9] * max_factories)  # Alive, Position X, Y, Ice, Ore, Water, Metal, Power, Lichen
        fact_high = np.asarray([[1, max_map_size, max_map_size, 100000, 100000, 100000, 100000, 100000, 100000]] * max_factories)

        unit_low = np.asarray([[0] * 8] * max_units)  # Alive, Position X, Y, Ice, Ore, Water, Metal, Power, Lichen
        unit_high = np.asarray([[1, max_map_size, max_map_size, 100000, 100000, 100000, 100000, 100000]] * max_units)

        self.observation_space = gym.spaces.Dict(dict(
            ore=gym.spaces.Box(low=0, high=1, shape=(env.env_cfg.map_size, env.env_cfg.map_size), dtype=int),
            ice=gym.spaces.Box(low=0, high=1, shape=(env.env_cfg.map_size, env.env_cfg.map_size), dtype=int),
            rubble=gym.spaces.Box(low=0, high=100, shape=(env.env_cfg.map_size, env.env_cfg.map_size), dtype=int),
            lichen=gym.spaces.Box(low=0, high=100, shape=(env.env_cfg.map_size, env.env_cfg.map_size), dtype=int),
            opponent_lichen=gym.spaces.Box(low=0, high=100, shape=(env.env_cfg.map_size, env.env_cfg.map_size), dtype=int),
            lichen_strains=gym.spaces.Box(low=0, high=100, shape=(env.env_cfg.map_size, env.env_cfg.map_size), dtype=int),

            # Alive, Position X, Y, Ice, Ore, Water, Metal, Power, Lichen
            factories=gym.spaces.Box(low=fact_low, high=fact_high, dtype=int),
            opponent_factories=gym.spaces.Box(low=fact_low, high=fact_high, dtype=int),
            # Alive, Position X, Y, Ice, Ore, Water, Metal, Power
            units=gym.spaces.Box(low=unit_low, high=unit_high, dtype=int),
            opponent_units=gym.spaces.Box(low=unit_low, high=unit_high, dtype=int),
            units_map=gym.spaces.Box(low=0, high=1, shape=(env.env_cfg.map_size, env.env_cfg.map_size), dtype=int),
            opponent_units_map=gym.spaces.Box(low=0, high=1, shape=(env.env_cfg.map_size, env.env_cfg.map_size), dtype=int),
        ))

    def observation(self, obs):
        obs_start = time()
        # self = env
        player = 'player_0'
        opponent = 'player_1'
        game_state = obs_to_game_state(self.lux_env.env_steps, self.lux_env.env_cfg, obs[player])
        self.factories_set = set(game_state.factories[player].keys())
        self.opponent_factories_set = set(game_state.factories[opponent].keys())
        self.units_set = set(game_state.units[player].keys())
        self.opponent_units_set = set(game_state.units[opponent].keys())

        observation = dict()
        observation['ore'] = game_state.board.ore
        observation['ice'] = game_state.board.ice
        observation['rubble'] = game_state.board.rubble
        observation['lichen'] = np.zeros_like(game_state.board.lichen, dtype=int)
        observation['opponent_lichen'] = np.zeros_like(game_state.board.lichen, dtype=int)
        observation['lichen_strains'] = game_state.board.lichen_strains

        factory_ids = list(map(lambda x: int(x.split('_')[1]), game_state.factories[player].keys()))
        factory_lichen = dict()
        for fid in factory_ids:
            lichen = bfs_copy(
                list(map(lambda x: tuple(x), np.argwhere(game_state.board.factory_occupancy_map == fid))),
                game_state.board.lichen,
                observation['lichen']
            )
            factory_lichen[fid] = lichen

        opponent_factory_ids = list(map(lambda x: int(x.split('_')[1]), game_state.factories[opponent].keys()))
        for fid in opponent_factory_ids:
            lichen = bfs_copy(
                list(map(lambda x: tuple(x), np.argwhere(game_state.board.factory_occupancy_map == fid))),
                game_state.board.lichen,
                observation['opponent_lichen']
            )
            factory_lichen[fid] = lichen

        observation['factories'] = np.zeros(self.observation_space['factories'].shape, dtype=int)
        observation['opponent_factories'] = np.zeros(self.observation_space['opponent_factories'].shape, dtype=int)

        for fid, factory in game_state.factories[player].items():
            fid = int(fid.split('_')[1])
            observation['factories'][fid] = [
                1,
                factory.pos[0],
                factory.pos[1],
                factory.cargo.ice,
                factory.cargo.ore,
                factory.cargo.water,
                factory.cargo.metal,
                factory.power,
                factory_lichen[fid],
            ]

        for fid, factory in game_state.factories[opponent].items():
            fid = int(fid.split('_')[1])
            observation['opponent_factories'][fid] = [
                1,
                factory.pos[0],
                factory.pos[1],
                factory.cargo.ice,
                factory.cargo.ore,
                factory.cargo.water,
                factory.cargo.metal,
                factory.power,
                factory_lichen[fid],
            ]

        observation['units'] = np.zeros(self.observation_space['units'].shape, dtype=int)
        observation['opponent_units'] = np.zeros(self.observation_space['opponent_units'].shape, dtype=int)
        observation['units_map'] = np.zeros_like(game_state.board.ice, dtype=int)
        observation['opponent_units_map'] = np.zeros_like(game_state.board.ice, dtype=int)

        for uid, unit in game_state.units[player].items():
            uid = int(uid.split('_')[1])
            observation['units'][uid] = [
                1,
                unit.pos[0],
                unit.pos[1],
                unit.cargo.ice,
                unit.cargo.ore,
                unit.cargo.water,
                unit.cargo.metal,
                unit.power,
            ]
            observation['units_map'][unit.pos[0], unit.pos[1]] = 1

        for uid, unit in game_state.units[opponent].items():
            uid = int(uid.split('_')[1])
            observation['opponent_units'][uid] = [
                1,
                unit.pos[0],
                unit.pos[1],
                unit.cargo.ice,
                unit.cargo.ore,
                unit.cargo.water,
                unit.cargo.metal,
                unit.power,
            ]
            observation['opponent_units_map'][unit.pos[0], unit.pos[1]] = 1
        self.time_stats['observation'].append(time() - obs_start)
        return observation

    def get_observation(self, convert=True):
        obs = self.lux_env.state.get_obs()
        if convert:
            return self.observation({agent: obs for agent in self.lux_env.agents})
        return {agent: obs for agent in self.lux_env.agents}

    def reset(self, **kwargs):
        player = 'player_0'
        opponent = 'player_1'
        observation = self.env.reset(**kwargs)
        self.last_game_state = obs_to_game_state(self.lux_env.env_steps, self.lux_env.env_cfg, observation[player])
        self.last_opponent_game_state = obs_to_game_state(self.lux_env.env_steps, self.lux_env.env_cfg, observation[opponent])
        return self.observation(observation)

    def factory_action(self, factory: Factory, action_id, game_state: GameState, n_units):
        if action_id == 0:
            return None
        elif action_id == 1:
            if factory.can_build_light(game_state) and n_units < 50:
                return factory.build_light()
        elif action_id == 2:
            if factory.can_build_heavy(game_state) and n_units < 50:
                return factory.build_heavy()
        return None

    def unit_action(self, unit: Unit, action_id, game_state: GameState):
        if action_id == 0:
            return None
        elif action_id == 1:
            return unit.move(DIRECTIONS.UP)
        elif action_id == 2:
            return unit.move(DIRECTIONS.RIGHT)
        elif action_id == 3:
            return unit.move(DIRECTIONS.DOWN)
        elif action_id == 4:
            return unit.move(DIRECTIONS.LEFT)
        elif action_id == 5:
            return unit.dig(0, 1)
        elif action_id == 6:
            return unit.recharge(0)
        elif action_id == 7:  # Disabled by num_unit_action
            return unit.self_destruct()
        return None

    def action(self, action, with_opponent=True):
        act_start = time()
        # self = envs.envs[0]
        # action = new_actions[0]
        player = 'player_0'
        opponent = 'player_1'
        actions = dict()
        actions[player] = dict()
        actions[opponent] = dict()
        for idx in range(action['factories'].shape[0]):
            fidx = f'factory_{idx}'
            if fidx in self.factories_set:
                factory = self.last_game_state.factories[player][fidx]
                act = self.factory_action(factory, action['factories'][idx], self.last_game_state, len(self.units_set))
                if act is not None:
                    actions[player][fidx] = act
            elif with_opponent and fidx in self.opponent_factories_set:
                factory = self.last_game_state.factories[opponent][fidx]
                act = self.factory_action(factory, action['factories'][idx], self.last_opponent_game_state, len(self.opponent_units_set))
                if act is not None:
                    actions[opponent][fidx] = act

        for idx in range(action['units'].shape[0]):
            uidx = f'unit_{idx}'
            if uidx in self.units_set:
                unit = self.last_game_state.units[player][uidx]
                act = self.unit_action(unit, action['units'][idx], self.last_game_state)
                if act is not None:
                    actions[player][uidx] = [act]
            elif with_opponent and  uidx in self.opponent_units_set:
                unit = self.last_game_state.units[opponent][uidx]
                act = self.unit_action(unit, action['units'][idx], self.last_opponent_game_state)
                if act is not None:
                    actions[opponent][uidx] = [act]
        self.time_stats['action'].append(time() - act_start)
        return actions

    def step(self, action, with_opponent=True):
        step_start = time()
        player = 'player_0'
        opponent = 'player_1'
        observation, reward, done, info = self.env.step(self.action(action, with_opponent))
        self.last_game_state = obs_to_game_state(self.lux_env.env_steps, self.lux_env.env_cfg, observation[player])
        self.last_opponent_game_state = obs_to_game_state(self.lux_env.env_steps, self.lux_env.env_cfg, observation[opponent])
        self.time_stats['step'].append(time() - step_start)
        return self.observation(observation), reward, done, info


def create_env():
    def thunk() -> VectorWrapper:
        env = LuxAI_S2()
        env = VectorWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def test():
    # =====================================================
    #    Test Vector Wrapper
    # =====================================================
    from factory_setup import find_best_position
    from lux.utils import draw
    env = create_env()()
    cus_obs = env.reset()

    # Skip bidding
    actions = {'player_0': {'faction': 'AlphaStrike', 'bid': 0}, 'player_1': {'faction': 'AlphaStrike', 'bid': 0}}
    env_obs, rewards, dones, infos = env.lux_env.step(actions)

    while env.state.real_env_steps < 0:
        # Place factory
        player = 'player_0'
        spawn_position = find_best_position(env.env_steps, env_obs[player], env.env_cfg, split=len(env_obs[player]['factories'][player]) > 0, space=len(env_obs[player]['factories'][player]) > 1)
        actions = {'player_0': {'spawn': spawn_position, 'metal': 150, 'water': 150}, 'player_1': {}}
        env_obs, rewards, dones, infos = env.lux_env.step(actions)
        # draw(env)

        player = 'player_1'
        spawn_position = find_best_position(env.env_steps, env_obs[player], env.env_cfg, split=len(env_obs[player]['factories'][player]) > 0, space=len(env_obs[player]['factories'][player]) > 1)
        actions = {'player_0': {}, 'player_1': {'spawn': spawn_position, 'metal': 150, 'water': 150}}
        env_obs, rewards, dones, infos = env.lux_env.step(actions)
        # draw(env)
    # draw(env)

    for _ in range(30):
        actions = dict()
        player = 'player_0'
        game_state_0 = obs_to_game_state(env.env_steps, env.env_cfg, env_obs[player])
        actions[player] = dict([(fac_id, factory.water()) for fac_id, factory in game_state_0.factories[player].items()])

        player = 'player_1'
        game_state_1 = obs_to_game_state(env.env_steps, env.env_cfg, env_obs[player])
        actions[player] = dict([(fac_id, factory.water()) for fac_id, factory in game_state_1.factories[player].items()])
        env_obs, rewards, dones, infos = env.lux_env.step(actions)
        # draw(env)
    # draw(env)

    actions = dict()
    player = 'player_0'
    game_state_0 = obs_to_game_state(env.env_steps, env.env_cfg, env_obs[player])
    actions[player] = dict([(fac_id, factory.build_light()) for fac_id, factory in game_state_0.factories[player].items()])

    player = 'player_1'
    game_state_1 = obs_to_game_state(env.env_steps, env.env_cfg, env_obs[player])
    actions[player] = dict([(fac_id, factory.build_heavy()) for fac_id, factory in game_state_1.factories[player].items()])
    env_obs, rewards, dones, infos = env.lux_env.step(actions)

    draw(env)

    env_obs = env.get_observation()

    import matplotlib.pyplot as plt

    plt.imshow(env_obs['lichen'].T)
    plt.imshow(env_obs['opponent_lichen'].T)
    plt.show()

    # =====================================================
    #    Test Parallel Environment
    # =====================================================
    num_envs = 10
    envs = gym.vector.SyncVectorEnv([create_env() for _ in range(num_envs)])
    pal_obs = envs.reset()

    obs = env.lux_env.observation_space('player_0')
    obs.keys()
    obs['board'].keys()
    obs['board']['ore']
    obs['board']['ice']
    obs['board']['rubble']
    obs['board']['lichen']
    obs['factories']['player_0']

