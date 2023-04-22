import copy
import gym
import numpy as np
from time import time

from luxai_s2.env import LuxAI_S2
from luxai_s2.state import StatsStateDict

from lux.kit import obs_to_game_state, Factory, Unit, GameState


class DIRECTIONS:
    CENTER = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


class RESOURCE:
    ICE = 0
    ORE = 1
    WATER = 2
    METAL = 3
    POWER = 4


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
    def __init__(self, env: LuxAI_S2, num_factories=1, num_units=1, with_opponent=False):
        super().__init__(env)

        self.lux_env = env
        self.lux_action_space = env.action_space
        self.lux_observation_space = env.observation_space
        self.factories_set = set()
        self.opponent_factories_set = set()
        self.units_set = set()
        self.opponent_units_set = set()
        self.prev_step_metrics = None
        self.last_game_state: GameState = None
        self.last_opponent_game_state: GameState = None

        self.time_stats = dict(
            observation=[],
            action=[],
            step=[],
        )

        max_map_size = env.env_cfg.map_size - 1
        self.num_factories = num_factories
        self.num_units = num_units
        self.with_opponent = with_opponent

        self.action_space = gym.spaces.Dict(dict(
            player_0=gym.spaces.Dict(dict(
                factories=gym.spaces.Box(low=0, high=3, shape=(self.num_factories, 1), dtype=int),
                units=gym.spaces.Box(low=0, high=1000, shape=(self.num_units, 6), dtype=int),
            )),
            player_1=gym.spaces.Dict(dict(
                factories=gym.spaces.Box(low=0, high=3, shape=(self.num_factories, 1), dtype=int),
                units=gym.spaces.Box(low=0, high=1000, shape=(self.num_units, 6), dtype=int),
            )),
        ))

        fact_low = np.asarray([[0] * 9] * self.num_factories)  # Alive, Position X, Y, Ice, Ore, Water, Metal, Power, Lichen
        fact_high = np.asarray([[1, max_map_size, max_map_size, 100000, 100000, 100000, 100000, 100000, 100000]] * self.num_factories)

        unit_low = np.asarray([[0] * 8] * self.num_units)  # Alive, Position X, Y, Ice, Ore, Water, Metal, Power, Lichen
        unit_high = np.asarray([[1, max_map_size, max_map_size, 100000, 100000, 100000, 100000, 100000]] * self.num_units)

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
        # self = env
        # obs = self.lux_env.reset()

        obs_start = time()
        player = 'player_0'
        opponent = 'player_1'
        self.factories_set = set(self.last_game_state.factories[player].keys())
        self.opponent_factories_set = set(self.last_game_state.factories[opponent].keys())
        self.units_set = set(self.last_game_state.units[player].keys())
        self.opponent_units_set = set(self.last_game_state.units[opponent].keys())

        observation = dict()
        observation['ore'] = self.last_game_state.board.ore
        observation['ice'] = self.last_game_state.board.ice
        observation['rubble'] = self.last_game_state.board.rubble
        observation['lichen'] = np.zeros_like(self.last_game_state.board.lichen, dtype=int)
        observation['opponent_lichen'] = np.zeros_like(self.last_game_state.board.lichen, dtype=int)
        observation['lichen_strains'] = self.last_game_state.board.lichen_strains

        factory_lichen = dict()
        for fid in self.last_game_state.factories[player].keys():
            fval = int(fid.split('_')[1])
            lichen = bfs_copy(
                list(map(lambda x: tuple(x), np.argwhere(self.last_game_state.board.factory_occupancy_map == fval))),
                self.last_game_state.board.lichen,
                observation['lichen']
            )
            factory_lichen[fid] = lichen

        for fid in self.last_game_state.factories[opponent].keys():
            fval = int(fid.split('_')[1])
            lichen = bfs_copy(
                list(map(lambda x: tuple(x), np.argwhere(self.last_game_state.board.factory_occupancy_map == fval))),
                self.last_game_state.board.lichen,
                observation['opponent_lichen']
            )
            factory_lichen[fid] = lichen

        observation['factories'] = np.zeros(self.observation_space['factories'].shape, dtype=int)
        observation['opponent_factories'] = np.zeros(self.observation_space['opponent_factories'].shape, dtype=int)

        for idx, (fid, factory) in enumerate(self.last_game_state.factories[player].items()):
            observation['factories'][idx] = [
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

        for idx, (fid, factory) in enumerate(self.last_game_state.factories[opponent].items()):
            observation['opponent_factories'][idx] = [
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
        observation['units_map'] = np.zeros_like(self.last_game_state.board.ice, dtype=int)
        observation['opponent_units_map'] = np.zeros_like(self.last_game_state.board.ice, dtype=int)

        for idx, (uid, unit) in enumerate(self.last_game_state.units[player].items()):
            observation['units'][idx] = [
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

        for idx, (uid, unit) in enumerate(self.last_game_state.units[opponent].items()):
            observation['opponent_units'][idx] = [
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

    def set_agent(self, agent, env_id=None):
        self.agent = agent
        agent.handshake(id(self), env_id)  # Support parallel reset

    def reset(self, **kwargs):
        # self = env
        player = 'player_0'
        opponent = 'player_1'
        self.lux_env.env_cfg.MIN_FACTORIES = self.num_factories
        self.lux_env.env_cfg.MAX_FACTORIES = self.num_factories
        obs = self.lux_env.reset()

        step = 0
        while self.lux_env.state.real_env_steps < 0:
            actions = dict()
            actions[self.agent.player] = self.agent.early_setup(step, obs[self.agent.player], env_id=id(self))
            actions[self.agent.opponent] = self.agent.early_setup(step, obs[self.agent.opponent], as_opponent=True, env_id=id(self))
            # print(actions)
            obs, _, _, _ = self.lux_env.step(actions)
            step += 1

        # Update last_game_state for VectorWrapper
        self.last_game_state = obs_to_game_state(self.lux_env.env_steps, self.lux_env.env_cfg, obs[player])
        self.last_opponent_game_state = obs_to_game_state(self.lux_env.env_steps, self.lux_env.env_cfg, obs[opponent])

        return self.observation(obs)

    def factory_action(self, factory: Factory, action_id, game_state: GameState, n_units):
        if action_id == 0:
            return None
        elif action_id == 1:
            if factory.can_build_light(game_state) and n_units < self.num_units:
                return factory.build_light()
        elif action_id == 2:
            if factory.can_build_heavy(game_state) and n_units < self.num_units:
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
        elif action_id == 7:
            return unit.transfer(DIRECTIONS.CENTER, RESOURCE.ICE, unit.cargo.ice)
        return None

    def action(self, input_action):
        act_start = time()
        # self = envs.envs[0]
        # action = new_actions[0]
        player = 'player_0'
        opponent = 'player_1'
        actions = dict()
        actions[player] = dict()
        actions[opponent] = dict()

        # Convert factories' actions
        for idx, fidx in enumerate(self.factories_set):
            factory = self.last_game_state.factories[player][fidx]
            act = self.factory_action(factory, input_action[player]['factories'][idx], self.last_game_state, len(self.units_set))
            if act is not None:
                actions[player][fidx] = act

        # Convert units' actions
        for idx, uidx in enumerate(self.units_set):
            unit = self.last_game_state.units[player][uidx]
            act = self.unit_action(unit, input_action[player]['units'][idx], self.last_game_state)
            if act is not None:
                actions[player][uidx] = [act]

        if self.with_opponent:
            for idx, fidx in enumerate(self.opponent_factories_set):
                factory = self.last_opponent_game_state.factories[opponent][fidx]
                act = self.factory_action(factory, input_action[opponent]['factories'][idx], self.last_opponent_game_state, len(self.opponent_units_set))
                if act is not None:
                    actions[opponent][fidx] = act

            for idx, uidx in enumerate(self.opponent_units_set):
                unit = self.last_opponent_game_state.units[opponent][uidx]
                act = self.unit_action(unit, input_action[opponent]['units'][idx], self.last_opponent_game_state)
                if act is not None:
                    actions[opponent][uidx] = [act]
        self.time_stats['action'].append(time() - act_start)
        return actions

    def step(self, action):
        '''
            Return single ('player_0') reward only
        '''
        # self = envs.envs[0]
        # action = actions[0]
        step_start = time()
        player = 'player_0'
        opponent = 'player_1'

        if not self.with_opponent:
            opp_factories = self.lux_env.state.factories[opponent]
            for k in opp_factories.keys():
                factory = opp_factories[k]
                factory.cargo.water = 1000  # Keep enemy alive

        observation, rewards, dones, _ = self.lux_env.step(self.action(action))

        infos = dict()
        if player in self.lux_env.state.stats:
            stats: StatsStateDict = self.lux_env.state.stats[player]
            metrics = dict()
            metrics["ice_dug"] = (stats["generation"]["ice"]["HEAVY"] + stats["generation"]["ice"]["LIGHT"])
            metrics["water_produced"] = stats["generation"]["water"]
            metrics["action_queue_updates_success"] = stats["action_queue_updates_success"]
            metrics["action_queue_updates_total"] = stats["action_queue_updates_total"]

            infos["metrics"] = metrics
            infos["stats"] = stats
        else:
            infos["metrics"] = None
            infos["stats"] = None

        reward = 0
        if self.prev_step_metrics is not None:
            # we check how much ice and water is produced and reward the agent for generating both
            ice_dug_this_step = infos["metrics"]["ice_dug"] - self.prev_step_metrics["ice_dug"]
            water_produced_this_step = (
                    infos["metrics"]["water_produced"] - self.prev_step_metrics["water_produced"]
            )
            # we reward water production more as it is the most important resource for survival
            reward = ice_dug_this_step / 100 + water_produced_this_step
        rewards[player] = reward if not dones[player] else rewards[player]

        if infos["metrics"] is not None:
            self.prev_step_metrics = copy.deepcopy(infos["metrics"])
        self.last_game_state = obs_to_game_state(self.lux_env.env_steps, self.lux_env.env_cfg, observation[player])
        self.last_opponent_game_state = obs_to_game_state(self.lux_env.env_steps, self.lux_env.env_cfg, observation[opponent])

        self.time_stats['step'].append(time() - step_start)

        return self.observation(observation), rewards[player], dones[player], infos


def create_env(num_factories=1, num_units=1, with_opponent=False):
    def thunk() -> VectorWrapper:
        env = LuxAI_S2(collect_stats=True)
        env = VectorWrapper(env, num_factories, num_units, with_opponent)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def test():
    # =====================================================
    #    Dev simplified actions
    # =====================================================
    from ppo_single_obj_agent import Agent
    from lux.utils import draw

    num_envs = 16
    num_steps = 200
    envs = gym.vector.SyncVectorEnv([
        create_env(
            num_factories=1,
            num_units=1,
            with_opponent=False
        ) for _ in range(num_envs)
    ])

    agent = Agent(envs)
    for eid, env in enumerate(envs.envs):
        env.set_agent(agent, env_id=eid)
    observations = envs.reset()

    time_stats = dict(
        turn=[],
        feed=[],
    )

    for _ in range(num_steps):
        turn_start = time()

        feed_start = time()
        actions, logprobs, entropies, values = agent.get_action_and_value(agent.transform(observations))
        time_stats['feed'].append(time() - feed_start)

        observations, rewards, dones, infos = envs.step(actions)
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


def test2():
    # =====================================================
    #    Test Vector Wrapper
    # =====================================================

    from factory_setup import find_best_position

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

