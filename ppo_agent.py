from time import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Categorical

from lux.config import EnvConfig
from lux.kit import obs_to_game_state, GameState
from lux.utils import my_turn_to_place_factory

from factory_setup import find_best_position
from nn_utils import layer_init


class ROLES:
    BASE = 0
    ENEMY = 1
    GATHER = (2, 3, 4)
    ICE = 2
    ORE = 3
    RUBBLE = 4

    @staticmethod
    def is_role(a):
        return ROLES.BASE <= a <= ROLES.RUBBLE


class Agent(nn.Module):
    def __init__(self, envs, player: str = 'player_0', is_cuda=False):
        super(Agent, self).__init__()

        players = ['player_0', 'player_1']
        assert player in players

        self.time_stats = dict(
            transform=[],
            feed_action=[],
            feed_value=[],
            tensor_cuda=[],
            action_map_1=[],
            action_map_2=[],
            action_map_3=[],
            action_unit_1=[],
            action_unit_2=[],
            action_unit_3=[],
            action_unit_4=[],
            find_factory=[],
            find_enemy=[],
            find_resource=[],
        )

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.player = player
        self.opponent = players[1 - players.index(player)]
        self.envs = envs
        self.num_envs = envs.num_envs
        self.envs_mapping = dict()
        self.is_cuda = is_cuda

        self.__init_model__()

        self.max_factory = 10
        self.max_unit = 100
        self.units_role = np.zeros((self.num_envs, self.max_unit), dtype=int) - 1
        self.units_base = np.zeros((self.num_envs, self.max_unit), dtype=int) - 1
        self.units_target = np.zeros((self.num_envs, self.max_unit), dtype=int) - 1
        self.units_gather_loc = np.zeros((self.num_envs, self.max_unit, 2), dtype=np.float32) - 1

    def __init_model__(self):
        self.conf_map_layers = [
            'ice', 'ore', 'rubble',
            'lichen', 'opponent_lichen', 'lichen_strains',
            'units_map', 'opponent_units_map',
        ]
        self.conf_map_layers_op = [
            'ice', 'ore', 'rubble',
            'opponent_lichen', 'lichen', 'lichen_strains',
            'opponent_units_map', 'units_map',
        ]  # as_opponent
        self.conf_map_features = 128

        self.critic_map = nn.Sequential(
            layer_init(nn.Conv2d(len(self.conf_map_layers), 32, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            layer_init(nn.Conv2d(32, 64, kernel_size=3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            layer_init(nn.Conv2d(64, self.conf_map_features, kernel_size=3)),
            nn.ReLU(),
            nn.MaxPool2d(9),
            nn.Flatten(),
        )

        self.actor_map = nn.Sequential(
            layer_init(nn.Conv2d(len(self.conf_map_layers), 32, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            layer_init(nn.Conv2d(32, 64, kernel_size=3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            layer_init(nn.Conv2d(64, self.conf_map_features, kernel_size=3)),
            nn.ReLU(),
            nn.MaxPool2d(9),
            nn.Flatten(),
        )

        self.actor_observe = nn.Sequential(
            layer_init(nn.Conv2d(len(self.conf_map_layers), 32, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, self.conf_map_features, kernel_size=3)),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Flatten(),
        )

        num_factory_status = 9
        num_unit_status = 8
        self.conf_factories_status_features = 64
        self.conf_units_status_features = 64
        self.conf_status_features = self.conf_factories_status_features + self.conf_units_status_features

        self.critic_factories_status = nn.Sequential(
            layer_init(nn.Linear(num_factory_status, self.conf_factories_status_features)),
            nn.Tanh(),
            layer_init(nn.Linear(self.conf_factories_status_features, self.conf_factories_status_features)),
            nn.Tanh(),
        )
        self.critic_units_status = nn.Sequential(
            layer_init(nn.Linear(num_unit_status, self.conf_units_status_features)),
            nn.Tanh(),
            layer_init(nn.Linear(self.conf_units_status_features, self.conf_units_status_features)),
            nn.Tanh(),
        )
        self.critic_status = nn.Sequential(
            layer_init(nn.Linear(self.conf_map_features + self.conf_status_features, self.conf_map_features + self.conf_status_features)),
            nn.ReLU(),
            layer_init(nn.Linear(self.conf_map_features + self.conf_status_features, 1), std=1),
        )

        self.actor_factories_status = nn.Sequential(
            layer_init(nn.Linear(num_factory_status, self.conf_factories_status_features)),
            nn.Tanh(),
            layer_init(nn.Linear(self.conf_factories_status_features, self.conf_factories_status_features)),
            nn.Tanh(),
        )
        self.actor_units_status = nn.Sequential(
            layer_init(nn.Linear(num_unit_status, self.conf_units_status_features)),
            nn.Tanh(),
            layer_init(nn.Linear(self.conf_units_status_features, self.conf_units_status_features)),
            nn.Tanh(),
        )

        self.num_factory_action = 4  # See scratch.py
        self.num_unit_action = 7  # See scratch.py
        self.num_unit_role = 5  # See scratch.py
        self.actor_factory = nn.Sequential(
            layer_init(nn.Linear(self.conf_map_features + self.conf_factories_status_features, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, self.num_factory_action), std=0.01),
        )
        self.actor_unit_base = nn.Sequential(  # Map, Observe, UnitStat, BaseStat
            layer_init(nn.Linear(self.conf_map_features * 2 + self.conf_units_status_features * 2, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, self.num_unit_action), std=0.01),
        )
        self.actor_unit_enemy = nn.Sequential(  # Map, Observe, UnitStat, EnemyStat
            layer_init(nn.Linear(self.conf_map_features * 2 + self.conf_units_status_features * 2, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, self.num_unit_action), std=0.01),
        )
        self.actor_unit_gather = nn.Sequential(  # Map, Observe, UnitStat, TargetPos
            layer_init(nn.Linear(self.conf_map_features * 2 + self.conf_units_status_features + 2, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, self.num_unit_action), std=0.01),
        )
        self.actor_unit_role = nn.Sequential(  # Map, Observe, UnitStat, BaseStat
            layer_init(nn.Linear(self.conf_map_features * 2 + self.conf_units_status_features * 2, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, self.num_unit_role), std=0.01),
        )

    def set_envs(self, envs):
        self.envs = envs

    def handshake(self, wrapper_id, env_id):
        self.envs_mapping[wrapper_id] = env_id

    def to_cuda(self, a):
        if self.is_cuda:
            tensor_cuda_start = time()
            res = a.to(self.device)
            self.time_stats['tensor_cuda'].append(time() - tensor_cuda_start)
            return res
        return a

    def transform(self, observation_input):
        transform_start = time()
        # self = agent
        # observation_input = observations
        transformed = dict()
        map_size = observation_input['ice'].shape[1]
        # Map
        transformed['ice'] = th.tensor(observation_input['ice'].astype(np.float32))
        transformed['ore'] = th.tensor(observation_input['ore'].astype(np.float32))
        transformed['rubble'] = th.tensor((observation_input['rubble'] / self.envs.observation_space['rubble'].high).astype(np.float32))
        transformed['units_map'] = th.tensor(observation_input['units_map'].astype(np.float32))
        transformed['opponent_units_map'] = th.tensor(observation_input['opponent_units_map'].astype(np.float32))
        transformed['lichen'] = th.tensor((observation_input['lichen'] / self.envs.observation_space['lichen'].high).astype(np.float32))
        transformed['opponent_lichen'] = th.tensor((observation_input['opponent_lichen'] / self.envs.observation_space['opponent_lichen'].high).astype(np.float32))
        transformed['lichen_strains'] = th.tensor((observation_input['lichen_strains'] / self.envs.observation_space['lichen_strains'].high).astype(np.float32))

        # Status
        transformed['factories'] = th.tensor((observation_input['factories'] / self.envs.observation_space['factories'].high).astype(np.float32))
        transformed['opponent_factories'] = th.tensor((observation_input['opponent_factories'] / self.envs.observation_space['opponent_factories'].high).astype(np.float32))
        transformed['units'] = th.tensor((observation_input['units'] / self.envs.observation_space['units'].high).astype(np.float32))
        transformed['opponent_units'] = th.tensor((observation_input['opponent_units'] / self.envs.observation_space['opponent_units'].high).astype(np.float32))

        # Position - DO NOT CONVERT TO TENSOR
        transformed['pos_factories'] = observation_input['factories'][:, :, 1: 3]
        transformed['pos_units'] = observation_input['units'][:, :, 1: 3]
        transformed['pos_opponent_units'] = observation_input['opponent_units'][:, :, 1: 3]

        # Unit observe
        observe_range = 3
        observe_range_full = observe_range * 2 + 1
        maps = ['ice', 'ore', 'rubble', 'units_map', 'opponent_units_map', 'lichen', 'opponent_lichen', 'lichen_strains']
        opponent_maps = ['ice', 'ore', 'rubble', 'opponent_units_map', 'units_map', 'opponent_lichen', 'lichen', 'lichen_strains']
        transformed['units_observe'] = np.zeros(observation_input['units'].shape[:2] + (len(maps), observe_range_full, observe_range_full), dtype=np.float32)
        transformed['units_opponent_observe'] = np.zeros(observation_input['opponent_units'].shape[:2] + (len(maps), observe_range_full, observe_range_full), dtype=np.float32)

        def calculate_padding(pos):
            padding = np.asarray([0, 0, 0, 0])  # top, bot, left, right
            xs, xe = pos[0] - observe_range, pos[0] + observe_range + 1
            ys, ye = pos[1] - observe_range, pos[1] + observe_range + 1
            if xs < 0:
                padding[0] = -xs
                xs = 0
            if xe > map_size:
                padding[1] = xe - map_size
                xe = map_size
            if ys < 0:
                padding[2] = -ys
                ys = 0
            if ye > map_size:
                padding[3] = ye - map_size
                ye = map_size

            return xs, xe, ys, ye, (padding[:2], padding[2:])

        for env_id in range(observation_input['units'].shape[0]):
            for unit_id, pack in enumerate(observation_input['units'][env_id, :, 0:3].tolist()):
                alive, pos = pack[0], pack[1:]
                if not alive:
                    continue
                xs, xe, ys, ye, padding = calculate_padding(pos)
                for idx, m in enumerate(maps):
                    temp = transformed[m][env_id][xs: xe, ys: ye]
                    transformed['units_observe'][env_id, unit_id, idx] = np.pad(temp, padding, mode='constant', constant_values=-1)

            for unit_id, pack in enumerate(observation_input['opponent_units'][env_id, :, 0:3].tolist()):
                alive, pos = pack[0], pack[1:]
                if not alive:
                    continue
                xs, xe, ys, ye, padding = calculate_padding(pos)
                for idx, m in enumerate(opponent_maps):
                    temp = transformed[m][env_id][xs: xe, ys: ye]
                    transformed['units_opponent_observe'][env_id, unit_id, idx] = np.pad(temp, padding, mode='constant', constant_values=-1)

        transformed['units_observe'] = th.tensor(transformed['units_observe'])
        transformed['units_opponent_observe'] = th.tensor(transformed['units_opponent_observe'])

        self.time_stats['transform'].append(time() - transform_start)
        return transformed

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60, env_id: id = None, as_opponent: bool = False):
        if step == 0:
            return dict(faction='AlphaStrike', bid=0)

        env_cfg = self.get_env_cfg(env_id)
        player = self.get_player(as_opponent)
        game_state: GameState = obs_to_game_state(step, env_cfg, obs)

        if my_turn_to_place_factory(game_state.teams[player].place_first, step)\
            and game_state.teams[player].factories_to_place > 0:
            num_factory = len(game_state.factories[player].values())
            action = dict()
            action['spawn'] = find_best_position(step, obs, env_cfg, split=num_factory > 0, space=game_state.teams[player].factories_to_place == 1)
            action['metal'] = game_state.teams[player].metal // game_state.teams[player].factories_to_place
            action['water'] = game_state.teams[player].water // game_state.teams[player].factories_to_place

            return action

        return dict()

    def get_env_cfg(self, env_id=None) -> EnvConfig:
        return self.envs.env_cfg if env_id is None else self.envs.envs[self.envs_mapping[env_id]].env_cfg

    def get_player(self, as_opponent=False):
        return self.opponent if as_opponent else self.player

    def get_value(self, observation_input):
        # self = agent
        # observation_input = agent.transform(observations)

        feed_value_start = time()
        # Feed map
        inp_m = self.to_cuda(th.stack([observation_input[k] for k in self.conf_map_layers], dim=1))
        # as_opponent = True -> inp_m = th.tensor(np.stack([obs[k] for k in self.conf_map_layers_op], axis=1))
        out_map = self.critic_map(inp_m)
        # out_map.shape

        # Feed factories
        inp_f = self.to_cuda(observation_input['factories'])
        # inp_f.shape
        out_f = th.sum(self.critic_factories_status(inp_f), dim=1) / \
                th.clip(inp_f[:, :, 0].sum(axis=1).reshape(-1, 1).repeat(1, self.conf_units_status_features), 1)
        # out_f.shape

        # Feed units
        in_u = self.to_cuda(observation_input['units'])
        # in_u.shape
        out_u = th.sum(self.critic_units_status(in_u), dim=1) / \
                th.clip(in_u[:, :, 0].sum(axis=1).reshape(-1, 1).repeat(1, self.conf_units_status_features), 1)
        # out_u.shape

        # Predict critic from all status
        out_status = th.stack([out_f, out_u], dim=2).flatten(1)
        res = self.critic_status(th.stack([out_map, out_status], dim=2).flatten(1))
        self.time_stats['feed_value'].append(time() - feed_value_start)

        return res

    def get_action_and_value(self, observation_input, actions=None):
        # self = agent
        # observation_input = agent.transform(observations)
        # actions = None
        feed_action_start = time()
        no_action = actions is None
        if no_action:
            actions = dict()

        log_probs = dict()
        entropies = dict()

        action_map_0 = time()
        # Feed map

        inp_m = self.to_cuda(th.stack([observation_input[k] for k in self.conf_map_layers], dim=1))
        # as_opponent = True -> inp_m = th.tensor(np.stack([observation_input[k] for k in self.conf_map_layers_op], axis=1))
        out_map = self.actor_map(inp_m)
        # out_map.shape

        action_map_1 = time()
        self.time_stats['action_map_1'].append(action_map_1 - action_map_0)
        # Feed factories
        inp_f = self.to_cuda(observation_input['factories'])
        # inp_f.shape
        out_f = self.actor_factories_status(inp_f)
        # out_f.shape

        action_map_2 = time()
        self.time_stats['action_map_2'].append(action_map_2 - action_map_1)
        # Predict factory action
        action_factories = self.actor_factory(th.concat([out_map.reshape(self.envs.num_envs, 1, -1).repeat(1, self.max_factory, 1), out_f], dim=2))
        action_factories_probs = Categorical(logits=action_factories)
        if no_action:
            actions['factories'] = action_factories_probs.sample()
        log_probs['factories'] = action_factories_probs.log_prob(actions['factories'])
        entropies['factories'] = action_factories_probs.entropy()
        # action_factories_probs.probs  # real probs
        action_map_3 = time()
        self.time_stats['action_map_3'].append(action_map_3 - action_map_2)
        # print(f'Feed Map: {(action_map_end - action_map_start) * 1000:4.0f} ms')

        # =====================
        action_unit_0 = time()
        # Feed unit area
        in_uob = self.to_cuda(observation_input['units_observe'])
        out_uob = self.actor_observe(in_uob.reshape((-1,) + in_uob.shape[2:])).reshape(in_uob.shape[:2] + (128,))
        # out_uob.shape

        # Feed units
        in_u = self.to_cuda(observation_input['units'])
        # in_u.shape
        out_u = self.actor_units_status(in_u)
        # out_u.shape

        # Feed enemies
        in_ue = self.to_cuda(observation_input['opponent_units'])
        # in_ue.shape
        out_ue = self.actor_units_status(in_ue)
        # out_ue.shape

        # Predict unit action
        in_pre_unit = th.concat([out_map.reshape(self.num_envs, 1, -1).repeat(1, self.max_unit, 1), out_uob, out_u], dim=2)
        in_pre_base = th.stack([out_f[eid][[self.units_base[eid]]] for eid in range(self.num_envs)])
        in_pre_enemy = th.stack([out_ue[eid][[self.units_target[eid]]] for eid in range(self.num_envs)])
        in_pre_gather = self.to_cuda(th.stack([th.tensor(self.units_gather_loc[eid]) for eid in range(self.num_envs)]))

        action_units_base = self.actor_unit_base(th.concat([in_pre_unit, in_pre_base], dim=2))
        action_units_enemy = self.actor_unit_enemy(th.concat([in_pre_unit, in_pre_enemy], dim=2))
        action_units_gather = self.actor_unit_gather(th.concat([in_pre_unit, in_pre_gather], dim=2))
        action_units_role = self.actor_unit_role(th.concat([in_pre_unit, in_pre_base], dim=2))

        action_unit_1 = time()
        self.time_stats['action_unit_1'].append(action_unit_1 - action_unit_0)
        # print(f'Feed Unit 1: {(action_unit_1 - action_unit_0) * 1000:4.0f} ms')

        action_units_base_probs = Categorical(logits=action_units_base)
        action_units_enemy_probs = Categorical(logits=action_units_enemy)
        action_units_gather_probs = Categorical(logits=action_units_gather)
        action_units_role_probs = Categorical(logits=action_units_role)

        factory_filters = [np.where(observation_input['factories'][eid, :, 0])[0] for eid in range(self.envs.num_envs)]
        units_filters = [np.where(observation_input['opponent_units'][eid, :, 0])[0] for eid in range(self.envs.num_envs)]
        resource_filters = dict([(res, [np.argwhere(observation_input[res][eid]).numpy().T for eid in range(self.envs.num_envs)]) for res in ['ice', 'ore', 'rubble']])

        def find_closest_factory(eid, uid):
            start = time()
            factory_filter = factory_filters[eid]
            factory_dist = abs(observation_input['pos_units'][eid, uid] - observation_input['pos_factories'][eid]).sum(axis=1)
            res = factory_filter[np.argmin(factory_dist[factory_filter])]
            self.time_stats['find_factory'].append(time() - start)
            return res

        def find_closest_enemy(eid, uid):
            start = time()
            units_filter = units_filters[eid]
            if len(units_filter) == 0:
                res = -1
            else:
                units_dist = abs(observation_input['pos_units'][eid, uid] - observation_input['pos_opponent_units'][eid]).sum(axis=1)
                res = units_filter[np.argmin(units_dist[units_filter])]
            self.time_stats['find_enemy'].append(time() - start)
            return res

        def find_closest_resource(eid, uid, resource_type):
            start = time()
            resource_locs = resource_filters[resource_type][eid]
            units_dist = abs(observation_input['pos_units'][eid, uid] - resource_locs).sum(axis=1)
            res = resource_locs[np.argmin(units_dist)] / 47  # 47 = map_size - 1
            self.time_stats['find_resource'].append(time() - start)
            return res

        for eid in range(self.num_envs):
            for uid, role in enumerate(self.units_role[eid]):
                # Initialize unit state
                if self.units_role[eid, uid] == -1 and observation_input['units'][eid, uid, 0] > 0:
                    self.units_role[eid, uid] = ROLES.BASE
                    self.units_base[eid, uid] = find_closest_factory(eid, uid)
                    self.units_target[eid, uid] = -1
                    self.units_gather_loc[eid, uid] = [-1, -1]
                # else:  # Dead, Reset
                #     self.units_role[eid, uid] = th.Tensor([-1, -1])
                #     self.units_base[eid, uid] = th.Tensor([-1, -1])
                #     self.units_target[eid, uid] = th.Tensor([-1, -1])
                #     self.units_gather_loc[eid, uid] = th.Tensor([-1, -1])

        action_unit_2 = time()
        self.time_stats['action_unit_2'].append(action_unit_2 - action_unit_1)
        # print(f'Feed Unit 2: {(action_unit_2 - action_unit_1) * 1000:4.0f} ms')

        if no_action:
            action_units = th.zeros((self.num_envs, self.max_unit), dtype=th.int64)
            action_units_base = action_units_base_probs.sample().cpu().numpy()
            action_units_enemy = action_units_enemy_probs.sample().cpu().numpy()
            action_units_gather = action_units_gather_probs.sample().cpu().numpy()
            for eid in range(self.num_envs):
                for uid, role in enumerate(self.units_role[eid]):
                    # Assign units' action
                    if role == ROLES.BASE:
                        action_units[eid, uid] = action_units_base[eid, uid]
                    elif role == ROLES.ENEMY:
                        action_units[eid, uid] = action_units_enemy[eid, uid]
                    else:
                        action_units[eid, uid] = action_units_gather[eid, uid]
            actions['units'] = action_units = self.to_cuda(action_units)
        else:
            action_units = actions['units']
        action_units_role = action_units_role_probs.sample().cpu().numpy()

        log_probs_base = action_units_base_probs.log_prob(action_units).cpu().numpy()
        log_probs_enemy = action_units_enemy_probs.log_prob(action_units).cpu().numpy()
        log_probs_gather = action_units_gather_probs.log_prob(action_units).cpu().numpy()

        log_probs_all = []
        for eid in range(self.num_envs):
            log_probs_all.append([])
            for uid, role in enumerate(self.units_role[eid]):
                if role == ROLES.BASE:
                    log_probs_all[eid].append(log_probs_base[eid, uid])
                elif role == ROLES.ENEMY:
                    log_probs_all[eid].append(log_probs_enemy[eid, uid])
                elif role in ROLES.GATHER:
                    log_probs_all[eid].append(log_probs_gather[eid, uid])
                else:
                    log_probs_all[eid].append(th.tensor(np.nan))
        log_probs['units'] = self.to_cuda(th.tensor(log_probs_all))

        entropy_base = action_units_base_probs.entropy().cpu().numpy()
        entropy_enemy = action_units_enemy_probs.entropy().cpu().numpy()
        entropy_gather = action_units_gather_probs.entropy().cpu().numpy()

        entropy_all = []
        for eid in range(self.num_envs):
            entropy_all.append([])
            for uid, role in enumerate(self.units_role[eid]):
                if role == ROLES.BASE:
                    entropy_all[eid].append(entropy_base[eid, uid])
                elif role == ROLES.ENEMY:
                    entropy_all[eid].append(entropy_enemy[eid, uid])
                elif role in ROLES.GATHER:
                    entropy_all[eid].append(entropy_gather[eid, uid])
                else:
                    entropy_all[eid].append(th.tensor(np.nan))
        entropies['units'] = self.to_cuda(th.tensor(entropy_all))

        action_unit_3 = time()
        self.time_stats['action_unit_3'].append(action_unit_3 - action_unit_2)
        # print(f'Feed Unit 3: {(action_unit_3 - action_unit_2) * 1000:4.0f} ms')

        for eid in range(self.num_envs):
            for uid, role in enumerate(self.units_role[eid]):
                # Change role & Assign new target
                new_role = action_units_role[eid, uid]
                if ROLES.is_role(action_units_role[eid, uid]) and role != new_role:  # Change role
                    if new_role == ROLES.BASE:  # Find Base
                        # Old base not exists > Find New
                        if observation_input['factories'][eid, self.units_base[eid, uid], 0] == 0:
                            self.units_base[eid, uid] = find_closest_factory(eid, uid)
                        self.units_role[eid, uid] = new_role
                    elif new_role == ROLES.ENEMY:  # Find Enemy
                        self.units_target[eid, uid] = find_closest_enemy(eid, uid)
                        # New target exists > change role
                        if self.units_target[eid, uid] != -1:
                            self.units_role[eid, uid] = new_role
                    elif new_role == ROLES.ICE:  # Find Ice
                        self.units_gather_loc[eid, uid] = find_closest_resource(eid, uid, 'ice')
                        self.units_role[eid, uid] = new_role
                    elif new_role == ROLES.ORE:  # Find Ore
                        self.units_gather_loc[eid, uid] = find_closest_resource(eid, uid, 'ore')
                        self.units_role[eid, uid] = new_role
                    elif new_role == ROLES.RUBBLE:  # Find Rubble
                        self.units_gather_loc[eid, uid] = find_closest_resource(eid, uid, 'rubble')
                        self.units_role[eid, uid] = new_role

        res_actions = self.convert_parallel_actions(actions, self.num_envs)
        action_unit_4 = time()
        self.time_stats['action_unit_4'].append(action_unit_4 - action_unit_3)
        # print(f'Feed Unit 4: {(action_unit_4 - action_unit_3) * 1000:4.0f} ms')
        self.time_stats['feed_action'].append(time() - feed_action_start)
        return res_actions, log_probs, entropies, self.get_value(observation_input)

    @staticmethod
    def convert_parallel_actions(actions, num_envs=10):
        new_actions = [dict() for _ in range(num_envs)]
        for k in actions.keys():
            for eid in range(num_envs):
                new_actions[eid][k] = actions[k][eid]
        return new_actions


def test():
    from wrappers import create_env
    from lux.utils import draw

    num_envs = 4
    envs = gym.vector.SyncVectorEnv([create_env(with_opponent=False) for _ in range(num_envs)])
    agent = Agent(envs)

    for env_id, env in enumerate(envs.envs):
        env.set_agent(agent, env_id)
    observations = envs.reset()
    for _ in range(100):
        actions, logprobs, entropies, values = agent.get_action_and_value(observations)
        observations, rewards, dones, infos = envs.step(actions)

    # =============
    # Debug
    for env_id, env in enumerate(envs.envs):
        draw(env.lux_env)

    # Number of parameters
    sum([np.prod(parameter.size()) for parameter in agent.parameters()])
