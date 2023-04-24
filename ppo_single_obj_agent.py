import copy
from time import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Categorical
from luxai_s2 import LuxAI_S2

from lux.config import EnvConfig
from lux.kit import obs_to_game_state, GameState
from lux.utils import my_turn_to_place_factory
from lux.const import Direction, Resource
from lux.factory import Factory
from lux.unit import Unit

from factory_setup import find_best_position
from nn_utils import layer_init


RESERVE_POWER = 50
FACTORY_POWER_CAPACITY = 10_000
FACTORY_CARGO_SPACE = 10_000
FACTORY_MAX_UNITS = 500


class Roles:
    BASE = 0
    GATHER_ICE = 1
    GATHER_ORE = 2
    DIG_RUBBLE = 3


def move_function(unit, direction):
    def f():
        return unit.move(direction)

    return f


def transfer_function(unit, direction, resource, amount):
    def f():
        return unit.transfer(direction, resource, amount)
    return f


def pickup_function(unit, resource, amount):
    def f():
        return unit.pickup(resource, amount)
    return f


def charge_function(unit, amount):
    def f():
        return unit.recharge(amount)
    return f


def get_transfer_direction(factory, unit) -> (bool, int):
    """
    Value of dist
         (X   ,  Y)
        +101-  +222+
        2===2  1===1
        2=F=2  0=F=0
        2===2  1===1
        +101-  -222-

    :return can_transfer, direction_to_transfer
    """
    dist = factory.pos[0] - unit.pos[0], factory.pos[1] - unit.pos[1]
    if abs(dist[0]) > 2 or abs(dist[1]) > 2 or (abs(dist[0]) == abs(dist[1]) == 2):
        return False, -1

    if abs(dist[0]) <= 1 and abs(dist[1] <=1):
        return True, Direction.CENTER
    elif dist[1] == -2:  # Direction.SHIFT[Direction.UP]
        return True, Direction.UP
    elif dist[1] == 2:  # Direction.SHIFT[Direction.DOWN]
        return True, Direction.DOWN
    elif dist[0] == -2:  # Direction.SHIFT[Direction.LEFT]
        return True, Direction.LEFT
    elif dist[0] == 2:  # Direction.SHIFT[Direction.RIGHT]
        return True, Direction.RIGHT

    return True, Direction.CENTER


def pprint(data):
    print(f'''====================================================================
                               Stats
====================================================================
[Action Queue]
  {data['action_queue_updates_total']:5d} ({data['action_queue_updates_total']:5d} Success)
[Generation]
   Power: LIGHT({data['generation']['power']['LIGHT']:5d}) HEAVY({data['generation']['power']['HEAVY']:5d}) FACTORY({data['generation']['power']['FACTORY']:5d})
     Ice: LIGHT({data['generation']['ice']['LIGHT']:5d}) HEAVY({data['generation']['ice']['HEAVY']:5d})  Water: {data['generation']['water']:5d}
     Ore: LIGHT({data['generation']['ore']['LIGHT']:5d}) HEAVY({data['generation']['ore']['HEAVY']:5d})  Metal: {data['generation']['metal']:5d}
   Built: LIGHT({data['generation']['built']['LIGHT']:5d}) HEAVY({data['generation']['built']['HEAVY']:5d}) Lichen: {data['generation']['lichen']:5d}
[Consumption]
   Power: LIGHT({data['consumption']['power']['LIGHT']:5d}) HEAVY({data['consumption']['power']['HEAVY']:5d}) FACTORY({data['consumption']['power']['FACTORY']:5d})
     Ice: LIGHT({data['consumption']['ice']['LIGHT']:5d}) HEAVY({data['consumption']['ice']['HEAVY']:5d})  Water: {data['consumption']['water']:5d}
     Ore: LIGHT({data['consumption']['ore']['LIGHT']:5d}) HEAVY({data['consumption']['ore']['HEAVY']:5d})  Metal: {data['consumption']['metal']:5d}
[Destroyed]
    Unit: LIGHT({data['destroyed']['LIGHT']:5d}) HEAVY({data['destroyed']['HEAVY']:5d}) FACTORY({data['destroyed']['FACTORY']:5d})
  Rubble: LIGHT({data['destroyed']['rubble']['LIGHT']:5d}) HEAVY({data['destroyed']['rubble']['HEAVY']:5d})
  Lichen: LIGHT({data['destroyed']['lichen']['LIGHT']:5d}) HEAVY({data['destroyed']['lichen']['HEAVY']:5d})
[Transfer]
   Power: {data['transfer']['power']:5d}  Ice: {data['transfer']['ice']:5d}  Water: {data['transfer']['water']:5d}  Ore: {data['transfer']['ore']:5d}  Metal: {data['transfer']['metal']:5d}
[Pickup]
   Power: {data['pickup']['power']:5d}  Ice: {data['pickup']['ice']:5d}  Water: {data['pickup']['water']:5d}  Ore: {data['pickup']['ore']:5d}  Metal: {data['pickup']['metal']:5d}
====================================================================
''')


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=None, device=None):
        self.masks = masks
        self.device = device
        if self.masks is None:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = self.to_device(self.masks)
            logits = th.where(self.masks, logits, self.to_device(th.tensor(-1e+8)))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def to_device(self, tensor):
        if self.device is not None:
            return tensor.to(self.device)
        return tensor

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = th.where(self.masks, p_log_p, self.to_device(th.tensor(0.)))
        return -p_log_p.sum(-1)


class Agent(nn.Module):
    def __init__(self, max_units):
        super(Agent, self).__init__()
        self.MAX_UNITS = max_units
        self.game_environments = dict()
        self.game_player = dict()
        self.game_opponent = dict()
        self.game_observation = dict()
        self.game_states = dict()
        self.game_actions = dict()
        self.game_actions_vec = dict()
        self.game_actions_log = dict()
        self.game_log_probs = dict()
        self.game_entropies = dict()
        self.game_values = dict()
        self.game_stats = dict()
        self.game_rewards = dict()
        self.game_memory = dict()
        self.default_env = None
        self.__init_model__()

    def register(self, env_id, env, as_player='player_0'):
        players = ['player_0', 'player_1']
        assert as_player in players

        self.game_environments[env_id] = env
        self.game_player[env_id] = as_player
        self.game_opponent[env_id] = players[1 - players.index(as_player)]
        self.reset(env_id)
        if self.default_env is None:
            self.default_env = env_id

    def reset(self, env_id):
        self.game_observation[env_id] = []
        self.game_states[env_id] = []
        self.game_actions[env_id] = []
        self.game_actions_vec[env_id] = []
        self.game_actions_log[env_id] = []
        self.game_log_probs[env_id] = []
        self.game_entropies[env_id] = []
        self.game_values[env_id] = []
        self.game_stats[env_id] = []
        self.game_rewards[env_id] = []
        self.game_memory[env_id] = dict(
            unit=dict(),
            unit_position=[],
            unit_roles=dict(),
            unit_base=dict(),
            unit_gather_pos=dict(),
            factory=dict(),
            factory_position=[],
            factory_num_units=dict(),
            factory_num_roles=dict()
        )

    def __init_model__(self):
        factory_features = 13
        factory_actions = 3
        self.model_factory_critic = nn.Sequential(
            layer_init(nn.Linear(factory_features, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1),
        )
        self.model_factory_action = nn.Sequential(
            layer_init(nn.Linear(factory_features, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, factory_actions), std=0.01),
        )

        unit_features = 28
        unit_actions = 13
        self.model_unit_critic = nn.Sequential(
            layer_init(nn.Linear(unit_features, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1),
        )
        self.model_unit_action = nn.Sequential(
            layer_init(nn.Linear(unit_features, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, unit_actions), std=0.01),
        )

    def early_setup(self, step: int, observation, remainingOverageTime: int = 60, env_id: str = None):
        if env_id is None:
            env_id = self.default_env
        if step == 0:
            return dict(faction='AlphaStrike', bid=0)

        env_cfg = self.game_environments[env_id].get_config()
        player = self.game_player[env_id]
        game_state: GameState = obs_to_game_state(step, env_cfg, observation)

        if my_turn_to_place_factory(game_state.teams[player].place_first, step)\
            and game_state.teams[player].factories_to_place > 0:
            num_factory = len(game_state.factories[player].values())
            action = dict()
            action['spawn'] = find_best_position(step, observation, env_cfg, split=num_factory > 0, space=game_state.teams[player].factories_to_place == 1)
            action['metal'] = game_state.teams[player].metal // game_state.teams[player].factories_to_place
            action['water'] = game_state.teams[player].water // game_state.teams[player].factories_to_place

            return action

        return dict()

    def act(self, step: int, observation, remainingOverageTime: int = 60, env_id: str = None):
        # self = agent
        # step = env.lux_env.state.real_env_steps
        # observation = env.last_observation['player_0']
        # env_id = env.env_id + '_' + 'player_0'
        if env_id is None:
            env_id = self.default_env

        game_state = obs_to_game_state(step, self.game_environments[env_id].get_config(), observation)
        vec_observation = self.observe_game_state(game_state, env_id)
        actions, log_probs, entropies, values, logs = self.get_action_and_value(vec_observation, env_id=env_id)

        # Start logging from first action state
        if values is not None:
            self.game_observation[env_id].append(vec_observation)
            self.game_states[env_id].append(game_state)
            self.game_actions[env_id].append(actions)
            self.game_actions_log[env_id].append(logs)
            self.game_log_probs[env_id].append(log_probs)
            self.game_entropies[env_id].append(entropies)
            self.game_values[env_id].append(values)

        return actions

    def observe_game_state(self, game_state, env_id: str = None):
        # self = agent
        if env_id is None:
            env_id = self.default_env

        player = self.game_player[env_id]
        opponent = self.game_opponent[env_id]

        # game_state = obs_to_game_state(0, self.game_environments[env_id].get_config(), last_observation)

        factories = list(game_state.factories[player].values())  # type: list[Factory]
        units = list(game_state.units[player].values())  # type: list[Unit]
        enemies = list(game_state.units[opponent].values())  # type: list[Unit]

        game_env = self.game_environments[env_id]
        env_cfg = game_env.get_config()  # type: EnvConfig
        map_size = env_cfg.map_size - 1

        game_memory = self.game_memory[env_id]
        unit_roles = game_memory['unit_roles']  # type: dict[str, int]
        unit_base = game_memory['unit_base']  # type: dict[str, Factory]
        unit_gather_pos = game_memory['unit_gather_pos']  # type: dict[str, np.ndarray]
        factory_num_units = game_memory['factory_num_units']
        factory_num_roles = game_memory['factory_num_roles']

        # ==============================================================================================================
        # Map position for collision checking
        units_pos_mapping = dict()
        game_memory['unit_position'] = []
        game_memory['enemy_position'] = []
        game_memory['factory_position'] = []
        for u in units:
            units_pos_mapping[tuple(u.pos)] = u
            game_memory['unit_position'].append(u.pos)
        for u in enemies:
            units_pos_mapping[tuple(u.pos)] = u
            game_memory['enemy_position'].append(u.pos)
        for f in factories:
            game_memory['factory_position'].append(f.pos)
        game_memory['unit_position'] = np.asarray(game_memory['unit_position'])
        game_memory['enemy_position'] = np.asarray(game_memory['enemy_position'])
        game_memory['factory_position'] = np.asarray(game_memory['factory_position'])
        game_memory['ice_position'] = np.argwhere(game_state.board.ice)
        game_memory['ore_position'] = np.argwhere(game_state.board.ore)

        # Initialize for a new factory
        for f in factories:
            if f.unit_id not in game_memory['factory']:
                game_memory['factory'][f.unit_id] = f
                factory_num_units[f.unit_id] = dict(
                    LIGHT=0,
                    HEAVY=0,
                )
                factory_num_roles[f.unit_id] = {
                    Roles.BASE: 0,
                    Roles.GATHER_ICE: 0,
                    Roles.GATHER_ORE: 0,
                    Roles.DIG_RUBBLE: 0,
                }

        # Initialize for a new unit
        for u in units:
            if u.unit_id not in game_memory['unit']:
                game_memory['unit'][u.unit_id] = u
                f = factories[self.find_closest_point(u.pos, game_memory['factory_position'], return_idx=True)]
                unit_roles[u.unit_id] = Roles.GATHER_ICE
                unit_base[u.unit_id] = f
                unit_gather_pos[u.unit_id] = self.find_closest_point(u.pos, game_memory['ice_position'])
                factory_num_units[f.unit_id][u.unit_type] += 1
                factory_num_roles[f.unit_id][Roles.GATHER_ICE] += 1

        # ==============================================================================================================
        # Create Observation & Action masking
        unit_mapping = dict()
        unit_ids = []
        unit_observation = []
        unit_flags = []
        unit_funcs = []
        unit_masks = []
        unit_logs = [
            'MOV_UP', 'MOV_RIGHT', 'MOV_DOWN', 'MOV_LEFT',
            'DIG_ICE', 'DIG_ORE', 'DIG_RUBBLE', 'TRANSFER_ICE', 'TRANSFER_ORE', 'CHARGE',
            'POW_50', 'POW_150', 'POW_300'
        ]
        idx = 0
        for u in units:
            # Skip queue unit
            if len(u.action_queue) > 0:
                # print('Skip unit', u.unit_id, 'with queue', u.action_queue)
                continue
            unit_ids.append(u.unit_id)
            unit_mapping[u.unit_id] = idx

            can_charge = u.power < env_cfg.ROBOTS[u.unit_type].BATTERY_CAPACITY
            charge_action = charge_function(u, min(
                u.power + env_cfg.ROBOTS[u.unit_type].CHARGE * 5,  # Charge 5 Turn
                env_cfg.ROBOTS['LIGHT'].BATTERY_CAPACITY  # Maximum
            ))

            move_flags = []
            move_actions = []
            for direction in Direction.all:
                new_pos = Direction.shift(u.pos, direction)
                in_map = 0 <= new_pos[0] <= map_size and 0 <= new_pos[1] <= map_size
                if in_map:
                    not_factory = game_state.board.factory_occupancy_map[new_pos[0], new_pos[1]]
                    not_unit = new_pos not in units_pos_mapping
                    can_move = not_unit and not_factory and \
                        u.power > u.move_cost(game_state, direction)
                else:
                    can_move = False
                move_flags.append(can_move)
                move_actions.append(move_function(u, direction))

            can_dig = u.dig_cost(game_state) <= u.power
            can_dig_ice = can_dig and game_state.board.ice[u.pos[0], u.pos[1]] == 1
            can_dig_ore = can_dig and game_state.board.ore[u.pos[0], u.pos[1]] == 1
            can_dig_rubble = can_dig and game_state.board.rubble[u.pos[0], u.pos[1]] == 1
            dig_act = u.dig

            can_transfer, transfer_direction = get_transfer_direction(unit_base[u.unit_id], u)
            can_transfer_ice = can_transfer and u.cargo.ice > 0
            can_transfer_ore = can_transfer and u.cargo.ore > 0
            tf_ice_action = transfer_function(u, transfer_direction, Resource.ICE, u.cargo.ice)
            tf_ore_action = transfer_function(u, transfer_direction, Resource.ORE, u.cargo.ore)

            if can_transfer and transfer_direction == Direction.CENTER:
                can_pickup_power = [unit_base[u.unit_id].power - RESERVE_POWER > p for p in [50, 150, 300]]
            else:
                can_pickup_power = [False] * 3
            pickup_power = [pickup_function(u, Resource.POWER, p) for p in [50, 150, 300]]
            action_flags = move_flags + [
                can_dig_ice, can_dig_ore, can_dig_rubble, can_transfer_ice, can_transfer_ore, can_charge
            ] + can_pickup_power

            action_funcs = move_actions + [
                dig_act, dig_act, dig_act, tf_ice_action, tf_ore_action, charge_action
            ] + pickup_power

            unit_flags.append(action_flags)
            unit_funcs.append(action_funcs)
            unit_masks.append(action_flags)

            power_capacity = env_cfg.ROBOTS[u.unit_type].BATTERY_CAPACITY
            cargo_space = env_cfg.ROBOTS[u.unit_type].CARGO_SPACE
            u_obs = np.hstack([u.pos/map_size, np.asarray([
                u.power/power_capacity,
                u.cargo.ice/cargo_space,
                u.cargo.ore/cargo_space,
                u.cargo.water/cargo_space,
                u.cargo.metal/cargo_space,
                1 if u.unit_type == 'LIGHT' else 0,
                1 if u.unit_type == 'HEAVY' else 0,
                1 if unit_roles[u.unit_id] == Roles.BASE else 0,
                1 if unit_roles[u.unit_id] == Roles.GATHER_ICE else 0,
                1 if unit_roles[u.unit_id] == Roles.GATHER_ORE else 0,
                1 if unit_roles[u.unit_id] == Roles.DIG_RUBBLE else 0,
            ])])
            unit_observation.append(u_obs)
            idx += 1

        factory_mapping = dict()
        factory_ids = []
        factory_observation = []
        factory_flags = []
        factory_funcs = []
        factory_masks = []
        factory_logs = ['WATER', 'BUILD_LIGHT', 'BUILD_HEAVY']
        for idx, f in enumerate(factories):
            factory_ids.append(f.unit_id)
            factory_mapping[f.unit_id] = idx
            number_of_units = sum(factory_num_units[f.unit_id].values()) < self.MAX_UNITS
            empty_slot = tuple(f.pos) not in units_pos_mapping
            action_flags = [
                f.can_water(game_state),
                empty_slot and f.can_build_light(game_state, reserve_power=RESERVE_POWER) and number_of_units,
                empty_slot and f.can_build_heavy(game_state, reserve_power=RESERVE_POWER) and number_of_units,
            ]
            action_funcs = [f.water, f.build_light, f.build_heavy]

            factory_flags.append(action_flags)
            factory_funcs.append(action_funcs)
            factory_masks.append(action_flags)

            f_obs = np.hstack([f.pos/map_size, np.asarray([
                f.power/FACTORY_POWER_CAPACITY,
                f.cargo.ice/FACTORY_CARGO_SPACE,
                f.cargo.ore/FACTORY_CARGO_SPACE,
                f.cargo.water/FACTORY_CARGO_SPACE,
                f.cargo.metal/FACTORY_CARGO_SPACE,
                factory_num_units[f.unit_id]['LIGHT']/FACTORY_MAX_UNITS,
                factory_num_units[f.unit_id]['HEAVY']/FACTORY_MAX_UNITS,
                factory_num_roles[f.unit_id][Roles.BASE]/FACTORY_MAX_UNITS,
                factory_num_roles[f.unit_id][Roles.GATHER_ICE]/FACTORY_MAX_UNITS,
                factory_num_roles[f.unit_id][Roles.GATHER_ORE]/FACTORY_MAX_UNITS,
                factory_num_roles[f.unit_id][Roles.DIG_RUBBLE]/FACTORY_MAX_UNITS,
            ])])
            factory_observation.append(f_obs)

        # Expand units' observation
        idx = 0
        for u in units:
            # Skip queue unit
            if len(u.action_queue) > 0:
                continue
            fidx = factory_mapping[unit_base[u.unit_id].unit_id]
            u_obs = [unit_observation[idx], factory_observation[fidx]]
            if unit_roles[u.unit_id] == Roles.GATHER_ICE:
                u_obs.append(self.find_closest_point(u.pos, game_memory['ice_position']))
            elif unit_roles[u.unit_id] == Roles.GATHER_ORE:
                u_obs.append(self.find_closest_point(u.pos, game_memory['ore_position']))
            else:
                u_obs.append(np.zeros(2,))
            unit_observation[idx] = np.hstack(u_obs)
            idx += 1

        unit_observation = th.Tensor(np.asarray(unit_observation))
        unit_masks = th.Tensor(np.asarray(unit_masks)).type(th.BoolTensor)
        factory_observation = th.Tensor(np.asarray(factory_observation))
        factory_masks = th.Tensor(np.asarray(factory_masks)).type(th.BoolTensor)

        return dict(
            factory=dict(
                ids=factory_ids,
                flag=factory_flags,
                observation=factory_observation,
                action_mask=factory_masks,
                func=factory_funcs,
                log=factory_logs,
            ),
            unit=dict(
                ids=unit_ids,
                flag=unit_flags,
                observation=unit_observation,
                action_mask=unit_masks,
                func=unit_funcs,
                log=unit_logs,
            )
        )

    # def get_action(self, game_state, env_id: str = None):
    #     if env_id is None:
    #         env_id = self.default_env
    #     player = self.game_player[env_id]
    #     opponent = self.game_opponent[env_id]
    #
    #     player_actions = dict()
    #     player_logs = dict()
    #     factories = list(game_state.factories[player].values())
    #
    #     units_pos_mapping = dict()
    #     for u in game_state.units[player].values():
    #         units_pos_mapping[tuple(u.pos)] = u
    #     for u in game_state.units[opponent].values():
    #         units_pos_mapping[tuple(u.pos)] = u
    #
    #     # player_team_id = game_state.teams[player].team_id
    #     units = list(game_state.units[player].values())
    #     base = factories[0]
    #
    #     # Unit Action
    #     for u in units:
    #         can_move = False
    #         move_flags = []
    #         move_actions = []
    #         for direction in Direction.all:
    #             move_flags.append(Direction.shift(u.pos, direction) not in units_pos_mapping)
    #             if move_flags[-1]:
    #                 can_move = True
    #             move_actions.append(move_function(u, direction))
    #         move_dict = dict(flags=move_flags, actions=move_actions, logs=Direction.logs[1:])
    #
    #         can_dig = u.dig_cost(game_state) <= u.power
    #         can_dig_ice = can_dig and game_state.board.ice[u.pos[0], u.pos[1]]
    #         can_dig_ore = can_dig and game_state.board.ore[u.pos[0], u.pos[1]]
    #         can_dig_rubble = can_dig and game_state.board.rubble[u.pos[0], u.pos[1]]
    #         dig_act = u.dig
    #
    #         can_transfer, transfer_direction = get_transfer_direction(base, u)
    #         can_transfer_ice = can_transfer and u.cargo.ice > 0
    #         can_transfer_ore = can_transfer and u.cargo.ore > 0
    #         tf_ice_action = transfer_function(u, transfer_direction, Resource.ICE, u.cargo.ice)
    #         tf_ore_action = transfer_function(u, transfer_direction, Resource.ORE, u.cargo.ore)
    #
    #         unit_actions = dict()
    #         unit_actions['flags'] = [can_move, can_dig_ice, can_dig_ore, can_dig_rubble, can_transfer_ice, can_transfer_ore]
    #         unit_actions['actions'] = [move_dict, dig_act, dig_act, dig_act, tf_ice_action, tf_ore_action]
    #         unit_actions['logs'] = ['MOV_', 'DIG', 'TF_ICE', 'TF_ORE']
    #
    #         act = unit_actions
    #         log = ''
    #         while not callable(act):
    #             p = act
    #             choices = np.argwhere(p['flags']).flatten()
    #             if len(choices) == 0:
    #                 act = None
    #                 break
    #             choose = np.random.choice(choices)
    #             act = p['actions'][choose]
    #             log += p['logs'][choose]
    #
    #         if act is not None:
    #             if log.startswith('MOV_'):
    #                 direction = log.split('_')[1]
    #                 new_pos = Direction.shift(u.pos, getattr(Direction, direction))
    #                 units_pos_mapping[new_pos] = units_pos_mapping[tuple(u.pos)]
    #                 # !! Collision can be happened if old_pos == others' new_pos
    #                 # del units_pos_mapping[tuple(u.pos)]
    #
    #             player_actions[u.unit_id] = [act()]
    #             player_logs[u.unit_id] = [log]
    #
    #     # Factory Action
    #     for f in factories:
    #         vacancy = tuple(f.pos) not in units_pos_mapping
    #         actions = [f.water, f.build_light, f.build_heavy]
    #         flags = [f.can_water(game_state), vacancy and f.can_build_light(game_state), vacancy and f.can_build_heavy(game_state)]
    #         logs = ['WATER', 'B_LIGHT', 'B_HEAVY']
    #         choices = np.argwhere(flags).flatten()
    #         if len(choices) > 0:
    #             choose = np.random.choice(choices)
    #             player_actions[f.unit_id] = actions[choose]()
    #             player_logs[f.unit_id] = logs[choose]
    #
    #     return player_actions, player_logs

    def get_reward(self, observation, stats, env_id: str = None):
        # self = agent
        if env_id is None:
            env_id = self.default_env
        # observation = last_observation['player_0']
        # stats = self.game_stats[env_id][-1]

        reward = 0
        player = self.game_player[env_id]
        opponent = self.game_opponent[env_id]

        # print('==============================================================================')
        # print(stats)
        # print(f'[{env_id}]', 'Stats', len(self.game_stats[env_id]), 'States', len(self.game_states[env_id]))
        # No previous stats

        # Score from states
        if len(self.game_states) > 0:
            previous_game_state = self.game_states[env_id][-1]  # type: GameState
            units = observation['units'][player]
            previous_units = previous_game_state.units[player]
            if len(units) > len(previous_units):
                reward = 1 - len(previous_units)/self.MAX_UNITS
            if len(units) < len(previous_units):
                reward -= 1

        # Score from stats
        if len(self.game_stats[env_id]) > 0:
            previous_stats = self.game_stats[env_id][-1]
            if stats['generation']['ice']['HEAVY'] + stats['generation']['ice']['LIGHT'] > \
                previous_stats['generation']['ice']['HEAVY'] + previous_stats['generation']['ice']['LIGHT']:
                reward += 1
            if stats['transfer']['water'] > previous_stats['transfer']['water']:
                reward += 1
            if stats['generation']['built']['LIGHT'] > previous_stats['generation']['built']['LIGHT']:
                reward += 1
            if stats['generation']['built']['HEAVY'] > previous_stats['generation']['built']['HEAVY']:
                reward += 1

        self.game_stats[env_id].append(copy.deepcopy(stats))
        self.game_rewards[env_id].append(reward)
        return reward

    def find_closest_point(self, unit_pos, target_pos, return_idx=False):
        # start = time()
        target_dist = abs(unit_pos - target_pos).sum(axis=1)
        res = np.argmin(target_dist)
        # self.time_stats['find_closest_point'].append(time() - start)
        if return_idx:
            return res
        return target_pos[res]

    def select_random_action(self, action_flags):
        choices = np.argwhere(action_flags).flatten()
        if len(choices) == 0:
            return None
        choose = np.random.choice(choices)
        return choose

    def get_value(self, vec_observation):
        vec_unit = vec_observation['unit']
        unit_obs = vec_unit['observation']
        vec_factory = vec_observation['factory']
        fact_obs = vec_factory['observation']

        if len(unit_obs) == 0:
            return self.model_factory_critic(fact_obs).sum()
        else:
            return self.model_factory_critic(fact_obs).sum() + self.model_unit_critic(unit_obs).sum()

    def get_action_and_value(self,
                             vec_observation,
                             factory_actions=None, unit_actions=None,
                             save_actions=True,
                             random_factory=False,
                             env_id: str=None
                             ):
        # self = agent
        # vec_actions=None

        actions = dict()
        logs = dict()
        vec_factory = vec_observation['factory']
        fact_log = vec_factory['log']
        # === Random factory action
        if random_factory:
            for idx, fid in enumerate(vec_factory['ids']):
                flag = vec_factory['flag'][idx]
                func = vec_factory['func'][idx]
                act_id = self.select_random_action(flag)
                if act_id is not None:
                    actions[fid] = func[act_id]()
                    logs[fid] = fact_log[act_id]
        else:
            fact_obs = vec_factory['observation']
            fact_mask = vec_factory['action_mask']

            fact_action_val = self.model_factory_action(fact_obs)
            fact_action_prob = CategoricalMasked(logits=fact_action_val, masks=fact_mask)
            if factory_actions is None:
                factory_actions = fact_action_prob.sample()

            for idx, fid in enumerate(vec_factory['ids']):
                func = vec_factory['func'][idx]
                act_id = factory_actions[idx].item()
                actions[fid] = func[act_id]()
                logs[fid] = fact_log[act_id]

        # === Get Unit action
        if len(vec_observation['unit']['ids']) == 0:
            sum_log_prob = fact_action_prob.log_prob(factory_actions).sum()
            sum_entropy = fact_action_prob.entropy().sum()
            return actions, sum_log_prob, sum_entropy, self.get_value(vec_observation), logs

        vec_unit = vec_observation['unit']
        unit_obs = vec_unit['observation']
        unit_mask = vec_unit['action_mask']
        unit_log = vec_unit['log']

        unit_action_val = self.model_unit_action(unit_obs)
        unit_action_prob = CategoricalMasked(logits=unit_action_val, masks=unit_mask)
        if unit_actions is None:
            unit_actions = unit_action_prob.sample()

        for idx, uid in enumerate(vec_unit['ids']):
            func = vec_unit['func'][idx]
            act_id = unit_actions[idx].item()
            actions[uid] = [
                func[act_id]()
            ]
            logs[uid] = unit_log[act_id]

        sum_log_prob = unit_action_prob.log_prob(unit_actions).sum()
        sum_entropy = unit_action_prob.entropy().sum()

        if save_actions:
            self.game_actions_vec[env_id].append(dict(
                factory=factory_actions,
                unit=unit_actions,
            ))

        return actions, sum_log_prob, sum_entropy, self.get_value(vec_observation), logs

