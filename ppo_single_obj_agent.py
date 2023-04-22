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

from factory_setup import find_best_position
from nn_utils import layer_init


def move_function(unit, direction):
    def f():
        return unit.move(direction)

    return f


def transfer_function(unit, direction, resource, amount):
    def f():
        return unit.transfer(direction, resource, amount)

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

    if dist[1] == -2:  # Direction.SHIFT[Direction.UP]
        return True, Direction.UP
    elif dist[1] == 2:  # Direction.SHIFT[Direction.DOWN]
        return True, Direction.DOWN
    elif dist[0] == -2:  # Direction.SHIFT[Direction.LEFT]
        return True, Direction.LEFT
    elif dist[0] == 2:  # Direction.SHIFT[Direction.RIGHT]
        return True, Direction.RIGHT

    return True, Direction.CENTER


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.game_environments = dict()
        self.game_player = dict()
        self.game_opponent = dict()
        self.game_states = dict()
        self.game_actions = dict()
        self.game_action_logs = dict()
        self.default_env = None

    def register(self, env_id, env, as_player='player_0'):
        players = ['player_0', 'player_1']
        assert as_player in players

        self.game_environments[env_id] = env
        self.game_player[env_id] = as_player
        self.game_opponent[env_id] = players[1 - players.index(as_player)]
        self.game_states[env_id] = []
        self.game_actions[env_id] = []
        self.game_action_logs[env_id] = []
        if self.default_env is None:
            self.default_env = env_id

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
        if env_id is None:
            env_id = self.default_env

        game_state = obs_to_game_state(step, self.game_environments[env_id].get_config(), observation)
        self.game_states[env_id].append(game_state)
        # vec_observation = self.transform_observation(observation, game_state, env_id)
        actions, logs = self.get_action(game_state, env_id)
        self.game_actions[env_id].append(actions)
        self.game_action_logs[env_id].append(logs)

        return actions

    def get_action(self, game_state, env_id: str = None):
        if env_id is None:
            env_id = self.default_env
        player = self.game_player[env_id]
        opponent = self.game_opponent[env_id]

        player_actions = dict()
        player_logs = dict()
        factories = list(game_state.factories[player].values())

        units_pos_mapping = dict()
        for u in game_state.units[player].values():
            units_pos_mapping[tuple(u.pos)] = u
        for u in game_state.units[opponent].values():
            units_pos_mapping[tuple(u.pos)] = u


        # player_team_id = game_state.teams[player].team_id
        units = list(game_state.units[player].values())
        base = factories[0]

        # Unit Action
        for u in units:
            can_move = False
            move_flags = []
            move_actions = []
            for direction in Direction.all:
                move_flags.append(Direction.shift(u.pos, direction) not in units_pos_mapping)
                if move_flags[-1]:
                    can_move = True
                move_actions.append(move_function(u, direction))
            move_dict = dict(flags=move_flags, actions=move_actions, logs=Direction.logs[1:])

            can_dig = u.dig_cost(game_state) <= u.power
            can_dig_ice = can_dig and game_state.board.ice[u.pos[0], u.pos[1]]
            can_dig_ore = can_dig and game_state.board.ore[u.pos[0], u.pos[1]]
            can_dig_rubble = can_dig and game_state.board.rubble[u.pos[0], u.pos[1]]
            dig_act = u.dig

            can_transfer, transfer_direction = get_transfer_direction(base, u)
            can_transfer_ice = can_transfer and u.cargo.ice > 0
            can_transfer_ore = can_transfer and u.cargo.ore > 0
            tf_ice_action = transfer_function(u, transfer_direction, Resource.ICE, u.cargo.ice)
            tf_ore_action = transfer_function(u, transfer_direction, Resource.ORE, u.cargo.ore)

            unit_actions = dict()
            unit_actions['flags'] = [can_move, can_dig_ice, can_dig_ore, can_dig_rubble, can_transfer_ice,
                                     can_transfer_ore]
            unit_actions['actions'] = [move_dict, dig_act, dig_act, dig_act, tf_ice_action, tf_ore_action]
            unit_actions['logs'] = ['MOV_', 'DIG', 'TF_ICE', 'TF_ORE']

            act = unit_actions
            log = ''
            while not callable(act):
                p = act
                choices = np.argwhere(p['flags']).flatten()
                if len(choices) == 0:
                    act = None
                    break
                choose = np.random.choice(choices)
                act = p['actions'][choose]
                log += p['logs'][choose]

            if act is not None:
                if log.startswith('MOV_'):
                    direction = log.split('_')[1]
                    new_pos = Direction.shift(u.pos, getattr(Direction, direction))
                    units_pos_mapping[new_pos] = units_pos_mapping[tuple(u.pos)]
                    # !! Collision can be happened if old_pos == others' new_pos
                    # del units_pos_mapping[tuple(u.pos)]

                player_actions[u.unit_id] = [act()]
                player_logs[u.unit_id] = [log]

        # Factory Action
        for f in factories:
            vacancy = tuple(f.pos) not in units_pos_mapping
            actions = [f.water, f.build_light, f.build_heavy]
            flags = [f.can_water(game_state), vacancy and f.can_build_light(game_state), vacancy and f.can_build_heavy(game_state)]
            logs = ['WATER', 'B_LIGHT', 'B_HEAVY']
            choices = np.argwhere(flags).flatten()
            if len(choices) > 0:
                choose = np.random.choice(choices)
                player_actions[f.unit_id] = actions[choose]()
                player_logs[f.unit_id] = logs[choose]

        return player_actions, player_logs

    def get_reward(self, observation, stats, env_id: str = None):
        if env_id is None:
            env_id = self.default_env

        return 0
