import sys
import os.path as osp
import numpy as np
import numpy.typing as npt
from typing import Any, Callable, Dict
from stable_baselines3.ppo import PPO

from gym import spaces
from lux.kit import obs_to_game_state, GameState
from lux.config import EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory


class Controller:
    def __init__(self, action_space: spaces.Space) -> None:
        self.action_space = action_space

    def action_to_lux_action(
        self, agent: str, obs: Dict[str, Any], action: npt.NDArray
    ):
        """
        Takes as input the current "raw observation" and the parameterized action and returns
        an action formatted for the Lux env
        """
        raise NotImplementedError()

    def action_masks(self, agent: str, obs: Dict[str, Any]):
        """
        Generates a boolean action mask indicating in each discrete dimension whether it would be valid or not
        """
        raise NotImplementedError()


class SimpleUnitDiscreteController(Controller):
    def __init__(self, env_cfg) -> None:
        """
        A simple controller that controls only the robot that will get spawned.
        Moreover, it will always try to spawn one heavy robot if there are none regardless of action given

        For the robot unit
        - 4 cardinal direction movement (4 dims)
        - a move center no-op action (1 dim)
        - transfer action just for transferring ice in 4 cardinal directions or center (5)
        - pickup action for power (1 dims)
        - dig action (1 dim)
        - no op action (1 dim) - equivalent to not submitting an action queue which costs power

        It does not include
        - self destruct action
        - recharge action
        - planning (via actions executing multiple times or repeating actions)
        - factory actions
        - transferring power or resources other than ice

        To help understand how to this controller works to map one action space to the original lux action space,
        see how the lux action space is defined in luxai_s2/spaces/action.py

        """
        self.env_cfg = env_cfg
        self.move_act_dims = 4
        self.transfer_act_dims = 5
        self.pickup_act_dims = 1
        self.dig_act_dims = 1
        self.no_op_dims = 1

        self.move_dim_high = self.move_act_dims
        self.transfer_dim_high = self.move_dim_high + self.transfer_act_dims
        self.pickup_dim_high = self.transfer_dim_high + self.pickup_act_dims
        self.dig_dim_high = self.pickup_dim_high + self.dig_act_dims
        self.no_op_dim_high = self.dig_dim_high + self.no_op_dims

        self.total_act_dims = self.no_op_dim_high
        action_space = spaces.Discrete(self.total_act_dims)
        super().__init__(action_space)

    def _is_move_action(self, id):
        return id < self.move_dim_high

    def _get_move_action(self, id):
        # move direction is id + 1 since we don't allow move center here
        return np.array([0, id + 1, 0, 0, 0, 1])

    def _is_transfer_action(self, id):
        return id < self.transfer_dim_high

    def _get_transfer_action(self, id):
        id = id - self.move_dim_high
        transfer_dir = id % 5
        return np.array([1, transfer_dir, 0, self.env_cfg.max_transfer_amount, 0, 1])

    def _is_pickup_action(self, id):
        return id < self.pickup_dim_high

    def _get_pickup_action(self, id):
        return np.array([2, 0, 4, self.env_cfg.max_transfer_amount, 0, 1])

    def _is_dig_action(self, id):
        return id < self.dig_dim_high

    def _get_dig_action(self, id):
        return np.array([3, 0, 0, 0, 0, 1])

    def action_to_lux_action(
        self, player: str, obs: Dict[str, Any], commands: Dict[str, npt.NDArray]
    ):
        # self = agent.controller
        # player = agent.player
        lux_action = dict()
        units = obs["units"][player]
        for unit_id in units.keys():
            unit = units[unit_id]
            choice = commands[unit_id]
            action_queue = []
            no_op = False
            if self._is_move_action(choice):
                action_queue = [self._get_move_action(choice)]
            elif self._is_transfer_action(choice):
                action_queue = [self._get_transfer_action(choice)]
            elif self._is_pickup_action(choice):
                action_queue = [self._get_pickup_action(choice)]
            elif self._is_dig_action(choice):
                action_queue = [self._get_dig_action(choice)]
            else:
                # action is a no_op, so we don't update the action queue
                no_op = True

            # simple trick to help agents conserve power is to avoid updating the action queue
            # if the agent was previously trying to do that particular action already
            if len(unit["action_queue"]) > 0 and len(action_queue) > 0:
                same_actions = (unit["action_queue"][0] == action_queue[0]).all()
                if same_actions:
                    no_op = True
            if not no_op:
                lux_action[unit_id] = action_queue

        factories = obs["factories"][player]
        if len(units) == 0:
            for unit_id in factories.keys():
                lux_action[unit_id] = 1  # build a single heavy

        return lux_action

    def action_masks(self, agent: str, obs: Dict[str, Any]):
        """
        Defines a simplified action mask for this controller's action space

        Doesn't account for whether robot has enough power
        """

        # compute a factory occupancy map that will be useful for checking if a board tile
        # has a factory and which team's factory it is.
        shared_obs = obs
        factory_occupancy_map = (
            np.ones_like(shared_obs["board"]["rubble"], dtype=int) * -1
        )
        factories = dict()
        for player in shared_obs["factories"]:
            factories[player] = dict()
            for unit_id in shared_obs["factories"][player]:
                f_data = shared_obs["factories"][player][unit_id]
                f_pos = f_data["pos"]
                # store in a 3x3 space around the factory position it's strain id.
                factory_occupancy_map[
                    f_pos[0] - 1 : f_pos[0] + 2, f_pos[1] - 1 : f_pos[1] + 2
                ] = f_data["strain_id"]

        units = shared_obs["units"][agent]
        action_mask = np.zeros((self.total_act_dims), dtype=bool)
        for unit_id in units.keys():
            action_mask = np.zeros(self.total_act_dims)
            # movement is always valid
            action_mask[:4] = True

            # transferring is valid only if the target exists
            unit = units[unit_id]
            pos = np.array(unit["pos"])
            # a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
            move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
            for i, move_delta in enumerate(move_deltas):
                transfer_pos = np.array(
                    [pos[0] + move_delta[0], pos[1] + move_delta[1]]
                )
                # check if theres a factory tile there
                if (
                    transfer_pos[0] < 0
                    or transfer_pos[1] < 0
                    or transfer_pos[0] >= len(factory_occupancy_map)
                    or transfer_pos[1] >= len(factory_occupancy_map[0])
                ):
                    continue
                factory_there = factory_occupancy_map[transfer_pos[0], transfer_pos[1]]
                if factory_there in shared_obs["teams"][agent]["factory_strains"]:
                    action_mask[
                        self.transfer_dim_high - self.transfer_act_dims + i
                    ] = True

            factory_there = factory_occupancy_map[pos[0], pos[1]]
            on_top_of_factory = (
                factory_there in shared_obs["teams"][agent]["factory_strains"]
            )

            # dig is valid only if on top of tile with rubble or resources or lichen
            board_sum = (
                shared_obs["board"]["ice"][pos[0], pos[1]]
                + shared_obs["board"]["ore"][pos[0], pos[1]]
                + shared_obs["board"]["rubble"][pos[0], pos[1]]
                + shared_obs["board"]["lichen"][pos[0], pos[1]]
            )
            if board_sum > 0 and not on_top_of_factory:
                action_mask[
                    self.dig_dim_high - self.dig_act_dims : self.dig_dim_high
                ] = True

            # pickup is valid only if on top of factory tile
            if on_top_of_factory:
                action_mask[
                    self.pickup_dim_high - self.pickup_act_dims : self.pickup_dim_high
                ] = True
                action_mask[
                    self.dig_dim_high - self.dig_act_dims : self.dig_dim_high
                ] = False

            # no-op is always valid
            action_mask[-1] = True
            break
        return action_mask


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.model = PPO.load(osp.join("logs/exp_1", "models/latest_model"))
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        self.controller = SimpleUnitDiscreteController(env_cfg)
        self.units_factory_base = dict()

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if step == 0:
            return dict(faction="AlphaStrike", bid=0)

        game_state: GameState = obs_to_game_state(step, self.env_cfg, obs)
        if my_turn_to_place_factory(game_state.teams[self.player].place_first, step)\
                and game_state.teams[self.player].factories_to_place > 0:
            water_left = game_state.teams[self.player].water
            metal_left = game_state.teams[self.player].metal

            action = self.place_near_random_ice(obs)
            action['metal'] = metal_left // game_state.teams[self.player].factories_to_place
            action['water'] = water_left // game_state.teams[self.player].factories_to_place
            return action
        else:
            return dict()

        # if step == 0:
        #     # bid 0 to not waste resources bidding and declare as the default faction
        #     return dict(faction="AlphaStrike", bid=0)
        # else:
        #     game_state = obs_to_game_state(step, self.env_cfg, obs)
        #     # factory placement period
        #
        #     # how much water and metal you have in your starting pool to give to new factories
        #     water_left = game_state.teams[self.player].water
        #     metal_left = game_state.teams[self.player].metal
        #
        #     # how many factories you have left to place
        #     factories_to_place = game_state.teams[self.player].factories_to_place
        #     # whether it is your turn to place a factory
        #     my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)
        #     if factories_to_place > 0 and my_turn_to_place:
        #         # we will spawn our factory in a random location with 150 metal and water if it is our turn to place
        #         potential_spawns = np.array(list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))
        #         spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
        #         return dict(spawn=spawn_loc, metal=150, water=150)
        #     return dict()

    def place_near_random_ice(self, obs):
        """
        This policy will place a single factory with all the starting resources
        near a random ice tile
        """
        if obs["teams"][self.player]["metal"] == 0:
            return dict()
        potential_spawns = list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
        potential_spawns_set = set(potential_spawns)
        done_search = False

        # simple numpy trick to find locations adjacent to ice tiles.
        ice_diff = np.diff(obs["board"]["ice"])
        pot_ice_spots = np.argwhere(ice_diff == 1)
        if len(pot_ice_spots) == 0:
            pot_ice_spots = potential_spawns

        # pick a random ice spot and search around it for spawnable locations.
        trials = 5
        while trials > 0:
            pos_idx = np.random.randint(0, len(pot_ice_spots))
            pos = pot_ice_spots[pos_idx]
            area = 3
            for x in range(area):
                for y in range(area):
                    check_pos = [pos[0] + x - area // 2, pos[1] + y - area // 2]
                    if tuple(check_pos) in potential_spawns_set:
                        done_search = True
                        pos = check_pos
                        break
                if done_search:
                    break
            if done_search:
                break
            trials -= 1

        if not done_search:
            spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
            pos = spawn_loc
        return dict(spawn=pos)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        units_obs = self.convert_obs(obs)
        commands = dict([
            (unit_key, self.model.predict(unit_obs)[0])
            for unit_key, unit_obs in units_obs.items()
        ])
        return self.controller.action_to_lux_action(self.player, obs, commands)
        # obs_new
        # model.observation_space
        # action = model.predict(obs_new[agent.player])
        #
        # controller = SimpleUnitDiscreteController(env.env_cfg)
        # controller.action_to_lux_action(agent=agent.player, obs=obs_new, action=action)

        # actions = dict()
        #
        # """
        # optionally do forward simulation to simulate positions of units, lichen, etc. in the future
        # from lux.forward_sim import forward_sim
        # forward_obs = forward_sim(obs, self.env_cfg, n=2)
        # forward_game_states = [obs_to_game_state(step + i, self.env_cfg, f_obs) for i, f_obs in enumerate(forward_obs)]
        # """
        #
        # game_state = obs_to_game_state(step, self.env_cfg, obs)
        # factories = game_state.factories[self.player]
        # game_state.teams[self.player].place_first
        # factory_tiles, factory_units = [], []
        # for unit_id, factory in factories.items():
        #     if factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and \
        #             factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST:
        #         actions[unit_id] = factory.build_heavy()
        #     if factory.water_cost(game_state) <= factory.cargo.water / 5 - 200:
        #         actions[unit_id] = factory.water()
        #     factory_tiles += [factory.pos]
        #     factory_units += [factory]
        # factory_tiles = np.array(factory_tiles)
        #
        # units = game_state.units[self.player]
        # ice_map = game_state.board.ice
        # ice_tile_locations = np.argwhere(ice_map == 1)
        # for unit_id, unit in units.items():
        #
        #     # track the closest factory
        #     closest_factory = None
        #     adjacent_to_factory = False
        #     if len(factory_tiles) > 0:
        #         factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
        #         closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
        #         closest_factory = factory_units[np.argmin(factory_distances)]
        #         adjacent_to_factory = np.mean((closest_factory_tile - unit.pos) ** 2) == 0
        #
        #         # previous ice mining code
        #         if unit.cargo.ice < 40:
        #             ice_tile_distances = np.mean((ice_tile_locations - unit.pos) ** 2, 1)
        #             closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
        #             if np.all(closest_ice_tile == unit.pos):
        #                 if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
        #                     actions[unit_id] = [unit.dig(repeat=0, n=1)]
        #             else:
        #                 direction = direction_to(unit.pos, closest_ice_tile)
        #                 move_cost = unit.move_cost(game_state, direction)
        #                 if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
        #                     actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
        #         # else if we have enough ice, we go back to the factory and dump it.
        #         elif unit.cargo.ice >= 40:
        #             direction = direction_to(unit.pos, closest_factory_tile)
        #             if adjacent_to_factory:
        #                 if unit.power >= unit.action_queue_cost(game_state):
        #                     actions[unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, repeat=0)]
        #             else:
        #                 move_cost = unit.move_cost(game_state, direction)
        #                 if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
        #                     actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
        # return actions

    def convert_obs(self, obs: Dict[str, Any]) -> Dict:
        # self = agent
        observation = dict()
        ice_map = obs["board"]["ice"]
        ice_tile_locations = np.argwhere(ice_map == 1)

        factories = obs["factories"][self.player]
        units = obs["units"][self.player]

        factory_key = list(factories.keys())
        factory_vec = np.zeros((len(factories), 2))
        obs_vec = dict()
        for i, k in enumerate(factories.keys()):
            obs_vec[k] = np.zeros(13)
            # here we track a normalized position of the first friendly factory
            factory = factories[k]
            factory_vec[i] = np.array(factory["pos"]) / self.env_cfg.map_size

        if len(units) == 0:
            return obs_vec

        obs_vec = dict()
        for k in units.keys():
            obs_vec[k] = np.zeros(13)
            unit = units[k]

            # store cargo+power values scaled to [0, 1]
            cargo_space = self.env_cfg.ROBOTS[unit["unit_type"]].CARGO_SPACE
            battery_cap = self.env_cfg.ROBOTS[unit["unit_type"]].BATTERY_CAPACITY
            cargo_vec = np.array(
                [
                    unit["power"] / battery_cap,
                    unit["cargo"]["ice"] / cargo_space,
                    unit["cargo"]["ore"] / cargo_space,
                    unit["cargo"]["water"] / cargo_space,
                    unit["cargo"]["metal"] / cargo_space,
                ]
            )
            unit_type = (
                0 if unit["unit_type"] == "LIGHT" else 1
            )  # note that build actions use 0 to encode Light

            # normalize the unit position
            pos = np.array(unit["pos"]) / self.env_cfg.map_size

            if k not in self.units_factory_base:
                factories_distance = factory_vec - pos
                owned_factory = self.units_factory_base[k] = factory_key[np.argmin(abs(factories_distance.sum(axis=1)))]
            else:
                owned_factory = self.units_factory_base[k]
                if owned_factory not in factory_key:  # Dead factory > move to new one:
                    factories_distance = factory_vec - pos
                    owned_factory = self.units_factory_base[k] = factory_key[np.argmin(abs(factories_distance.sum(axis=1)))]

            unit_vec = np.concatenate(
                [pos, [unit_type], cargo_vec, [unit["team_id"]]], axis=-1
            )

            # we add some engineered features down here
            # compute closest ice tile
            ice_tile_distances = np.mean(
                (ice_tile_locations - np.array(unit["pos"])) ** 2, 1
            )
            # normalize the ice tile location
            closest_ice_tile = (
                ice_tile_locations[np.argmin(ice_tile_distances)] / self.env_cfg.map_size
            )
            obs_vec[k] = np.concatenate(
                [unit_vec, factory_vec[factory_key.index(owned_factory)] - pos, closest_ice_tile - pos], axis=-1
            )
        return obs_vec
