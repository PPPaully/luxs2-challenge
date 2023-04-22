import numpy as np

from luxai_s2 import LuxAI_S2
from factory_setup import find_best_position
from lux.utils import draw, close_draw
from lux.kit import obs_to_game_state
from lux.const import Direction, Resource


def skip():
    global observations, rewards, dones, infos
    # Skip bidding
    actions = {'player_0': {'faction': 'AlphaStrike', 'bid': 0}, 'player_1': {'faction': 'AlphaStrike', 'bid': 0}}
    observations, rewards, dones, infos = env.step(actions)

    while env.state.real_env_steps < 0:
        # Place factory
        player = 'player_0'
        spawn_position = find_best_position(env.env_steps, observations[player], env.env_cfg,
                                            split=len(observations[player]['factories'][player]) > 0,
                                            space=len(observations[player]['factories'][player]) > 1)
        actions = {'player_0': {'spawn': spawn_position, 'metal': 150, 'water': 150}, 'player_1': {}}
        observations, rewards, dones, infos = env.step(actions)
        # draw(env)

        player = 'player_1'
        spawn_position = find_best_position(env.env_steps, observations[player], env.env_cfg,
                                            split=len(observations[player]['factories'][player]) > 0,
                                            space=len(observations[player]['factories'][player]) > 1)
        actions = {'player_0': {}, 'player_1': {'spawn': spawn_position, 'metal': 150, 'water': 150}}
        observations, rewards, dones, infos = env.step(actions)


def keep_alive(player=None):
    if player is None:
        for pk in env.state.factories.keys():
            for factory in env.state.factories[pk].values():
                factory.cargo.water = 1000
    else:
        for factory in env.state.factories[player].values():
            factory.cargo.water = 1000


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


# ==========================
def random_action():
    global player_actions, player_actions_log, game_state
    player_actions = {player: dict()}
    player_actions_log = {player: dict()}
    factories = list(game_state.factories[player].values())

    units_pos_mapping = dict()
    for u in game_state.units[player].values():
        units_pos_mapping[tuple(u.pos)] = u
    for u in game_state.units[opponent].values():
        units_pos_mapping[tuple(u.pos)] = u

    def move(unit, direction):
        def f():
            return unit.move(direction)
        return f

    def transfer(unit, direction, resource, amount):
        def f():
            return unit.transfer(direction, resource, amount)
        return f

    player_team_id = game_state.teams[player].team_id
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
            move_actions.append(move(u, direction))
        move_dict = dict(flags=move_flags, actions=move_actions, logs=Direction.logs[1:])

        can_dig = u.dig_cost(game_state) <= u.power
        can_dig_ice = can_dig and game_state.board.ice[u.pos[0], u.pos[1]]
        can_dig_ore = can_dig and game_state.board.ore[u.pos[0], u.pos[1]]
        can_dig_rubble = can_dig and game_state.board.rubble[u.pos[0], u.pos[1]]
        dig_act = u.dig

        can_transfer, transfer_direction = get_transfer_direction(base, u)
        can_transfer_ice = can_transfer and u.cargo.ice > 0
        can_transfer_ore = can_transfer and u.cargo.ore > 0
        tf_ice_action = transfer(u, transfer_direction, Resource.ICE, u.cargo.ice)
        tf_ore_action = transfer(u, transfer_direction, Resource.ORE, u.cargo.ore)

        unit_actions = dict()
        unit_actions['flags'] = [can_move, can_dig_ice, can_dig_ore, can_dig_rubble, can_transfer_ice, can_transfer_ore]
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

            player_actions[player][u.unit_id] = [act()]
            player_actions_log[player][u.unit_id] = [log]

    # Factory Action
    for f in factories:
        empty_slot = tuple(f.pos) not in units_pos_mapping
        action = [f.water, f.build_light, f.build_heavy]
        action_flags = [f.can_water(game_state), empty_slot and f.can_build_light(game_state), empty_slot and f.can_build_heavy(game_state)]
        action_logs = ['WATER', 'B_LIGHT', 'B_HEAVY']
        choices = np.argwhere(action_flags).flatten()
        if len(choices) > 0:
            choose = np.random.choice(choices)
            player_actions[player][f.unit_id] = action[choose]()
            player_actions_log[player][f.unit_id] = action_logs[choose]


def step():
    global player_actions, game_state, observations, rewards, dones, infos
    observations, rewards, dones, infos = env.step(player_actions)
    print(f'''==========================
Step: {env.state.real_env_steps}
Actions: {player_actions_log['player_0']}
Rewards: {rewards}
Info: {infos}
Dones: {dones}
''')

    all_actions_log.append(player_actions_log)

    game_state = obs_to_game_state(env.state.real_env_steps, env.env_cfg, observations[player])
    all_game_state.append(game_state)


def pprint(data):
    print(f'''
====================================================================
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


# ==========================
env = LuxAI_S2(collect_stats=True)
env.env_cfg.MIN_FACTORIES = env.env_cfg.MAX_FACTORIES = 1


player = 'player_0'
opponent = 'player_1'
player_actions = None
player_actions_log = None
all_game_state = []
all_actions_log = []
observations = env.reset(seed=855488599)
rewards = dones = infos = None
skip()

game_state = obs_to_game_state(env.state.real_env_steps, env.env_cfg, observations[player])
all_game_state.append(game_state)

while True:
    keep_alive(player=opponent)
    random_action()
    step()
    pprint(env.state.stats[player])
    if env.state.stats[player]['destroyed']['HEAVY'] or env.state.stats[player]['destroyed']['LIGHT']:
        break
    if dones[player] or dones[opponent]:
        break
draw(env)

all_game_state[10].units[player]['unit_2'].pos
all_game_state[10].units[player]['unit_2'].power
all_game_state[11].units[player]['unit_2'].pos
all_game_state[11].units[player]['unit_2'].power

all_game_state[-2].factories
set(all_game_state[-1].board.lichen_strains.flatten())

# all_game_state[-3].units[player].keys()
# all_game_state[-2].units[player].keys()
# all_game_state[-1].units[player].keys()
# game_state.units[player].keys()
#
# all_game_state[-1].units[player]['unit_2'].pos
#
# all_actions_log[-2][player]
# all_actions_log[-1][player]
#
# game_state.factories[player]['factory_0'].pos
#
# Direction.logs[1]
# Direction.logs[4]
# Direction.SHIFT[Direction.UP]
# Direction.SHIFT[Direction.LEFT]
#
# for k, u in all_game_state[-1].units[player].items():
#     print(k, u.pos)
# print('==========')
# for k, u in game_state.units[player].items():
#     print(k, u.pos)
