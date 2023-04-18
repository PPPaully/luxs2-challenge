import time
import numpy as np
import matplotlib.pyplot as plt

from luxai_s2.env import LuxAI_S2
from lux.kit import obs_to_game_state
from lux.utils import draw


def conv2d(a, f, loop=0):
    a = np.pad(a, pad_width=f.shape[0]//2)
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape=s, strides=a.strides * 2)
    if loop > 0:
        return conv2d(np.einsum('ij,ijkl->kl', f, subM), f, loop - 1)
    return np.einsum('ij,ijkl->kl', f, subM)


def norm(a):
    mn = a.min()
    mx = a.max()
    return (a - mn)/(mx - mn)


def clip(a):
    return np.clip(a, 0, 1)


def find_best_position(step, obs, env_cfg, split=False, space=False):
    # player = 'player_0'
    # player = 'player_1'
    # step, obs, env_cfg = env.env_steps, env_obs[player], env.env_cfg
    game_state = obs_to_game_state(step, env_cfg, obs)

    enlarge_filter = np.ones((3, 3))/9
    value_sum_filter = np.pad(np.zeros((3, 3)), 1, 'constant', constant_values=1)
    value_sum_filter = value_sum_filter / value_sum_filter.sum()
    average_area_filter = np.ones((5, 5))/49


    ore_area = conv2d(norm(clip(conv2d(clip(conv2d(game_state.board.ore, enlarge_filter*9, 3)), enlarge_filter, 5))), value_sum_filter)
    ice_area = conv2d(norm(clip(conv2d(clip(conv2d(game_state.board.ice, enlarge_filter*9, 3)), enlarge_filter, 5))), value_sum_filter)
    open_area = conv2d(norm(game_state.board.rubble.max() - game_state.board.rubble), value_sum_filter)

    if space:  # For lichen focus
        ice_weight, ore_weight, open_weight = (3, 3, 10)
    else:
        ice_weight, ore_weight, open_weight = (5, 5, 1)
    sum_weight = sum([ice_weight, ore_weight, open_weight])
    ice_weight = ice_weight / sum_weight
    ore_weight = ore_weight / sum_weight
    open_weight = open_weight / sum_weight

    if split:  # Remove near factory spot
        land_value = clip(
            ore_area * ore_weight +
            ice_area * ice_weight +
            open_area * open_weight
        ) * game_state.board.valid_spawns_mask
        if space:  # For lichen focus
            land_value = conv2d(land_value, average_area_filter)
    else:
        land_value = conv2d(
            clip(
                ore_area * ore_weight +
                ice_area * ice_weight +
                open_area * open_weight
            ),
            average_area_filter
        )
    valid_land_value = land_value * game_state.board.valid_spawns_mask

    # Show map
    # plt.imshow(ore_area.T)
    # plt.imshow(ice_area.T)
    # plt.imshow(open_area.T)
    # plt.imshow(land_value.T)
    # plt.imshow(valid_land_value.T)
    # plt.show()

    # draw(env)

    x = np.argmax(valid_land_value) // env_cfg.map_size
    y = np.argmax(valid_land_value) % env_cfg.map_size

    return x, y


def test():
    env = LuxAI_S2()
    env_obs = env.reset()
    # Skip bidding
    actions = {'player_0': {'faction': 'AlphaStrike', 'bid': 0}, 'player_1': {'faction': 'AlphaStrike', 'bid': 0}}
    env_obs, rewards, dones, infos = env.step(actions)

    while env.state.real_env_steps < 0:
        # Place factory
        player = 'player_0'
        spawn_position = find_best_position(
            env.env_steps, env_obs[player], env.env_cfg,
            split=len(env_obs[player]['factories'][player]) > 0,
            space=len(env_obs[player]['factories'][player]) > 1
        )
        actions = {'player_0': {'spawn': spawn_position, 'metal': 150, 'water': 150}, 'player_1': {}}
        env_obs, rewards, dones, infos = env.step(actions)
        # draw(env)

        player = 'player_1'
        spawn_position = find_best_position(env.env_steps, env_obs[player], env.env_cfg, split=len(env_obs[player]['factories'][player]) > 0, space=len(env_obs[player]['factories'][player]) > 1)
        actions = {'player_0': {}, 'player_1': {'spawn': spawn_position, 'metal': 150, 'water': 150}}
        env_obs, rewards, dones, infos = env.step(actions)
        # draw(env)

    for _ in range(30):
        actions = dict()
        player = 'player_0'
        game_state_0 = obs_to_game_state(env.env_steps, env.env_cfg, env_obs[player])
        actions[player] = dict([(fac_id, factory.water()) for fac_id, factory in game_state_0.factories[player].items()])

        player = 'player_1'
        game_state_1 = obs_to_game_state(env.env_steps, env.env_cfg, env_obs[player])
        actions[player] = dict([(fac_id, factory.water()) for fac_id, factory in game_state_1.factories[player].items()])
        env_obs, rewards, dones, infos = env.step(actions)
        # draw(env)

    actions = dict()
    player = 'player_0'
    game_state_0 = obs_to_game_state(env.env_steps, env.env_cfg, env_obs[player])
    actions[player] = dict([(fac_id, factory.build_light()) for fac_id, factory in game_state_0.factories[player].items()])

    player = 'player_1'
    game_state_1 = obs_to_game_state(env.env_steps, env.env_cfg, env_obs[player])
    actions[player] = dict([(fac_id, factory.build_heavy()) for fac_id, factory in game_state_1.factories[player].items()])
    env_obs, rewards, dones, infos = env.step(actions)
    draw(env)

    list(game_state_0.factories[player].values())[0].power
    list(game_state_0.units[player].values())[0].recharge(2, 1, 10)

    game_state_1.board.lichen.sum()


    env.state.board.map.rubble = np.zeros((env.env_cfg.map_size, env.env_cfg.map_size), dtype=int)
    draw(env)
