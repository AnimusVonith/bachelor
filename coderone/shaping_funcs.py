import pandas as pd
import numpy as np

POS_ADVANTAGE = None
INPUT_ARR_X = None
INPUT_ARR_Y = None
FIRE_PATTERN = None
MAX_GAME_LEN = None

def get_fire_pattern(game_state):
    SIZE = game_state["world"]["height"]

    x = np.zeros((SIZE, SIZE), np.float32)

    gap = 0
    curr_x = 0

    LAST_SIZE = SIZE

    current_len = 0
    fire_interval = game_state["config"]["fire_spawn_interval_ticks"]
    multiplier = fire_interval * 2

    for i in range(SIZE):
        if i % 2 == 0:
            curr_x += 1
            if i % 4 == 0:
                #0
                if gap != 0:
                    x[curr_x-1][gap:-gap] = (np.arange(LAST_SIZE) + current_len) * multiplier
                    x[-curr_x][gap:-gap] = (np.flip(np.arange(LAST_SIZE)) + current_len) * multiplier + fire_interval
                else:
                    x[curr_x-1] = (np.arange(LAST_SIZE) + current_len) * multiplier
                    x[-curr_x] = (np.flip(np.arange(LAST_SIZE)) + current_len) * multiplier + fire_interval
            else:
                #2
                x[curr_x-1][gap:-gap] = (np.arange(LAST_SIZE) + current_len) * multiplier + fire_interval
                x[-curr_x][gap:-gap] = np.flip(np.arange(LAST_SIZE) + current_len) * multiplier
        else:
            LAST_SIZE -= 2
            gap += 1
            if (i-1) % 4 == 0:
                #1
                x.T[curr_x-1][gap:-gap] = (np.flip(np.arange(LAST_SIZE)) + current_len) * multiplier + fire_interval
                x.T[-curr_x][gap:-gap] = (np.arange(LAST_SIZE) + current_len) * multiplier
            else:
                #3
                x.T[curr_x-1][gap:-gap] = (np.flip(np.arange(LAST_SIZE)) + current_len) * multiplier
                x.T[-curr_x][gap:-gap] = (np.arange(LAST_SIZE) + current_len) * multiplier + fire_interval
        current_len += LAST_SIZE

    x += game_state["config"]["game_duration_ticks"]
    x = np.flip(x, 1)
    return x

def get_pos_score(game_state):
    width = game_state["world"]["width"]
    height = game_state["world"]["height"]
    to_return = []

    for x in range(width):
        column = []
        for y in range(height):
            column.append(width - (abs(x-int(width/2)) + abs(y-int(height/2))))
        to_return.append(column)
    return to_return

def get_shaped(game_state, my_agent_id=None, my_unit_id=None):

    if my_agent_id is None:
        my_agent_id = "a"

    current_tick = game_state["tick"]
    shape = (game_state["world"]["height"], game_state["world"]["width"])
    ORE_MAX_HP = 3
    PLAYER_MAX_HP = 3
    MAX_BOMBS = 5
    MAX_BLAST_DIAMETER = 15
    MAX_INVULNERABILITY = 200
    MAX_EXPIRES = 200
    
    df = pd.DataFrame.from_dict(game_state["unit_state"])
    dfh = df.copy().T

    my_units = dfh[dfh["agent_id"]==my_agent_id]["unit_id"]
    enemy_units = dfh[dfh["agent_id"]!=my_agent_id]["unit_id"]

    input_shape = (24,15,15)
    input_arr = np.zeros(input_shape, dtype=np.float32)
    shape = input_arr[0].shape

    global POS_ADVANTAGE
    global INPUT_ARR_X
    global INPUT_ARR_Y
    global FIRE_PATTERN
    global MAX_GAME_LEN

    if MAX_GAME_LEN is None:
        max_prefire_duration = game_state["config"]["game_duration_ticks"]
        max_fire_duration = game_state["config"]["fire_spawn_interval_ticks"] * game_state["world"]["height"] * game_state["world"]["width"]
        MAX_GAME_LEN  = max_fire_duration + max_fire_duration

    if POS_ADVANTAGE is None:
        holder = np.array(get_pos_score(game_state))
        POS_ADVANTAGE = ((holder/holder.max())*254).copy()

    if INPUT_ARR_X is None or INPUT_ARR_Y is None:
        holder = np.zeros(shape)
        holder[::] = np.array(range(holder.shape[0]))
        INPUT_ARR_X = ((holder/holder.max())*254).copy()
        INPUT_ARR_Y = ((holder.T/holder.T.max())*254).copy()

    if FIRE_PATTERN is None:
        FIRE_PATTERN = get_fire_pattern(game_state)

    input_arr[19] = POS_ADVANTAGE
    input_arr[20] = INPUT_ARR_X
    input_arr[21] = INPUT_ARR_Y
    input_arr[22] = ((FIRE_PATTERN-game_state["tick"]).clip(0)/MAX_GAME_LEN)*254
    input_arr[23] = (np.full(shape, game_state["tick"])/MAX_GAME_LEN)*254


    if not my_units.empty and my_unit_id is None:
        my_unit_id = my_units[0]

    if my_unit_id is not None:
        my_unit = dfh[dfh["unit_id"] == my_unit_id][["coordinates", "hp"]].copy()
        np.put(input_arr[0], np.ravel_multi_index(np.array(my_unit["coordinates"].to_list()).T, shape), ((my_unit["hp"]/PLAYER_MAX_HP)*254).to_list())

    dfh = pd.DataFrame.from_dict(game_state["entities"])

    wh = dfh[dfh["type"]=="w"][["x", "y"]]
    mh = dfh[dfh["type"]=="m"][["x", "y"]]
    ex = dfh[dfh["type"]=="x"][["x", "y"]]

    if "hp" in dfh:
        oh = dfh[dfh["type"]=="o"][["x", "y", "hp"]]
        np.put(input_arr[12], np.ravel_multi_index(np.array(oh[["x", "y"]]).T, shape), ((oh["hp"]/ORE_MAX_HP)*254).to_list())

    if "expires" in dfh:
        dfhe = dfh.copy()
        dfhe = dfhe[dfhe["expires"].notna()]
        dfhe["expires"] = dfhe["expires"] - current_tick
        a = dfhe[dfhe["type"]=="a"]     #ammo
        be = dfhe[dfhe["type"]=="b"]    #bomb    
        bp = dfhe[dfhe["type"]=="bp"]   #blast_powerup
        x = dfhe[dfhe["type"]=="x"]     #fire
        np.put(input_arr[13], np.ravel_multi_index(np.array(a[["x", "y"]]).T, shape), ((a["expires"]/MAX_EXPIRES)*254).to_list())
        np.put(input_arr[14], np.ravel_multi_index(np.array(be[["x", "y"]]).T, shape), ((be["expires"]/MAX_EXPIRES)*254).to_list())
        np.put(input_arr[16], np.ravel_multi_index(np.array(bp[["x", "y"]]).T, shape), ((bp["expires"]/MAX_EXPIRES)*254).to_list())
        np.put(input_arr[17], np.ravel_multi_index(np.array(x[["x", "y"]]).T, shape), ((x["expires"]/MAX_EXPIRES)*254).to_list())

    if "blast_diameter" in dfh:
        bd = dfh[dfh["type"]=="b"]      #bomb blast diameter
        np.put(input_arr[15], np.ravel_multi_index(np.array(bd[["x", "y"]]).T, shape), ((bd["blast_diameter"]/MAX_BLAST_DIAMETER)*254).to_list())
 
    np.put(input_arr[10], np.ravel_multi_index(np.array(wh[["x", "y"]]).T, shape), 1*254)   #wood
    np.put(input_arr[11], np.ravel_multi_index(np.array(mh[["x", "y"]]).T, shape), 1*254)   #metal
    np.put(input_arr[18], np.ravel_multi_index(np.array(ex[["x", "y"]]).T, shape), 1*254)   #fire

    df = pd.DataFrame.from_dict(game_state["unit_state"]).T
    dfhh = df.copy()
    dfhh["bombs"] = [dfhh["inventory"][unit_id]["bombs"] for unit_id in dfhh["unit_id"].to_list()]
    dead_units = dfhh[dfhh["hp"]<1][["coordinates"]]
    dfha = dfhh[dfhh["hp"] >= 1].copy()

    teammates = dfha[dfha["agent_id"]==my_agent_id][["coordinates", "hp", "invulnerability", "blast_diameter", "bombs"]]
    enemies = dfha[dfha["agent_id"]!=my_agent_id][["coordinates", "hp", "invulnerability", "blast_diameter", "bombs"]]

    if not teammates.empty:
        np.put(input_arr[1], np.ravel_multi_index(np.array(teammates["coordinates"].to_list()).T, shape), ((teammates["hp"]/PLAYER_MAX_HP)*254).to_list())
        np.put(input_arr[2], np.ravel_multi_index(np.array(teammates["coordinates"].to_list()).T, shape), ((teammates["blast_diameter"]/MAX_BLAST_DIAMETER)*254).to_list())
        np.put(input_arr[3], np.ravel_multi_index(np.array(teammates["coordinates"].to_list()).T, shape), ((teammates["bombs"]/MAX_BOMBS)*254).to_list())
        np.put(input_arr[4], np.ravel_multi_index(np.array(teammates["coordinates"].to_list()).T, shape), ((teammates["invulnerability"]/MAX_INVULNERABILITY)*254).to_list())
    if not enemies.empty:
        np.put(input_arr[5], np.ravel_multi_index(np.array(enemies["coordinates"].to_list()).T, shape), ((enemies["hp"]/PLAYER_MAX_HP)*254).to_list())
        np.put(input_arr[6], np.ravel_multi_index(np.array(enemies["coordinates"].to_list()).T, shape), ((enemies["blast_diameter"]/MAX_BLAST_DIAMETER)*254).to_list())
        np.put(input_arr[7], np.ravel_multi_index(np.array(enemies["coordinates"].to_list()).T, shape), ((enemies["bombs"]/MAX_BOMBS)*254).to_list())
        np.put(input_arr[8], np.ravel_multi_index(np.array(enemies["coordinates"].to_list()).T, shape), ((enemies["invulnerability"]/MAX_INVULNERABILITY)*254).to_list())
    if not dead_units.empty:
        np.put(input_arr[9], np.ravel_multi_index(np.array(dead_units["coordinates"].to_list()).T, shape), 1*254)

    #from simple reward faster learning
    input_arr[:19] = np.transpose(input_arr[:19], (0,2,1))

    return input_arr.reshape(input_shape)


def unit_channel_shaping(game_state, my_unit_id, input_arr):
    shape = input_arr[0].shape
    input_arr[0] = np.zeros(shape)
    PLAYER_MAX_HP = 3
    
    coords = game_state["unit_state"][my_unit_id]["coordinates"]
    input_values = game_state["unit_state"][my_unit_id]["hp"]
    
    np.put(input_arr[0], np.ravel_multi_index(np.array(coords).T, shape), (np.array(input_values)/PLAYER_MAX_HP)*254)
    return input_arr


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    test_state = {
            "agents": {
                "a": {
                    "agent_id": "a",
                    "unit_ids": [
                        "c",
                        "e",
                        "g"
                    ]
                },
                "b": {
                    "agent_id": "b",
                    "unit_ids": [
                        "d",
                        "f",
                        "h"
                    ]
                }
            },
            "unit_state": {
                "c": {
                    "coordinates": [
                        4,
                        10
                    ],
                    "hp": 3,
                    "inventory": {
                        "bombs": 3
                    },
                    "blast_diameter": 3,
                    "unit_id": "c",
                    "agent_id": "a",
                    "invulnerability": 0
                },
                "d": {
                    "coordinates": [
                        10,
                        10
                    ],
                    "hp": 3,
                    "inventory": {
                        "bombs": 3
                    },
                    "blast_diameter": 3,
                    "unit_id": "d",
                    "agent_id": "b",
                    "invulnerability": 0
                },
                "e": {
                    "coordinates": [
                        3,
                        0
                    ],
                    "hp": 3,
                    "inventory": {
                        "bombs": 3
                    },
                    "blast_diameter": 3,
                    "unit_id": "e",
                    "agent_id": "a",
                    "invulnerability": 0
                },
                "f": {
                    "coordinates": [
                        11,
                        0
                    ],
                    "hp": 3,
                    "inventory": {
                        "bombs": 3
                    },
                    "blast_diameter": 3,
                    "unit_id": "f",
                    "agent_id": "b",
                    "invulnerability": 0
                },
                "g": {
                    "coordinates": [
                        10,
                        3
                    ],
                    "hp": 3,
                    "inventory": {
                        "bombs": 3
                    },
                    "blast_diameter": 3,
                    "unit_id": "g",
                    "agent_id": "a",
                    "invulnerability": 0
                },
                "h": {
                    "coordinates": [
                        4,
                        3
                    ],
                    "hp": 3,
                    "inventory": {
                        "bombs": 3
                    },
                    "blast_diameter": 3,
                    "unit_id": "h",
                    "agent_id": "b",
                    "invulnerability": 0
                }
            },
            "entities": [
                {
                    "created": 0,
                    "x": 0,
                    "y": 2,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 14,
                    "y": 2,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 8,
                    "y": 12,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 6,
                    "y": 12,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 13,
                    "y": 14,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 1,
                    "y": 14,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 6,
                    "y": 5,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 8,
                    "y": 5,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 2,
                    "y": 5,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 12,
                    "y": 5,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 12,
                    "y": 7,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 2,
                    "y": 7,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 12,
                    "y": 8,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 2,
                    "y": 8,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 8,
                    "y": 11,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 6,
                    "y": 11,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 0,
                    "y": 7,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 14,
                    "y": 7,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 10,
                    "y": 4,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 4,
                    "y": 4,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 0,
                    "y": 6,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 14,
                    "y": 6,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 3,
                    "y": 8,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 11,
                    "y": 8,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 11,
                    "y": 7,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 3,
                    "y": 7,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 3,
                    "y": 3,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 11,
                    "y": 3,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 13,
                    "y": 9,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 1,
                    "y": 9,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 14,
                    "y": 1,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 0,
                    "y": 1,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 10,
                    "y": 12,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 4,
                    "y": 12,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 11,
                    "y": 6,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 3,
                    "y": 6,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 0,
                    "y": 9,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 14,
                    "y": 9,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 9,
                    "y": 7,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 5,
                    "y": 7,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 14,
                    "y": 5,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 0,
                    "y": 5,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 9,
                    "y": 10,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 5,
                    "y": 10,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 9,
                    "y": 4,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 5,
                    "y": 4,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 3,
                    "y": 13,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 11,
                    "y": 13,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 2,
                    "y": 4,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 12,
                    "y": 4,
                    "type": "m"
                },
                {
                    "created": 0,
                    "x": 14,
                    "y": 0,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 0,
                    "y": 0,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 8,
                    "y": 6,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 6,
                    "y": 6,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 10,
                    "y": 11,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 4,
                    "y": 11,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 8,
                    "y": 14,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 6,
                    "y": 14,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 1,
                    "y": 1,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 13,
                    "y": 1,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 3,
                    "y": 9,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 11,
                    "y": 9,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 10,
                    "y": 5,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 4,
                    "y": 5,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 6,
                    "y": 13,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 8,
                    "y": 13,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 14,
                    "y": 14,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 0,
                    "y": 14,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 5,
                    "y": 8,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 9,
                    "y": 8,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 12,
                    "y": 9,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 2,
                    "y": 9,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 8,
                    "y": 8,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 6,
                    "y": 8,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 3,
                    "y": 5,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 11,
                    "y": 5,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 13,
                    "y": 12,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 1,
                    "y": 12,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 0,
                    "y": 11,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 14,
                    "y": 11,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 13,
                    "y": 8,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 1,
                    "y": 8,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 3,
                    "y": 14,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 11,
                    "y": 14,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 12,
                    "y": 12,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 2,
                    "y": 12,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 2,
                    "y": 3,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 12,
                    "y": 3,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 6,
                    "y": 4,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 8,
                    "y": 4,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 9,
                    "y": 2,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 5,
                    "y": 2,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 6,
                    "y": 2,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 8,
                    "y": 2,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 10,
                    "y": 1,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 4,
                    "y": 1,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 9,
                    "y": 0,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 5,
                    "y": 0,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 12,
                    "y": 6,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 2,
                    "y": 6,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 14,
                    "y": 4,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 0,
                    "y": 4,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 4,
                    "y": 6,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 10,
                    "y": 6,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 14,
                    "y": 8,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 0,
                    "y": 8,
                    "type": "w",
                    "hp": 1
                },
                {
                    "created": 0,
                    "x": 8,
                    "y": 10,
                    "type": "o",
                    "hp": 3
                },
                {
                    "created": 0,
                    "x": 6,
                    "y": 10,
                    "type": "o",
                    "hp": 3
                },
                {
                    "created": 0,
                    "x": 5,
                    "y": 1,
                    "type": "o",
                    "hp": 3
                },
                {
                    "created": 0,
                    "x": 9,
                    "y": 1,
                    "type": "o",
                    "hp": 3
                },
                {
                    "created": 0,
                    "x": 6,
                    "y": 3,
                    "type": "o",
                    "hp": 3
                },
                {
                    "created": 0,
                    "x": 8,
                    "y": 3,
                    "type": "o",
                    "hp": 3
                },
                {
                    "created": 0,
                    "x": 3,
                    "y": 4,
                    "type": "o",
                    "hp": 3
                },
                {
                    "created": 0,
                    "x": 11,
                    "y": 4,
                    "type": "o",
                    "hp": 3
                },
                {
                    "created": 0,
                    "x": 9,
                    "y": 6,
                    "type": "o",
                    "hp": 3
                },
                {
                    "created": 0,
                    "x": 5,
                    "y": 6,
                    "type": "o",
                    "hp": 3
                },
                {
                    "created": 0,
                    "x": 6,
                    "y": 0,
                    "type": "o",
                    "hp": 3
                },
                {
                    "created": 0,
                    "x": 8,
                    "y": 0,
                    "type": "o",
                    "hp": 3
                },
                {
                    "created": 0,
                    "x": 5,
                    "y": 11,
                    "type": "o",
                    "hp": 3
                },
                {
                    "created": 0,
                    "x": 9,
                    "y": 11,
                    "type": "o",
                    "hp": 3
                }
            ],
            "world": {
                "width": 15,
                "height": 15
            },
            "tick": 0,
            "config": {
                "tick_rate_hz": 10,
                "game_duration_ticks": 300,
                "fire_spawn_interval_ticks": 2
            }
        }

    x = np.zeros((5,15,15) ,dtype=np.float32)
    unit_channel_shaping(test_state, "c", x)
    """
    holder = get_shaped(test_state)
    for i,x in enumerate(holder):
        print(i)
        print("\n")
        print(x)
        print("\n\n\n\n")
    """