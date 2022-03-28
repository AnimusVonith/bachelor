import asyncio
from typing import Dict
from gym_lib import Gym
import os
import gym
import gym_lib
import random
import pandas as pd
import numpy as np

#from stable_baselines3.common import make_vec_env
#from stable_baselines3 import PPO


fwd_model_uri = os.environ.get(
    "FWD_MODEL_CONNECTION_STRING") or "ws://127.0.0.1:6969/?role=admin"

mock_6x6_state: Dict = {
"game_id": "dev", 
"agents": {
    "a": {"agent_id": "a", "unit_ids": ["c", "e", "g"]}, 
    "b": {"agent_id": "b", "unit_ids": ["d", "f", "h"]}}, 
"unit_state": {
    "c": {"coordinates": [0, 1], "hp": 3, "inventory": {"bombs": 3}, "blast_diameter": 3, "unit_id": "c", "agent_id": "a", "invulnerability": 0}, 
    "d": {"coordinates": [5, 1], "hp": 3, "inventory": {"bombs": 3}, "blast_diameter": 3, "unit_id": "d", "agent_id": "b", "invulnerability": 0}, 
    "e": {"coordinates": [3, 3], "hp": 3, "inventory": {"bombs": 3}, "blast_diameter": 3, "unit_id": "e", "agent_id": "a", "invulnerability": 0}, 
    "f": {"coordinates": [2, 3], "hp": 3, "inventory": {"bombs": 3}, "blast_diameter": 3, "unit_id": "f", "agent_id": "b", "invulnerability": 0}, 
    "g": {"coordinates": [2, 4], "hp": 3, "inventory": {"bombs": 3}, "blast_diameter": 3, "unit_id": "g", "agent_id": "a", "invulnerability": 0}, 
    "h": {"coordinates": [3, 4], "hp": 3, "inventory": {"bombs": 3}, "blast_diameter": 3, "unit_id": "h", "agent_id": "b", "invulnerability": 0}}, 
"entities": [
    {"created": 0, "x": 0, "y": 3, "type": "m"}, {"created": 0, "x": 5, "y": 3, "type": "m"}, {"created": 0, "x": 4, "y": 3, "type": "m"}, 
    {"created": 0, "x": 1, "y": 3, "type": "m"}, {"created": 0, "x": 3, "y": 5, "type": "m"}, {"created": 0, "x": 2, "y": 5, "type": "m"}, 
    {"created": 0, "x": 5, "y": 4, "type": "m"}, {"created": 0, "x": 0, "y": 4, "type": "m"}, 
    {"created": 0, "x": 1, "y": 1, "type": "w", "hp": 1}, {"created": 0, "x": 4, "y": 1, "type": "w", "hp": 1}, 
    {"created": 0, "x": 3, "y": 0, "type": "w", "hp": 1}, {"created": 0, "x": 2, "y": 0, "type": "w", "hp": 1}, 
    {"created": 0, "x": 5, "y": 5, "type": "w", "hp": 1}, {"created": 0, "x": 0, "y": 5, "type": "w", "hp": 1}, 
    {"created": 0, "x": 4, "y": 0, "type": "w", "hp": 1}, {"created": 0, "x": 1, "y": 0, "type": "w", "hp": 1}, 
    {"created": 0, "x": 5, "y": 0, "type": "w", "hp": 1}, {"created": 0, "x": 0, "y": 0, "type": "w", "hp": 1}], 
"world": {"width": 15, "height": 15}, 
"tick": 0, 
"config": {"tick_rate_hz": 10, "game_duration_ticks": 300, "fire_spawn_interval_ticks": 2}}


def calculate_reward(state: Dict):
    # custom reward function
    return 1


def get_bombs_coords(game_state, unit_id):
    dfh = pd.DataFrame.from_dict(game_state["entities"])
    dfh = dfh[dfh["type"] == "b"]
    if "unit_id" in dfh:
        dfhc = dfh[dfh["unit_id"] == unit_id][["x", "y"]].values.tolist()
        if dfhc != []:
            return dfhc
    return None


def unit_channel_shaping(self, game_state, my_unit_id, input_arr):
    df = pd.DataFrame.from_dict(game_state["unit_state"])
    dfh = df.copy().T
    shape = input_arr[0].shape
    my_unit = dfh[dfh["unit_id"]==my_unit_id][["coordinates", "hp"]]
    np.put(input_arr[0], np.ravel_multi_index(np.array(my_unit["coordinates"].to_list()).T, shape), my_unit["hp"].to_list())
    return input_arr


def get_actions(my_units, agent_id, game_state):
    actions = []
    action_s = ["move", "bomb", "detonate", "noop"]
    directions = ["down", "left", "right", "up"]
    for unit_id in my_units:
        u_action = random.choice(action_s)
        if u_action == "move":
            action = {"action": {"type":"move", "unit_id": unit_id, "move": random.choice(directions)}, "agent_id": agent_id}
            actions.append(action)
        elif u_action == "bomb":
            action = {"action": {"type": "bomb", "unit_id": unit_id}, "agent_id": agent_id}
            actions.append(action)
        elif u_action == "detonate":
            bomb_coords = get_bombs_coords(game_state, unit_id)
            if (bomb_coords != []) and (bomb_coords is not None):
                if type(bomb_coords[0]) is list:
                    bomb_coords = random.choice(bomb_coords)
                x, y = bomb_coords
                action = {"action": {"type": "detonate", "unit_id": unit_id, "coordinates": [x, y]}, "agent_id": agent_id}
                actions.append(action)
            pass
        else:
            pass
    return actions


def get_shaped(game_state, my_agent_id):
    shape = (game_state["world"]["height"], game_state["world"]["width"])

    df = pd.DataFrame.from_dict(game_state["agents"])
    dfh = df.copy().T
    my_units = dfh[dfh["agent_id"]==my_agent_id]["unit_ids"]
    enemy_units = dfh[dfh["agent_id"]!=my_agent_id]["unit_ids"]

    input_shape = (22,15,15)
    input_arr = np.zeros(input_shape, dtype="int8")
    shape = input_arr[0].shape

    dfh = pd.DataFrame.from_dict(game_state["entities"])

    wh = dfh[dfh["type"]=="w"][["x", "y"]]
    mh = dfh[dfh["type"]=="m"][["x", "y"]]
    ex = dfh[dfh["type"]=="x"][["x", "y"]]

    if "hp" in dfh:
        oh = dfh[dfh["type"]=="o"][["x", "y", "hp"]]
        o1 = oh[oh["hp"]==1][["x", "y"]]
        o2 = oh[oh["hp"]==2][["x", "y"]]
        o3 = oh[oh["hp"]==3][["x", "y"]]
        np.put(input_arr[12], np.ravel_multi_index(np.array(oh[["x", "y"]]).T, shape), oh["hp"].to_list())
        np.put(input_arr[13], np.ravel_multi_index(np.array(o1[["x", "y"]]).T, shape), 1)
        np.put(input_arr[14], np.ravel_multi_index(np.array(o2[["x", "y"]]).T, shape), 1)
        np.put(input_arr[15], np.ravel_multi_index(np.array(o3[["x", "y"]]).T, shape), 1)
    else:
        input_arr[12] = np.zeros(shape)
        input_arr[13] = np.zeros(shape)
        input_arr[14] = np.zeros(shape)
        input_arr[15] = np.zeros(shape)

    if "expires" in dfh:
        a = dfh[dfh["type"]=="a"]
        be = dfh[dfh["type"]=="b"]
        bp = dfh[dfh["type"]=="bp"]
        x = dfh[dfh["type"]=="x"]
        x = x[x["expires"].notna()]
        np.put(input_arr[16], np.ravel_multi_index(np.array(a[["x", "y"]]).T, shape), a["expires"].to_list())
        np.put(input_arr[17], np.ravel_multi_index(np.array(be[["x", "y"]]).T, shape), be["expires"].to_list())
        np.put(input_arr[19], np.ravel_multi_index(np.array(bp[["x", "y"]]).T, shape), bp["expires"].to_list())
        np.put(input_arr[20], np.ravel_multi_index(np.array(x[["x", "y"]]).T, shape), x["expires"].to_list())
    else:
        input_arr[16] = np.zeros(shape)
        input_arr[17] = np.zeros(shape)
        input_arr[19] = np.zeros(shape)
        input_arr[20] = np.zeros(shape)

    if "blast_diameter" in dfh:
        bd = dfh[dfh["type"]=="b"]
        np.put(input_arr[18], np.ravel_multi_index(np.array(bd[["x", "y"]]).T, shape), bd["blast_diameter"].to_list())
    else:
        input_arr[18] = np.zeros(shape)    
    
    np.put(input_arr[10], np.ravel_multi_index(np.array(wh[["x", "y"]]).T, shape), 1)
    np.put(input_arr[11], np.ravel_multi_index(np.array(mh[["x", "y"]]).T, shape), 1)
    np.put(input_arr[21], np.ravel_multi_index(np.array(ex[["x", "y"]]).T, shape), 1)
    
    df = pd.DataFrame.from_dict(game_state["unit_state"]).T
    dfhh = df.copy()
    dfhh["bombs"] = [dfhh["inventory"][unit_id]["bombs"] for unit_id in dfhh["unit_id"].to_list()]
    dead_units = dfhh[dfhh["hp"]<1][["coordinates"]]
    dfha = dfhh[dfhh["hp"] >= 1].copy()

    teammates = dfha[dfha["agent_id"]==my_agent_id][["coordinates", "hp", "invulnerability", "blast_diameter", "bombs"]]
    enemies = dfha[dfha["agent_id"]!=my_agent_id][["coordinates", "hp", "invulnerability", "blast_diameter", "bombs"]]

    if not teammates.empty:
        np.put(input_arr[1], np.ravel_multi_index(np.array(teammates["coordinates"].to_list()).T, shape), teammates["hp"].to_list())
        np.put(input_arr[2], np.ravel_multi_index(np.array(teammates["coordinates"].to_list()).T, shape), teammates["blast_diameter"].to_list())
        np.put(input_arr[3], np.ravel_multi_index(np.array(teammates["coordinates"].to_list()).T, shape), teammates["bombs"].to_list())
        np.put(input_arr[4], np.ravel_multi_index(np.array(teammates["coordinates"].to_list()).T, shape), teammates["invulnerability"].to_list())
    if not enemies.empty:
        np.put(input_arr[5], np.ravel_multi_index(np.array(enemies["coordinates"].to_list()).T, shape), enemies["hp"].to_list())
        np.put(input_arr[6], np.ravel_multi_index(np.array(enemies["coordinates"].to_list()).T, shape), enemies["blast_diameter"].to_list())
        np.put(input_arr[7], np.ravel_multi_index(np.array(enemies["coordinates"].to_list()).T, shape), enemies["bombs"].to_list())
        np.put(input_arr[8], np.ravel_multi_index(np.array(enemies["coordinates"].to_list()).T, shape), enemies["invulnerability"].to_list())
    if not dead_units.empty:
        np.put(input_arr[9], np.ravel_multi_index(np.array(dead_units["coordinates"].to_list()).T, shape), 1)

    return input_arr


async def main():

    gym = Gym(fwd_model_uri)
    await gym.connect()
    env = gym.make("bomberland-open-ai-gym", mock_6x6_state)


    observation = mock_6x6_state
    my_units = mock_6x6_state.get("agents").get("a").get("unit_ids")
    my_agent = "a"
    enemy_units = mock_6x6_state.get("agents").get("b").get("unit_ids")
    enemy_agent = "b"
    tick = 0

    i=0
    while i < 1000:
        shaped_obs = get_shaped(observation, my_agent)
        print(shaped_obs.shape)
        my_actions = get_actions(my_units, my_agent, observation)
        enemy_actions = get_actions(enemy_units, enemy_agent, observation)
        actions = my_actions + enemy_actions
        observation, done, info = await env.step(actions)
        reward = calculate_reward(observation)
        if done:
            print(f"""{observation.keys()}\n\n{done}\n\n{info}\n\n\n\n""")
            tick += observation["tick"]
            i+=1
            await env.reset()
    await gym.close()
    

if __name__ == "__main__":
    asyncio.run(main())