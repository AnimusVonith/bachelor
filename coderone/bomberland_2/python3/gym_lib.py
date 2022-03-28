import asyncio
import json
from typing import Callable, Dict, List
import pandas as pd
import numpy as np
import gym
import random
import os
from gym import spaces
from itertools import product
import torch as th
import time


import websockets
from forward_model import ForwardModel
from asgiref.sync import async_to_sync


json_action_format = {
    "move": {
        "enum": [
            "down",
            "left",
            "right",
            "up"
        ],"type": "string"},
    "type": {
        "enum": ["move"],"type": "string"},
    "unit_id": {"type": "string"},

    "type": {
        "enum": ["bomb"],"type": "string"},
    "unit_id": {"type": "string"},

    "coordinates": {
        "items": [
            {"type": "number"},
            {"type": "number"}
        ],"maxItems": 2,"minItems": 2,"type": "array"},
    "type": {"enum": ["detonate"],"type": "string"},
    "unit_id": {"type": "string"}
}

def get_shaped(game_state, my_agent_id=None, my_unit_id=None):

    if my_agent_id is None:
        my_agent_id = "a"

    current_tick = game_state["tick"]
    shape = (game_state["world"]["height"], game_state["world"]["width"])
    ORE_MAX_HP = 3
    
    df = pd.DataFrame.from_dict(game_state["unit_state"])
    dfh = df.copy().T

    my_units = dfh[dfh["agent_id"]==my_agent_id]["unit_id"]
    enemy_units = dfh[dfh["agent_id"]!=my_agent_id]["unit_id"]

    input_shape = (19,15,15)
    input_arr = np.zeros(input_shape, dtype=np.uint8)
    shape = input_arr[0].shape

    if not my_units.empty and my_unit_id is None:
        my_unit_id = my_units[0]

    if my_unit_id is not None:
        my_unit = dfh[dfh["unit_id"] == my_unit_id][["coordinates", "hp"]].copy()
        np.put(input_arr[0], np.ravel_multi_index(np.array(my_unit["coordinates"].to_list()).T, shape), my_unit["hp"].to_list())

    dfh = pd.DataFrame.from_dict(game_state["entities"])

    wh = dfh[dfh["type"]=="w"][["x", "y"]]
    mh = dfh[dfh["type"]=="m"][["x", "y"]]
    ex = dfh[dfh["type"]=="x"][["x", "y"]]

    if "hp" in dfh:
        oh = dfh[dfh["type"]=="o"][["x", "y", "hp"]]
        np.put(input_arr[12], np.ravel_multi_index(np.array(oh[["x", "y"]]).T, shape), ((oh["hp"]/ORE_MAX_HP)*255).to_list())

    if "expires" in dfh:
        dfhe = dfh.copy()
        dfhe = dfhe[dfhe["expires"].notna()]
        dfhe["expires"] = dfhe["expires"] - current_tick
        a = dfhe[dfhe["type"]=="a"]     #ammo
        be = dfhe[dfhe["type"]=="b"]    #bomb    
        bp = dfhe[dfhe["type"]=="bp"]   #blast_powerup
        x = dfhe[dfhe["type"]=="x"]     #fire
        np.put(input_arr[13], np.ravel_multi_index(np.array(a[["x", "y"]]).T, shape), a["expires"].to_list())
        np.put(input_arr[14], np.ravel_multi_index(np.array(be[["x", "y"]]).T, shape), be["expires"].to_list())
        np.put(input_arr[16], np.ravel_multi_index(np.array(bp[["x", "y"]]).T, shape), bp["expires"].to_list())
        np.put(input_arr[17], np.ravel_multi_index(np.array(x[["x", "y"]]).T, shape), x["expires"].to_list())

    if "blast_diameter" in dfh:
        bd = dfh[dfh["type"]=="b"]      #bomb blast diameter
        np.put(input_arr[15], np.ravel_multi_index(np.array(bd[["x", "y"]]).T, shape), bd["blast_diameter"].to_list())
 
    np.put(input_arr[10], np.ravel_multi_index(np.array(wh[["x", "y"]]).T, shape), 1)   #wood
    np.put(input_arr[11], np.ravel_multi_index(np.array(mh[["x", "y"]]).T, shape), 1)   #metal
    np.put(input_arr[18], np.ravel_multi_index(np.array(ex[["x", "y"]]).T, shape), 1)   #fire

    
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

    return input_arr.reshape(19,15,15)
    

def get_random_coords(current, sign):
    HEIGHT, WIDTH = current.shape
    holder_x = random.randint(0,WIDTH-1)
    holder_y = random.randint(0,HEIGHT-1)
    while current[holder_y][holder_x] != " ":
        holder_x = random.randint(0,WIDTH-1)
        holder_y = random.randint(0,HEIGHT-1)
    current[holder_y][holder_x] = sign
    return current, [holder_y, holder_x]


def generate_random_env():
    print("generating env")
    holder = {
        "game_id": "dev",
        "agents": {
            "a": {"agent_id": "a","unit_ids": ["c","e","g"]},   
            "b": {"agent_id": "b","unit_ids": ["d","f","h"]}},
        "unit_state": {
            #"c": {"coordinates": [2,3],"hp": 1,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "c","agent_id": "a","invulnerability": 0},
            "d": {"coordinates": [7,7],"hp": 1,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "d","agent_id": "b","invulnerability": 0}
        },
        "entities": [
            #{"created": 0,"x": 6,"y": 7,"type": "m"},
            #{"created": 0,"x": 7,"y": 6,"type": "w","hp": 1},
            #{"created": 0,"x": 6,"y": 3,"type": "o","hp": 3},
        ],
        "world": {"width": 15,"height": 15},
        "tick": 0,
        "config": {"tick_rate_hz": 100,"game_duration_ticks": 300,"fire_spawn_interval_ticks": 2}
    }


    WIDTH = holder["world"]["width"]
    HEIGHT = holder["world"]["height"]

    current = np.full((WIDTH,HEIGHT), " ")
    current[7][7] = "d"

    N_wood = 10
    N_metal = 5
    N_ore = 5

    #create units
    unit_state_holder = {
        "coordinates": [2,3],"hp": 1,"inventory": {"bombs": 3},"blast_diameter": 3,
        "unit_id": "c",
        "agent_id": "a","invulnerability": 0}

    for agent in holder["agents"]:
        for unit in holder["agents"][agent]["unit_ids"]:
            if unit not in holder["unit_state"]:
                current, unit_state_holder["coordinates"] = get_random_coords(current,unit)
                unit_state_holder["unit_id"] = unit
                unit_state_holder["agent_id"] = agent
                holder["unit_state"][unit] = unit_state_holder.copy()

    #create wood
    wood_holder = {"created": 0,"x": 7,"y": 6,"type": "w","hp": 1}
    for i in range(N_wood):
        current, coords = get_random_coords(current, "w")
        wood_holder["y"], wood_holder["x"] = coords
        holder["entities"].append(wood_holder.copy())


    #create metal
    metal_holder = {"created": 0,"x": 6,"y": 7,"type": "m"}
    for i in range(N_metal):
        current, coords = get_random_coords(current, "m")
        metal_holder["y"], metal_holder["x"] = coords
        holder["entities"].append(metal_holder.copy())

    #create ore
    ore_holder = {"created": 0,"x": 6,"y": 3,"type": "o","hp": 3}
    for i in range(N_ore):
        current, coords = get_random_coords(current, "o")
        ore_holder["y"], ore_holder["x"] = coords
        holder["entities"].append(ore_holder.copy())

    return holder


class GymEnv(gym.Env):
    def __init__(self, fwd_model: ForwardModel, channel: int, send_next_state: Callable[[Dict, List[Dict],  int], Dict], initial_state=None):
        
        #np.set_printoptions(threshold=np.inf)
        self._initial_state = initial_state

        if initial_state is None:
            initial_state = generate_random_env()

        self._state = initial_state
        self.shaped_state = get_shaped(self._state)

        self._fwd = fwd_model
        self._channel = channel
        self._send = send_next_state

        self.GAME_LEN = 1000
        
        self.actions = ["right", "left", "up", "down", "bomb", "detonate-1", "detonate-2", "detonate-3", "noop"]
        self.total_actions = []
        
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0, high=254, shape=(19,15,15), dtype=np.uint8)

        self.metadata = {"render.modes": ["human"]}
        self.last_reward = 0
        self.current_reward = 0

        self.MIN_REWARD = -10000
        self.MAX_REWARD = 10000

        self.reward_range = (self.MIN_REWARD, self.MAX_REWARD)

        self.position_advantage = self.get_pos_score()

        self.done = False
        self.info = {"state":self._state, "events":{}}
        self.holder = [self.shaped_state, self.current_reward, self.done, self.info]


    def get_pos_score(self):
        width = self._state["world"]["width"]
        height = self._state["world"]["height"]
        to_return = []

        for x in range(width):
            column = []
            for y in range(height):
                column.append(width - (abs(x-int(width/2)) + abs(y-int(height/2))))
            to_return.append(column)
        return to_return

    def get_state(self):
        return self._state

    def get_bombs_coords(self, unit_id):
        dfh = pd.DataFrame.from_dict(self._state["entities"])
        if ("type" in dfh) and ("unit_id" in dfh):
            dfh = dfh[dfh["type"] == "b"]
            dfhc = dfh[dfh["unit_id"] == unit_id][["x", "y"]].values.tolist()
            if dfhc != []:
                return dfhc
        return None


    async def get_actions(self, actions=None, agent_id=None, unit_id=None):
        directions = ["right", "left", "up", "down"]

        enemy_actions = ["right", "left", "up", "down", "noop"]

        if isinstance(actions,(int,np.int64)) and actions < len(self.actions):
            actions = self.actions[actions]

        if actions is None: #get random enemy move
            d_action = random.choice(enemy_actions)
            if agent_id is None:
                agent_id = "b"
            if unit_id is None:
                unit_id = "d"

        else: #get our move
            d_action = actions
            if agent_id is None:
                agent_id = "a"
            if unit_id is None:
                unit_id = "c"
        actions = []

        d_action = d_action.split("-")
        u_action = d_action[0]

        if u_action in directions:
            action = {"action": {"type":"move", "unit_id": unit_id, "move": u_action}, "agent_id": agent_id}
            actions.append(action)
        elif u_action == "bomb":
            action = {"action": {"type": "bomb", "unit_id": unit_id}, "agent_id": agent_id}
            actions.append(action)
        elif u_action == "detonate":
            bomb_coords = self.get_bombs_coords(unit_id)
            if (bomb_coords is not None):
                if type(bomb_coords[0]) is list:
                    d_action[1] = int(d_action[1])
                    if len(bomb_coords) >= d_action[1]:
                        bomb_coords = bomb_coords[-d_action[1]]
                    else:
                        return actions
                action = {"action": {"type": "detonate", "unit_id": unit_id, "coordinates": bomb_coords}, "agent_id": agent_id}
                actions.append(action)
        return actions


    def calculate_reward(self, agent_id=None, unit_id=None):
        current_tick = self._state["tick"]
        unit_states = pd.DataFrame.from_dict(self._state["unit_state"]).T

        if agent_id is None:
            agent_id = "a"

        #get alive teammates
        my_units = unit_states[unit_states["agent_id"] == agent_id]

        if unit_id is None:
            unit_id = "c"

        coords = list(my_units[my_units["unit_id"] == unit_id]["coordinates"][0])
        posx, posy = coords

        my_units = my_units[my_units["hp"] > 0]["hp"]

        #get alive enemies
        enemy_units = unit_states[unit_states["agent_id"] != agent_id]
        enemy_units = enemy_units[enemy_units["hp"] > 0]["hp"]

        sum_my_hp = sum(my_units)
        sum_enemy_hp = sum(enemy_units)
        n_my_units = len(my_units)
        n_enemy_units = len(enemy_units)
        
        position_reward = self.position_advantage[posy][posx]

        if not self.done:
            to_return = ((sum_my_hp - sum_enemy_hp)*10) - self.last_reward #((sum_my_hp - sum_enemy_hp)*10) + position_reward 
            self.last_reward = ((sum_my_hp - sum_enemy_hp)*10) #((sum_my_hp - sum_enemy_hp)*10) + position_reward 
            return to_return #if i dont lose hp +10, if i lose hp -10
        
        if n_my_units > 0 and n_enemy_units == 0: #i win
            print("i win")
            reward_now = (100*n_my_units + 10*sum_my_hp)*(self.GAME_LEN/current_tick)
            return min(reward_now, self.MAX_REWARD)
        elif n_my_units == 0 and n_enemy_units > 0: #enemy win
            print("i lose")
            reward_now = -(100*n_enemy_units + 10*sum_enemy_hp)*(self.GAME_LEN/current_tick)
            return max(reward_now, self.MIN_REWARD)
        print("i draw")
        return -100 #0 #draw


    async def async_reset(self):
        print("Resetting")

        if self._initial_state is None:
            print("none in reset")
            self._state = generate_random_env()

        new_coords = [random.randint(0,14),random.randint(0,14)]

        while new_coords[0] == 7 or new_coords[1] == 7:
            new_coords = [random.randint(0,14),random.randint(0,14)]

        self._state["unit_state"]["c"]["coordinates"] = new_coords
        self.info["state"] = self._state
        self.info["events"] = {}
        self.shaped_state = get_shaped(self._state)
        self.last_reward = 0
        return self.shaped_state

    async def await_reset(self):
        return await self.async_reset()

    def reset(self):
        self.total_actions = []

        loop = asyncio.get_event_loop()
        task = loop.create_task(self.await_reset())
        loop.run_until_complete(task)
        return self.shaped_state


    async def async_step(self, actions, get_enemy_actions=True):

        all_actions = []

        if (type(actions) is list) and (type(actions[0]) is list):
            for action in actions:
                all_actions += await self.get_actions(action[0], action[1], action[2])
        else:
            all_actions += await self.get_actions(actions)

        if get_enemy_actions:
            for enemy_unit in self._state["agents"]["b"]["unit_ids"]:
                if self._state["unit_state"][enemy_unit]["hp"] > 0 and self._state["unit_state"][enemy_unit]["coordinates"] != [7,7]:
                    all_actions += await self.get_actions(None, enemy_unit, "b")

        state = await self._send(self._state, all_actions, self._channel)
        self._state = state.get("next_state")
        #render
        self.shaped_state = get_shaped(state.get("next_state"))
        self.done = state.get("is_complete")
        self.info["events"] = state.get("tick_result").get("events")
        self.info["state"] = self._state

        self.current_reward = self.calculate_reward()

        self.holder = [self.shaped_state, self.current_reward, self.done, self.info]
        return self.holder


    def step(self, actions):
        if type(actions) is int:
            self.total_actions.append(self.actions[actions])
        else:
            self.total_actions.append(actions)

        loop = asyncio.get_event_loop()
        task = loop.create_task(self.async_step(actions))
        loop.run_until_complete(task)

        if self.done:
            if not os.path.exists("my_logs"):
                os.makedirs("my_logs")
            with open(f"my_logs/log-{int(time.time())}.txt", "w") as f:
                f.write(str(self.total_actions))

            print(self.total_actions)

        return self.holder


    def render(self, mode="human", close=False):
        render_array = np.full((self.shape)," ")




class Gym():
    def __init__(self, fwd_model_uri: str):
        self._client_fwd = ForwardModel(fwd_model_uri)
        self._channel_counter = 0
        self._channel_is_busy_status: Dict[int, bool] = {}
        self._channel_buffer: Dict[int, Dict] = {}
        self._client_fwd.set_next_state_callback(self._on_next_game_state)
        self._environments: Dict[str, GymEnv] = {}


    async def connect(self):
        loop = asyncio.get_event_loop()

        client_fwd_connection = await self._client_fwd.connect()

        loop = asyncio.get_event_loop()
        loop.create_task(
            self._client_fwd._handle_messages(client_fwd_connection))


    async def close(self):
        await self._client_fwd.close()


    async def _on_next_game_state(self, state):
        channel = state.get("sequence_id")
        self._channel_is_busy_status[channel] = False
        self._channel_buffer[channel] = state


    def make(self, name: str, initial_state=None) -> GymEnv:
        if self._environments.get(name) is not None:
            raise Exception(
                f"environment \"{name}\" has already been instantiated")
        self._environments[name] = GymEnv(
            self._client_fwd, self._channel_counter, self._send_next_state, initial_state)
        self._channel_counter += 1
        return self._environments[name]


    async def _send_next_state(self, state, actions, channel: int):
        self._channel_is_busy_status[channel] = True
        await self._client_fwd.send_next_state(channel, state, actions)
        while self._channel_is_busy_status[channel] == True:
            # TODO figure out why packets are not received without some sleep
            await asyncio.sleep(0.0001)
        result = self._channel_buffer[channel]
        del self._channel_buffer[channel]
        return result
