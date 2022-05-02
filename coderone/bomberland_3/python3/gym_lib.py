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
import pickle

import websockets
from forward_model import ForwardModel
from model_funcs import load_model
from shaping_funcs import get_shaped, get_pos_score, unit_channel_shaping
from misc_funcs import get_bombs_coords

def show_state(game_state, print_state=True):
    current = np.full((15,15), " ")
    #print(game_state["entities"])
    for unit in game_state["unit_state"]:
        unit = game_state["unit_state"][unit]
        x, y = unit["coordinates"]
        #print(f"""unit {unit} at {[x,y]}""")
        current[y][x] = unit["unit_id"] 
    for entity in game_state["entities"]:
        current[entity["y"]][entity["x"]] = entity["type"]
        #print(f"""entity {entity["type"]} at {[entity["x"],entity["y"]]}""")
    if print_state:
        print(np.flip(current,0))
    return np.flip(current,0)
    

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

    initial_game_state = {
        "game_id": "dev",
        "agents": {
            "a": {"agent_id": "a", "unit_ids": ["c","e","g"]},   
            "b": {"agent_id": "b", "unit_ids": ["d","f","h"]}},
        "unit_state": {
            #"c": {"coordinates": [2,3],"hp": 1,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "c","agent_id": "a","invulnerability": 0},
            #"d": {"coordinates": [7,7],"hp": 1,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "d","agent_id": "b","invulnerability": 0}
        },
        "entities": [
            #{"created": 0,"x": 6,"y": 7,"type": "m"},
            #{"created": 0,"x": 7,"y": 6,"type": "w","hp": 1},
            #{"created": 0,"x": 6,"y": 3,"type": "o","hp": 3},
        ],
        "world": {"width": 15,"height": 15},
        "tick": 0,
        "config": {"tick_rate_hz": 50,"game_duration_ticks": 1,"fire_spawn_interval_ticks": 1} #default 300 game_dur, 2 fire
    }

    holder = initial_game_state
    WIDTH = holder["world"]["width"]
    HEIGHT = holder["world"]["height"]

    current = np.full((WIDTH,HEIGHT), " ")
    #current[7][7] = "d"

    N_wood = 20
    N_metal = 8
    N_ore = 8

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
        self.position_advantage = get_pos_score(self._state)

        self.shaped_state = get_shaped(self._state)

        self._fwd = fwd_model
        self._channel = channel
        self._send = send_next_state

        max_prefire_duration = self._state["config"]["game_duration_ticks"]
        max_fire_duration = self._state["config"]["fire_spawn_interval_ticks"] * self._state["world"]["height"] * self._state["world"]["width"]
        self.GAME_LEN  = max_fire_duration + max_fire_duration
        
        self.actions = ["right", "left", "up", "down", "bomb", "detonate-1", "detonate-2", "detonate-3", "noop"]
        self.total_actions = []
        self.last_action = {}

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0, high=254, shape=(24,15,15), dtype=np.float32)

        self.metadata = {"render.modes": ["human"]}
        self.last_reward = {}
        self.current_reward = 0
        self.MIN_REWARD = -10000
        self.MAX_REWARD = 10000

        self.model_name = "no_name"

        self.current_render = []

        self.reward_range = (self.MIN_REWARD, self.MAX_REWARD)
        self.last_position = {}

        self.done = False
        self.info = {"state":self._state, "events":{}}
        self.holder = [self.shaped_state, self.current_reward, self.done, self.info]

    def get_state(self):
        return self._state

    def set_model_name(self, model_name):
        self.model_name = model_name

    async def get_actions(self, actions=None, agent_id=None, unit_id=None):
        directions = ["right", "left", "up", "down"]

        enemy_actions = ["right", "left", "up", "down", "noop", "noop", "noop", "noop", "noop", "noop"]

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

        if self.last_action.get(unit_id) is None:
            self.last_action[unit_id] = ""
        if self.last_position.get(unit_id) is None:
            self.last_position[unit_id] = self._state["unit_state"][unit_id]["coordinates"]

        holder = actions
        actions = []

        d_action = d_action.split("-")
        u_action = d_action[0]

        self.last_position[unit_id] = self._state["unit_state"][unit_id]["coordinates"]

        if u_action in directions:
            action = {"action": {"type":"move", "unit_id": unit_id, "move": u_action}, "agent_id": agent_id}
            actions.append(action)
            self.last_action[unit_id] = "move"
            self.last_position[unit_id] = self._state["unit_state"][unit_id]["coordinates"]
        elif u_action == "bomb":
            if (self._state["unit_state"][unit_id]["inventory"]["bombs"] >= 1) :
                action = {"action": {"type": "bomb", "unit_id": unit_id}, "agent_id": agent_id}
                actions.append(action)
                if self.last_action[unit_id] != "bomb" and self.last_action[unit_id] != "fail" and self.last_action[unit_id] != "noop":
                    self.last_action[unit_id] = "bomb"
                else:
                    self.last_action[unit_id] = "fail"
            else:
                self.last_action[unit_id] = "fail"
                print("fail bomb")
        elif u_action == "detonate":
            bomb_coords = get_bombs_coords(self._state, unit_id)
            if bomb_coords is not None:
                if type(bomb_coords[0]) is list:
                    d_action[1] = int(d_action[1])

                    #if there are less bombs than requested, fail, else return bomb coordinates of the selected bomb (-1, -2, -3) from end
                    #selected from the end as to "last placed" to "first placed" so most recent bomb is always -1, etc.
                    if len(bomb_coords) >= d_action[1]:
                        bomb_coords = bomb_coords[-d_action[1]]
                    else:
                        self.last_action[unit_id] = "fail"
                        return actions
                action = {"action": {"type": "detonate", "unit_id": unit_id, "coordinates": bomb_coords}, "agent_id": agent_id}
                actions.append(action)
                self.last_action[unit_id] = "detonate"
            else:
                self.last_action[unit_id] = "fail"
        else:
            self.last_action[unit_id] = "noop"
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
        
        position_reward = self.position_advantage[posy][posx]*3

        if self.last_reward[unit_id] is None:
            self.last_reward[unit_id] = ((sum_my_hp - sum_enemy_hp)*100) + position_reward

        if not self.done:
            to_return = (((sum_my_hp - sum_enemy_hp)*100) - self.last_reward[unit_id] + position_reward)
            self.last_reward[unit_id] = ((sum_my_hp - sum_enemy_hp)*100) + position_reward

            if self.last_action[unit_id] == "bomb": # or self.last_action[unit_id] == "detonate"
                to_return += 10
            elif self.last_action[unit_id] == "fail":
                to_return -= 20
            elif self.last_action[unit_id] == "move" and self.last_position[unit_id] == self._state["unit_state"][unit_id]["coordinates"]:
                to_return -= 2
                print("fail move")
                self.last_action[unit_id] = "fail"
            print(to_return)
            return to_return

        """
        if n_my_units > 0 and n_enemy_units == 0: #i win
            print("i win")
            reward_now = (1000*n_my_units + 100*sum_my_hp)*(self.GAME_LEN/(current_tick+1))
            return min(reward_now, self.MAX_REWARD)
            
        elif n_my_units == 0 and n_enemy_units > 0: #enemy win
            print("i lose")
            reward_now = -(1000*n_enemy_units + 100*sum_enemy_hp)*(self.GAME_LEN/(current_tick+1))
            return max(reward_now, self.MIN_REWARD)
        """
        reward_now = (-(1000*n_enemy_units + 100*sum_enemy_hp)+(500*n_my_units + 50*sum_my_hp))*(self.GAME_LEN/(current_tick+1))
        if reward_now > 0:
            print("i win")
            return min(reward_now, self.MAX_REWARD)
        elif reward_now < 0:
            print("i lose")
            return max(reward_now, self.MIN_REWARD)
        
        print("i draw")
        return 0 #draw

    def alive_reward(self, unit_id):
        if self.last_action[unit_id] == "move" and self.last_position[unit_id] != self._state["unit_state"][unit_id]["coordinates"]:
            return 2
        return 0

    def alive_reward_v2(self, unit_id, training_mode):
        agent_id = "a"
        if self.done and training_mode:
            if self._state["unit_state"][unit_id]["hp"] < 1:
                return -1 #lose -1
            else:
                return 1 #win 1
        elif self.done:
            unit_states = pd.DataFrame.from_dict(self._state["unit_state"]).T
            my_units = unit_states[unit_states["agent_id"] == agent_id]
            my_units = my_units[my_units["hp"] > 0]["hp"]
            enemy_units = unit_states[unit_states["agent_id"] != agent_id]
            enemy_units = enemy_units[enemy_units["hp"] > 0]["hp"]
            final_count = len(my_units) - len(enemy_units)
            if final_count > 0:
                return 1 #win 1
            elif final_count == 0:
                return 0 #draw 0
            else:
                return -1 #lose -1
        if self.last_action[unit_id] == "move" and self.last_position[unit_id] != self._state["unit_state"][unit_id]["coordinates"]:
            return 2 #move 2
        #noop 0
        return 0

    async def async_reset(self):
        print("Resetting")

        if self._initial_state is None:
            print("none in reset")
            self._state = generate_random_env()

        self.info["state"] = self._state
        self.info["events"] = {}
        self.shaped_state = get_shaped(self._state)
        self.last_action = {}
        self.last_reward = {}
        self.last_position = {}
        self.current_render = []
        self.current_reward = 0
        return self.shaped_state

    async def await_reset(self):
        return await self.async_reset()

    def reset(self):
        self.total_actions = []

        loop = asyncio.get_event_loop()
        task = loop.create_task(self.await_reset())
        loop.run_until_complete(task)
        return self.shaped_state


    async def async_step(self, actions, training_mode=True, get_enemy_actions=True):

        #print(pd.DataFrame(self._state["unit_state"]).T)
        self.current_render.append(show_state(self._state, False))

        all_actions = []

        if (type(actions) is list) and (type(actions[0]) is list):
            for action in actions:
                all_actions += await self.get_actions(action[0], action[1], action[2])
        else:
            all_actions += await self.get_actions(actions)

        if get_enemy_actions:
            for enemy_unit in self._state["agents"]["b"]["unit_ids"]:
                if self._state["unit_state"][enemy_unit]["hp"] > 0 and self._state["unit_state"][enemy_unit]["coordinates"] != [7,7]:
                    all_actions += await self.get_actions(None, "b", enemy_unit)

        current_tick = self._state["tick"]
        str_actions = str(current_tick) +str("}\n{".join(str(all_actions).strip("[]").split("}, {")))

        self.total_actions.append(str_actions)

        state = await self._send(self._state, all_actions, self._channel)
        self._state = state.get("next_state")
        self.shaped_state = get_shaped(self._state)

        self.done = state.get("is_complete")

        if training_mode and self._state["unit_state"]["c"]["hp"] < 1:
            self.done = True

        self.info["events"] = state.get("tick_result").get("events")
        self.info["state"] = self._state

        self.current_reward = self.alive_reward_v2("c", training_mode)

        print(f"{self.current_reward}")

        self.holder = [self.shaped_state, self.current_reward, self.done, self.info]
        return self.holder


    def step(self, actions, training_mode=True):

        loop = asyncio.get_event_loop()
        task = loop.create_task(self.async_step(actions, training_mode))
        loop.run_until_complete(task)

        if self.done:
            """
            if not os.path.exists("my_logs"):
                os.makedirs("my_logs")
            if not os.path.exists(f"my_logs/{self.model_name}"):
                os.makedirs(f"my_logs/{self.model_name}")
            with open(f"my_logs/{self.model_name}/log-{int(time.time())}.txt", "wb") as f:
                pickle.dump(self.total_actions, f)
            """

            if not os.path.exists("renders"):
                os.makedirs("renders")
            if not os.path.exists(f"renders/{self.model_name}"):
                os.makedirs(f"renders/{self.model_name}")
            with open(f"renders/{self.model_name}/render-{int(time.time())}.txt", "wb") as f:
                pickle.dump(self.current_render, f)

            #print(self.total_actions)
        return self.holder


    def render(self, mode="human", close=False):
        show_state(self._state)


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
