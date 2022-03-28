from typing import Union
from game_state import GameState
import asyncio
import random
import os
import pandas as pd
import numpy as np
import json
import gym

uri = os.environ.get(
    'GAME_CONNECTION_STRING') or "ws://127.0.0.1:3000/?role=agent&agentId=agentA&name=defaultName"


class MyEnv():
    def __init__(self):
        self.action_space = actions

    def step(self):
        ...

    def reset(self):
        ...


class Agent():
    def __init__(self):
        self._client = GameState(uri)

        # any initialization code can go here
        self._client.set_game_tick_callback(self._on_game_tick)

        loop = asyncio.get_event_loop()
        connection = loop.run_until_complete(self._client.connect())
        tasks = [
            asyncio.ensure_future(self._client._handle_messages(connection)),
        ]
        loop.run_until_complete(asyncio.wait(tasks))

    # returns coordinates of the first bomb placed by a unit
    def _get_bomb_to_detonate(self, unit) -> Union[int, int] or None:
        entities = self._client._state.get("entities")
        bombs = list(filter(lambda entity: entity.get(
            "unit_id") == unit and entity.get("type") == "b", entities))
        bomb = next(iter(bombs or []), None)
        if bomb != None:
            return [bomb.get("x"), bomb.get("y")]
        else:
            return None


    def unit_channel_shaping(self, game_state, my_unit_id, input_arr):
        df = pd.DataFrame.from_dict(game_state["unit_state"])
        dfh = df.copy().T
        shape = input_arr[0].shape
        my_unit = dfh[dfh["unit_id"]==my_unit_id][["coordinates", "hp"]]
        np.put(input_arr[0], np.ravel_multi_index(np.array(my_unit["coordinates"].to_list()).T, shape), my_unit["hp"].to_list())
        return input_arr


    def shaping(self, game_state, my_agent_id):
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
        oh = dfh[dfh["type"]=="o"][["x", "y", "hp"]]
        o1 = oh[oh["hp"]==1][["x", "y"]]
        o2 = oh[oh["hp"]==2][["x", "y"]]
        o3 = oh[oh["hp"]==3][["x", "y"]]
        ex = dfh[dfh["type"]=="x"][["x", "y"]]

        if "expires" in dfh:
            a = dfh[dfh["type"]=="a"]
            be = dfh[dfh["type"]=="b"]
            bp = dfh[dfh["type"]=="bp"]
            x = dfh[dfh["type"]=="x"]
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
        np.put(input_arr[12], np.ravel_multi_index(np.array(oh[["x", "y"]]).T, shape), oh["hp"].to_list())
        np.put(input_arr[13], np.ravel_multi_index(np.array(o1[["x", "y"]]).T, shape), 1)
        np.put(input_arr[14], np.ravel_multi_index(np.array(o2[["x", "y"]]).T, shape), 1)
        np.put(input_arr[15], np.ravel_multi_index(np.array(o3[["x", "y"]]).T, shape), 1)
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


    def magic_happens(self, input_arr):
        #do something with input_arr
        action = random.choice(self.actions)
        return action


    def tick_save(self, game_state):
        with open(f"replays/{game_state['game_id']}.json", "w+", encoding="utf-8") as f:
            json.dump(game_state, f, ensure_ascii=False, indent=4)


    async def _on_game_tick(self, tick_number, game_state):
        
        # get my units
        my_agent_id = game_state.get("connection").get("agent_id")
        my_units = game_state.get("agents").get(my_agent_id).get("unit_ids")

        self.tick_save(game_state)

        input_arr = self.shaping(game_state, my_agent_id)

        # send each unit a random action
        for unit_id in my_units:
            if game_state["unit_state"][unit_id]["hp"] < 1:
                continue
            
            self.actions = ["up", "down", "left", "right", "bomb", "detonate", "noop"]

            input_arr = self.unit_channel_shaping(game_state, unit_id, input_arr)

            action = self.magic_happens(input_arr)

            if action in ["up", "left", "right", "down"]:
                await self._client.send_move(action, unit_id)
            elif action == "bomb":
                await self._client.send_bomb(unit_id)
            elif action == "detonate":
                bomb_coordinates = self._get_bomb_to_detonate(unit_id)
                if bomb_coordinates != None:
                    x, y = bomb_coordinates
                    await self._client.send_detonate(x, y, unit_id)
            else:
                print(f"Unhandled action: {action} for unit {unit_id}")


def main():
    Agent()


if __name__ == "__main__":
    main()
