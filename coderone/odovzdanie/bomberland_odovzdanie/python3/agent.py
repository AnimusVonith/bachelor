# Jakub Adamčiak
# Bakalárska práca
# Posilňované učenie na hre Bomberman

from typing import Union
from game_state import GameState
import asyncio
import random
import os
import sys
import pandas as pd
import numpy as np

from model_funcs import load_model
from shaping_funcs import get_shaped, unit_channel_shaping
from misc_funcs import get_bombs_coords

class Agent():
    def __init__(self, agentID):
        uri = os.environ.get('GAME_CONNECTION_STRING') or f"ws://127.0.0.1:3000/?role=agent&agentId={agentID}&name=holder"

        self._client = GameState(uri)

        self._client.set_game_tick_callback(self._on_game_tick)

        self.my_agent_id = agentID
        self.actions = ["right", "left", "up", "down", "noop", "noop", "noop", "noop", "noop", "noop"]

        if agentID == "agentA":
            self.actions = ["right", "left", "up", "down", "bomb", "detonate-1", "detonate-2", "detonate-3", "noop"]
            self.model = load_model()[0]
            self.obs = None
            
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

    async def _on_game_tick(self, tick_number, game_state):

        # get my units
        my_agent_id = game_state.get("connection").get("agent_id")
        my_units = game_state.get("agents").get(my_agent_id).get("unit_ids")
        
        if self.my_agent_id == "agentA":
            self.obs = get_shaped(game_state)

        # send each unit a random action
        for unit_id in my_units:

            if game_state["unit_state"][unit_id]["hp"] < 1:
                continue

            if self.my_agent_id == "agentA":
                self.obs = unit_channel_shaping(game_state, unit_id, self.obs)
                action = self.model.predict(self.obs)
                action = self.actions[action[0]]
                print(f"{unit_id} - {action}")
            else:
                action = random.choice(self.actions)

            if isinstance(action,(int,np.int64)) and action < len(self.actions):
                action = self.actions[action]
            
            if action in ["up", "left", "right", "down"]:
                await self._client.send_move(action, unit_id)
            elif action == "bomb":
                await self._client.send_bomb(unit_id)
            elif action[:8] == "detonate":
                bomb_coords = get_bombs_coords(game_state, unit_id)
                if bomb_coords is not None:
                    if type(bomb_coords[0]) is list:
                        holder = int(action[-1])
                        #if there are less bombs than requested, fail, else return bomb coordinates of the selected bomb (-1, -2, -3) from end
                        #selected from the end as to "last placed" to "first placed" so most recent bomb is always -1, etc.
                        if len(bomb_coords) >= holder:
                            bomb_coords = bomb_coords[-holder]
                            x, y = bomb_coords
                            await self._client.send_detonate(x, y, unit_id)


def main(agentID):
    Agent(agentID)


if __name__ == "__main__":
    main(sys.argv[-1])
