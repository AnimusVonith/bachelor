from typing import Union
from game_state import GameState
import asyncio
import random
import os
from stable_baselines3 import PPO, A2C
import pandas as pd
import numpy as np

from model_funcs import load_model
from shaping_funcs import get_shaped, get_pos_score, unit_channel_shaping
from misc_funcs import get_bomb_coords

uri = os.environ.get(
    'GAME_CONNECTION_STRING') or "ws://127.0.0.1:3000/?role=agent&agentId=agentId&name=defaultName"


actions = ["up", "down", "left", "right", "bomb", "detonate-1", "detonate-2", "detonate-3" "noop"]


class Agent():
    def __init__(self):
        self._client = GameState(uri)

        self.model, self.steps_learnt = load_model()
        
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

    async def _on_game_tick(self, tick_number, game_state):

        # get my units
        my_agent_id = game_state.get("connection").get("agent_id")
        my_units = game_state.get("agents").get(my_agent_id).get("unit_ids")

        shaped_state = get_shaped(game_state, my_agent_id)

        # send each unit a random action
        for unit_id in my_units:

            if game_state["unit_state"][unit_id]["hp"] < 1:
                continue

            shaped_state = unit_channel_shaping(game_state, unit_id, shaped_state)

            action = self.model.predict(shaped_state)

            if action in ["up", "left", "right", "down"]:
                await self._client.send_move(action, unit_id)
            elif action == "bomb":
                await self._client.send_bomb(unit_id)
            elif action[:8] == "detonate":
                bomb_coordinates = self.get_bomb_coords(game_state, unit_id)
                if bomb_coordinates is not None:
                    if len(bomb_coordinates)
                    x, y = bomb_coordinates
                    await self._client.send_detonate(x, y, unit_id)
            else:
                print(f"Unhandled action: {action} for unit {unit_id}")


def main():
    Agent()


if __name__ == "__main__":
    main()
