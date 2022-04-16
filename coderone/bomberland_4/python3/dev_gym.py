import asyncio
from typing import Dict
from gym_lib import Gym
import os
import sys
import gym
import gym_lib
import random
import pandas as pd
import numpy as np
import time
import torch as th
import torch.nn as nn
import argparse

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


#https://stable-baselines3.readthedocs.io/_/downloads/en/master/pdf/ 1.9.3 (page 45)

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
    This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 32):
    
        cnn_opt = 3

        if os.path.exists("info.txt"):
            with open("info.txt", "r") as f:
                holder = f.read().split("\n")
                cnn_opt = int(holder[0])

        if cnn_opt > 2:
            features_dim = 64

        print(f"cnn_opt: {cnn_opt}--------------features_dim: {features_dim}")

        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        #OUTPUT = ((INPUT - KERNEL + 2*PADDING) / STRIDE) + 1
        #OUTPUT = ((15-3+2*1)/1)+1
        #OUTPUT = 15

        self.nn_options = {
        1: nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, padding=1, stride=1), 
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()),

        2: nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, padding=1, stride=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()),

        3: nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, padding=1, stride=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()),

        4: nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, padding=1, stride=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Flatten()),
        }

        self.cnn = self.nn_options[cnn_opt]
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


fwd_model_uri = os.environ.get(
    "FWD_MODEL_CONNECTION_STRING") or "ws://127.0.0.1:6969/?role=admin"

arena_templates = {

    "mock_15x15_state" : {
        "game_id": "dev",
        "agents": {
            "a": {"agent_id": "a","unit_ids": ["c"]},   #,"e","g"
            "b": {"agent_id": "b","unit_ids": ["d"]}},  #,"f","h"
        "unit_state": {
            "c": {"coordinates": [4,0],"hp": 1,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "c","agent_id": "a","invulnerability": 0},
            "d": {"coordinates": [7,7],"hp": 1,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "d","agent_id": "b","invulnerability": 0},
            #"e": {"coordinates": [5,12],"hp": 3,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "e","agent_id": "a","invulnerability": 0},
            #"f": {"coordinates": [9,12],"hp": 3,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "f","agent_id": "b","invulnerability": 0},
            #"g": {"coordinates": [1,10],"hp": 3,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "g","agent_id": "a","invulnerability": 0},
            #"h": {"coordinates": [13,10],"hp": 3,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "h","agent_id": "b","invulnerability": 0}
        },
        "entities": [
            {"created": 0,"x": 14,"y": 1,"type": "m"},{"created": 0,"x": 0,"y": 1,"type": "m"},{"created": 0,"x": 1,"y": 2,"type": "m"},
            {"created": 0,"x": 13,"y": 2,"type": "m"},{"created": 0,"x": 11,"y": 0,"type": "m"},{"created": 0,"x": 3,"y": 0,"type": "m"},
            {"created": 0,"x": 12,"y": 12,"type": "m"},{"created": 0,"x": 2,"y": 12,"type": "m"},{"created": 0,"x": 11,"y": 14,"type": "m"},
            {"created": 0,"x": 3,"y": 14,"type": "m"},{"created": 0,"x": 6,"y": 8,"type": "m"},{"created": 0,"x": 8,"y": 8,"type": "m"},
            {"created": 0,"x": 14,"y": 7,"type": "m"},{"created": 0,"x": 0,"y": 7,"type": "m"},{"created": 0,"x": 2,"y": 4,"type": "m"},
            {"created": 0,"x": 12,"y": 4,"type": "m"},{"created": 0,"x": 8,"y": 14,"type": "m"},{"created": 0,"x": 6,"y": 14,"type": "m"},
            {"created": 0,"x": 6,"y": 7,"type": "m"},{"created": 0,"x": 8,"y": 7,"type": "m"},{"created": 0,"x": 9,"y": 9,"type": "m"},
            {"created": 0,"x": 5,"y": 9,"type": "m"},{"created": 0,"x": 9,"y": 2,"type": "m"},{"created": 0,"x": 5,"y": 2,"type": "m"},
            {"created": 0,"x": 11,"y": 8,"type": "m"},{"created": 0,"x": 3,"y": 8,"type": "m"},{"created": 0,"x": 5,"y": 5,"type": "m"},
            {"created": 0,"x": 9,"y": 5,"type": "m"},{"created": 0,"x": 5,"y": 11,"type": "m"},{"created": 0,"x": 9,"y": 11,"type": "m"},
            {"created": 0,"x": 5,"y": 14,"type": "m"},{"created": 0,"x": 9,"y": 14,"type": "m"},{"created": 0,"x": 2,"y": 0,"type": "m"},
            {"created": 0,"x": 12,"y": 0,"type": "m"},{"created": 0,"x": 10,"y": 6,"type": "m"},{"created": 0,"x": 4,"y": 6,"type": "m"},
            {"created": 0,"x": 1,"y": 4,"type": "m"},{"created": 0,"x": 13,"y": 4,"type": "m"},{"created": 0,"x": 3,"y": 10,"type": "m"},
            {"created": 0,"x": 11,"y": 10,"type": "m"},{"created": 0,"x": 10,"y": 7,"type": "m"},{"created": 0,"x": 4,"y": 7,"type": "m"},
            {"created": 0,"x": 1,"y": 13,"type": "m"},{"created": 0,"x": 13,"y": 13,"type": "m"},{"created": 0,"x": 0,"y": 10,"type": "m"},
            {"created": 0,"x": 14,"y": 10,"type": "m"},{"created": 0,"x": 10,"y": 1,"type": "m"},{"created": 0,"x": 4,"y": 1,"type": "m"},
            {"created": 0,"x": 12,"y": 14,"type": "m"},{"created": 0,"x": 2,"y": 14,"type": "m"},
            {"created": 0,"x": 9,"y": 6,"type": "w","hp": 1},{"created": 0,"x": 5,"y": 6,"type": "w","hp": 1},
            {"created": 0,"x": 2,"y": 10,"type": "w","hp": 1},{"created": 0,"x": 12,"y": 10,"type": "w","hp": 1},
            {"created": 0,"x": 2,"y": 7,"type": "w","hp": 1},{"created": 0,"x": 12,"y": 7,"type": "w","hp": 1},
            {"created": 0,"x": 3,"y": 6,"type": "w","hp": 1},{"created": 0,"x": 11,"y": 6,"type": "w","hp": 1},
            {"created": 0,"x": 12,"y": 1,"type": "w","hp": 1},{"created": 0,"x": 2,"y": 1,"type": "w","hp": 1},
            {"created": 0,"x": 4,"y": 14,"type": "w","hp": 1},{"created": 0,"x": 10,"y": 14,"type": "w","hp": 1},
            {"created": 0,"x": 4,"y": 9,"type": "w","hp": 1},{"created": 0,"x": 10,"y": 9,"type": "w","hp": 1},
            {"created": 0,"x": 13,"y": 5,"type": "w","hp": 1},{"created": 0,"x": 1,"y": 5,"type": "w","hp": 1},
            {"created": 0,"x": 2,"y": 2,"type": "w","hp": 1},{"created": 0,"x": 12,"y": 2,"type": "w","hp": 1},
            {"created": 0,"x": 12,"y": 13,"type": "w","hp": 1},{"created": 0,"x": 2,"y": 13,"type": "w","hp": 1},
            {"created": 0,"x": 14,"y": 6,"type": "w","hp": 1},{"created": 0,"x": 0,"y": 6,"type": "w","hp": 1},
            {"created": 0,"x": 6,"y": 13,"type": "w","hp": 1},{"created": 0,"x": 8,"y": 13,"type": "w","hp": 1},
            {"created": 0,"x": 2,"y": 6,"type": "w","hp": 1},{"created": 0,"x": 12,"y": 6,"type": "w","hp": 1},
            {"created": 0,"x": 1,"y": 12,"type": "w","hp": 1},{"created": 0,"x": 13,"y": 12,"type": "w","hp": 1},
            {"created": 0,"x": 10,"y": 2,"type": "w","hp": 1},{"created": 0,"x": 4,"y": 2,"type": "w","hp": 1},
            {"created": 0,"x": 3,"y": 12,"type": "w","hp": 1},{"created": 0,"x": 11,"y": 12,"type": "w","hp": 1},
            {"created": 0,"x": 8,"y": 11,"type": "w","hp": 1},{"created": 0,"x": 6,"y": 11,"type": "w","hp": 1},
            {"created": 0,"x": 8,"y": 10,"type": "w","hp": 1},{"created": 0,"x": 6,"y": 10,"type": "w","hp": 1},
            {"created": 0,"x": 6,"y": 9,"type": "w","hp": 1},{"created": 0,"x": 8,"y": 9,"type": "w","hp": 1},
            {"created": 0,"x": 14,"y": 14,"type": "w","hp": 1},{"created": 0,"x": 0,"y": 14,"type": "w","hp": 1},
            {"created": 0,"x": 4,"y": 13,"type": "w","hp": 1},{"created": 0,"x": 10,"y": 13,"type": "w","hp": 1},
            {"created": 0,"x": 6,"y": 12,"type": "w","hp": 1},{"created": 0,"x": 8,"y": 12,"type": "w","hp": 1},
            {"created": 0,"x": 0,"y": 0,"type": "w","hp": 1},{"created": 0,"x": 14,"y": 0,"type": "w","hp": 1},
            {"created": 0,"x": 6,"y": 6,"type": "w","hp": 1},{"created": 0,"x": 8,"y": 6,"type": "w","hp": 1},
            {"created": 0,"x": 5,"y": 7,"type": "w","hp": 1},{"created": 0,"x": 9,"y": 7,"type": "w","hp": 1},
            {"created": 0,"x": 10,"y": 8,"type": "w","hp": 1},{"created": 0,"x": 4,"y": 8,"type": "w","hp": 1},
            {"created": 0,"x": 11,"y": 9,"type": "w","hp": 1},{"created": 0,"x": 3,"y": 9,"type": "w","hp": 1},
            {"created": 0,"x": 10,"y": 3,"type": "w","hp": 1},{"created": 0,"x": 4,"y": 3,"type": "w","hp": 1},
            {"created": 0,"x": 8,"y": 4,"type": "o","hp": 3},{"created": 0,"x": 6,"y": 4,"type": "o","hp": 3},
            {"created": 0,"x": 10,"y": 4,"type": "o","hp": 3},{"created": 0,"x": 4,"y": 4,"type": "o","hp": 3},
            {"created": 0,"x": 13,"y": 8,"type": "o","hp": 3},{"created": 0,"x": 1,"y": 8,"type": "o","hp": 3},
            {"created": 0,"x": 2,"y": 5,"type": "o","hp": 3},{"created": 0,"x": 12,"y": 5,"type": "o","hp": 3},
            {"created": 0,"x": 10,"y": 10,"type": "o","hp": 3},{"created": 0,"x": 4,"y": 10,"type": "o","hp": 3},
            {"created": 0,"x": 8,"y": 2,"type": "o","hp": 3},{"created": 0,"x": 6,"y": 2,"type": "o","hp": 3},
            {"created": 0,"x": 6,"y": 3,"type": "o","hp": 3},{"created": 0,"x": 8,"y": 3,"type": "o","hp": 3}
        ],
        "world": {"width": 15,"height": 15},
        "tick": 0,
        "config": {"tick_rate_hz": 10,"game_duration_ticks": 300,"fire_spawn_interval_ticks": 2}
    },

    "training_arena_1" : {
        "game_id": "dev",
        "agents": {
            "a": {"agent_id": "a","unit_ids": ["c"]},   #,"e","g"
            "b": {"agent_id": "b","unit_ids": ["d"]}},  #,"f","h"
        "unit_state": {
            "c": {"coordinates": [2,3],"hp": 1,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "c","agent_id": "a","invulnerability": 0},
            "d": {"coordinates": [7,7],"hp": 1,"inventory": {"bombs": 0},"blast_diameter": 3,"unit_id": "d","agent_id": "b","invulnerability": 0},
            #"e": {"coordinates": [5,12],"hp": 3,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "e","agent_id": "a","invulnerability": 0},
            #"f": {"coordinates": [9,12],"hp": 3,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "f","agent_id": "b","invulnerability": 0},
            #"g": {"coordinates": [1,10],"hp": 3,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "g","agent_id": "a","invulnerability": 0},
            #"h": {"coordinates": [13,10],"hp": 3,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "h","agent_id": "b","invulnerability": 0}
        },
        "entities": [
            {"created": 0,"x": 6,"y": 7,"type": "m"},
            {"created": 0,"x": 8,"y": 7,"type": "m"},
            {"created": 0,"x": 7,"y": 6,"type": "w","hp": 1},
            {"created": 0,"x": 7,"y": 8,"type": "w","hp": 1}
        ],
        "world": {"width": 15,"height": 15},
        "tick": 0,
        "config": {"tick_rate_hz": 100,"game_duration_ticks": 100,"fire_spawn_interval_ticks": 1}
    }

}
def unit_channel_shaping(game_state, my_unit_id, input_arr):
    shape = input_arr[0].shape
    input_arr[0] = np.zeros(shape)
    
    coords = game_state["unit_state"][my_unit_id]["coordinates"]
    input_values = game_state["unit_state"][my_unit_id]["hp"]

    np.put(input_arr[0], np.ravel_multi_index(np.array(coords).T, shape), input_values)
    return input_arr

def get_model(env, alg_opt, path="."):
    model = None
    steps_learnt = 0
    try:
        holder = pd.DataFrame(os.listdir(path))
        if not holder.empty:
            holder[["n", "steps", "time"]] = holder[0].str.split("-", expand=True)
            holder["steps"] = holder["steps"].astype("int64")
            holder.sort_values("steps", inplace=True)
            last_model = holder.iloc[-1][0]
            steps_learnt = holder.iloc[-1]["steps"]
            model = alg_opt.load(f"{path}/{last_model}", env=env, policy=CustomCNN)
            print(f"loaded {last_model}")
    except Exception as e:
        raise e
    return model, steps_learnt

def learning(model, model_name, learn_step=2000, steps_learnt=0, iterations=10, path="."):
    for i in range(iterations):
        steps_learnt += learn_step
        last_timestamp = int(time.time())
        last_name = f"{model_name}-{steps_learnt}-{last_timestamp}"

        model.learn(total_timesteps=learn_step, reset_num_timesteps=False, tb_log_name=f"tb_log-{model_name}")
        model.save(f"{path}/{last_name}")

    return model, steps_learnt

def testing(env, model, n_of_games=10, agent_id="a"):
    obs = env.reset()
    rewards = 0
    score = [0,0,0]
    i = 0

    while i < n_of_games:
        state = env.get_state()
        actions = []
        alive_units = pd.DataFrame(state["unit_state"]).T
        alive_units = alive_units[alive_units["agent_id"]==agent_id]
        alive_units = alive_units[alive_units["hp"] > 0]
        alive_units = alive_units["unit_id"].to_list()
        for unit in alive_units:
            unit_channel_shaping(state, unit, obs)
            action, _states = model.predict(obs)
            actions.append([action, agent_id, unit])
        obs, reward, done, info = env.step(actions)
        rewards += reward
        if done:
            i+=1
            if reward > 0:
                score[0]+=1
            elif reward < 0:
                score[2]+=1
            else:
                score[1]+=1

            print(info)
            print(reward, rewards)
            obs = env.reset()
    return rewards, score

def main():

    gym = Gym(fwd_model_uri)

    loop = asyncio.get_event_loop()
    task = loop.create_task(gym.connect())
    loop.run_until_complete(task)

    #None = randomly generated
    CURRENT_ARENA = None

    env = gym.make("bomberland-open-ai-gym", CURRENT_ARENA)

    cnn_opt = 1
    alg_opt = PPO
    alg_str = "ppo"
    name = ""

    if os.path.exists("info.txt"):
        with open("info.txt", "r") as f:
            holder = f.read().split("\n")
            cnn_opt, alg_str = holder[:2]
            if alg_str == "a2c":
                alg_opt = A2C
            elif alg_str == "ppo":
                alg_opt = PPO
            else:
                alg_str = "ppo"
                alg_opt = PPO
            cnn_opt = int(cnn_opt)
            if len(holder) == 3:
                name = holder[2]
    
    print(f"alg_str: {alg_str}------------------cnn_opt: {cnn_opt}")


    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=32),
    )

    #get model
    model = None
    steps_learnt = 0
    if name != "":
        name = "_"+str(name)
    model_name = f"{alg_str}_cnn_{cnn_opt}{name}"
    model_dir = f"models/{model_name}"
    log_dir = f"logs/{model_name}"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model, steps_learnt = get_model(env, alg_opt, model_dir)

    if model is None:
        model = alg_opt("CnnPolicy", env, verbose=1, tensorboard_log=log_dir, policy_kwargs=policy_kwargs)
        print("creating new model")
        steps_learnt = 0


    ITERATIONS = 10
    LEARN_STEP = 10000
    N_OF_GAMES = 10
    EPOCHS = 100

    final_score = [0,0,0]
    best_avg = -100000

    for i in range(EPOCHS):
        #learning
        model, steps_learnt = learning(model, model_name, LEARN_STEP, steps_learnt, ITERATIONS, model_dir)
        print(f"finished learning after {steps_learnt} steps")
        
        #testing
        rewards, score = testing(env, model, N_OF_GAMES)
        final_score[0] += score[0]
        final_score[1] += score[1]
        final_score[2] += score[2]
        avg = int(rewards/N_OF_GAMES)
        best_avg = max(avg, best_avg)
        print(f"finished testing with score: {score} and final reward: {rewards} in {N_OF_GAMES} games.")
        print(f"final score: {final_score}\nthis average: {avg}\nbest average: {best_avg}")
        if score[0] == N_OF_GAMES:
            print("might have overtrained - ending process")
            break
    loop = asyncio.get_event_loop()
    task = loop.create_task(gym.close())
    loop.run_until_complete(task)


if __name__ == "__main__":
    main()