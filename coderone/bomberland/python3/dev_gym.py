import asyncio
from typing import Dict
from gym_lib import Gym
import sys
import gym
import gym_lib
import random
import pandas as pd
import numpy as np
import os
import time

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from model_funcs import load_model
from shaping_funcs import get_shaped, get_pos_score, unit_channel_shaping
from arena_templates import get_arena_templates

fwd_model_uri = os.environ.get(
    "FWD_MODEL_CONNECTION_STRING") or "ws://127.0.0.1:6969/?role=admin"

def learning(model, model_name, learn_step=2000, steps_learnt=0, iterations=10, path="."):
    for i in range(iterations):
        steps_learnt += learn_step
        last_timestamp = int(time.time())
        last_name = f"{model_name}-{steps_learnt}-{last_timestamp}"

        model.learn(total_timesteps=learn_step, reset_num_timesteps=False, tb_log_name=f"tb_log-{model_name}")
        model.save(f"{path}/{last_name}")

    return model, steps_learnt

def testing(env, model, n_of_games=10, actions_counter=None, agent_id="a"):

    rewards = 0
    score = [0,0,0]
    i = 0

    obs = env.reset()

    avg_rewards = []
    steps = 0

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

            if actions_counter is not None and action < len(actions_counter):
                actions_counter[action] += 1

            actions.append([action, agent_id, unit])
        obs, reward, done, info = env.step(actions, False)
        
        if done:
            steps = state["tick"]
            rewards += reward
            avg_rewards.append(rewards/float(steps))
            i+=1
            steps = 0
            if reward > 0:
                score[0]+=1
            elif reward < 0:
                score[2]+=1
            else:
                score[1]+=1

            print(info)
            print(reward, rewards)
            obs = env.reset()
    return rewards, score, actions_counter, avg_rewards

def main():

    gym = Gym(fwd_model_uri)

    loop = asyncio.get_event_loop()
    task = loop.create_task(gym.connect())
    loop.run_until_complete(task)

    #None = randomly generated
    CURRENT_ARENA = None
    #CURRENT_ARENA = get_arena_templates()

    actions_counter = np.zeros(9, np.uint32)

    env = gym.make("bomberland-open-ai-gym", CURRENT_ARENA)

    model, steps_learnt, model_name, model_dir = load_model(env)

    env.set_model_name(model_name)

    ITERATIONS = 10
    LEARN_STEP = 10000
    N_OF_GAMES = 10
    TRAINING_LOOPS = 1000

    final_score = [0,0,0]
    best_avg = -100000

    for i in range(TRAINING_LOOPS):
        #learning
        model, steps_learnt = learning(model, model_name, LEARN_STEP, steps_learnt, ITERATIONS, model_dir)
        print(f"finished learning after {steps_learnt} steps")
        
        #testing
        if i == 10:
            N_OF_GAMES = 100
        else:
            N_OF_GAMES = 10
            
        rewards, score, actions_counter, avg_rewards = testing(env, model, N_OF_GAMES, actions_counter)
        final_score[0] += score[0]
        final_score[1] += score[1]
        final_score[2] += score[2]
        avg = int(rewards/N_OF_GAMES)
        best_avg = max(avg, best_avg)

        print(f"finished testing with score: {score} and final reward: {rewards} in {N_OF_GAMES} games.")
        print(f"final score: {final_score}\nthis average: {avg}\nbest average: {best_avg}")

        if not os.path.exists("result_logs"):
            os.makedirs("result_logs")
        with open(f"result_logs/results-{model_name}.txt", "w") as f:
            f.write(f"""{model_name}\nwin/draw/lose\ntotal_score:{final_score}\ncurrent_winrate:{(final_score[0]/float(sum(final_score)))*100.0}%\nthis_score:{score}\navg_per_step:{avg_rewards}\nthis_avg:{avg}\nbest_avg:{best_avg}\nactions_counter:{actions_counter}\n{env.actions}""")

    loop = asyncio.get_event_loop()
    task = loop.create_task(gym.close())
    loop.run_until_complete(task)


if __name__ == "__main__":
    print(sys.argv)
    exit(0)
    main()