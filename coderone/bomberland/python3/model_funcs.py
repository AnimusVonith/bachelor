import pandas as pd
from nn_setup import CustomCNN
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os


def get_model(alg_opt, path=".\\models", env=None):
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
            model = alg_opt.load(f"{path}/{last_model}", policy=CustomCNN)
            if env is not None:
                model.set_env(env)
            print(f"loaded {last_model}")
    except Exception as e:
        raise e
    return model, steps_learnt

def load_model(env = None):
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

    model, steps_learnt = get_model(alg_opt, model_dir, env)

    """
    until simple_reward
    if model is None and env is not None:
        model = alg_opt("CnnPolicy", env, verbose=1, tensorboard_log=log_dir, learning_rate=0.00005, policy_kwargs=policy_kwargs)
        print("creating new model")
        steps_learnt = 0
    """

    if model is None and env is not None:
        model = alg_opt("CnnPolicy", env, verbose=1, tensorboard_log=log_dir, learning_rate=0.001, policy_kwargs=policy_kwargs)
        print("creating new model")
        steps_learnt = 0

    return model, steps_learnt, model_name, model_dir