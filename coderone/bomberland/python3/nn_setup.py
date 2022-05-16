#https://stable-baselines3.readthedocs.io/_/downloads/en/master/pdf/ 1.9.3 (page 45)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import gym
import os

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