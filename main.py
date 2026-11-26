import math

import gymnasium as gym
from model import FFN
import torch
import torch.nn as nn
import multiprocessing as mp
from multi_envs import run
from ppo_implimentation import PPO

class GlobalRolloutBuffer:
    def __init__(self, steps, obs_dim, n_envs):
        self.states = torch.zeros((steps+1, n_envs, obs_dim), dtype=torch.float32)  # to accommodate last state
        self.values = torch.zeros((steps+1, n_envs), dtype=torch.float32)

        self.rewards = torch.zeros((steps, n_envs), dtype=torch.float32)
        self.actions = torch.zeros((steps, n_envs), dtype=torch.int64)
        self.log_probs = torch.zeros((steps, n_envs), dtype=torch.float32)

        self.done = torch.zeros((steps, n_envs), dtype=torch.int8)  # -1 truncated, 1 terminated, 0 neither, 2 both


# // ENV PARAMS //
SEED = 67
ROLLOUT_STEPS = 500
N_ENVS = 6
ACT_SPACE = 2
OBS_DIM = 4

# // MODEL PARAMS //
HIDDEN_DIM = 128
HIDDEN_LAYERS = 3

# // PPO PARAMS
GAMMA = 0.99
LMBDA = 0.95
EPSILON = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
PPO_EPOCHS = 5
POLICY_LR = 3e-4
VALUE_LR = 3e-4
MINIBATCH = 64


if __name__ == '__main__':

    agent = PPO(
        obs_dim=OBS_DIM,
        act_dim=ACT_SPACE,
        hidden_dim=HIDDEN_DIM,
        num_hidden=HIDDEN_LAYERS,
        gamma=GAMMA,
        lmbda=LMBDA,
        eps=EPSILON,
        ent_coef=ENTROPY_COEF,
        value_coef=VALUE_COEF,
        epochs=PPO_EPOCHS,
        policy_lr=POLICY_LR,
        value_lr=VALUE_LR,
        minibatch=MINIBATCH,
    )

    buffer = run(agent.actor, agent.critic)
    advantage, returns = agent.gae(buffer)
    print(advantage.shape, returns.shape)


    for name, tensor in vars(buffer).items():
        print(f"{name}: {tensor.shape}")

    print("Rewards:", buffer.rewards.min(), buffer.rewards.max())
    print("Actions:", torch.unique(buffer.actions))
    print("Done flags:", torch.unique(buffer.done, return_counts=True))
    print("NaN values:", torch.isnan(buffer.values).any())

