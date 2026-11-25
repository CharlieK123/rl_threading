import math

import gymnasium as gym
from model import FFN
import torch
from torch.distributions import Categorical
import torch.nn as nn
import multiprocessing as mp
from multi_envs import run

class GlobalRolloutBuffer:
    def __init__(self, steps, obs_dim, n_envs):
        self.states = torch.zeros((steps+1, n_envs, obs_dim), dtype=torch.float32)  # to accommodate last state
        self.values = torch.zeros((steps+1, n_envs), dtype=torch.float32)

        self.rewards = torch.zeros((steps, n_envs), dtype=torch.float32)
        self.actions = torch.zeros((steps, n_envs), dtype=torch.int64)
        self.log_probs = torch.zeros((steps, n_envs), dtype=torch.float32)

        self.done = torch.zeros((steps, n_envs), dtype=torch.int8)  # -1 truncated, 1 terminated, 0 neither, 2 both


class RolloutBuffer:
    def __init__(self, steps, obs_dim):
        self.states = torch.zeros((steps+1, obs_dim), dtype=torch.float32)  # to accommodate last state
        self.values = torch.zeros(steps+1, dtype=torch.float32)

        self.rewards = torch.zeros(steps, dtype=torch.float32)
        self.actions = torch.zeros(steps, dtype=torch.int64)
        self.log_probs = torch.zeros(steps, dtype=torch.float32)

        self.done = torch.zeros(steps, dtype=torch.int8)  # -1 truncated, 1 terminated, 0 neither, 2 both

class Actor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_hidden):
        super().__init__()

        self.policy = FFN(in_dim, out_dim, hidden_dim, num_hidden, actor=True)

    def get_action(self, state):
        logits = self.policy(state)
        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob  # return action tensor and respective log prob


class Critic(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_hidden):
        super().__init__()

        self.value = FFN(in_dim, out_dim, hidden_dim, num_hidden, actor=False)

    def get_value(self, state):
        return self.value(state).squeeze(-1)  # return predicted value scalar


# Params
SEED = 67
ROLLOUT_STEPS = 500
N_ENVS = 6
ACT_SPACE = 2

# // FFN Params //
INPUT_DIM = 4
OUTPUT_DIM = 120
HIDDEN_DIM = 128
HIDDEN_LAYERS = 3


def run_single_rollout(env, steps):
    obs, info = env.reset(seed=SEED)

    for i in range(steps):
        state = torch.as_tensor(obs, dtype=torch.float32)

        action, log_prob = actor.get_action(state)  # tensor, tensor
        pred_value = critic.get_value(state)  # scalar
        print(type(action),type(log_prob), type(pred_value.item()))

        next_obs, reward, terminated, truncated, _ = env.step(action.item())


        if terminated or truncated:
            obs, info = env.reset(seed=SEED)
        else:
            obs = next_obs

def run_parallel_rollout(conn, num_env):
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=SEED+num_env)

    conn.send(obs)

    while True:
        action = conn.recv()

        if action is None:
            break

        next_obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()
        else:
            obs = next_obs

        conn.send((obs, reward, terminated, truncated))

    env.close()
    conn.close()

def build_processes():
    connections = []
    processes = []

    for i in range(N_ENVS):
        main_conn, process_conn = mp.Pipe()

        process = mp.Process(
            target=run_parallel_rollout,
            args=(process_conn, i)
        )

        process.start()
        process_conn.close()

        connections.append(main_conn)
        processes.append(process)

    return connections, processes


def run_processes(connections):
    # initialise the starting states for all processes
    states = torch.stack([
        torch.as_tensor(conn.recv(), dtype=torch.float32)
        for conn in connections
    ])

    with torch.inference_mode():
        for i in range(ROLLOUT_STEPS):
            # compute agent actions, log_probs and critic values
            actions, log_probs = actor.get_action(states)
            values = critic.get_value(states)

            # send actions to envs to rollout
            for j, conn in enumerate(connections):
                conn.send(actions[j].item())

            # envs send new states
            new_states = [conn.recv() for conn in connections]

            # Extract new observations
            states = torch.stack([
                torch.as_tensor(result[0], dtype=torch.float32)
                for result in new_states
            ])






if __name__ == '__main__':
    actor = Actor(
        in_dim=INPUT_DIM,
        out_dim=ACT_SPACE,
        hidden_dim=HIDDEN_DIM,
        num_hidden=HIDDEN_LAYERS
    )

    critic = Critic(
        in_dim=INPUT_DIM,
        out_dim=1,
        hidden_dim=HIDDEN_DIM,
        num_hidden=HIDDEN_LAYERS
    )

    buffer = run(actor, critic)

    for name, tensor in vars(buffer).items():
        print(f"{name}: {tensor.shape}")

