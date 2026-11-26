
import gymnasium as gym
import torch
import multiprocessing as mp
import numpy as np

SEED = 67
ROLLOUT_STEPS = 10
N_ENVS = 6
ACT_SPACE = 2
OBS_DIM = 4

class GlobalRolloutBuffer:
    def __init__(self, steps, obs_dim, n_envs):
        self.states = torch.zeros((steps, n_envs, obs_dim), dtype=torch.float32)
        self.next_states = torch.zeros((steps, n_envs, obs_dim), dtype=torch.float32)
        self.values = torch.zeros((steps, n_envs), dtype=torch.float32)

        self.rewards = torch.zeros((steps, n_envs), dtype=torch.float32)
        self.actions = torch.zeros((steps, n_envs), dtype=torch.int64)
        self.log_probs = torch.zeros((steps, n_envs), dtype=torch.float32)

        self.done = torch.zeros((steps, n_envs), dtype=torch.int8)  # -1 truncated, 1 terminated, 0 neither, 2 both

        self.steps = steps
        self.dim = obs_dim
        self.n_envs = n_envs

    def add(self, step, s, n, v, r, a, l, d):
        self.states[step]= s
        self.next_states[step] = n
        self.values[step]= v
        self.rewards[step]= r
        self.actions[step] = a
        self.log_probs[step] = l
        self.done[step] = d




def run_parallel_rollout(conn, num_env):
    """

    :param conn: connection object between main process and specific process
    :param num_env: the index of process this is
    :return: None, all information is sent through the connection

    This function will wait simulate the steps for the main process
    at init this subprocess will send the starting state to the main process which will return an action
    to take, this is repeated until we reach the desired rollout steps, in which the main process will return
    a None action, leading this subprocess to halt and close itself

    """
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=SEED+num_env)

    conn.send(obs)

    while True:
        action = conn.recv()   # wait for action to be computed and sent via main process

        if action is None:  # rollout limit reached
            break

        next_obs, reward, terminated, truncated, _ = env.step(action)

        # -1 truncated, 1 terminated, 0 neither, 2 both
        def define_done(term, trunc):
            if term and trunc:
                done = 2
            elif term and not trunc:
                done = 1
            elif not term and trunc:
                done = -1
            else:
                done = 0
            return done  #   #

        done = define_done(terminated, truncated)


        if terminated or truncated:  #
            obs, info = env.reset()
            main_next_state = obs

        else:
            main_next_state = next_obs

        conn.send((main_next_state, next_obs, reward, done))

    env.close()
    conn.close()

def build_processes():
    """

    :return: connections is a list of the conn objects used to communicate between the main process
    and the subprocesses running the parallel envs, processes is the list of the process objects themselves

    This function builds N sub-processes to run env rollouts in parallel
    """
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


def run_processes(connections, actor, critic):
    """

    :param connections: list of conn objs
    :param actor: Actor model class
    :param critic: Critic model class
    :return: buffer is the saved trajectory data necessary for PPO

    This function acts as the main process and coordinates all the sub-processes, it runs the MLPs to
    get the actions, log_probs and predicted values sends the necessary data to the rollout workers
    and saves their results in the GlobalRolloutBuffer object which saves the data
    """
    buffer = GlobalRolloutBuffer(ROLLOUT_STEPS, OBS_DIM, N_ENVS)

    # initialise the starting states for all processes
    states_t = torch.stack([
        torch.as_tensor(conn.recv(), dtype=torch.float32)
        for conn in connections
    ])

    with torch.inference_mode():
        for i in range(ROLLOUT_STEPS):
            # compute agent actions, log_probs and critic values
            actions_t, log_probs_t = actor.get_action(states_t)
            pred_values_t = critic.get_value(states_t)

            # send actions to envs to rollout
            for j, conn in enumerate(connections):
                conn.send(actions_t[j].item())

            # envs send new states (n_envs, 4)
            trajectories = [conn.recv() for conn in connections]

            # place trajectory data into clean, ordered tensors
            next_state_t = torch.from_numpy(
                np.stack([result[1] for result in trajectories])
            ).float()

            rewards_t = torch.tensor(
                [result[2] for result in trajectories],
                dtype=torch.float32
            )

            dones_t = torch.tensor(
                [result[3] for result in trajectories],
                dtype=torch.int8
            )

            buffer.add(i, states_t, next_state_t, pred_values_t, rewards_t, actions_t, log_probs_t, dones_t)

            # Extract new observations
            states_t = torch.stack([
                torch.as_tensor(result[0], dtype=torch.float32)
                for result in trajectories
            ])



    return buffer


def run(actor, critic):
    connections, processes = build_processes()

    try:
        buffer = run_processes(connections, actor, critic)

    finally:
        for conn in connections:  # Shut down the workers
            conn.send(None)

        for process in processes:
            process.join()

        for conn in connections:
            conn.close()

    return buffer