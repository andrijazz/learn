import time

import numpy as np


def episode(env, agent):
    pass


def train(env, agent):
    # collect experience
    num_episodes = 10000
    experience = ReplayBuffer(1000000, 5, 1)
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            env.render()
            action = env.action_space.sample()
            next_state, reward, next_done, info = env.step(action)
            experience.append(state, action, reward, done, next_state)
            state = next_state
            done = next_done


class ReplayBuffer:
    def __init__(self, size, minibatch_size, seed):
        """
        Args:
            size (integer): The size of the replay buffer.
            minibatch_size (integer): The sample size.
            seed (integer): The seed for the random number generator.
        """
        self.buffer = []
        self.minibatch_size = minibatch_size
        self.rand_generator = np.random.RandomState(seed)
        self.max_size = size

    def append(self, state, action, reward, terminal, next_state):
        """
        Args:
            state (Numpy array): The state.
            action (integer): The action.
            reward (float): The reward.
            terminal (integer): 1 if the next state is a terminal state and 0 otherwise.
            next_state (Numpy array): The next state.
        """
        if len(self.buffer) == self.max_size:
            del self.buffer[0]
        self.buffer.append([state, action, reward, terminal, next_state])

    def sample(self):
        """
        Returns:
            A list of transition tuples including state, action, reward, terinal, and next_state
        """
        idxs = self.rand_generator.choice(np.arange(len(self.buffer)), size=self.minibatch_size)
        return [self.buffer[idx] for idx in idxs]

    def size(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, actions):
        self.actions = actions

    def step(self, observation, reward, done):
        return 0
