import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from lunar_lander.dqn import DQN
from lunar_lander.replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

# Lunar lander state description
# https://github.com/openai/gym/blob/074bc269b5405c22e95856920e43a067a14302b1/gym/envs/box2d/lunar_lander.py#L304-L315


def argmax(q_values):
    top = float("-inf")
    ties = []

    for i in range(len(q_values)):
        if q_values[i] > top:
            top = q_values[i]
            ties = []

        if q_values[i] == top:
            ties.append(i)

    return np.random.choice(ties)


def episode(env, agent, update=False):
    writer = SummaryWriter()

    num_episodes = 1000
    experience = ReplayBuffer(10000, 32, 1)
    for i in range(num_episodes):
        state = env.reset()
        total_reward = 0
        while True:
            # env.render()
            action = agent.step(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            experience.append(state, action, reward, next_state, done)
            state = next_state

            # update
            if update:
                samples = experience.sample()
                agent.update(samples)

            if done:
                break

        writer.add_scalar('reward', total_reward, global_step=i)


class DQNAgent:
    def __init__(self, action_space, observation_space):
        self.num_actions = action_space.n
        self.eps_max = 1
        self.eps_min = 0.1
        self.decay_interval = 300
        self.increment = (self.eps_max - self.eps_min) / self.decay_interval
        self.eps = self.eps_max
        self.n = 0
        self.gamma = 1

        # pytorch vars
        self.q_net = DQN(8, self.num_actions)

        # self.device = 0
        # self.q_net = self.q_net.to(self.device)
        params_to_update = self.q_net.parameters()
        self.optimizer = optim.Adam(params_to_update, lr=0.01)
        self.criterion = nn.MSELoss()

    def step(self, observation):
        self.eps = self.eps - self.n * self.increment
        rand = np.random.uniform()

        if rand > self.eps:  # act greedy
            # do the forward pass to compute q
            q = self.get_q_values(observation)
            action = argmax(q.to_numpy())
        else:  # explore by taking random action
            action = np.random.choice(self.num_actions)

        return action

    def get_q_values(self, s):
        self.q_net.eval()
        s_t = torch.from_numpy(s)
        q = self.q_net(s_t)
        return q

    def update(self, samples):
        self.q_net.train()
        self.optimizer.zero_grad()

        x = torch.empty(len(samples), dtype=torch.float)
        y = torch.empty(len(samples), dtype=torch.float)
        for j, [s, a, r, sp, terminal] in enumerate(samples):
            q_s = self.get_q_values(s)
            x[j] = q_s[a]

            if terminal:
                y[j] = r
                continue

            q_sp = self.get_q_values(sp)
            q_max, _ = torch.max(q_sp, dim=0)
            y[j] = r + self.gamma * q_max

        # print(self.q_net.fc1.weight)
        loss = self.criterion(x, y)
        loss.backward()
        self.optimizer.step()
        # print(self.q_net.fc1.weight)
