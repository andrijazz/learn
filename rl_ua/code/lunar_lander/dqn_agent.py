import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from lunar_lander.dqn import DQN
from lunar_lander.replay_buffer import ReplayBuffer


# Lunar lander state description
# https://github.com/openai/gym/blob/074bc269b5405c22e95856920e43a067a14302b1/gym/envs/box2d/lunar_lander.py#L304-L315


writer = SummaryWriter()


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
    num_episodes = 300
    batch_size = 8
    replay_buffer_size = 50000
    num_replay_updates_per_step = 4
    experience = ReplayBuffer(replay_buffer_size, batch_size, 1)
    for i in range(num_episodes):
        state = env.reset()
        total_reward = 0
        step = 0
        agent.update_eps()
        while True:
            env.render()
            action = agent.step(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            experience.append(state, action, reward, next_state, done)
            state = next_state
            step += 1

            # update
            if update and experience.size() > batch_size:
                for _ in range(num_replay_updates_per_step):
                    samples = experience.sample()
                    agent.update(samples)

            if done:
                break

        writer.add_scalar('episode/reward', total_reward, global_step=i)
        writer.add_scalar('episode/steps', step, global_step=i)
        writer.add_scalar('agent/eps', agent.eps, global_step=i)
        writer.add_scalar('agent/loss', agent.last_loss, global_step=i)

    # save weights
    model_name = 'model-{}.pth'.format(num_episodes)
    checkpoint_file = os.path.join(writer.log_dir, model_name)
    torch.save(agent.q_net.state_dict(), checkpoint_file)


class DQNAgent:
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.num_actions = action_space.n
        self.eps_max = 1
        self.eps_min = 0.1
        self.eps_decay_in_episodes = 200
        self.increment = (self.eps_max - self.eps_min) / self.eps_decay_in_episodes
        self.eps = 0
        self.n = 0
        self.gamma = 0.99

        # pytorch vars
        self.q_net = DQN(8, self.num_actions)

        self.device = 0
        # self.q_net = self.q_net.to(self.device)
        params_to_update = self.q_net.parameters()
        self.optimizer = optim.Adam(params_to_update, lr=1e-3)
        self.criterion = nn.MSELoss()
        self.last_loss = 0

    def update_eps(self):
        if self.eps > self.eps_min:
            self.eps = self.eps - self.increment

    def step(self, observation):
        self.n += 1
        rand = np.random.uniform()

        if rand > self.eps:  # act greedy
            # do the forward pass to compute q
            q = self.get_q_values(observation)
            # q = q.detach().cpu().numpy()
            action = argmax(q.data.numpy())
        else:  # explore by taking random action
            action = np.random.choice(self.num_actions)

        return action

    def get_q_values(self, s):
        self.q_net.eval()
        s_t = torch.from_numpy(s)
        # s_t = s_t.to(self.device)
        q = self.q_net(s_t)
        return q

    def update(self, samples):
        self.q_net.train()
        self.optimizer.zero_grad()

        y_est = torch.empty(len(samples), dtype=torch.float, device=self.device)
        y = torch.empty(len(samples), dtype=torch.float, device=self.device)
        for j, [s, a, r, sp, terminal] in enumerate(samples):
            q_s = self.get_q_values(s)
            y_est[j] = q_s[a]

            if terminal:
                y[j] = r
                continue

            q_sp = self.get_q_values(sp)
            q_max, _ = torch.max(q_sp, dim=0)
            y[j] = r + self.gamma * q_max

        # x = x.to(self.device)
        # y = y.to(self.device)
        loss = self.criterion(y_est, y)
        loss.backward()
        self.optimizer.step()
        self.last_loss = loss.data

