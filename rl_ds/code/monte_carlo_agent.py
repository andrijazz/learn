import numpy as np

from base_agent import BaseAgent
from utils import argmax


def episode(env, agent):
    history = list()

    state = env.start()
    action = agent.start(state)

    while True:
        reward, new_state, done = env.step(action)

        # collect reward
        agent.total_reward += reward
        history.append((state, action, reward))

        if done:
            agent.update(history)
            break

        action = agent.step(new_state, reward)
        state = new_state

    env.end()
    agent.end()
    # last reward is result of the game
    return reward


class MonteCarloAgent(BaseAgent):
    def __init__(self, agent_settings=None):
        super().__init__(agent_settings)
        self.actions = agent_settings['actions']

        # state is (dealer_sum, player_sum) tuple
        # state -> number of times state is visited
        self.n = np.zeros((11, 22))
        # state, action -> number of times action a is selected from state
        self.N = np.zeros((11, 22, len(self.actions)))
        # state -> state value
        self.Q = np.zeros((11, 22, len(self.actions)))

        # total reward over a single episode
        self.total_reward = 0

        # eps parameter
        self.n0 = 100.

        # gamma
        self.gamma = 1

    def start(self, state):
        """
        Starting the agent
        :return: Returns initial action
        """

        self.n[state] = self.n[state] + 1

        rand = np.random.uniform()
        eps = self.n0 / (self.n0 + self.n[state])
        # eps = 0.01
        if rand > eps:    # be greedy
            q_hit = self.Q[state][0]
            q_stick = self.Q[state][1]
            action = argmax([q_hit, q_stick])
        else:   # explore by taking random action
            action = np.random.choice(len(self.actions))

        self.N[state][action] = self.N[state][action] + 1
        return self.actions[action]

    def step(self, state, reward):
        """
        Agent step.

        :param state:
        :param reward:
        :return: next action
        """

        self.n[state] = self.n[state] + 1

        rand = np.random.uniform()
        eps = self.n0 / (self.n0 + self.n[state])
        # eps = 0.01
        if rand > eps:    # be greedy
            q_hit = self.Q[state][0]
            q_stick = self.Q[state][1]
            action = argmax([q_hit, q_stick])
        else:   # explore by taking random action
            action = np.random.choice(len(self.actions))

        self.N[state][action] = self.N[state][action] + 1
        return self.actions[action]

    def end(self):
        """
        Cleans up the agent properties
        :return:
        """

        # reset total reward over an episode
        self.total_reward = 0

    def update(self, history):
        # Monte Carlo computations
        G = 0
        for (state, action, reward) in reversed(history):
            G = self.gamma * G + reward
            a = self.actions.index(action)
            alpha_t = 1. / self.N[state[0], state[1], a]
            self.Q[state[0], state[1], a] = self.Q[state[0], state[1], a] + alpha_t * (G - self.Q[state[0], state[1], a])

    def print_Q(self):
        # print action value function
        for i in range(self.Q.shape[0]):
            for j in range(self.Q.shape[1]):
                for k in range(self.Q.shape[2]):
                    print('({}, {}, {}) = {}'.format(i, j, self.actions[k], self.Q[i, j, k]))
