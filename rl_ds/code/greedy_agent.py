from base_agent import BaseAgent
from utils import argmax


class GreedyAgent(BaseAgent):
    def __init__(self, Q, agent_settings=None):
        super().__init__(agent_settings)
        self.actions = agent_settings['actions']
        self.Q = Q
        self.total_reward = 0

    def start(self, state):
        """
        Starting the agent
        :return: Returns initial action
        """
        q_hit = self.Q[state][0]
        q_stick = self.Q[state][1]
        action = argmax([q_hit, q_stick])
        return self.actions[action]

    def step(self, state, reward):
        """
        Agent step.

        :param state:
        :param reward:
        :return: next action
        """
        q_hit = self.Q[state][0]
        q_stick = self.Q[state][1]
        action = argmax([q_hit, q_stick])
        return self.actions[action]

    def end(self):
        """
        Cleans up the agent properties
        :return:
        """
        self.total_reward = 0
