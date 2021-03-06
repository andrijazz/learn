import numpy as np

from base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, agent_settings=None):
        super().__init__(agent_settings)
        self.actions = agent_settings['actions']
        self.total_reward = 0

    def start(self, state):
        """
        Starting the agent
        :return: Returns initial action
        """
        action = np.random.choice(self.actions)
        return action

    def step(self, state, reward):
        """
        Agent step.

        :param state:
        :param reward:
        :return: next action
        """
        action = np.random.choice(self.actions)
        return action

    def end(self):
        """
        Cleans up the agent properties
        :return:
        """
        self.total_reward = 0
