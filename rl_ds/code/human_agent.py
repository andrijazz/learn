import numpy as np

from base_agent import BaseAgent


class HumanAgent(BaseAgent):
    def __init__(self, agent_settings=None):
        super().__init__(agent_settings)
        self.actions = agent_settings['actions']
        self.Q = None
        self.total_reward = 0

    def start(self, state):
        """
        Starting the agent
        :return: Returns initial action
        """
        if state[1] < 12:
            return self.actions[0]

        if state[1] > 18:
            return self.actions[1]

        action = np.random.choice(self.actions)
        return action

    def step(self, state, reward):
        """
        Agent step.

        :param state:
        :param reward:
        :return: next action
        """
        # if state[1] <= 11:
        #     return self.actions[0]

        if state[1] >= 20:
            return self.actions[1]

        action = np.random.choice(self.actions)
        return action

    def end(self):
        """
        Cleans up the agent properties
        :return:
        """
        self.total_reward = 0
