from __future__ import print_function

from abc import abstractmethod


class BaseAgent:
    def __init__(self, agent_settings=None):
        self.settings = agent_settings

    @abstractmethod
    def start(self, state):
        """
        Starting the agent
        :return: Returns initial action
        """

    @abstractmethod
    def step(self, state, reward):
        """
        Agent step.

        :param state:
        :param reward:
        :return: next action
        """

    @abstractmethod
    def end(self):
        """
        Cleans up the agent properties
        :return:
        """
