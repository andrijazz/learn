from __future__ import print_function

from abc import abstractmethod


class BaseEnvironment:
    def __init__(self, env_settings=None):
        self.settings = env_settings

    @abstractmethod
    def start(self):
        """
        Starting environment
        :return: initial state
        """

    @abstractmethod
    def step(self, action):
        """
        Taking a env step.

        :param action:
        :return: (new state, reward, is_terminating) obtained by taking action
        """

    @abstractmethod
    def end(self):
        """
        Cleans up the env
        :return:
        """
