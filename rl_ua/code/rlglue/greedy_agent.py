import numpy as np
import utils

from agent import BaseAgent


class GreedyAgent(BaseAgent):
    def __init__(self):
        super(BaseAgent, self).__init__()
        self.actions = None
        self.q_values = None
        self.seed = None
        self.last_action = None
        self.arm_count = None
        self.step_size = None

    def agent_init(self, agent_info=None):
        """Setup for the agent called when the experiment first starts."""
        self.actions = agent_info['actions']
        self.q_values = np.zeros(len(self.actions))
        self.arm_count = np.zeros(len(self.actions))
        self.step_size = 0.1

    def agent_start(self, observation):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the environment's evn_start function.
        Returns:
            The first action the agent takes.
        """

        choosen_action = utils.argmax(self.q_values)
        self.last_action = choosen_action
        return choosen_action

    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        self.arm_count[self.last_action] += 1
        step_size = 1 / self.arm_count[self.last_action]
        self.q_values[self.last_action] = self.q_values[self.last_action] + step_size * (
                reward - self.q_values[self.last_action])
        choosen_action = utils.argmax(self.q_values)
        self.last_action = choosen_action
        return choosen_action
