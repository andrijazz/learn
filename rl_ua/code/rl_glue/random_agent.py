import numpy as np

from rl_glue.agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self):
        super(BaseAgent, self).__init__()
        self.actions = None
        # self.seed = None
        self.last_action = None

    def agent_init(self, agent_info=None):
        """Setup for the agent called when the experiment first starts."""
        self.actions = agent_info['actions']
        # self.seed = agent_info['random_seed']

    def agent_start(self, observation):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        # np.random.seed(self.seed)

        choosen_action = np.random.choice(len(self.actions))
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
        choosen_action = np.random.choice(len(self.actions))
        self.last_action = choosen_action
        return choosen_action
