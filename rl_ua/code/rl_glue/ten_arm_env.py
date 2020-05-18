import numpy as np
from rl_glue.environment import BaseEnvironment


class TenArmEnviroment(BaseEnvironment):
    def __init__(self):
        super(BaseEnvironment, self).__init__()
        self.arms = None
        self.seed = None

    def env_init(self, env_info=None):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """
        self.seed = env_info.get("random_seed", None)
        np.random.seed(self.seed)
        self.arms = np.random.randn(10)     # [np.random.normal(0.0, 1.0) for _ in range(10)]
        self.reward_obs_term = (0.0, 0, False)

    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        s0 = self.reward_obs_term[1]
        return s0

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """
        sigma = 1.0
        mu = self.arms[action]
        reward = sigma * np.random.randn() + mu
        self.reward_obs_term = (reward, 0, False)
        return self.reward_obs_term

    def env_cleanup(self):
        """Cleanup done after the environment ends"""

    def env_message(self, message):
        """A message asking the environment for information

        Args:
            message: the message passed to the environment

        Returns:
            the response (or answer) to the message
        """
