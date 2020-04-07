from rl_glue import RLGlue
from environment import BaseEnvironment
from random_agent import RandomAgent
from greedy_agent import GreedyAgent
import numpy as np
import tqdm
import matplotlib.pyplot as plt


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


def run(agent_class):
    env_init = {
        'random_seed': None
    }

    agent_init = {
        'actions': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }

    num_runs = 200
    num_steps = 1000
    avgs = []
    for k in tqdm.tqdm(range(num_runs)):
        env = TenArmEnviroment()
        env.env_init(env_init)
        agent = agent_class()
        agent.agent_init(agent_init)
        init_state = env.env_start()
        action = agent.agent_start(init_state)

        # rl_glue = RLGlue(TenArmEnviroment, agent_class)
        # rl_glue.rl_init(agent_init, env_init)
        # rl_glue.rl_start()

        scores = [0]
        averages = []

        for i in range(num_steps):
            # The environment and agent take a step and return
            # reward, new_state, action, done = rl_glue.rl_step()

            (reward, state, done) = env.env_step(action)
            action = agent.agent_step(reward, state)

            scores.append(scores[-1] + reward)
            averages.append(scores[-1] / (i + 1))
        avgs.append(averages)
    return avgs


greedy_avgs = run(GreedyAgent)
random_avgs = run(RandomAgent)

plt.figure(figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')
# plt.plot([1.55 for _ in range(num_steps)], linestyle="--")
plt.plot(np.mean(random_avgs, axis=0), color='blue')
plt.plot(np.mean(greedy_avgs, axis=0), color='orange')
plt.title("Average Reward of Agents")
plt.xlabel("Steps")
plt.ylabel("Average reward")
plt.show()
