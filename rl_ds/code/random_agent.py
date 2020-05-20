import numpy as np

from base_agent import BaseAgent


def episode(env, agent):
    state = env.start()
    action = agent.start(state)
    while True:
        reward, new_state, done = env.step(action)

        # collect reward
        agent.total_reward += reward

        if done:
            break

        action = agent.step(new_state, reward)

    env.end()
    agent.end()

    # last reward is result of the game
    return reward


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
