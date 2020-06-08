import gym
from lunar_lander.dqn_agent import DQNAgent, episode as dqn_episode_fn


def main():
    env = gym.make("LunarLander-v2")
    agent = DQNAgent(env.action_space)
    dqn_episode_fn(env, agent)
    env.close()


if __name__ == "__main__":
    main()
