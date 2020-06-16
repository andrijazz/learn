import torch
import gym
from lunar_lander.dqn_agent import DQNAgent, episode as dqn_episode_fn
from lunar_lander.random_agent import RandomAgent, episode as rnd_episode_fn


def main():
    env = gym.make("LunarLander-v2")

    agent = DQNAgent(env.action_space, env.observation_space)
    state_dict = torch.load('runs/Jun11_11-09-19_miles/model-300.pth')
    agent.q_net.load_state_dict(state_dict)
    dqn_episode_fn(env, agent, update=False)

    # agent = RandomAgent(env.action_space, env.observation_space)
    # rnd_episode_fn(env, agent)

    env.close()


if __name__ == "__main__":
    main()
