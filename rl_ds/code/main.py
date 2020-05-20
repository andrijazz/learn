from easy21_environment import Easy21Environment
from monte_carlo_agent import MonteCarloAgent, monte_carlo_episode
from random_agent import RandomAgent, episode
from human_agent import HumanAgent


def training(env, agent, episode_fn, num_episodes=50000):
    for i in range(num_episodes):
        episode_fn(env, agent)


def testing(env, agent, episode_fn, num_episodes=10000):
    won = 0
    lose = 0
    draw = 0
    for i in range(num_episodes):
        result = episode_fn(env, agent)
        if result > 0:
            won += 1
        elif result < 0:
            lose += 1
        else:
            draw += 1
    print('{} - Won {} / Draw {} / Lose {}'.format(agent.__class__.__name__, won, draw, lose))


def main():
    env = Easy21Environment()

    # training
    mc_agent = MonteCarloAgent({'actions': ['hit', 'stick']})
    training(env, mc_agent, monte_carlo_episode)

    # test agents
    random_agent = RandomAgent({'actions': ['hit', 'stick']})
    human_agent = HumanAgent({'actions': ['hit', 'stick']})

    # testing
    testing(env, mc_agent, monte_carlo_episode)
    testing(env, random_agent, episode)
    testing(env, human_agent, episode)


if __name__ == "__main__":
    main()
