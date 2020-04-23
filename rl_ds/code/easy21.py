import numpy as np

from easy21_environment import Easy21Environment
from easy21_mc_agent import Easy21MonteCarloAgent
from greedy_agent import GreedyAgent
from random_agent import RandomAgent
from human_agent import HumanAgent


def episode(env, agent, train=True):
    history = list()

    state = env.start()
    # print("Start state - Dealer = {}, Player = {}".format(state[0], state[1]))
    action = agent.start(state)
    # print(action)

    while True:
        reward, new_state, done = env.step(action)

        # collect reward
        agent.G += reward
        history.append((state, action, agent.G))

        if done:
            break

        action = agent.step(new_state, reward)
        state = new_state
        # print(action)

    if isinstance(agent, Easy21MonteCarloAgent) and train:
        agent.monte_carlo_update(history)

    env.end()
    agent.end()

    return reward


def play(env, agent, num_games=10000):
    won = 0
    lose = 0
    draw = 0
    for i in range(num_games):
        result = episode(env, agent, train=False)
        if result > 0:
            won += 1
        elif result < 0:
            lose += 1
        else:
            draw += 1
    print('{} - Won {} / Draw {} / Lose {}'.format(agent.__class__.__name__, won, draw, lose))


def test(agent):
    # player should hit in every state less then 11 for sure
    for i in range(12):
        v = agent.Q[:, i, 0] > agent.Q[:, i, 1]
        assert np.all(v)

    # player should stick every time he has 21
    v = agent.Q[:, 21, 0] < agent.Q[:, 21, 1]
    assert np.all(v)


def main():
    # train
    num_episodes = 1000000
    env = Easy21Environment()
    agent = Easy21MonteCarloAgent({'actions': ['hit', 'stick']})

    for i in range(num_episodes):
        episode(env, agent)

    # test
    # test(agent)

    random_agent = RandomAgent({'actions': ['hit', 'stick']})
    greedy_agent = GreedyAgent({'actions': ['hit', 'stick']})
    human_agent = HumanAgent({'actions': ['hit', 'stick']})
    greedy_agent.Q = agent.Q
    play(env, random_agent)
    play(env, human_agent)
    play(env, agent)
    play(env, greedy_agent)


if __name__ == "__main__":
    main()
