from easy21_environment import Easy21Environment
from greedy_agent import GreedyAgent
from human_agent import HumanAgent
from monte_carlo_agent import MonteCarloAgent, episode as monte_carlo_episode
from random_agent import RandomAgent
from sarsa_agent import SarsaAgent, episode as sarsa_episode
from qlearning_agent import QLearningAgent


def training(env, agent, episode_fn, num_episodes=50000):
    for i in range(num_episodes):
        episode_fn(env, agent)


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


def testing(env, agent, episode_fn, num_episodes=10000, name=None):
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

    if name is None:
        name = agent.__class__.__name__
    print('{} - Won {} / Draw {} / Lose {}'.format(name, won, draw, lose))


def main():
    env = Easy21Environment()

    # training
    mc_agent = MonteCarloAgent(agent_settings={'actions': ['hit', 'stick']})
    training(env, mc_agent, monte_carlo_episode, num_episodes=100000)

    sarsa_agent = SarsaAgent(agent_settings={'actions': ['hit', 'stick'], 'lambd': 0.1})
    training(env, sarsa_agent, sarsa_episode, num_episodes=100000)

    q_agent = QLearningAgent(agent_settings={'actions': ['hit', 'stick'], 'lambd': 0.1})
    training(env, q_agent, sarsa_episode, num_episodes=100000)

    # test agents
    greedy_mc_agent = GreedyAgent(mc_agent.Q, agent_settings={'actions': ['hit', 'stick']})
    greedy_sarsa_agent = GreedyAgent(sarsa_agent.Q, agent_settings={'actions': ['hit', 'stick']})
    greedy_q_agent = GreedyAgent(q_agent.Q, agent_settings={'actions': ['hit', 'stick']})
    random_agent = RandomAgent(agent_settings={'actions': ['hit', 'stick']})
    human_agent = HumanAgent(agent_settings={'actions': ['hit', 'stick']})

    # testing
    testing(env, greedy_mc_agent, episode, name="MonteCarlo")
    testing(env, greedy_sarsa_agent, episode, name="Sarsa")
    testing(env, greedy_q_agent, episode, name="QLearning")
    testing(env, random_agent, episode)
    testing(env, human_agent, episode)


if __name__ == "__main__":
    main()
