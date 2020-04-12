from rl_glue import RLGlue
from random_agent import RandomAgent
from greedy_agent import GreedyAgent
from eps_greedy_agent import EpsGreedyAgent
from ten_arm_env import TenArmEnviroment
import numpy as np
import tqdm
import matplotlib.pyplot as plt


def run(agent_class):
    env_init = {
        'random_seed': None
    }

    agent_init = {
        'actions': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'eps': 0.1
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
eps_avgs = run(EpsGreedyAgent)

plt.figure(figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')
# plt.plot([1.55 for _ in range(num_steps)], linestyle="--")
plt.plot(np.mean(random_avgs, axis=0), color='blue')
plt.plot(np.mean(greedy_avgs, axis=0), color='orange')
plt.plot(np.mean(eps_avgs, axis=0), color='red')

plt.title("Average Reward of Agents")
plt.xlabel("Steps")
plt.ylabel("Average reward")
plt.show()
