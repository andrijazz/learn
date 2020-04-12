# Gridworld City, a thriving metropolis with a booming technology industry, has recently experienced an influx of
# grid-loving software engineers. Unfortunately, the city's street parking system, which charges a fixed rate,
# is struggling to keep up with the increased demand. To address this, the city council has decided to modify the
# pricing scheme to better promote social welfare. In general, the city considers social welfare higher when more
# parking is being used, the exception being that the city prefers that at least one spot is left unoccupied (so that
# it is available in case someone really needs it). The city council has created a Markov decision process (MDP) to
# model the demand for parking with a reward function that reflects its preferences. Now the city has hired you — an
# expert in dynamic programming — to help determine an optimal policy.

from parking_world_utils import plot, ParkingWorld
import numpy as np

num_spaces = 3  # 0, 1, 2, 3 can be occupied
num_prices = 3
env = ParkingWorld(num_spaces, num_prices)

# init value function
V = np.zeros(num_spaces + 1)

# init policy
# state, action -> probability
# rows sum to 1
pi = np.ones((num_spaces + 1, num_prices)) / num_prices

V[0] = 10
pi[0] = np.array([0.75, 0.21, 0.04])
plot(V, pi)


def bellman_update(env, V, pi, s, gamma):
    """Mutate ``V`` according to the Bellman update equation."""
    actions = pi[s]
    G = [0] * len(actions)
    for action in env.A:
        transitions = env.transitions(s, action)

        for s_, (r, p) in enumerate(transitions):
            G[action] += p * (r + gamma * V[s_])

    V[s] = np.sum(G * actions)


def evaluate_policy(env, V, pi, gamma, theta):
    while True:
        delta = 0
        for s in env.S:
            v = V[s]
            bellman_update(env, V, pi, s, gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V


V = evaluate_policy(env, V, pi, 0.9, 0.1)
plot(V, pi)
