import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import main_agent
import ten_arm_env
from rl_glue import RLGlue


# First we are going to implement the argmax function, which takes in a list of action values and returns an action with
# the highest value. Why are we implementing our own instead of using the argmax function that numpy uses? Numpy's
# argmax function returns the first instance of the highest value. We do not want that to happen as it biases the agent
# to choose a specific action in the case of ties. Instead we want to break ties between the highest values randomly.
# So we are going to implement our own argmax function. You may want to look at np.random.choice to randomly select
# from a list of values.


def argmax(q_values):
    top = float("-inf")
    ties = []
    for i in range(len(q_values)):
        if q_values[i] > top:
            top = q_values[i]
            ties = [i]
        elif q_values[i] == top:
            ties.append(i)

    return np.random.choice(ties)


def test_argmax():
    test_array = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    assert argmax(test_array) == 8, "Check your argmax implementation returns the index of the largest value"

    test_array = [1, 0, 0, 1]
    total = 0
    for i in range(100):
        total += argmax(test_array)

    assert total > 0, "Make sure your argmax implementation randomly choooses among the largest values. Make sure you "\
                      "are not setting a random seed (do not use np.random.seed)"
    assert total != 300, "Make sure your argmax implementation randomly choooses among the largest values."


class GreedyAgent(main_agent.Agent):
    def agent_step(self, reward, observation):
        """
        Takes one step for the agent. It takes in a reward and observation and
        returns the action the agent chooses at that time step.

        Arguments:
        reward -- float, the reward the agent received from the environment after taking the last action.
        observation -- float, the observed state the agent is in. Do not worry about this for this assignment
        as you will not use it until future lessons.
        Returns:
        current_action -- int, the action chosen by the agent at the current time step.
        """
        ### Useful Class Variables ###
        # self.q_values : An array with the agent’s value estimates for each action.
        # self.arm_count : An array with a count of the number of times each arm has been pulled.
        # self.last_action : The action that the agent took on the previous time step.
        #######################

        # current action = ? # Use the argmax function you created above
        # (~2 lines)
        ### START CODE HERE ###
        self.last_action = int(self.last_action)
        ### END CODE HERE ###

        # Update action values. Hint: Look at the algorithm in section 2.4 of the textbook.
        # Increment the counter in self.arm_count for the action from the previous time step
        # Update the step size using self.arm_count
        # Update self.q_values for the action from the previous time step
        # (~3-5 lines)
        ### START CODE HERE ###
        self.arm_count[self.last_action] += 1
        step_size = 1 / self.arm_count[self.last_action]
        self.q_values[self.last_action] = self.q_values[self.last_action] + step_size * (
                reward - self.q_values[self.last_action])

        current_action = argmax(self.q_values)
        ### END CODE HERE ###

        self.last_action = current_action

        return current_action


def run_ten_arm_experiment():
    num_runs = 200  # The number of times we run the experiment
    num_steps = 1000  # The number of steps each experiment is run for
    env = ten_arm_env.Environment  # We the environment to use
    agent = GreedyAgent  # We choose what agent we want to use
    agent_info = {"num_actions": 10}  # Pass the agent the information it needs;
    # here it just needs the number of actions (number of arms).
    env_info = {}  # Pass the environment the information it needs; in this case, it is nothing.

    all_averages = []

    for i in tqdm(range(num_runs)):  # tqdm is what creates the progress bar below once the code is run
        rl_glue = RLGlue(env, agent)  # Creates a new RLGlue experiment with the env and agent we chose above
        rl_glue.rl_init(agent_info, env_info)  # Pass RLGlue what it needs to initialize the agent and environment
        rl_glue.rl_start()  # Start the experiment

        scores = [0]
        averages = []

        for i in range(num_steps):
            reward, _, action, _ = rl_glue.rl_step()  # The environment and agent take a step and return
            # the reward, and action taken.
            scores.append(scores[-1] + reward)
            averages.append(scores[-1] / (i + 1))
        all_averages.append(averages)

    plt.figure(figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')
    plt.plot([1.55 for _ in range(num_steps)], linestyle="--")
    plt.plot(np.mean(all_averages, axis=0))
    plt.legend(["Best Possible", "Greedy"])
    plt.title("Average Reward of Greedy Agent")
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.show()
    greedy_scores = np.mean(all_averages, axis=0)
    print("greedy_scores: {}".format(greedy_scores))


run_ten_arm_experiment()

