from operator import add

import numpy as np

from agent import BaseAgent
from environment import BaseEnvironment
from rl_glue import RLGlue
from manager import Manager


class CliffWalkEnvironment(BaseEnvironment):
    def __init__(self):
        super(BaseEnvironment, self).__init__()

    def state_to_idx(self, state):
        h = state[0]
        w = state[1]
        idx = h * self.width + w
        return idx

    def idx_to_state(self, idx):
        pass

    def env_init(self, env_info=None):
        self.height = env_info['grid_height']
        self.width = env_info['grid_width']
        self.grid = np.zeros((self.height, self.width))
        self.states = self.grid.flatten()
        self.start = (self.height - 1, 0)
        self.term = (self.height - 1, self.width - 1)
        self.cliff = [(self.height - 1, i) for i in range(1, (self.width - 1))]

    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        # keep track of last state
        self.last_state = self.start

        reward = 0
        state = self.state_to_idx(self.last_state)
        done = False
        self.reward_state_term = (reward, state, done)
        return state

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """
        new_state = self.last_state
        if action == 1 and self.last_state[1] - 1 >= 0:     # left = 1
            new_state = tuple(map(add, self.last_state, (0, -1)))
        elif action == 3 and self.last_state[1] + 1 <= self.width - 1:  # right = 3
            new_state = tuple(map(add, self.last_state, (0, 1)))
        elif action == 0 and self.last_state[0] - 1 >= 0:   # up = 0
            new_state = tuple(map(add, self.last_state, (-1, 0)))
        elif action == 2 and self.last_state[0] + 1 <= self.height - 1:     # down = 2
            new_state = tuple(map(add, self.last_state, (1, 0)))

        reward = -1
        done = False
        if new_state in self.cliff:
            reward = -100
            new_state = self.start
        elif new_state == self.term:
            done = True

        self.reward_state_term = (reward, self.state_to_idx(new_state), done)
        self.last_state = new_state

        return self.reward_state_term

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        self.last_state = self.start

    def env_message(self, message):
        """A message asking the environment for information

        Args:
            message: the message passed to the environment

        Returns:
            the response (or answer) to the message
        """
        pass


class TDAgent(BaseAgent):
    def __init__(self):
        super(BaseAgent, self).__init__()
        self.num_actions = 4
        # self.seed = None
        self.last_action = None
        self.values = None

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""

        # Create a random number generator with the provided seed to seed the agent for reproducibility.
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))

        # Policy will be given, recall that the goal is to accurately estimate its corresponding value function.
        self.policy = agent_info.get("policy")
        # Discount factor (gamma) to use in the updates.
        self.discount = agent_info.get("discount")
        # The learning rate or step size parameter (alpha) to use in updates.
        self.step_size = agent_info.get("step_size")

        # Initialize an array of zeros that will hold the values.
        # Recall that the policy can be represented as a (# States, # Actions) array. With the
        # assumption that this is the case, we can use the first dimension of the policy to
        # initialize the array for values.
        self.values = np.zeros((self.policy.shape[0],))

    def agent_start(self, state):
        """The first method called when the episode starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the environment's env_start function.
        Returns:
            The first action the agent takes.
        """
        # The policy can be represented as a (# States, # Actions) array. So, we can use
        # the second dimension here when choosing an action.
        action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])
        self.last_state = state
        return action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step after the last action, i.e., where the agent ended up after the
                last action
        Returns:
            The action the agent is taking.
        """
        # Hint: We should perform an update with the last state given that we now have the reward and
        # next state. We break this into two steps. Recall for example that the Monte-Carlo update
        # had the form: V[S_t] = V[S_t] + alpha * (target - V[S_t]), where the target was the return, G_t.
        target = reward + self.discount * self.values[state]
        self.values[self.last_state] = self.values[self.last_state] + self.step_size * (
                    target - self.values[self.last_state])

        # Having updated the value for the last state, we now act based on the current
        # state, and set the last state to be current one as we will next be making an
        # update with it when agent_step is called next once the action we return from this function
        # is executed in the environment.
        action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])
        self.last_state = state
        return action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the terminal state.
        """
        # Hint: Here too, we should perform an update with the last state given that we now have the
        # reward. Note that in this case, the action led to termination. Once more, we break this into
        # two steps, computing the target and the update itself that uses the target and the
        # current value estimate for the state whose value we are updating.
        target = reward
        self.values[self.last_state] = self.values[self.last_state] + self.step_size * (
                    target - self.values[self.last_state])

    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """
        if message == "get_values":
            return self.values
        else:
            raise Exception("TDAgent.agent_message(): Message not understood!")

    def agent_cleanup(self):
        """Cleanup done after the agent ends."""
        self.last_state = None


def test_state():
    env = CliffWalkEnvironment()
    env.env_init({'grid_height': 4, 'grid_width': 12})
    coords_to_test = [(0, 0), (0, 11), (1, 5), (3, 0), (3, 9), (3, 11)]
    true_states = [0, 11, 17, 36, 45, 47]
    output_states = [env.state_to_idx(coords) for coords in coords_to_test]
    assert(output_states == true_states)


def test_action_up():
    env = CliffWalkEnvironment()
    env.env_init({'grid_height': 4, 'grid_width': 12})
    env.last_state = (0, 0)
    env.env_step(0)
    assert (env.last_state == (0, 0))

    env.last_state = (1, 0)
    env.env_step(0)
    assert (env.last_state == (0, 0))


def test_td_updates():
    # The following test checks that the TD check works for a case where the transition
    # garners reward -1 and does not lead to a terminal state. This is in a simple two state setting
    # where there is only one action. The first state's current value estimate is 0 while the second is 1.
    # Note the discount and step size if you are debugging this test.
    agent = TDAgent()
    policy_list = np.array([[1.], [1.]])
    agent.agent_init({"policy": np.array(policy_list), "discount": 0.99, "step_size": 0.1})
    agent.values = np.array([0., 1.])
    agent.agent_start(0)
    reward = -1
    next_state = 1
    agent.agent_step(reward, next_state)
    assert (np.isclose(agent.values[0], -0.001) and np.isclose(agent.values[1], 1.))

    # The following test checks that the TD check works for a case where the transition
    # garners reward -100 and lead to a terminal state. This is in a simple one state setting
    # where there is only one action. The state's current value estimate is 0.
    # Note the discount and step size if you are debugging this test.
    agent = TDAgent()
    policytrace_list = np.array([[1.]])
    agent.agent_init({"policy": np.array(policy_list), "discount": 0.99, "step_size": 0.1})
    agent.values = np.array([0.])
    agent.agent_start(0)
    reward = -100
    next_state = 0
    agent.agent_end(reward)
    assert (np.isclose(agent.values[0], -10))


def run_experiment(env_info, agent_info,
                   num_episodes=5000,
                   experiment_name=None,
                   plot_freq=100,
                   true_values_file=None,
                   value_error_threshold=1e-8):
    env = CliffWalkEnvironment
    agent = TDAgent
    rl_glue = RLGlue(env, agent)

    rl_glue.rl_init(agent_info, env_info)

    manager = Manager(env_info, agent_info, true_values_file=true_values_file, experiment_name=experiment_name)
    for episode in range(1, num_episodes + 1):
        rl_glue.rl_episode(0)  # no step limit
        if episode % plot_freq == 0:
            values = rl_glue.agent.agent_message("get_values")
            manager.visualize(values, episode)

    values = rl_glue.agent.agent_message("get_values")
    if true_values_file is not None:
        # Grading: The Manager will check that the values computed using your TD agent match
        # the true values (within some small allowance) across the states. In addition, it also
        # checks whether the root mean squared value error is close to 0.
        manager.run_tests(values, value_error_threshold)

    return values


env_info = {"grid_height": 4, "grid_width": 12}
agent_info = {"discount": 1, "step_size": 0.01}

# The Optimal Policy that strides just along the cliff
policy = np.ones(shape=(env_info['grid_width'] * env_info['grid_height'], 4)) * 0.25
policy[36] = [1, 0, 0, 0]
for i in range(24, 35):
    policy[i] = [0, 0, 0, 1]
policy[35] = [0, 0, 1, 0]

agent_info.update({"policy": policy})

true_values_file = "optimal_policy_value_fn.npy"
run_experiment(env_info, agent_info, num_episodes=5000, experiment_name="Policy Evaluation on Optimal Policy",
               plot_freq=500, true_values_file=true_values_file)
