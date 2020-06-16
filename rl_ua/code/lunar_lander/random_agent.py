from torch.utils.tensorboard import SummaryWriter


def episode(env, agent):
    writer = SummaryWriter()

    num_episodes = 1000
    for i in range(num_episodes):
        state = env.reset()
        total_reward = 0
        while True:
            env.render()
            action = agent.step()
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state

            if done:
                break

        writer.add_scalar('reward', total_reward, global_step=i)


class RandomAgent:
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.num_actions = action_space.n

    def step(self):
        return self.action_space.sample()
