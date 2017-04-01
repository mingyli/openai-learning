import gym
from gym import wrappers
from DDPGLearner import DDPGLearner

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    env = wrappers.Monitor(env, './experiments/pendulum-ddpg-v0', force=True)
    max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    learner = DDPGLearner(state_dim, action_dim, action_bound)

    total = 0
    for episode in range(1000):
        obs0 = env.reset()
        ep_reward = 0
        for t in range(max_steps):
            if episode % 25 == 0:
                env.render()
                pass
            action = learner.act(obs0)
            obs1, reward, done, info = env.step(action)
            doneint = 1.0 if done else 0.
            learner.replay_buffer.add(obs0, action, reward, t, doneint, obs1)
            learner.train()

            obs0 = obs1
            ep_reward += reward
            if done:
                break
        total += ep_reward
        print("Episode {0:8d}: {1:4d} timesteps, {2:4f} ".format(episode, t, ep_reward))