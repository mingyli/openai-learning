import gym
from gym import wrappers
import numpy as np

env = gym.make('ReversedAddition-v0')

Q = np.random.rand(env.observation_space.n, env.action_space.spaces[0].n,
                   env.action_space.spaces[1].n, env.action_space.spaces[2].n)
Q = Q / 10 ** 5

def train(alpha=0.8, gamma=0.9, num_episodes=9999, epsilon=0.5):
    rewards = [0 for _ in range(100)]
    for i in range(num_episodes):
        obs0 = env.reset()
        ep_reward = 0
        for t in range(200):
            # env.render()
            if np.random.random_sample() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[obs0,...])
                action = np.unravel_index(action, Q.shape)
                action = action[1:]
            obs1, reward, done, info = env.step(action)
            Q[obs0,action[0],action[1],action[2]] += \
                   alpha*(reward + gamma*np.max(Q[obs1,...]) - Q[obs0,action[0],action[1],action[2]])
            ep_reward += reward
            obs0 = obs1
            if reward > 0:
                epsilon -= 9 * 10 ** -5
            if done:
                break
        rewards.append(ep_reward)
        rewards.pop(0)
    print("moving average: " + str(sum(rewards) / 100))
    # print(Q)

def simulate():
    obs0 = env.reset()
    for t in range(200):
        env.render()
        action = np.argmax(Q[obs0,...])
        action = np.unravel_index(action, Q.shape)
        action = action[1:]
        obs1, reward, done, info = env.step(action)
        obs0 = obs1
        if done:
            env.render()
            break

if __name__ == '__main__':
    train()
    # simulate()
