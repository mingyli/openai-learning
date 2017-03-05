import gym
from gym import wrappers
import numpy as np

env = gym.make('Taxi-v2')
# env = wrappers.Monitor(env, './experiments/taxi-v2', force=True)

Q = np.random.rand(env.observation_space.n, env.action_space.n)
Q = Q / 10 ** 7

def train(alpha=0.2, gamma=0.95, num_episodes=4000, epsilon=0.6):
    rewards = [0 for _ in range(100)]
    for i in range(num_episodes):
        obs0 = env.reset()
        ep_reward = 0
        for t in range(200):
            # env.render()
            if np.random.random_sample() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[obs0,:])
            obs1, reward, done, info = env.step(action)
            Q[obs0,action] += alpha*(reward + gamma*np.max(Q[obs1,:]) - Q[obs0,action])
            ep_reward += reward
            obs0 = obs1
            epsilon -= 3 * 10 ** -5
            if done:
                break
        rewards.append(ep_reward)
        rewards.pop(0)
    print("moving average: " + str(sum(rewards) / len(rewards)))
    print(Q)

def simulate():
    obs0 = env.reset()
    for t in range(200):
        env.render()
        action = np.argmax(Q[obs0,:])
        obs1, reward, done, info = env.step(action)
        obs0 = obs1
        if done:
            env.render()
            break

if __name__ == '__main__':
    train()
    # simulate()
