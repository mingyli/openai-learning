import gym
from gym import wrappers
import numpy as np

env = gym.make('FrozenLake-v0')
env = wrappers.Monitor(env, './experiments/frozenlake-v0')

Q = np.zeros([env.observation_space.n, env.action_space.n])
#0 left, 1 down, 2 right, 3 up

lr = 0.8
gamma = 0.9
num_episodes = 2000

def train():
    rewards = []
    for i in range(num_episodes):
        obs0 = env.reset()
        ep_reward = 0
        for t in range(100):
            # env.render()
            action = np.argmax(Q[obs0,:] + np.random.randn(1, env.action_space.n) * 1./(i+1))
            # action = env.action_space.sample();
            obs1, reward, done, info = env.step(action)
            Q[obs0, action] = Q[obs0, action] + lr*(reward+gamma*np.max(Q[obs1,:]) - Q[obs0, action])
            ep_reward += reward
            obs0 = obs1
            if done:
                break
        rewards.append(ep_reward)

    print("Score over time: " + str(sum(rewards) / num_episodes))
    print("Final Q-Table Values")
    print(Q)

def simulate():
    obs0 = env.reset()
    for t in range(100):
        env.render()
        action = np.argmax(Q[obs0,:])
        obs1, reward, done, info = env.step(action)
        obs0 = obs1
        if done:
            env.render()
            break
    print(obs0, reward)

if __name__ == '__main__':
    train()
    simulate()
