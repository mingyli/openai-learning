import gym
from gym import wrappers
import numpy as np

env = gym.make('FrozenLake-v0')
# env = wrappers.Monitor(env, './experiments/frozenlake-v0', force=True)

Q = np.random.rand(env.observation_space.n, env.action_space.n)
Q = Q / 10 ** 5
#0 left, 1 down, 2 right, 3 up

alpha = 0.3
gamma = .9
num_episodes = 4000

def train():
    epsilon = .5
    rewards = [0 for _ in range(100)] #moving average over 100 episodes
    for i in range(num_episodes):
        obs0 = env.reset()
        ep_reward = 0
        for t in range(200):
            # env.render()
            if np.random.random_sample() < epsilon:
                action = np.random.randint(0, env.action_space.n)
            else:
                action = np.argmax(Q[obs0,:])
            obs1, reward, done, info = env.step(action)
            Q[obs0, action] += alpha*(reward + gamma*np.max(Q[obs1,:]) - Q[obs0, action])
            #penalize if fall into hole
            if done and reward == 0:
                Q[obs0, action] -= 1 * 10 ** -2
            ep_reward += reward
            obs0 = obs1
            epsilon -= 10 ** -4
            if done:
                break
        rewards.append(ep_reward)
        rewards.pop(0)

    print("Average reward over 100 episodes: " + str(sum(rewards) / len(rewards)))
    print("Final Q-Table Values")
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
    print(obs0, reward)

if __name__ == '__main__':
    train()
    # simulate()
