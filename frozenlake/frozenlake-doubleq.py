import gym
from gym import wrappers
import numpy as np

env = gym.make('FrozenLake-v0')
# env = wrappers.Monitor(env, './experiments/frozenlake-doubleq', force=True)

Q = (np.zeros([env.observation_space.n, env.action_space.n]),
     np.zeros([env.observation_space.n, env.action_space.n]))
#0 left, 1 down, 2 right, 3 up

alpha = 0.8
gamma = 0.99
num_episodes = 3000

def train():
    rewards = [0 for _ in range(100)] #moving average over 100 episodes
    alt = 1
    for i in range(num_episodes):
        obs0 = env.reset()
        ep_reward = 0
        for t in range(200):
            # env.render()
            action = np.argmax(Q[alt][obs0,:] + np.random.randn(1, env.action_space.n) * 1./(i+1))
            # action = env.action_space.sample();
            obs1, reward, done, info = env.step(action)
            Q[alt][obs0, action] = Q[alt][obs0, action] + alpha*(reward + gamma*np.max(Q[1-alt][obs1,:]) - Q[alt][obs0, action])
            ep_reward += reward
            obs0 = obs1
            alt = 1 - alt
            if done:
                break
        rewards.append(ep_reward)
        rewards.pop(0)
    print("Average reward over 100 episodes: " + str(sum(rewards) / len(rewards)))
    print("Final Q-Table Values")
    print(Q[alt])

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
