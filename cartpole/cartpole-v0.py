import gym
from gym import wrappers
import numpy as np

def train(alpha=0.3, gamma=0.99, num_episodes=5001, epsilon=0.6):
    rewards = [0 for _ in range(100)]
    for i in range(num_episodes):
        obs0 = env.reset()
        ep_reward = 0
        for t in range(600):
            # env.render()
            disc = d.discretize(obs0)
            if np.random.random_sample() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[disc])
            obs1, reward, done, info = env.step(action)
            Q[disc + [action]] += \
                   alpha*(reward + gamma*np.max(Q[d.discretize(obs1)]) - Q[disc + [action]])
            #penalize or reward heavily if complete
            if done:
                if t < i * 300/num_episodes:
                    Q[disc + [action]] -= 10 ** 1
                elif t > i * 450/num_episodes:
                    Q[disc + [action]] += 10 ** 1
            ep_reward += reward
            obs0 = obs1
            epsilon -= 2 * 10 ** -6
            if done:
                break
        rewards.append(ep_reward)
        rewards.pop(0)
    print("moving average: " + str(sum(rewards) / len(rewards)))
    # print(Q)

def simulate():
    obs0 = env.reset()
    for t in range(200):
        env.render()
        action = np.argmax(Q[d.discretize(obs0)])
        obs1, reward, done, info = env.step(action)
        obs0 = obs1
        if done:
            env.render()
            break

class Discretizer:
    '''
    [-4.8,4.8] -> {0,1,...,14} through {-4.48,-4.16,...,4.48,4.8}
    '''
    def __init__(self, intervals, limits):
        self.intervals = intervals
        self.limits = limits
        self.bins = []
        for (num_intervals, limit) in zip(intervals, limits):
            self.bins.append([-limit + 2 * k * limit / num_intervals for k in range(1, num_intervals + 1)])

    def discretize(self, observation):
        res = []
        for (i, bucket) in zip(observation, self.bins):
            res.append(np.digitize(i, bucket, right=True))
        return res

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, './experiments/cartpole-v0', force=True)
    intervals = (20,20,20,20)
    limits = (4.8, 10, 0.42, 10)
    Q = np.random.rand(*intervals, env.action_space.n) / 10 ** 7
    d = Discretizer(intervals, limits)
    train()
    simulate()
