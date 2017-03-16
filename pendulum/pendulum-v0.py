import gym
from gym import wrappers
import numpy as np
from collections import deque

class Discretizer:
	'''
	[-1.2,0.6] -> {0,1,...,19} 
	'''
	def __init__(self, intervals, limits):
		self.intervals = intervals
		self.limits = limits
		self.bins = []
		for (num_intervals, limit) in zip(intervals, limits):
			low, high = limit
			self.bins.append([low + k * (high-low) / num_intervals for k in range(1, num_intervals + 1)])

	def discretize(self, reals):
		res = []
		for (i, bucket) in zip(reals, self.bins):
			res.append(np.digitize(i, bucket, right=True))
		return tuple(res)

	def continuous(self, indices):
		res = []
		for (i, bucket) in zip(indices, self.bins):
			delta = bucket[1] - bucket[0]
			res.append(bucket[i] - delta / 2)
		return res

class QLearner:
	def __init__(self, observations, actions, alpha=0.3, gamma=0.99, epsilon=0.5, d_epsilon=-10**-5):
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.d_epsilon = d_epsilon
		self.Q = np.random.rand(*observations, *actions)

	def act(self, obs):
		self.epsilon = max(0.0, self.epsilon + self.d_epsilon)
		if np.random.random() < self.epsilon:
			return env.action_space.sample()
		unraveled = np.unravel_index(np.argmax(self.Q[obs]), self.Q.shape)
		return act_disc.continuous(unraveled[len(obs):])

	def learn(self, obs0, obs1, action, reward, done, t):
		target = reward + self.gamma * np.max(self.Q[obs1])
		self.Q[obs0 + action] += self.alpha * (target - self.Q[obs0 + action])

	def __str__(self):
		return "alpha: {}\ngamma: {}\nepsilon: {}".format(self.alpha, self.gamma, self.epsilon)

def simulate(episodes, render=False):
	moving_average = deque([0 for _ in range(episodes)])
	for _ in range(episodes):
		obs0 = env.reset()
		obs0 = obs_disc.discretize(obs0)
		ep_reward = 0
		for t in range(999):
			if render:
				env.render()
			action = learner.act(obs0)
			obs1, reward, done, info = env.step(action)
			ep_reward += reward
			obs0 = obs_disc.discretize(obs1)
			if done:
				break
		moving_average.append(ep_reward)
		moving_average.popleft()
	print("moving average: ", sum(moving_average) / len(moving_average))

if __name__ == '__main__':
	env = gym.make('Pendulum-v0')
	observations = (99, 99, 99)
	actions = (99,)
	obs_limits = tuple([(l, h) for (l, h) in zip(env.observation_space.low, env.observation_space.high)])
	obs_disc = Discretizer(observations, obs_limits)
	act_limits = tuple([(l, h) for (l, h) in zip(env.action_space.low, env.action_space.high)])
	act_disc = Discretizer(actions, act_limits)
	learner = QLearner(observations, actions, alpha=0.2, epsilon=1, d_epsilon=-1*10**-5)
	for i in range(1000):
		obs0 = env.reset()
		obs0 = obs_disc.discretize(obs0)
		for t in range(999):
			action = learner.act(obs0)
			obs1, reward, done, info = env.step(action)
			obs1 = obs_disc.discretize(obs1)
			action = act_disc.discretize(action)
			learner.learn(obs0, obs1, action, reward, done, t)
			obs0 = obs1
			if done:
				break
	simulate(100)
	simulate(1, render=True)