import gym
import numpy as np
from gym import wrappers
from collections import deque

env = gym.make('MountainCar-v0')
env = wrappers.Monitor(env, './experiments/mountaincar-v0', force=True)

class QLearner:
	def __init__(self, observations, actions, alpha=0.3, gamma=0.99, epsilon=0.5, d_epsilon=-10**-5):
		self.observations = observations
		self.actions = actions
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.d_epsilon = d_epsilon
		self.Q = np.random.rand(*observations, actions)

	def act(self, obs):
		self.epsilon = max(0.1, self.epsilon + self.d_epsilon)
		if np.random.random() < self.epsilon:
			return env.action_space.sample()
		return np.argmax(self.Q[obs])

	def learn(self, obs0, obs1, action, reward, done, t):
		target = reward + self.gamma * np.max(self.Q[obs1])
		self.Q[obs0 + (action,)] += self.alpha * (target - self.Q[obs0 + (action,)])
		if done:
			self.Q[obs0 + (action,)] += (180 / (t-20) - 1) * 10 ** 3
			if t > 195:
				self.Q[obs0 + (action,)] -= 10

	def __str__(self):
		return "alpha: {}\ngamma: {}\nepsilon: {}".format(self.alpha, self.gamma, self.epsilon)

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

	def discretize(self, observation):
		res = []
		for (i, bucket) in zip(observation, self.bins):
			res.append(np.digitize(i, bucket, right=True))
		return tuple(res)

def simulate(episodes, render=False):
	moving_average = deque([0 for _ in range(episodes)])
	for _ in range(episodes):
		obs0 = env.reset()
		disc0 = d.discretize(obs0)
		ep_reward = 0
		for t in range(200):
			if render:
				env.render()
			action = learner.act(disc0)
			obs1, reward, done, info = env.step(action)
			ep_reward += reward
			disc0 = d.discretize(obs1)
			if done:
				break
		moving_average.append(ep_reward)
		moving_average.popleft()
	print("moving average: ", sum(moving_average) / len(moving_average))

if __name__ == '__main__':
	intervals = (15,20)
	limits = tuple([(l, h) for (l, h) in zip(env.observation_space.low, env.observation_space.high)])
	d = Discretizer(intervals, limits)
	learner = QLearner(intervals, env.action_space.n, epsilon=1.0, alpha=0.2, gamma=0.99, d_epsilon=-1*10**-5)
	for i in range(4000):
		obs0 = env.reset()
		disc0 = d.discretize(obs0)
		for t in range(200):
			action = learner.act(disc0)
			obs1, reward, done, info = env.step(action)
			disc1 = d.discretize(obs1)
			learner.learn(disc0, disc1, action, reward, done, t)
			disc0 = disc1
			if done:
				break
	simulate(100)
	simulate(1, render=True)
	# print(learner)