import gym
import numpy as np
from collections import deque

env = gym.make('MountainCar-v0')

class QLearner:
	def __init__(self, observations, actions, epsilon=0.5):
		self.observations = observations
		self.actions = actions
		self.epsilon = epsilon
		self.Q = np.random.rand(*observations, actions) / 10**3

	def act(self, obs, d_epsilon=-10**-5):
		self.epsilon = max(0.0, self.epsilon + d_epsilon)
		if np.random.random() < self.epsilon:
			return env.action_space.sample()
		return np.argmax(self.Q[obs])

	def learn(self, obs0, obs1, action, reward, done, t, alpha=0.3, gamma=0.99):
		d_Q = alpha * (reward + gamma * np.max(self.Q[obs1]) - self.Q[obs0 + (action,)])
		self.Q[obs0 + (action,)] += d_Q
		if done and t < 180:
			self.Q[obs0 + (action,)] += 10

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

def simulate(episodes):
	moving_average = deque([0 for _ in range(episodes)])
	for _ in range(episodes):
		obs0 = env.reset()
		disc0 = d.discretize(obs0)
		ep_reward = 0
		for t in range(200):
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
	# print(learner.Q)

if __name__ == '__main__':
	intervals = (20,20)
	limits = tuple([(l, h) for (l, h) in zip(env.observation_space.low, env.observation_space.high)])
	d = Discretizer(intervals, limits)
	learner = QLearner(intervals, env.action_space.n, epsilon=0.4)
	for i in range(1000):
		obs0 = env.reset()
		disc0 = d.discretize(obs0)
		for t in range(200):
			action = learner.act(disc0, d_epsilon=-2*10**-5)
			obs1, reward, done, info = env.step(action)
			disc1 = d.discretize(obs1)
			learner.learn(disc0, disc1, action, reward, done, t, alpha=0.2, gamma=0.99)
			disc0 = disc1
			if done:
				break
	simulate(1)