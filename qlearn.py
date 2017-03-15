import gym
import numpy as np

class QLearner:
	def __init__(self, observations, actions, epsilon=0.5):
		self.observations = observations
		self.actions = actions
		self.epsilon = epsilon
		self.Q = np.random.rand(*observations, actions)

	def act(self, observation, d_epsilon=-10**5):
		self.epsilon += d_epsilon
		if np.random.random() < self.epsilon:
			return np.random.randint(self.actions)
		return np.argmax(self.Q[observation])

	def learn(self, obs0, obs1, action, reward, alpha=0.3, gamma=0.99):
		self.Q[obs0 + [action]] += \
			alpha * (reward + gamma * np.max(self.Q[obs1]) - self.Q[obs0 + [action]])
			