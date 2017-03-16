import gym
import numpy as np

class QLearner:
	def __init__(self, observations, actions, alpha=0.3, gamma=0.99, epsilon=0.5, d_epsilon=-10**-5):
		self.observations = observations
		self.actions = actions
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.d_epsilon = d_epsilon
		self.Q = np.random.rand(*observations, actions) 

	def act(self, observation, d_epsilon=-10**5):
		self.epsilon += d_epsilon
		if np.random.random() < self.epsilon:
			return np.random.randint(self.actions)
		return np.argmax(self.Q[observation])

	def learn(self, obs0, obs1, action, reward, done, t):
		d_Q = self.alpha * (reward + self.gamma * np.max(self.Q[obs1]) - self.Q[obs0 + (action,)])
		self.Q[obs0 + (action,)] += d_Q

	def __str__(self):
		return "alpha: {}\ngamma: {}\nepsilon: {}".format(self.alpha, self.gamma, self.epsilon)