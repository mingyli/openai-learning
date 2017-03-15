import gym
import numpy as np
from collections import deque

env = gym.make('RepeatCopy-v0')

class QLearner:
	def __init__(self, observations, actions, epsilon=0.5):
		self.observations = observations
		self.actions = actions
		self.epsilon = epsilon
		self.Q = np.random.rand(observations, *actions) / 10**5

	def act(self, observation, d_epsilon=-10**-5):
		self.epsilon = max(0.01, self.epsilon + d_epsilon)
		if np.random.random() < self.epsilon:
			return env.action_space.sample()
		action = np.argmax(self.Q[observation])
		action = np.unravel_index(action, self.Q.shape)
		return action[1:]

	def learn(self, obs0, obs1, action, reward, done, t, alpha=0.3, gamma=0.99):
		self.Q[(obs0,) + action] += \
			alpha * (reward + gamma * np.max(self.Q[obs1]) - self.Q[(obs0,) + action])

def simulate():
	moving_average = deque([0 for _ in range(100)])
	for i in range(100):
		obs0 = env.reset()
		ep_reward = 0
		for t in range(200):
			# env.render()
			action = learner.act(obs0)
			obs1, reward, done, info = env.step(action)
			ep_reward += reward
			obs0 = obs1
			if done:
				break
		moving_average.append(ep_reward)
		moving_average.popleft()
	print("moving average: ", str(sum(moving_average) / len(moving_average)))
	print(learner.Q)

if __name__ == '__main__':
	action_space = tuple([space.n for space in env.action_space.spaces])
	learner = QLearner(env.observation_space.n, action_space, epsilon=0.8)
	for i in range(4000):
		obs0 = env.reset()
		for t in range(200):
			action = learner.act(obs0, d_epsilon=-3*10**-4)
			obs1, reward, done, info = env.step(action)
			learner.learn(obs0, obs1, action, reward, done, t, alpha=0.5, gamma=0.8)
			obs0 = obs1
			if done:
				break
	simulate()