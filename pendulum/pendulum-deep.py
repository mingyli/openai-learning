import gym
import numpy as np
import keras
# from deep import DeepQLearner
from keras.models import Sequential
from keras.layers import Dense


	

if __name__ == '__main__':
	env = gym.make('Pendulum-v0')
	max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
	learner = DoubleDeepQLearner(len(env.observation_space.high), (8, 16, 32, len(env.action_space.high)), 256, 100000)
	total = 0
	for _ in range(1000):
		obs0 = env.reset()
		ep_reward = 0
		for t in range(max_steps):
			action = learner.act(obs0)
			obs1, reward, done, info = env.step(action)
			learner.learn(obs0, obs1, action, reward, done, t)
			obs0 = obs1
			ep_reward += reward
			if done:
				break
		total += ep_reward
		print("Episode {0:8d}: {1:4d} timesteps, {2:4f} average".format(i_episode, t+1, total/(i_episode+1)))
