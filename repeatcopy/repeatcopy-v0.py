import gym
from qlearn import QLearner

env = gym.make('RepeatCopy-v0')

if __name__ == '__main__':
	learner = QLearner()