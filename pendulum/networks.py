import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Dense



class ActorNetwork:
	"""
	takes in state vector and returns an action 
	"""
	def __init__(self, state_dim, action_dim, action_bound, lr=0.001, tau=0.001):
		self.state_dim = state_dim # 3 because 3x1 vector
		self.action_dim = action_dim # 1 because 1x1 vector
		self.action_bound = action_bound # 2 because [-2,2]
		self.lr = lr
		self.tau = tau # what proportion of new weights to copy into target

		self.actor = Sequential()
		self.target = Sequential()
		first_layer = Dense(16, 
							input_dim = self.state_dim,
							init = 'uniform',
							activation = 'relu')
		self.actor.add(first_layer)
		self.target.add(first_layer)
		self.actor.add(Dense(16, activation='relu', init='uniform'))
		self.target.add(Dense(16, activation='relu', init='uniform'))
		last_layer = Dense(self.action_dim,
						   init = 'uniform',
						   activation = lambda x: self.action_bound * K.tanh(x))
		self.actor.add(last_layer)
		self.target.add(last_layer)
		self.actor.compile(optimizer = 'rmsprop',
						   loss = 'mse')
		self.target.compile(optimizer = 'rmsprop',
							loss = 'mse')

	def act(self, state):
		pred = self.actor.predict(state.reshape(1, self.state_dim))
		return pred.reshape(1, self.action_dim)

	def learn(self, batch):
		pass

	def target_predict(self, state_batch):
		return self.target.predict_on_batch(state_batch)

class CriticNetwork:
	"""
	Q network. takes in state and action as vector and returns a Q value
	"""
	def __init__(self, state_dim, action_dim, lr=0.001, tau=0.001):
		self.state_dim = state_dim
		self.action_dim = action_dim 
		self.lr = lr
		self.tau = tau

		self.critic = Sequential()
		self.target = Sequential()
		first_layer = Dense(16, 
							input_dim = self.state_dim + self.action_dim,
							init = 'uniform',
							activation = 'relu')
		self.critic.add(first_layer)
		self.target.add(first_layer)
		self.critic.add(Dense(16, activation='relu', init='uniform'))
		self.target.add(Dense(16, activation='relu', init='uniform'))
		last_layer = Dense(self.action_dim,
						   init = 'uniform',
						   activation = 'linear') # linear activation for Q values
		self.critic.add(last_layer)
		self.target.add(last_layer)

		self.critic.compile(optimizer = 'rmsprop',
						   loss = 'mse')
		self.target.compile(optimizer = 'rmsprop',
							loss = 'mse')

	def target_predict(self, batch):
		"""
		takes in a batch of (state, action) vectors and returns Q values
		"""
		return self.target.predict_on_batch(batch)

	def learn(self, target_batch, s0_batch, a_batch):
		input_batch = np.hstack((s0_batch, a_batch))
		return self.critic.train_on_batch(input_batch, target_batch)