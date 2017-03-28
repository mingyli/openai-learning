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

    def predict_target(self, state_batch):
        return self.target.predict_on_batch(state_batch)
    
    def update_target(self):
        actor_weights = self.actor.get_weights()
        target_weights = self.target.get_weights()
        for i in xrange(len(actor_weights)):
            target_weights[i] = self.tau * actor_weights[i] + (1-self.tau) * target_weights[i]
        self.target.set_weights(target_weights)

class CriticNetwork:
    """
    Q network. takes in (state, action) as vector and returns a Q value
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

    def predict_target(self, batch):
        """
        takes in a batch of (state, action) vectors and returns Q values
        """
        return self.target.predict_on_batch(batch)

    def learn(self, input_batch, target_batch):
        """
        takes in a batch of (state, action) vectors and target Q values
        """
        self.critic.train_on_batch(input_batch, target_batch)

    def update_target(self):
        critic_weights = self.critic.get_weights()
        target_weights = self.target.get_weights()
        for i in xrange(len(critic_weights)):
            target_weights[i] = self.tau * critic_weights[i] + (1-self.tau) * target_weights[i]
        self.target.set_weights(target_weights)