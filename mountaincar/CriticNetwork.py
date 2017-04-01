import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten

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
        first_layer = Dense(256, 
                            input_dim = self.state_dim + self.action_dim,
                            init = 'uniform',
                            activation = 'relu')
        self.critic.add(first_layer)
        self.target.add(first_layer)
        self.critic.add(Dense(512, activation='relu', init='uniform'))
        self.target.add(Dense(512, activation='relu', init='uniform'))
        last_layer = Dense(self.action_dim,
                           init = 'uniform',
                           activation = 'linear') # linear activation for Q values
        self.critic.add(last_layer)
        self.target.add(last_layer)

        self.critic.compile(optimizer = 'rmsprop',
                           loss = 'mse')
        self.target.compile(optimizer = 'rmsprop',
                            loss = 'mse')
        self.target.set_weights(self.critic.get_weights())

    def learn(self, input_batch, target_batch):
        """
        takes in a batch of (state, action) vectors and target Q values
        """
        return self.critic.train_on_batch(input_batch, target_batch)
    
    def predict_target(self, batch):
        """
        takes in a batch of (state, action) vectors and returns Q values
        """
        return self.target.predict_on_batch(batch)

    def update_target(self):
        critic_weights = self.critic.get_weights()
        target_weights = self.target.get_weights()
        for i in range(len(critic_weights)):
            target_weights[i] = self.tau * critic_weights[i] + (1-self.tau) * target_weights[i]
        self.target.set_weights(target_weights)

