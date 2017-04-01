import numpy as np
import keras
from keras.models import *
from keras.layers import *
from ReplayBuffer import ReplayBuffer

class DDPGLearner:
    def __init__(self, state_dim, action_dim, action_bound):
        self.replay_buffer = ReplayBuffer(100000)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.actor = self.create_actor(state_dim, action_dim)
        # print(self.actor.summary())
        self.critic, self.frozen_critic = self.create_critic(state_dim, action_dim)
        
        self.actor_target = self.create_actor(state_dim, action_dim)
        self.critic_target, _ = self.create_critic(state_dim, action_dim)
        # self.actor_target.set_weights(self.actor.get_weights())
        # self.critic_target.set_weights(self.critic.get_weights())
        self.update_targets(tau=1.)

        # make actor trainer
        state_input = Input(shape=(state_dim,))
        pred_action = self.actor(state_input)
        pred_q = self.frozen_critic([state_input, pred_action])
        self.actor_trainer = Model(input=state_input, output=pred_q)
        def neg_q(y_true,y_pred):
            return -y_pred
        self.actor_trainer.compile(loss=neg_q, optimizer='rmsprop')
        # print(self.actor_trainer.summary())

    def create_actor(self, input_dim, output_dim):
        input_layer = Input(shape=(input_dim,))
        i = Dense(32, activation='relu')(input_layer)
        i = Dense(32, activation='relu')(i)
        i = Dense(output_dim, activation='tanh')(i)
        i = Lambda(lambda x: x * self.action_bound, output_shape=(output_dim,))(i)
        model = Model(input=input_layer, output=i)
        model.compile(loss='mse', optimizer='rmsprop')
        return model

    def create_critic(self, state_dim, action_dim):
        state_input = Input(shape=(state_dim,))
        action_input = Input(shape=(action_dim,))
        i = merge([state_input, action_input], mode='concat')
        i = Dense(32, activation='relu')(i)
        i = Dense(32, activation='relu')(i)
        i = Dense(1, activation='linear')(i)
        out = i
        model = Model(input=[state_input, action_input], output=out)
        model.compile(loss='mse', optimizer='rmsprop')

        for i in model.layers:
            i.trainable = False 

        frozen_model = Model(input=[state_input,action_input],output=out)
        frozen_model.compile(loss='mse', optimizer='rmsprop')
        return model, frozen_model

    def update_targets(self, tau=0.001):
        actor_weights, critic_weights = self.actor.get_weights(), self.critic.get_weights()
        actor_target_weights, critic_target_weights = self.actor_target.get_weights(), self.critic_target.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = tau * actor_weights[i] + (1-tau) * actor_target_weights[i]
        for i in range(len(critic_weights)):
            critic_target_weights[i] = tau * critic_weights[i] + (1-tau) * critic_target_weights[i]
        self.actor_target.set_weights(actor_target_weights)
        self.critic_target.set_weights(critic_target_weights)

    def act(self, state):
        state = np.reshape(state, (1, self.state_dim))
        return self.actor.predict(state)[0]

    def train(self, MINIBATCH_SIZE=64):
        replay_buffer = self.replay_buffer
        if replay_buffer.size() > MINIBATCH_SIZE:
            minibatch = replay_buffer.sample_batch(MINIBATCH_SIZE)
            s0_batch, a_batch, r_batch, t_batch, d_batch, s1_batch = minibatch

            action_target = self.actor_target.predict_on_batch(s1_batch)
            q_target = self.critic_target.predict_on_batch([s1_batch, action_target])
            targets = r_batch.reshape(64,1) + (1.0 - d_batch.reshape(64,1)) * 0.99 * q_target
            # targets = r_batch.reshape(64,1) + 0.99 * q_target

            # import pdb
            # pdb.set_trace()

            self.critic.train_on_batch([np.array(s0_batch), np.array(a_batch)], targets)#, batch_size=MINIBATCH_SIZE)
            self.actor_trainer.train_on_batch(np.array(s0_batch), targets)#, batch_size=MINIBATCH_SIZE)
            self.update_targets()
    


