import gym
import numpy as np
from CriticNetwork import CriticNetwork
from ReplayBuffer import ReplayBuffer

MINIBATCH_SIZE = 64
EPSILON = 1.0
GAMMA = 0.99

def action_max(state):
    batch = np.array([np.hstack((state, [action])) for action in range(env.action_space.n)])
    return np.argmax(critic.predict_target(batch))

def action_max_batch(state_batch):
    """
    returns a batch of best actions for a batch of states
    """
    return np.array([[action_max(state)] for state in state_batch])

def act(state):
    if np.random.random() < EPSILON:
        return env.action_space.sample()
    return action_max(state)

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    state_dim = env.observation_space.shape[0]
    action_dim = 1 # env.action_space.n = 2, so values (0,1,2)
    
    critic = CriticNetwork(state_dim, action_dim, tau=0.5)
    replay_buffer = ReplayBuffer(10000)

    total = 0
    for episode in range(1000):
        obs0 = env.reset()
        ep_reward = 0
        EPSILON = max(10e-4, EPSILON * 0.99)
        for t in range(max_steps):
            if episode % 25 == 0:
                env.render()
            action = act(obs0)
            obs1, reward, done, info = env.step(action)
            ep_reward += reward
            if done:
                break

            replay_buffer.add(obs0.reshape(state_dim), [action], reward, t, obs1.reshape(state_dim))
            if replay_buffer.size() > MINIBATCH_SIZE:
                minibatch = replay_buffer.sample_batch(MINIBATCH_SIZE)
                s0_batch, a_batch, r_batch, t_batch, s1_batch = minibatch
                action_target_batch = action_max_batch(s1_batch)
                q_target_batch = critic.predict_target(np.hstack((s1_batch, action_target_batch)))
                target_batch = r_batch.reshape(MINIBATCH_SIZE, 1) + GAMMA * q_target_batch
                loss = critic.learn(np.hstack((s0_batch, a_batch)), target_batch)
                critic.update_target()

            obs0 = obs1
            
        total += ep_reward
        print("Episode {0:8d}: {1:4d} timesteps, {2:4f}, {3:4f}".format(episode, t, ep_reward, EPSILON))
        print(loss)