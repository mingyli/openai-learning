import gym
import numpy as np
import networks
from networks import ActorNetwork, CriticNetwork
from replay_buffer import ReplayBuffer

MINIBATCH_SIZE = 64
GAMMA = 0.99

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    actor = ActorNetwork(state_dim, action_dim, action_bound)
    critic = CriticNetwork(state_dim, action_dim)
    replay_buffer = ReplayBuffer(10000)

    total = 0
    for episode in range(1000):
        obs0 = env.reset()
        ep_reward = 0
        for t in range(max_steps):
            if episode % 25 == 0:
                env.render()
            action = actor.act(obs0) # TODO add noise for exploration
            obs1, reward, done, info = env.step(action)
            replay_buffer.add(obs0.reshape(state_dim), action.reshape(action_dim), reward, t, obs1.reshape(state_dim))

            if replay_buffer.size() > MINIBATCH_SIZE:
                minibatch = replay_buffer.sample_batch(MINIBATCH_SIZE)
                s0_batch, a_batch, r_batch, t_batch, s1_batch = minibatch

                actor_target_batch = actor.predict_target(s1_batch)
                q_target_batch = critic.predict_target(np.hstack((s1_batch, actor_target_batch)))
                target_batch = r_batch + GAMMA * q_target_batch

                loss = critic.learn(np.hstack((s0_batch, a_batch)), target_batch)
                # TODO update actor policy
                
                actor.update_target()
                critic.update_target()

            obs0 = obs1
            ep_reward += reward[0]
            if done:
                break        
        total += ep_reward
        print("Episode {0:8d}: {1:4d} timesteps, {2:4f} average".format(episode, t, total/(episode+1)))
        print(loss)
