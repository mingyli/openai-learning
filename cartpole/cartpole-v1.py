import gym
import numpy as np

env = gym.make('CartPole-v1')

#TODO count for each pixel?
Q = np.zeros([env.observation_space.high-env.observation_space.low, env.action_space.n])

lr = 0.85
gamma = 0.99
num_episodes = 500

rList = []
for i_episode in range(num_episodes):
    observation = env.reset()
    total_reward = 0
    for t in range(100):
        env.render()
        print(observation)
        # action = env.action_space.sample()
        action = np.argmax(Q[observation,:] + np.random.randn(1, env.action_space.n) * (1./(i+1)))
        new_observation, reward, done, info = env.step(action)
        Q[observation, action] += lr * (reward + gamma * np.max(Q[new_observation,:]) - Q[observation,action])
        total_reward += reward
        observation = new_observation
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    rList.append(total_reward)

print("Score over time: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)
