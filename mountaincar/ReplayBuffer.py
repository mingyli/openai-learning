from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, state0, action, reward, t, state1):
        experience = (state0, action, reward, t, state1)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        s0_batch = np.array([e[0] for e in batch])
        a_batch = np.array([e[1] for e in batch])
        r_batch = np.array([e[2] for e in batch])
        t_batch = np.array([e[3] for e in batch])
        s1_batch = np.array([e[4] for e in batch])
        return s0_batch, a_batch, r_batch, t_batch, s1_batch

    def clear(self):
        self.deque.clear()
        self.count = 0