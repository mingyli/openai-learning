import keras

class Agent:
    def __init__(self, state_dim, action_dim, action_bound=2, tau=0.001, gamma=0.99):
        