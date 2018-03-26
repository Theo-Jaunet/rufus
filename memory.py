from random import sample

import numpy as np


class ReplayMemory:
    def __init__(self,flags):
        channels = 1
        self.capacity = flags.replay_memory_size
        state_shape = (self.capacity, channels, flags.resolution[0], flags.resolution[1])
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(self.capacity, dtype=np.int32)
        self.r = np.zeros(self.capacity, dtype=np.float32)
        self.isterminal = np.zeros(self.capacity, dtype=np.float32)

        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        """# Push transition into buffer memory, remove oldest if the buffer is full """

        self.s1[self.pos, :, :, :] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, :] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        """# Get random sample from buffer memory """

        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]