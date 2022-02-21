from collections import deque
import random


class ReplayBuffer(object):

    def __init__(self, buffer_size, min_buffer_size):
        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size

    def initialize_buffer(self, buff=deque(), n_exp=0):
        self.buffer = buff
        self.num_experiences = n_exp

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if len(self.buffer) < self.min_buffer_size:
            return 'small buffer'
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

    def get_buffer_and_exp(self):
        return (self.buffer, [self.num_experiences])
