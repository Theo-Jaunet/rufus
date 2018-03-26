import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from network import TNet
from memory import ReplayMemory


class RdqnAgent():
    def __init__(self, flags):

        # actions = [[True, False, False, False], [False, True, False, False], [False, False, True, False],
        #          [False, True, True, False], [True, False, True, False], [True, False, False, True],
        #
        #          [False, True, False, True], [False, False, False, True]]

        actions = [[True, False], [False, True]]
        self.action_map = {i: act for i, act in enumerate(actions)}

        if flags.load_model:
            print("Loading model from: ", flags.model_savedfile)
            self.model = torch.load(flags.model_savedfile).cuda()
        else:
            self.model = TNet(len(self.action_map), False).cuda()

        self.optimizer = torch.optim.SGD(self.model.parameters(), flags.learning_rate)
        self.memory = ReplayMemory(flags)
        self.criterion = nn.SmoothL1Loss()
        self.gamma = flags.discount_factor
        self.batch_size = flags.batch_size

    def learn(self, s1, target_q):
        """# Save model delta from memory replay learning"""

        s1 = torch.from_numpy(s1)
        target_q = torch.from_numpy(target_q)
        s1, target_q = Variable(s1).cuda(), Variable(target_q).cuda()
        output = self.model(s1)
        loss = self.criterion(output, target_q)

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model.restore_hidden()
        self.model.mem = False
        return loss

    def get_q_values(self, state):
        """# Get model predictions of Q-values"""

        state = torch.from_numpy(state)
        state = Variable(state).cuda()
        return self.model(state)

    def get_best_action(self, state):
        """# Get action with highest Q-value"""

        q = self.get_q_values(state)
        m, index = torch.max(q, 2)
        action = index.cpu().data.numpy()[0, 0]
        return action

    def learn_from_memory(self):
        """# Learns from a single transition (making use of replay memory).
        s2 is ignored if s2_isterminal """

        # Get a random minibatch from the replay memory and learns from it.
        if self.memory.size > self.batch_size:
            s1, a, s2, isterminal, r = self.memory.get_sample(self.batch_size)
            self.model.mem = True
            self.model.store_hidden()

            q = self.get_q_values(s2).cpu().data.numpy()
            q2 = np.max(q, axis=1)
            q2 = np.max(q2, axis=1)
            target_q = self.get_q_values(s1).cpu().data.numpy()

            target_q = target_q.reshape(self.batch_size, len(self.action_map))
            target_q[np.arange(target_q.shape[0]), a] = r + self.gamma * (1 - isterminal) * q2

            self.learn(s1, target_q)

    def add_mem_and_learn(self, s1, a, s2, isterminal, reward):
        """# Add experience to memory buffer & learn"""

        self.memory.add_transition(s1, a, s2, isterminal, reward)

        self.learn_from_memory()


    def add_mem(self, s1, a, s2, isterminal, reward):
        """# Add experience to memory buffer"""

        self.memory.add_transition(s1, a, s2, isterminal, reward)

    def save_model(self, path):
        """# Save trained model"""

        print("Saving the network ... ", end="")
        torch.save(self.model, path)
        print(" ... done !")
