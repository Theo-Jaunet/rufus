from random import sample, randint
from torch.autograd import Variable
from vizdoom import *
import skimage.color, skimage.transform
import numpy as np

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from log import Log


def init(cfg):
    game = DoomGame()
    game.load_config(cfg)
    game.set_doom_scenario_path("scenario/my_way_home.wad")
    game.set_doom_map("map01")
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.RGB24)
    # game.set_screen_resolution(ScreenResolution.RES_320X240)
    game.add_available_button(Button.TURN_LEFT)
    game.add_available_button(Button.TURN_RIGHT)
    game.add_available_button(Button.MOVE_FORWARD)
    game.add_available_button(Button.MOVE_BACKWARD)

    game.set_window_visible(False)
    game.init()

    return game


def imgform(img):
    img = skimage.transform.resize(img, (3, 60, 80))
    img = img.astype(np.float32)
    return img


class Net(nn.Module):
    def __init__(self, available_actions_count, mem):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.lstm_size = 512
        self.lstm = nn.LSTM(1536, 512, batch_first=True)
        self.mem = mem
        self.h0 = Variable(torch.zeros(1, 1, self.lstm_size), requires_grad=True).cuda()
        self.c0 = Variable(torch.zeros(1, 1, self.lstm_size), requires_grad=True).cuda()

        self.action_value_layer = nn.Linear(self.lstm_size, available_actions_count)

    def forward(self, x):

        if self.mem:
            batch_size, sequence_length = x.size()[0:2]

            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))

            x = x.view(batch_size, 1, -1)

            h0 = Variable(torch.zeros(1, batch_size, self.lstm_size), requires_grad=True).cuda()
            c0 = Variable(torch.zeros(1, batch_size, self.lstm_size), requires_grad=True).cuda()

            x, (h0, c0) = self.lstm(x, (h0, c0))
            temp = self.action_value_layer(x)
            return temp
        else:
            batch_size = x.size()[0]
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))

            x = x.view(1, batch_size, -1)

            x, (self.h0, self.c0) = self.lstm(x, (self.h0, self.c0))

            return self.action_value_layer(x)

    def reset_hidden(self):
        self.h0 = Variable(torch.zeros(1, 1, self.lstm_size), requires_grad=True).cuda()
        self.c0 = Variable(torch.zeros(1, 1, self.lstm_size), requires_grad=True).cuda()

    def store_hidden(self):
        self.h0b = self.h0[:]
        self.c0b = self.c0[:]

    def restore_hidden(self):
        self.h0 = self.h0b[:]
        self.c0 = self.c0b[:]


class ReplayMemory:
    def __init__(self, capacity):
        channels = 3
        state_shape = (capacity, channels, 60, 80)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, :, :, :] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, :] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]


def learn_from_memory():
    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > 5000:
        s1, a, s2, isterminal, r = memory.get_sample(64)
        model.store_hidden()
        s2 = torch.from_numpy(s2)
        s2 = Variable(s2).cuda()

        model.mem = True
        q = model(s2).cpu().data.numpy()
        q2 = np.max(q, axis=1)
        q2 = np.max(q2, axis=1)

        s1 = torch.from_numpy(s1)
        s1 = Variable(s1).cuda()
        target_q = model(s1).cpu().data.numpy()
        target_q = target_q.reshape((64, 8))

        target_q[np.arange(target_q.shape[0]), a] = r + 0.99 * (1 - isterminal) * q2

        target_q = torch.from_numpy(target_q)
        s1, target_q = s1, Variable(target_q).cuda()
        output = model(s1)
        loss = criterion(output, target_q)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.restore_hidden()
        model.mem = False


def cap(epsilon, itmax, it):
    if it < itmax and epsilon > 0.1:
        return epsilon - 0.0000025
    else:
        return 0.01


def get_pos(game):
    return str(game.get_game_variable(GameVariable.POSITION_X)) + "," + str(game.get_game_variable(
        GameVariable.POSITION_Y)) + "," + str(game.get_game_variable(GameVariable.ANGLE))


def set_mess(q_values, rng, episode, action, reward, game):
    q = "["
    for elem in q_values[0, 0]:
        q += str(elem) + ";"
    q = q[:-1] + "]"
    log.write_track(
        q + "," + str(rng) + "," + str(episode) + "," + str(action) + "," + str(reward) + "," + get_pos(game)+"\n")


if __name__ == '__main__':

    game = init("scenario/my_way_home.cfg")
    log = Log()
    log.file_creation()

    actions = [[True, False, False, False],
               [False, True, False, False],
               [False, False, True, False],
               [False, False, False, True],
               [True, False, True, False],
               [True, False, False, True],
               [False, True, True, False],
               [False, True, False, True]]

    action_map = {i: act for i, act in enumerate(actions)}

    model = Net(len(action_map), False).cuda()

    optimizer = optim.SGD(model.parameters(), 0.00025)
    criterion = nn.SmoothL1Loss()

    memory = ReplayMemory(50000)

    epsilon = 1.0

    for i in range(50000):
        print("Episode #" + str(i + 1))
        game.new_episode()
        model.reset_hidden()
        w = 0

        while not game.is_episode_finished():
            rng = True
            s1 = imgform(game.get_state().screen_buffer)
            s1 = s1.reshape([1, 3, 60, 80])
            state = Variable(torch.from_numpy(s1)).cuda()
            q = model(state)
            m, index = torch.max(q, 1)

            # epsilon update
            if i < 0:
                epsilon = cap(epsilon, 10, i)

            if random.random() > epsilon:
                rng = False
                action = index.cpu().data.numpy()[0]
            else:
                action = randint(0, len(actions) - 1)

            reward = game.make_action(actions[action], 4)
            set_mess(q.cpu().data.numpy(), rng, i, w, reward, game)
            isterminal = game.is_episode_finished()
            s2 = imgform(game.get_state().screen_buffer) if not isterminal else None
            memory.add_transition(s1, action, s2, isterminal, reward)
            learn_from_memory()

            w += 1

        print("Episode finished.")
        print("Total reward:", game.get_total_reward())
        print("************************")

    game.close()
