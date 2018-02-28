from random import sample, randint
from torch.autograd import Variable
from vizdoom import *
from time import sleep

import itertools as it
import skimage.color, skimage.transform
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def init(cfg):
    game = DoomGame()
    game.load_config(cfg)
    game.set_doom_scenario_path("scenario/my_way_home.wad")
    game.set_doom_map("map01")
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    # game.set_screen_resolution(ScreenResolution.RES_320X240)
    game.add_available_button(Button.MOVE_FORWARD)
    game.add_available_button(Button.TURN_LEFT)
    game.add_available_button(Button.TURN_RIGHT)
    game.add_available_button(Button.TURN180)

    game.set_episode_timeout(2600)
    game.set_episode_start_time(20)
    game.set_window_visible(False)
    game.init()

    return game


def imgform(img):
    img = skimage.transform.resize(img, (84, 84))
    img = img.astype(np.float32)
    return img


class Net(nn.Module):
    def __init__(self, available_actions_count):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32,64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, available_actions_count)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3136)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayMemory:
    def __init__(self, capacity):
        channels = 1
        state_shape = (capacity, channels, 84, 84)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, 0, :, :] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, 0, :, :] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]


def learn_from_memory():

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > 64:
        s1, a, s2, isterminal, r = memory.get_sample(64)
        print(isterminal.shape,"ist")
        s2 = torch.from_numpy(s2)
        s2 = Variable(s2)
        q = model(s2).data.numpy()

        q2 = np.max(q, axis=1)
        print(q2.shape,"q2")
        s1 = torch.from_numpy(s1)
        s1 = Variable(s1)
        target_q = model(s1).data.numpy()
        print(target_q.shape,"target")
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
        target_q[np.arange(target_q.shape[0]), a] = r + 0.99 * (1 - isterminal) * q2
        print(target_q.shape,"fsfsd")
        target_q = torch.from_numpy(target_q)
        s1, target_q = s1, Variable(target_q)
        output = model(s1)
        loss = criterion(output, target_q)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def cap(it, itmax):
    pur = it / itmax

    if pur < 0.3:
        return 80
    elif it < 0.6:
        return 50
    else:
        return 20


if __name__ == '__main__':

    game = init("scenario/my_way_home.cfg")

    actions = [[True, False, False, False],
               [False, True, False, False],
               [False, False, True, False],
               [False, False, False, True],
               [True, False, True, False],
               [True, False, False, True],
               [False, True, True, False],
               [False, True, False, True]]

    action_map = {i: act for i, act in enumerate(actions)}

    model = Net(len(action_map))

    optimizer = optim.SGD(model.parameters(), 0.00025)
    criterion = nn.MSELoss()

    memory = ReplayMemory(25000)
    with open("logs/log.csv", "w") as f:
        f.write("episode,action,reward,posx,posy\n")
        for i in range(5000):
            print("Episode #" + str(i + 1))
            game.new_episode()
            w = 0
            while not game.is_episode_finished():
                s1 = imgform(game.get_state().screen_buffer)
                s1 = s1.reshape([1, 1, 84, 84])

                if randint(0, 100) > cap(i, 200):

                    state = torch.from_numpy(s1)
                    state = Variable(state)

                    q = model(state)
                    m, index = torch.max(q, 1)
                    action = index.data.numpy()[0]
                else:

                    action = randint(0, len(actions) - 1)

                reward = game.make_action(actions[action], 4)

                isterminal = game.is_episode_finished()
                s2 = imgform(game.get_state().screen_buffer) if not isterminal else None

                memory.add_transition(s1, action, s2, isterminal, reward)
                learn_from_memory()
                f.write(str(i) + "," + str(w) + "," + str(reward) + "," + str(
                    game.get_game_variable(GameVariable.POSITION_X)) + "," + str(game.get_game_variable(
                    GameVariable.POSITION_Y)) + "\n")
                w += 1
            # Check how the episode went.
            print("Episode finished.")
            print("Total reward:", game.get_total_reward())
            print("************************")
    game.close()