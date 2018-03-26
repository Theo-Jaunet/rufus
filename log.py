import datetime
import os

import numpy as np
from PIL import Image


class Log:

    def __init__(self):
        self.date = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(
            ':', '.')
        self.agent_filename = "agent_" + self.date + "_log.csv"
        self.net_filename = "net_" + self.date + "_log.csv"

        self.std_dir = "logs/" + self.date
        self.weight_dir = self.std_dir + "/weights/"
        self.features_dir = self.std_dir + "/features/"
        self.models_dir = self.std_dir + "/models/"
        self.csv_dir = self.std_dir + "/csv/"
        self.temp_dir = ""
        self.episode_dir = ""
        self.image_dir = ""
        self.preimage_dir = ""
        self.hidden_dir = ""
        self.acts_dir = ""

        self.init_dir()

    def init_dir(self):
        """# Create directories for logging & model information gathering """

        os.makedirs(self.std_dir)
        # os.makedirs(self.weight_dir)
        # os.makedirs(self.features_dir)
        os.makedirs(self.models_dir)
        os.makedirs(self.csv_dir)
        self.make_reward()

    def file_creation(self):
        """# Create both csv files for logging """

        with open(self.csv_dir + self.agent_filename, "w") as f:
            f.write("learning_step,train,random,episode,action,reward,posx,posy,angle\n")

        with open(self.csv_dir + self.net_filename, "w") as f:
            f.write("learning_step,train,hidden,cell,q_values\n")

    def write_agent_track(self, message):
        """# Write agent trace from his actions """

        with open(self.csv_dir + self.agent_filename, "a") as f:
            f.write(message)

    def write_net_track(self, message):
        """# Write net trace for visualizations """
        with open(self.csv_dir + self.net_filename, "a") as f:
            f.write(message)

    def set_agent_mess(self, step, train, rng, episode, action, reward, pos):
        """# Setup and format one row of agent csv from gathered information """

        self.write_agent_track(
            step + "," + str(train) + "," + str(rng) + "," + str(episode) + "," + str(action) + "," + str(
                reward) + "," + pos + "\n")

    def save_weight_gray(self, img, name):
        """# Save weight matrix into gray shades images """

        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(self.weight_dir + name)

    def save_feature_gray(self, img, name):
        """# Save weight matrix into gray shades images """

        img = (img * 255).astype(np.uint8).reshape(img.shape[1], img.shape[0])
        img = Image.fromarray(img)
        img.save(self.features_dir + name)

    def set_net_agent(self, hidden, cell, q_values, step, w_conv1, w_conv2, w_conv3, relu_1, relu_2, relu_3):
        """# Setup and format one row of network csv from gathered information """

        # TODO: not functional yet, is this csv really needed ?

        q = "["
        for elem in q_values[0, 0]:
            q += str(elem) + ";"
        q = q[:-1] + "]"

        c = "["
        for elem in cell[0, 0]:
            c += str(elem) + ";"
        c = c[:-1] + "]"

        h = "["
        for elem in hidden[0, 0]:
            h += str(elem) + ";"
        h = h[:-1] + "]"

        print(self.temp_dir)

    def make_epoch_dir(self, epoch):

        if not os.path.exists(self.std_dir + "/epoch_" + str(epoch)):
            os.makedirs(self.std_dir + "/epoch_" + str(epoch))
            self.temp_dir = self.std_dir + "/epoch_" + str(epoch) + "/"

    def make_dir(self, dir):

        if not os.path.exists(self.temp_dir + dir):
            os.makedirs(self.temp_dir + dir)
            if dir == "weights":
                self.weight_dir = self.temp_dir + dir + "/"

    def save_input(self, img, name, num):
        """# Save weight matrix into gray shades images """

        if num == 0:
            img = Image.fromarray(img)
            img.save(self.image_dir + name)
        elif num == 1:
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
            img.save(self.preimage_dir + name)

    def make_episode(self, episode):

        if not os.path.exists(self.temp_dir + "episode_" + str(episode)):
            os.makedirs(self.temp_dir + "episode_" + str(episode))
            self.episode_dir = self.temp_dir + "episode_" + str(episode) + "/"

    def make_instance_dir(self, dir):

        if not os.path.exists(self.episode_dir + dir):
            os.makedirs(self.episode_dir + dir)
        if dir == "features":
            self.features_dir = self.episode_dir + dir + "/"
        elif dir == "images":
            self.image_dir = self.episode_dir + dir + "/"
        elif dir == "preimages":
            self.preimage_dir = self.episode_dir + dir + "/"
        elif dir == "hidden":
            self.hidden_dir = self.episode_dir + dir + "/"
        elif dir == "acts":
            self.acts_dir = self.episode_dir + dir + "/"

    def save_vector(self, hid, vec, name):
        if vec == "hidden":
            with open(self.hidden_dir + name, "w") as f:
                f.write(hid)
        elif vec == "acts":
            with open(self.acts_dir + name, "w") as f:
                f.write(hid)

    def make_reward(self):

        with open(self.csv_dir + "rewards.csv", 'w') as f:
            f.write("epoch,episode,reward\n")

    def save_reward(self, epoch, episode, reward):
        with open(self.csv_dir + "rewards.csv", 'a') as f:
            f.write(str(epoch) + "," + str(episode) + "," + str(reward)+"\n")
