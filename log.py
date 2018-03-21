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

        self.make_dir()

    def make_dir(self):
        """# Create directories for logging & model information gathering """

        os.makedirs(self.std_dir)
        os.makedirs(self.weight_dir)
        os.makedirs(self.features_dir)
        os.makedirs(self.models_dir)
        os.makedirs(self.csv_dir)

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

        img = (img * 255).astype(np.uint8)
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
