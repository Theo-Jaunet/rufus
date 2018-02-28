import datetime


class Log:

    def __init__(self):
        self.uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(
            ':', '.')
        self.uniq_filename = self.uniq_filename + "_log.csv"
        self.dir = "logs/"

    def file_creation(self):
        with open(self.dir + self.uniq_filename, "w") as f:
            f.write("q_values,random,episode,action,reward,posx,posy,angle\n")

    def write_track(self, message):
        with open(self.dir + self.uniq_filename, "a") as f:
            f.write(message)
