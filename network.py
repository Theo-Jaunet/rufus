import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TNet(nn.Module):
    def __init__(self, available_actions_count, mem):
        super(TNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=16, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=8, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1)

        self.lstm_size = 512
        self.lstm = nn.LSTM(10816, 512, batch_first=True)
        self.mem = mem
        self.h0 = Variable(torch.zeros(1, 1, self.lstm_size), requires_grad=True).cuda()
        self.c0 = Variable(torch.zeros(1, 1, self.lstm_size), requires_grad=True).cuda()

        self.x1 = None
        self.x2 = None
        self.x3 = None

        self.action_value_layer = nn.Linear(self.lstm_size, available_actions_count)

    def forward(self, x):

        # if the agent is using his memory to learn
        if self.mem:
            batch_size, sequence_length = x.size()[0:2]

            self.x1 = F.relu(self.conv1(x))
            self.x2 = F.relu(self.conv2(self.x1))
            self.x3 = F.relu(self.conv3(self.x2))

            x = self.x3.view(batch_size, 1, -1)

            h0 = Variable(torch.zeros(1, batch_size, self.lstm_size), requires_grad=True).cuda()
            c0 = Variable(torch.zeros(1, batch_size, self.lstm_size), requires_grad=True).cuda()

            x, (h0, c0) = self.lstm(x, (h0, c0))

            temp = self.action_value_layer(x)
            return temp
        else:
            batch_size = x.size()[0]

            self.x1 = F.relu(self.conv1(x))
            self.x2 = F.relu(self.conv2(self.x1))
            self.x3 = F.relu(self.conv3(self.x2))


            x = self.x3.view(batch_size, 1, -1)

            x, (self.h0, self.c0) = self.lstm(x, (self.h0, self.c0))
            return self.action_value_layer(x)

    def reset_hidden(self):
        """# Reset values of hidden state and cell """

        self.h0 = Variable(torch.zeros(1, 1, self.lstm_size), requires_grad=True).cuda()
        self.c0 = Variable(torch.zeros(1, 1, self.lstm_size), requires_grad=True).cuda()

    def store_hidden(self):
        """# Save values of hidden state and cell """

        self.h0b = self.h0[:]
        self.c0b = self.c0[:]

    def restore_hidden(self):
        """# Restore values of hidden state and cell """

        self.h0 = self.h0b[:]
        self.c0 = self.c0b[:]

    def get_info(self):
        """# Gather needed information about network such as features, filters, states, etc ... """

        a = self.conv1.cpu().weight.data.numpy()
        b = self.conv2.cpu().weight.data.numpy()
        c = self.conv3.cpu().weight.data.numpy()
        g = self.h0.cpu().data.numpy()
        h = self.c0.cpu().data.numpy()
        d = self.x1.cpu().data.numpy()
        e = self.x2.cpu().data.numpy()
        f = self.x3.cpu().data.numpy()

        # Needed to use GPU and still be able to get info
        self.conv1.cuda()
        self.conv2.cuda()
        self.conv3.cuda()
        self.h0.cuda()
        self.c0.cuda()

        return [a, b, c, d, e, f, g, h]
