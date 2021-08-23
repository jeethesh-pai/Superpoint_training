import torch.nn as nn
from torchsummary import summary
import torch
from torch.utils.tensorboard import SummaryWriter


model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8
            nn.BatchNorm2d(128),

            nn.Flatten(),
            nn.Linear(8*8*128, 1024),
            nn.ReLU(),

            nn.Linear(1024, 10))
summary(model.cuda(), input_size=(3, 32, 32))
writer = SummaryWriter(log_dir='logs_new')
writer.add_graph(model.cuda(), torch.ones(size=(2, 3, 320, 240)).cuda())
writer.flush()
writer.close()
