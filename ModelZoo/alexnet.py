import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, dataset='cifar10',num_classes=10):
        super(AlexNet, self).__init__()
        assert dataset in ['cifar10', 'cifar100']
        if dataset in ['cifar10', 'cifar100']:
            data_mean = [0.5, 0.5, 0.5]
            data_std = [0.2, 0.2, 0.2]

        self.mean = nn.Parameter(torch.tensor(data_mean).unsqueeze(0).unsqueeze(2).unsqueeze(3), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(data_std).unsqueeze(0).unsqueeze(2).unsqueeze(3), requires_grad=False)

        conv_layer_1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=5)
        conv_layer_2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        conv_layer_3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)

        self.conv1 = nn.Sequential(
            conv_layer_1,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            conv_layer_2,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            conv_layer_3,
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x