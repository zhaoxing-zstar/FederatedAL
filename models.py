"""
Initializing neural networks for server and clients.
It's better to validate models before running experiments.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import ModelZoo


class InitNet(object):
    def initial_nets(self, dataset, model):
        if model in ["simple-cnn", "moderate-cnn"]:
            net = self.simple_cnn(dataset, model)
        elif model in ["resnet18", "resnet50"]:
            net = self.resnet(model)
        elif model in ["wideresnet"]:
            net = self.wide_resnet()
        elif model in ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
                       'vgg19_bn', 'vgg19']:
            net = self.vgg_model(model)
        elif model in ["alexnet"]:
            net = self.alexnet(dataset)
        elif model in ["nin"]:
            net = self.nin(num_classes=10)
        elif model in ["small-cnn"]:
            net = self.small_cnn(3, 10)
        else:
            return self.default()
        return net

    def simple_cnn(self, dataset, model):
        if model == "simple-cnn":
            if dataset in ["cifar10", "cinic10", "svhn"]:
                # return ModelZoo.simplecnn.SimpleCNN(input_dim=(16 * 5 * 5),
                #                                     hidden_dims=[120, 84], output_dim=10)
                return ModelZoo.simplecnn.CNNCifar()
            elif dataset in ["mnist", 'femnist', 'fmnist']:
                return ModelZoo.simplecnn.SimpleCNNMNIST(input_dim=(
                    16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            elif dataset == 'celeba':
                return ModelZoo.simplecnn.SimpleCNN(input_dim=(16 * 5 * 5),
                                                    hidden_dims=[120, 84], output_dim=2)

        elif model == "moderate-cnn":
            if dataset in ["mnist", 'femnist']:
                return ModelZoo.simplecnn.ModerateCNNMNIST()
            elif dataset in ["cifar10", "cinic10", "svhn"]:
                # print("in moderate cnn")
                return ModelZoo.simplecnn.ModerateCNN()
            elif dataset == 'celeba':
                return ModelZoo.simplecnn.ModerateCNN(output_dim=2)

    def resnet(self, model):
        """
        Only supports CIFAR10 dataset?
        """
        if model == "resnet18":
            return ModelZoo.resnetcifar.ResNet18_cifar10()
        elif model == "resnet50":
            return ModelZoo.resnetcifar.ResNet50_cifar10()

    def wide_resnet(self):
        """
        Only supports dataset with 10 classes.
        """
        return ModelZoo.wideresnet.WideResNet()

    def alexnet(self, dataset):
        return ModelZoo.alexnet.AlexNet(dataset=dataset)

    def vgg_model(self, model):
        return eval(f"ModelZoo.vggmodel.{model}()")

    def nin(self, num_classes):
        return ModelZoo.nin.NIN(num_classes=num_classes)

    def small_cnn(self, channels, classes):
        return ModelZoo.small_cnn.SmallCNN(num_channels=channels, num_classes=classes)

    def default(self):
        raise NotImplementedError


if __name__ == "__main__":
    net = InitNet().initial_nets("mnist", "vgg13")
    print(net)
