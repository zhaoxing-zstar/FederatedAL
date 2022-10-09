# -*- encoding: utf-8 -*-
'''
@ description: perform attacks on trained models.
'''

# here put the import lib

import torch
import random
import argparse
from torch.autograd import Variable
import torch.optim as optim
import models
from torchvision import datasets, transforms
from datasets import MNIST_truncated, CIFAR10_truncated, SVHN_custom, FashionMNIST_truncated, CustomTensorDataset, CelebA_custom, FEMNIST, Generated, genData
import numpy as np
from tqdm import tqdm
import os
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from email import header
import torchattacks

parser = argparse.ArgumentParser(
    description='PyTorch CIFAR10 PGD Attack Evaluation')
parser.add_argument('-b', '--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('-d', '--dataset', default='cifar10', help='image dataset')
parser.add_argument('-m', '--model', default='vgg16',
                    help='model architecture')
parser.add_argument('-s', '--random-seed', type=int,
                    default=0, help="random seed")
parser.add_argument('--model-path',
                    default='./epoch_100.pth',
                    help='model for white-box attack evaluation')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(
    root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=args.test_batch_size, shuffle=False, num_workers= 4, drop_last=False)


def set_seed():
    seed = args.random_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def create_model():
    model = models.InitNet().initial_nets(args.dataset, args.model)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()
    # preprocessing = dict(mean = 0.1307, std = 0.3081)

    # bounds is useful if you woek with different models that have different bounds
    return model


def accuracy(model, inputs, labels):
    net_out = model(inputs)
    pred = net_out.data.max(1)[1]
    correct = pred.eq(labels.data).sum()
    return correct.item()


def preformAttack(fmodel):
    """
    evaluate model by white-box attack
    """
    attacks = [
        torchattacks.FGSM(model, eps=8/255),
        torchattacks.CW(model, c=2, kappa=0, steps=1000, lr=0.01),
        torchattacks.PGD(model, eps=8/255, alpha=2/255,
                         steps=20, random_start=True),
        torchattacks.MIFGSM(model, eps=8/255, steps=20, decay=1.0),
        torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=20),
        torchattacks.AutoAttack(
            model, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=args.random_seed, verbose=False)
        # torchattacks.DeepFool(model, steps=50, overshoot=0)
        # fa.L2CarliniWagnerAttack(binary_search_steps=3, initial_const=0.0001, steps=1000),
    ]
    attack_success = np.zeros((len(attacks), 2))
    for i, attack in enumerate(attacks):
        print(f"Attack Method: {attack}")
        natural_acc = []
        robust_acc = []
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            nat_acc = accuracy(model, images, labels)
            natural_acc.append(nat_acc)
            # check the meaning of each term at: https://foolbox.jonasrauber.de/guide/getting-started.html#multiple-epsilons
            adv_images = attack(images, labels)
            adv_images.to(device)
            rob_acc = accuracy(model, adv_images, labels)
            robust_acc.append(rob_acc)
        attack_success[i][0] = 100. * sum(natural_acc)/len(test_loader.dataset)
        attack_success[i][1] = 100. * sum(robust_acc)/len(test_loader.dataset)
        print(
            f"natural accuracy: {attack_success[i][0]}, robust accuracy: {attack_success[i][1]}")
    np.save('./attack_success.npy', attack_success)


if __name__ == "__main__":
    print(len(test_loader), len(test_loader.dataset))
    set_seed()
    model = create_model()
    preformAttack(model)
