# -*- encoding: utf-8 -*-
'''
@ description: perform attacks on trained models.
'''

# here put the import lib

import torch
import argparse
from torch.autograd import Variable
import torch.optim as optim
import models
from torchvision import datasets, transforms
import foolbox.attacks as fa
import foolbox as fb
import numpy as np
from tqdm import tqdm
import os
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from email import header
from fastapi import Header


header

parser = argparse.ArgumentParser(
    description='PyTorch CIFAR10 PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model-path',
                    default='./epoch_100.pth',
                    help='model for white-box attack evaluation')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(), ])
testset = torchvision.datasets.CIFAR10(
    root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def create():
    model = models.Cifar10Net('VGG16').to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    # preprocessing = dict(mean = 0.1307, std = 0.3081)

    # bounds is useful if you woek with different models that have different bounds
    fmodel = fb.models.PyTorchModel(model, bounds=(0, 1))
    return fmodel


def preformAttack(fmodel):
    """
    evaluate model by white-box attack
    """
    attacks = [
        fa.LinfFastGradientAttack(),
        fa.LinfProjectedGradientDescentAttack(steps=20, random_start=True),
        fa.LinfDeepFoolAttack(),
        # fa.L2CarliniWagnerAttack(binary_search_steps=3, initial_const=0.0001, steps=1000),
    ]
    attack_success = np.zeros((len(attacks), 2))
    for i, attack in enumerate(attacks):
        print(f"Attack Method: {attack}")
        natural_acc = []
        robust_acc = []
        for images, labels in tqdm(test_loader, position=0, leave=True):
            images, labels = next(iter(test_loader))
            images, labels = images.to(device), labels.to(device)
            nat_acc = fb.utils.accuracy(fmodel, images, labels)
            natural_acc.append(nat_acc)
            # check the meaning of each term at: https://foolbox.jonasrauber.de/guide/getting-started.html#multiple-epsilons
            raw, clipped, is_adv = attack(
                fmodel, images, labels, epsilons=0.031)
            rob_acc = 1 - is_adv.detach().cpu().numpy().mean(axis=-1)
            robust_acc.append(rob_acc)
            # print(f"natural accuracy: {nat_acc}, robust accuracy: {rob_acc}")
        attack_success[i][0] = sum(natural_acc)/len(test_loader)
        attack_success[i][1] = sum(robust_acc)/len(test_loader)
    np.save('./attack_success.npy', attack_success)


if __name__ == "__main__":
    # fmodel = create()
    # preformAttack(fmodel)
    print(np.load('./attack_success.npy'))
    # images, labels = samples(fmodel, dataset='mnist', batchsize=128)
    # print(accuracy(fmodel, images, labels))
