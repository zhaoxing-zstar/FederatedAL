from utils import *
import utils
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from config import get_cfg_defaults     # local variable usage pattern
import logging
import argparse
from cgitb import handler
import server
import user
import numpy as np
import json
import os
import random
import datetime
import sys
import torch
import pdb
os.sys.path.append(os.path.dirname(__file__))
writer = SummaryWriter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def select_server_algo(server, algo):
    algodict = {
        "FedAvg": server.fedAvg,
        "FedProx": server.fedProx,
        "SCAFFOLD": server.SCAFFOLD,
        "FedNova": server.fedNova,
        "FedWAvg": server.fedWAvg
    }
    func = algodict.get(algo)
    return func


def federated_train(cfg, logger):
    logger.info("Partitioning Data")
    # net_dataidx_map: partition results for training data
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        cfg.DATASET, cfg.PATH.DATADIR, cfg.PATH.LOGDIR, cfg.PARTITION, cfg.SERVER.CLIENTS, cfg.BETA
    )

    #  global dl: dataloader, global ds: dataset
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(
        cfg.DATASET, cfg.PATH.DATADIR, cfg.USER.BATCH_SIZE, cfg.USER.BATCH_SIZE
    )
    # distillation data loader
    # distill_dataset = utils.load_generate_data(
    #     cfg.PATH.DISTILLDATA, fraction=0.01, resolution=32)
    distill_dataset = utils.load_stl10_data(
        cfg.PATH.DISTILLDATA, resolution=32)
    distill_loader = torch.utils.data.DataLoader(
        distill_dataset, batch_size=64, shuffle=True, drop_last=False)

    logger.info("Initializaing Users")
    users = []
    for user_id in range(cfg.SERVER.CLIENTS):
        # cfg.USER.BATCH_SIZE, cfg.SERVER.CLIENTS,
        users.append(user.User(user_id, device, net_dataidx_map,
                     traindata_cls_counts, cfg))
        # momentum, cfg.DATASET, cfg.MODEL, device, net_dataidx_map, cfg.USER.ADV))

    the_server = server.Server(
        users, device, net_dataidx_map, cfg, test_dl_global, distill_loader)
    # test_size = len(the_server.test_loader.dataset)

    logger.info("Starting Training!")
    best_adv_accuracy = 0
    TEST_STEP = 1
    for cur_round in range(cfg.SERVER.COMMU_ROUND):
        logger.info("In communication round:"+str(cur_round))
        if cur_round % cfg.SERVER.HELP_ROUND == 0 and cfg.HELP:
            the_server.assign_helper_clients(cfg.USER.HELPERS)

        selected = np.random.choice(cfg.SERVER.CLIENTS, int(
            cfg.SERVER.CLIENTS*cfg.SERVER.SAMPLE), replace=False)
        if cur_round == cfg.CKPT:
            the_server.save_model(cur_round)

        algo_func = select_server_algo(the_server, cfg.SERVER.ALGO)
        algo_func(cur_round, selected)  # containing trian_clients.

        # if round % TEST_STEP == 0:
        #     test_loss, accuracy = the_server.test()
        #     print(
        #         f'Round: [{round+1}/{cfg.SERVER.COMMU_ROUND}] Average loss: {test_loss:.4f}, Accuracy: ({accuracy:.2f}%)')

        #     writer.add_scalars(
        #         'Natutal Test', {'Loss': test_loss, 'Acc': accuracy}, round)
        # logger.info(f"global model natural accuracy {accuracy}")
        if (cur_round % TEST_STEP == 0) and (cfg.USER.ADV):
            nat_accuracy, adv_accuracy, stability = the_server.adv_test()
            print(
                f'Round, [{cur_round+1}/{cfg.SERVER.COMMU_ROUND}], Natural accuracy, {nat_accuracy}, Robust accuracy, {adv_accuracy}, Perturbation Stability, {stability}')

            # save the model with best robust accuracy during training.
            if cur_round == 0:
                best_adv_accuracy = adv_accuracy
            if adv_accuracy >= best_adv_accuracy:
                the_server.save_model(cur_round, best=True)
                best_adv_accuracy = adv_accuracy

            writer.add_scalars(
                'Robust Test', {'Nat Acc': nat_accuracy, 'Rob Acc': adv_accuracy, 'Perturbation Stability': stability}, cur_round)
            logger.info(
                f'Global Round, [{cur_round+1}/{cfg.SERVER.COMMU_ROUND}], Natural accuracy, {nat_accuracy}, Robust accuracy, {adv_accuracy}, Perturbation Stability, {stability}')

        logger.info('global n_training: %d' % len(train_dl_global.dataset))
        logger.info('global n_test: %d' % len(test_dl_global.dataset))

        # train_acc = compute_accuracy(
        #     the_server.global_net, train_dl_global, device=device)
        # test_acc, conf_matrix = compute_accuracy(
        #     the_server.global_net, test_dl_global, get_confusion_matrix=True, device=device)

        # logger.info('>> Global Model Train accuracy: %f' % train_acc)
        # logger.info('>> Global Model Test accuracy: %f' % test_acc)

    writer.flush()

    the_server.save_model(cfg.SERVER.COMMU_ROUND)
    writer.close()


def get_args():
    """Argument Parsing Funtion

    Not needed anymore.
    """
    parser = argparse.ArgumentParser(
        description='Adversarial Robusteness in Federated Learning Learning')
    parser.add_argument('--partition', type=str, default='homo',
                        help="the data partitioning strategy")
    parser.add_argument("--algo", type=str, default="fedavg",
                        help="communication strategy")
    parser.add_argument('--logdir', type=str, required=False,
                        default="./logs/", help='Log directory path')
    parser.add_argument('-s', '--dataset', default='CIFAR10', choices=[
                        'MNIST', 'CIFAR10', 'CIFAR100', 'SVHN', 'FEMNIST'], help='image dataset')
    parser.add_argument('-m', '--model', default='VGG16',
                        choices=['VGG16', 'AlexNet', 'Resnet', 'WideResnet'], help='model architecture')
    parser.add_argument('-n', '--users-count', default=10,
                        type=int, help='number of clients')
    parser.add_argument("--comm-round", type=int, default=50,
                        help="number of maximum communication rounds")
    parser.add_argument('-b', '--batch_size', default=128,
                        type=int, help='batch_size')
    parser.add_argument('-e', '--epochs', default=200,
                        type=int, help='number of local epochs')
    parser.add_argument('-l', '--learning_rate', default=0.01,
                        type=float, help='initial learning rate')
    parser.add_argument('-r', '--resume', default=0, type=int,
                        help='resume training from some checkpoint')
    parser.add_argument('-ckpt', '--checkpoint', default=100,
                        type=int, help='create a checkpoint at this epoch')
    parser.add_argument('-adv', '--adversarial_training', default=0,
                        type=bool, help='whether to do adversarial training on client side')
    args = parser.parse_args()
    return args

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

if __name__ == "__main__":
    cfg = get_cfg_defaults()
    yaml_file = sys.argv[1]    # the yaml file to merge from, relative path
    cfg.merge_from_file(yaml_file)
    cfg.freeze()

    mkdirs(cfg.PATH.LOGDIR)
    logger = set_logger(cfg.PATH.LOGDIR, yaml_file)
    logger.info(cfg)
    logger.info("#" * 100)

    seed_everything(cfg.SEED)

    # args = get_args()
    federated_train(cfg, logger)
