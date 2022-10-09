import functools
import torch
import torchattacks
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
import models
import copy
from typing import Optional, Any
from utils import *
import pdb
from Loss.trades import trades_loss, trades_variant, trades_global
from Loss.pgd import madry_loss
from Loss.mart import mart_loss
from Loss.calibrated import calibrated_loss, calibrated_global


def flatten_params(params):
    return np.concatenate([i.data.cpu().numpy().flatten() for i in params])


def row_into_parameters(row, parameters):
    offset = 0
    for param in parameters:
        new_size = functools.reduce(lambda x, y: x*y, param.shape)
        current_data = row[offset:offset + new_size]

        param.data[:] = torch.from_numpy(current_data.reshape(param.shape))
        offset += new_size

# the data loader will be given here, and optimizer, criterion(loss function).


class User:
    def __init__(self, user_id, device, net_dataidx_map, traindata_cls_counts, cfg):
        self.user_id = user_id
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.learning_rate = cfg.USER.LR
        self.grads = None
        self.epochs = cfg.USER.EPOCHS
        self.weight_decay = cfg.USER.WEIGHT_DECAY
        self.data_set = cfg.DATASET
        self.adversarial_training = cfg.USER.ADV
        self.momentum = cfg.USER.MOMENTUM
        self.ratio = cfg.USER.RATIO
        self.mu = cfg.USER.MU
        self.device = device
        self.model = cfg.MODEL
        self.toHelp = cfg.HELP
        self.net = models.InitNet().initial_nets(
            self.data_set, cfg.MODEL)   # initial local networks
        self.c_net = models.InitNet().initial_nets(
            self.data_set, self.model)   # initial local c_i network
        self.c_net = self.c_net.to(self.device)
        self.net = self.net.to(device)
        self.optimizer = self.choose_optimizer(cfg.USER.OPTIMIZER)
        self.fl_algo = cfg.SERVER.ALGO
        self.original_params = None
        dataidxs = net_dataidx_map[self.user_id]
        self.train_loader, self.test_loader, _, _ = get_dataloader(
            cfg.DATASET, cfg.PATH.DATADIR, cfg.USER.BATCH_SIZE, 32, dataidxs)
        # try assign helper clients
        self.helper_clients = {}
        # set \pi_i for CalFAT
        # self.pi_weight = compute_pi(traindata_cls_counts[self.user_id])
        # self.pi_weight = self.pi_weight.to(self.device)
        weight_dict = dict(sorted(traindata_cls_counts[self.user_id].items()))
        weight_lst = list(weight_dict.values())
        # weight_mean = np.mean(weight_lst)
        # weight_lst = [(weight_mean/weight_lst[i])**(1/4) if weight_lst[i]<weight_mean else (weight_mean/weight_lst[i])**(1/2) for i in range(len(weight_lst))]
        self.pi_weight = torch.cuda.FloatTensor(weight_lst)
<<<<<<< HEAD
=======

    # def local_train_fedavg(self, global_net):
    #     for epoch in range(self.epochs):
    #         epoch_loss = []
    #         for data, target in (self.train_loader):
    #             data, target = data.to(self.device), target.to(self.device)
    #             # if self.adversarial_training:
    #             #     data = self.mix_adversarial(data, target)
    #             # net_out = self.net(data)
    #             # loss = self.criterion(net_out, target)
    #             loss = calibrated_loss(self.net,
    #                                    data,
    #                                    target,
    #                                    self.optimizer,
    #                                    self.pi_weight,
    #                                    step_size=2/255,
    #                                    epsilon=8/255,
    #                                    perturb_steps=10,
    #                                    beta=1.0,
    #                                    distance='l_inf')
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()
    #             epoch_loss.append(loss.item())

    #         epoch_loss = sum(epoch_loss) / len(epoch_loss)
    #         logger.info(f'Epoch: {epoch} Loss: {epoch_loss}')
>>>>>>> 28f221242897b67bcbce7ef764cbb7f2046f1945

    def local_train_fedavg(self, global_net):
        for epoch in range(self.epochs):
            epoch_loss = []
            for data, target in (self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                # if self.adversarial_training:
<<<<<<< HEAD
                #     data = self.mix_adversarial(data, target)
                # net_out = self.net(data)
                # loss = self.criterion(net_out, target)
                loss = calibrated_global(self.net,
=======
                #     adv_data = self.mix_adversarial(data, target)
                if self.toHelp:
                    loss_natural, loss_robust, loss_commute = trades_global(self.net,
                                                                            global_net,
                                                                            data,
                                                                            target,
                                                                            self.optimizer,
                                                                            step_size=2/255,
                                                                            epsilon=8/255,
                                                                            perturb_steps=10,
                                                                            beta=0.0,
                                                                            mu=1.0,
                                                                            distance='l_inf')
                else:
                    # net_out = self.net(adv_data)
                    loss = trades_loss(self.net,
>>>>>>> 28f221242897b67bcbce7ef764cbb7f2046f1945
                                       data,
                                       target,
                                       self.optimizer,
                                       global_net,
                                       self.pi_weight,
                                       step_size=2/255,
                                       epsilon=8/255,
                                       perturb_steps=10,
                                       mu=1.0,
                                       distance='l_inf')
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.item())

            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            logger.info(f'Epoch: {epoch} Loss: {epoch_loss}')

    # def local_train_fedavg(self, global_net):
    #     for epoch in range(self.epochs):
    #         epoch_loss = []
    #         epoch_natural_loss = []
    #         epoch_robust_loss = []
    #         epoch_commute_loss = []
    #         for data, target in (self.train_loader):
    #             data, target = data.to(self.device), target.to(self.device)
    #             # if self.adversarial_training:
    #             #     adv_data = self.mix_adversarial(data, target)
    #             if self.toHelp:
    #                 loss_natural, loss_robust, loss_commute = trades_global(self.net,
    #                                                                         global_net,
    #                                                                         data,
    #                                                                         target,
    #                                                                         self.optimizer,
    #                                                                         step_size=2/255,
    #                                                                         epsilon=8/255,
    #                                                                         perturb_steps=10,
    #                                                                         beta=0.0,
    #                                                                         mu=1.0,
    #                                                                         distance='l_inf')
    #             else:
    #                 # net_out = self.net(adv_data)
    #                 loss = trades_loss(self.net,
    #                                    data,
    #                                    target,
    #                                    self.optimizer,
    #                                    step_size=2/255,
    #                                    epsilon=8/255,
    #                                    perturb_steps=10,
    #                                    beta=6.0,
    #                                    distance='l_inf')
    #                 # loss = self.criterion(net_out, target)
    #             loss = loss_natural + loss_robust + loss_commute
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()
    #             epoch_loss.append(loss.item())
    #             # epoch_natural_loss.append(loss_natural.item())
    #             # epoch_robust_loss.append(loss_robust.item())
    #             # epoch_commute_loss.append(loss_commute.item())

    #         epoch_loss = sum(epoch_loss) / len(epoch_loss)

    #         # if epoch == (self.epochs-1) and self.user_id == 1:
    #         #     epoch_natural_loss = sum(
    #         #         epoch_natural_loss) / len(epoch_natural_loss)
    #         #     epoch_robust_loss = sum(
    #         #         epoch_robust_loss)/len(epoch_robust_loss)
    #         #     epoch_commute_loss = sum(
    #         #         epoch_commute_loss)/len(epoch_commute_loss)
    #         #     logger.info(
    #         #         f'User id, {self.user_id}, Natural_loss, {epoch_natural_loss}, Robust_loss, {epoch_robust_loss}, Commmute_loss, {epoch_commute_loss}')
            logger.info(f'Epoch: {epoch} Loss: {epoch_loss}')

    def local_train_fedprox(self, global_net):
        proximal_term = 0.0
        for w, w_t in zip(self.net.parameters(), global_net.parameters()):
            proximal_term += torch.pow(
                torch.norm(w.data - w_t.data), 2)
        for epoch in range(self.epochs):
            epoch_loss = []
            for data, target in (self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                target = target.long()
                if self.adversarial_training:
                    data = self.mix_adversarial(data, target)
                net_out = self.net(data)
                loss = self.criterion(net_out, target)
                fed_prox_reg = 0.0
                for param_index, param in enumerate(self.net.parameters()):
                    fed_prox_reg += ((self.mu / 2) * torch.norm((param -
                                                                 list(global_net.parameters())[param_index]))**2)
                loss += fed_prox_reg
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.item())
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            logger.info(f'Epoch: {epoch} Loss: {epoch_loss}')

    def local_train_scaffold(self, global_net, c_global, cur_round):
        if cur_round == 0:
            # initialized c_i as c_global in the first round.
            self.c_net.load_state_dict(c_global.state_dict())

        cnt = 0
        c_global_para = c_global.state_dict()
        c_local_para = self.c_net.state_dict()
        for epoch in range(self.epochs):
            epoch_loss = []
            for data, target in (self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                if self.adversarial_training:
                    data = self.mix_adversarial(data, target)
                net_out = self.net(data)
                loss = self.criterion(net_out, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                net_para = self.net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - self.learning_rate * \
                        (c_global_para[key] - c_local_para[key])
                self.net.load_state_dict(net_para)

                cnt += 1
                epoch_loss.append(loss.item())
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            logger.info(f'Epoch: {epoch} Loss: {epoch_loss}')
        c_new_para = self.c_net.state_dict()
        self.c_delta_para = copy.deepcopy(self.c_net.state_dict())
        global_model_para = global_net.state_dict()
        net_para = self.net.state_dict()
        for key in net_para:
            c_new_para[key] = c_new_para[key] - c_global_para[key] + \
                (global_model_para[key] - net_para[key]) / \
                (cnt * self.learning_rate)
            self.c_delta_para[key] = c_new_para[key] - c_local_para[key]
        self.c_net.load_state_dict(c_new_para)

    def local_train_fednova(self, global_net, cur_round):
        """
        This FedNova algorithm only works with SGD optimizer
        """
        tau = 0
        for epoch in range(self.epochs):
            epoch_loss = []
            for data, target in (self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                if self.adversarial_training:
                    data = self.mix_adversarial(data, target)
                net_out = self.net(data)
                loss = self.criterion(net_out, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                tau += 1
                epoch_loss.append(loss.item())
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            logger.info(f'Epoch: {epoch} Loss: {epoch_loss}')
        self.a_i = (tau - self.momentum * (1 - pow(self.momentum, tau)) /
                    (1 - self.momentum)) / (1 - self.momentum)
        global_model_para = global_net.state_dict()
        net_para = self.net.state_dict()
        self.norm_grad = copy.deepcopy(global_net.state_dict())
        for key in self.norm_grad:
            self.norm_grad[key] = torch.true_divide(
                global_model_para[key]-net_para[key], self.a_i)

    def step(self, current_params, learning_rate, cur_round,  global_net: Optional[Any] = None, c_global: Optional[Any] = None):
        """
        step function is directly called from server side.
        notice that test loader is the same between server and clients.
        """
        self.optimizer.param_groups[0]["lr"] = learning_rate
        self.net.load_state_dict(current_params)
        self.net.to(self.device)
        for helper in self.helper_clients.values():
            helper.to(self.device)
        logger.info(
            f"Training Network {self.user_id}, Number of training samples:{len(self.train_loader.dataset)}, Number of testing samples: {len(self.test_loader.dataset)}")
        # train_acc = compute_accuracy(
        #     self.net, self.train_loader, device=self.device)
        # test_acc, conf_matrix = compute_accuracy(
        #     self.net, self.test_loader, get_confusion_matrix=True, device=self.device)
        # logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
        # logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

        if self.fl_algo == "FedAvg":
            self.local_train_fedavg(global_net)
        elif self.fl_algo == "FedProx":
            self.local_train_fedprox(global_net)
        elif self.fl_algo == "SCAFFOLD":
            self.local_train_scaffold(global_net, c_global, cur_round)
        elif self.fl_algo == "FedNova":
            self.local_train_fednova(global_net, cur_round)
        elif self.fl_algo == "FedWAvg":
            self.local_train_fedavg()

        # test_nat_accuracy, test_adv_accuracy, test_smooth_accuracy = self.adv_test()

        # # train_acc = compute_accuracy(
        # #     self.net, self.train_loader, device=self.device)
        # # test_acc, conf_matrix = compute_accuracy(
        # # self.net, self.test_loader, get_confusion_matrix=True, device=self.device)

        # # logger.info('>> Training accuracy: %f' % train_acc)
        # # logger.info('>> Test accuracy: %f' % test_nat_accuracy)
        # # logger.info('>> Robust test loss: %f' % test_loss)
        # # logger.info(
        # #     f'>> Network: {self.user_id} Robust accuracy under PGD attack: {adv_accuracy}')
        # logger.info(
        #     f'>> Network, {self.user_id}, Natural accuracy, {test_nat_accuracy}, Robust accuracy, {test_adv_accuracy}, Perturbation stability, {test_smooth_accuracy}')

        logger.info(' ** Training complete **')

    def choose_optimizer(self, optimizer):
        if optimizer == 'Adam':
            return optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif optimizer == 'Adagrad':
            return optim.Adagrad(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif optimizer == 'SGD':
            return optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)

    def adv_test(self):
        """check accuracy in test dataset

        return:
            acc_nat: natural accuracy
            acc_adv: robust accuracy
            stability: perturbation stability, defined at: https://arxiv.org/abs/2010.01279
        """
        robust_right_total = 0
        natural_right_total = 0
        smooth_right_total = 0
        self.net.eval()
        for data, target in self.test_loader:
            data, target = data.to(self.device), target.to(self.device)
            adv_data = self.mix_adversarial(data, target)

            net_pred = self.net(data).data.max(1)[1]
            net_adv_pred = self.net(adv_data).data.max(1)[1]
            natural_right_total += net_pred.eq(target.data).sum()
            robust_right_total += net_adv_pred.eq(target.data).sum()
            smooth_right_total += net_adv_pred.eq(net_pred.data).sum()

        acc_nat = 100. * float(natural_right_total) / \
            len(self.test_loader.dataset)
        acc_adv = 100. * float(robust_right_total) / \
            len(self.test_loader.dataset)
        stability = 100. * float(smooth_right_total) / \
            len(self.test_loader.dataset)
        return acc_nat, acc_adv, stability

    def mix_adversarial(self, x, y):
        # Replacing the cfg.USER.RATIO of the examples by Adversarial Examples
        rand_perm = np.random.choice(
            x.size(0), int(x.size(0)*self.ratio), replace=False)
        x_adv, y_adv = x[rand_perm, :], y[rand_perm]

        # Vary the PGD attack parameters
        attacker = torchattacks.PGD(
            self.net, eps=8/255,  alpha=2/255, steps=10, random_start=False)
        # attacker = torchattacks.AutoAttack(
        #     self.net, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False)
        x_adv = attacker(x_adv, y_adv)
        x_tmp = x.clone().detach()
        x_tmp[rand_perm, :] = x_adv
        return x_tmp

    def predict_logit(self, input):
        # for ditillation process
        self.net.train()
        with torch.no_grad():
            y_ = self.net(input)
        return y_
