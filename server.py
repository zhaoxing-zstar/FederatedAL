import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from utils import *
import copy
import models
import user
import pdb
import torchattacks
from typing import Optional, Any
from scipy.stats import truncnorm
from scipy import spatial
from tqdm import tqdm
from Loss.rslad import rslad_loss, iad_loss
from Loss.trades import trades_loss
import pdb

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.autograd.set_detect_anomaly(True)


class Server:
    def __init__(self, users, device, net_dataidx_map, cfg, test_dl_global, distill_loader):
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.users = users
        self.device = device
        self.num_clients = cfg.SERVER.CLIENTS
        self.net_dataidx_map = net_dataidx_map
        self.momentum = cfg.USER.MOMENTUM
        self.data_set = cfg.DATASET
        self.userepoch = cfg.USER.EPOCHS
        self.modelpath = cfg.PATH.LOGDIR
        self.model = cfg.MODEL
        self.net_dataidx_map = net_dataidx_map
        self.learning_rate = cfg.USER.LR
        # initialize global models, same as local model.
        self.global_net = models.InitNet().initial_nets(self.data_set, cfg.MODEL)
        self.global_net = self.global_net.to(device)
        self.test_loader = test_dl_global
        self.current_weights = self.global_net.state_dict()
        # control variates in SCAFFOLD algorithm
        self.c_global_net = models.InitNet().initial_nets(self.data_set, self.model)
        self.c_global_net.to(self.device)
        # d params in FedNova algorithm
        self.d_list = [copy.deepcopy(self.global_net.state_dict())
                       for i in range(self.num_clients)]
        self.d_total_round = copy.deepcopy(self.global_net.state_dict())
        # q: scale factor in FedWAvg algorithm
        self.scale_factor = cfg.SERVER.SCALE_FACTOR
        # distillation fake data loader
        self.distill_loader = distill_loader

        # optimizer for distillation
        self.distill_learning_rate = cfg.SERVER.LR
        self.optimizer = self.choose_optimizer(cfg.USER.OPTIMIZER)

    def save_model(self, rounds, best: Optional[bool] = False):
        if best:
            filename = "round_best.pth"
        else:
            filename = "round_{}".format(rounds) + '.pth'
        fileloc = os.path.join(self.modelpath, filename)

        with open(fileloc, 'wb') as file:
            torch.save(self.global_net.state_dict(), file)
        print("\n", "model saved")
        return

    def load_model(self, round):
        filename = "round_{}".format(round) + '.pth'
        fileloc = os.path.join(self.modelpath, filename)
        self.global_net.load_state_dict(torch.load(
            fileloc, map_location=torch.device(self.device)))
        self.current_weights = self.global_net.state_dict()
        return

    def calc_learning_rate(self, cur_round):
        """ learning rate scheduler
        Currently Unavailable
        """
        return self.learning_rate
        if cur_round * self.userepoch >= 150:
            return self.learning_rate*0.1
        elif cur_round * self.userepoch >= 200:
            return self.learning_rate*0.1*0.1
        return self.learning_rate

    def train_client(self, cur_round, selected):
        for usr in selected:
            self.users[usr].step(self.current_weights,
                                 self.calc_learning_rate(cur_round))

    def collect_gradients(self, selected, defined_weight: Optional[Any] = None):
        """
        collect local gradients and update global model
        """
        # if defined_weight.any():
        #     weight = defined_weight
        # else:
        weight = [len(self.net_dataidx_map[i]) for i in selected]
        weight = np.array(weight) / sum(weight)
        selectedUsr = [self.users[usr] for usr in selected]
        for idx, usr in enumerate(selectedUsr):
            net_para = usr.net.cpu().state_dict()
            if idx == 0:
                for key in net_para:
                    self.current_weights[key] = net_para[key] * weight[idx]
            else:
                for key in net_para:
                    self.current_weights[key] += net_para[key] * weight[idx]
        # copying all the parameters
        self.global_net.load_state_dict(self.current_weights)

    def fedAvg(self, cur_round, selected):
        for usr in selected:
            self.users[usr].step(self.current_weights,
                                 self.calc_learning_rate(cur_round),
                                 cur_round,
                                 self.global_net)

        # use average results as a starting point for distillation
        self.collect_gradients(selected)
        # nat_accuracy, adv_accuracy, stability = self.adv_test()
        # logger.info(
        #     f'Before Distillation, Global Round, [{cur_round+1}], Natural accuracy, {nat_accuracy}, Robust accuracy, {adv_accuracy}, Perturbation Stability, {stability}')
        self.distill(selected, epochs=5, num_classes=10)

    def fedProx(self, cur_round, selected):
        for usr in selected:
            self.users[usr].step(self.current_weights,
                                 self.calc_learning_rate(cur_round),
                                 cur_round,
                                 self.global_net)
        self.collect_gradients(selected)

    def SCAFFOLD(self, cur_round, selected):
        total_delta = copy.deepcopy(self.global_net.state_dict())
        for key in total_delta:
            total_delta[key] = 0.0

        for usr in selected:
            self.users[usr].step(self.current_weights,
                                 self.calc_learning_rate(cur_round),
                                 cur_round,
                                 self.global_net,
                                 self.c_global_net)
            for key in total_delta:
                total_delta[key] += self.users[usr].c_delta_para[key]

        # update the global c_net
        for key in total_delta:
            total_delta[key] /= len(selected)
        c_global_para = self.c_global_net.state_dict()
        for key in c_global_para:
            if c_global_para[key].type() == 'torch.LongTensor':
                c_global_para[key] += total_delta[key].type(torch.LongTensor)
            elif c_global_para[key].type() == 'torch.cuda.LongTensor':
                c_global_para[key] += total_delta[key].type(
                    torch.cuda.LongTensor)
            else:
                # print(c_global_para[key].type())
                c_global_para[key] += total_delta[key]
        self.c_global_net.load_state_dict(c_global_para)
        self.collect_gradients(selected)

    def fedNova(self, cur_round, selected):
        for key in self.d_total_round:
            self.d_total_round[key] = 0
        a_list = []
        d_list = []
        n_list = []
        for usr in selected:
            self.users[usr].step(self.current_weights,
                                 self.calc_learning_rate(cur_round),
                                 cur_round,
                                 self.global_net)
            a_list.append(self.users[usr].a_i)
            d_list.append(self.users[usr].norm_grad)
            n_list.append(len(self.users[usr].train_loader.dataset))
        for i in range(len(selected)):
            d_para = d_list[i]
            for key in d_para:
                self.d_total_round[key] += d_para[key] * \
                    n_list[i] / sum(n_list)

        coeff = 0.0
        for i in range(len(selected)):
            coeff += a_list[i] * n_list[i] / sum(n_list)

        # update the global model
        updated_model = self.global_net.state_dict()
        for key in updated_model:
            # print(updated_model[key])
            if updated_model[key].type() == 'torch.LongTensor':
                updated_model[key] -= (coeff * self.d_total_round[key]
                                       ).type(torch.LongTensor)
            elif updated_model[key].type() == 'torch.cuda.LongTensor':
                updated_model[key] -= (coeff * self.d_total_round[key]
                                       ).type(torch.cuda.LongTensor)
            else:
                updated_model[key] -= coeff * self.d_total_round[key]
        self.current_weights = updated_model
        self.global_net.load_state_dict(updated_model)

    def fedWAvg(self, cur_round, selected):
        # cosine similarity list
        c_k = []
        for usr in selected:
            self.users[usr].step(self.current_weights,
                                 self.calc_learning_rate(cur_round),
                                 cur_round)
            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
            c_k.append(cos(list(self.global_net.parameters())[-1],
                           list(self.users[usr].net.parameters())[-1]).item())
        print(c_k)
        weight = F.softmax(self.scale_factor * torch.Tensor(c_k), dim=0)
        print(weight)
        self.collect_gradients(selected, weight)

    def test(self):
        test_loss = 0
        correct = 0

        self.global_net.eval()
        with torch.no_grad():
            for data, target in self.test_loader:

                data, target = data.to(self.device), target.to(self.device)
                net_out = self.global_net(data)
                loss = self.criterion(net_out, target)
                test_loss += loss.data.item()
                pred = net_out.data.max(1)[1]
                correct += pred.eq(target.data).sum()

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * float(correct) / len(self.test_loader.dataset)

        return test_loss, accuracy

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
        self.global_net.eval()
        for data, target in self.test_loader:
            data, target = data.to(self.device), target.to(self.device)
            adv_data = self.mix_adversarial(data, target)

            net_pred = self.global_net(data).data.max(1)[1]
            net_adv_pred = self.global_net(adv_data).data.max(1)[1]
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
        attacker = torchattacks.PGD(
            self.global_net, eps=8/255, alpha=2/255, steps=10, random_start=False)
        # attacker = torchattacks.AutoAttack(
        #     self.global_net, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False)
        x_adv = attacker(x, y)
        return x_adv

    def assign_helper_clients(self, helpers: int):
        for usr in self.users:
            usr.helper_clients = {}

        # generate random gauss noise.
        mu, std, lower, upper = 125, 125, 0, 255
        random_gauss = truncnorm(
            (lower - mu)/std, (upper - mu)/std, loc=mu, scale=std).rvs((1, 3, 32, 32))
        random_gauss = random_gauss.astype(np.float32)/255
        random_gauss = torch.from_numpy(random_gauss).float().to(self.device)

        embed_result = {}
        for id, usr in enumerate(self.users):
            usr.net.to(self.device)
            embed_result[id] = np.squeeze(
                usr.net(random_gauss).detach().cpu().numpy())

        # construct the KDtree for nearest neighbor search
        tree = spatial.KDTree(list(embed_result.values()))

        # Nearest
        for id, usr in enumerate(self.users):
            user_cout = embed_result[id]
            similar_clients = tree.query(user_cout, helpers+1)
            for sim_id in similar_clients[1]:
                if sim_id == id:
                    continue
                usr.helper_clients[sim_id] = self.users[sim_id].net

        # Furthest
        # for id, usr in enumerate(self.users):
        #     user_cout = embed_result[id]
        #     similar_clients = tree.query(user_cout, self.num_clients)
        #     for sim_id in similar_clients[1][:-helpers-1:-1]:
        #         if sim_id == id:
        #             continue
        #         usr.helper_clients[sim_id] = self.users[sim_id].net

    def choose_optimizer(self, optimizer):
        if optimizer == 'Adam':
            return optim.Adam(self.global_net.parameters(), lr=self.distill_learning_rate)
        elif optimizer == 'Adagrad':
            return optim.Adagrad(self.global_net.parameters(), lr=self.distill_learning_rate)
        elif optimizer == 'SGD':
            return optim.SGD(self.global_net.parameters(), lr=self.distill_learning_rate, momentum=self.momentum)

    # def distill(self, selected, epochs, num_classes):
    #     # choose the global distillation epochs and number of classes
    #     print("Distilling global model")
    #     self.global_net.train()

    #     for epoch in range(epochs):
    #         epoch_loss = []
    #         for data, target in self.distill_loader:
    #             # target won't be used in distillation process
    #             data = data.to(self.device)
    #             teacher_logits = torch.zeros([data.shape[0], num_classes],
    #                                          device=self.device)
    #             for usr in selected:
    #                 self.users[usr].net.to(self.device)
    #                 y_c = self.users[usr].predict_logit(data)
    #                 teacher_logits += (y_c/len(selected)).detach()
    #             # loss = rslad_loss(self.global_net,
    #             #                   teacher_logits,
    #             #                   data,
    #             #                   self.optimizer,
    #             #                   step_size=2/255,
    #             #                   epsilon=8/255,
    #             #                   perturb_steps=10,
    #             #                   alpha=5.0/6.0)

    #             # loss = iad_loss(self.global_net,
    #             #                 teacher_logits,
    #             #                 data,
    #             #                 self.optimizer,
    #             #                 step_size=2/255,
    #             #                 epsilon=8/255,
    #             #                 perturb_steps=10,
    #             #                 alpha=0.8)
    #             teacher_logits = nn.Softmax(1)(teacher_logits)
    #             y_ = nn.LogSoftmax(1)(self.global_net(data))
    #             self.optimizer.zero_grad()
    #             loss = torch.nn.KLDivLoss(reduction="batchmean")(y_, teacher_logits.detach())
    #             loss.backward()
    #             self.optimizer.step()
    #             epoch_loss.append(loss.item())
    #         epoch_loss = sum(epoch_loss) / len(epoch_loss)
    #         logger.info(f"Distillation Epoch: {epoch} Loss: {epoch_loss}")
    #     # update self.current_weights after distillation
    #     self.current_weights = self.global_net.state_dict()

    
    def distill(self, selected, epochs, num_classes):
        # choose the global distillation epochs and number of classes
        print("Distilling global model")
        self.global_net.train()

        for epoch in range(epochs):
            epoch_loss = []
            for data, target in self.distill_loader:
                # target won't be used in distillation process
                data = data.to(self.device)

                # generate pseudo labels for unlabeled data
                y_ = self.global_net(data).max(1)[1]
                self.optimizer.zero_grad()
                loss = trades_loss(self.global_net,
                                    data,
                                    y_,
                                    self.optimizer,
                                    step_size=2/255,
                                    epsilon=8/255,
                                    perturb_steps=10,
                                    beta=1.0,
                                    distance='l_inf')
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.item())
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            logger.info(f"Distillation Epoch: {epoch} Loss: {epoch_loss}")
        # update self.current_weights after distillation
        self.current_weights = self.global_net.state_dict()
