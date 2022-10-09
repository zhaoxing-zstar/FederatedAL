import os
import shutil
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import random
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
from datasets import MNIST_truncated, CIFAR10_truncated, SVHN_custom, FashionMNIST_truncated, CustomTensorDataset, CelebA_custom, FEMNIST, Generated, genData, ImageNetDS, IdxSubset
from math import sqrt

import torch.nn as nn

import torch.optim as optim
import torchvision.utils as vutils
import time
import random
import datetime
import sklearn.datasets as sk
from sklearn.datasets import load_svmlight_file

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def mkdirs(dirpath):
    """ make a new directory
    """
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def set_logger(logdir, yaml_file):
    """ set up a Logger
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logs_file_name = f'ExpLog-{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")}.log'
    logging.basicConfig(
        filename=os.path.join(logdir, logs_file_name),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    """get the train/test dataloader for different Datasets, data augmentation methods can be added here.
    """
    if dataset in ('mnist', 'femnist', 'fmnist', 'cifar10', 'svhn', 'generated'):
        if dataset == 'mnist':
            dl_obj = MNIST_truncated

            transform_train = transforms.Compose([
                transforms.ToTensor()])

            transform_test = transforms.Compose([
                transforms.ToTensor()])

        elif dataset == 'femnist':
            dl_obj = FEMNIST
            transform_train = transforms.Compose([
                transforms.ToTensor()])
            transform_test = transforms.Compose([
                transforms.ToTensor()])

        elif dataset == 'fmnist':
            dl_obj = FashionMNIST_truncated
            transform_train = transforms.Compose([
                transforms.ToTensor()])
            transform_test = transforms.Compose([
                transforms.ToTensor()])

        elif dataset == 'svhn':
            dl_obj = SVHN_custom
            transform_train = transforms.Compose([
                transforms.ToTensor()])
            transform_test = transforms.Compose([
                transforms.ToTensor()])

        elif dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor()])

        else:
            dl_obj = Generated
            transform_train = None
            transform_test = None

        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True,
                          transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False,
                         transform=transform_test, download=True)

        train_dl = data.DataLoader(
            dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
        test_dl = data.DataLoader(
            dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl, train_ds, test_ds


def load_mnist_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(
        datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(
        datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_fmnist_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = FashionMNIST_truncated(
        datadir, train=True, download=True, transform=transform)
    mnist_test_ds = FashionMNIST_truncated(
        datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_svhn_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    svhn_train_ds = SVHN_custom(
        datadir, train=True, download=True, transform=transform)
    svhn_test_ds = SVHN_custom(
        datadir, train=False, download=True, transform=transform)

    X_train, y_train = svhn_train_ds.data, svhn_train_ds.target
    X_test, y_test = svhn_test_ds.data, svhn_test_ds.target

    # X_train = X_train.data.numpy()
    # y_train = y_train.data.numpy()
    # X_test = X_test.data.numpy()
    # y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar10_data(datadir):

    test_transform = transforms.Compose([transforms.ToTensor()])
    train_transform = transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                ])

    cifar10_train_ds = CIFAR10_truncated(
        datadir, train=True, download=True, transform=train_transform)
    cifar10_test_ds = CIFAR10_truncated(
        datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_celeba_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    celeba_train_ds = CelebA_custom(
        datadir, split='train', target_type="attr", download=True, transform=transform)
    celeba_test_ds = CelebA_custom(
        datadir, split='test', target_type="attr", download=True, transform=transform)

    gender_index = celeba_train_ds.attr_names.index('Male')
    y_train = celeba_train_ds.attr[:, gender_index:gender_index+1].reshape(-1)
    y_test = celeba_test_ds.attr[:, gender_index:gender_index+1].reshape(-1)

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (None, y_train, None, y_test)


def load_femnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = FEMNIST(
        datadir, train=True, transform=transform, download=True)
    mnist_test_ds = FEMNIST(datadir, train=False,
                            transform=transform, download=True)

    X_train, y_train, u_train = mnist_train_ds.data, mnist_train_ds.targets, mnist_train_ds.users_index
    X_test, y_test, u_test = mnist_test_ds.data, mnist_test_ds.targets, mnist_test_ds.users_index

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    u_train = np.array(u_train)
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()
    u_test = np.array(u_test)

    return (X_train, y_train, u_train, X_test, y_test, u_test)


def load_stl10_data(datadir,  resolution, split="train"):
    is_trian = split == "train"

    transform = transforms.Compose(
        [torchvision.transforms.Resize((resolution, resolution))]
        + [transforms.RandomCrop(resolution, 4), transforms.RandomHorizontalFlip()]
        + [transforms.ToTensor()]
    )
    return datasets.STL10(
        root=datadir,
        split=split,
        transform=transform,
        download=True
    )


def load_generate_data(datadir, fraction=0.15,  resolution=32):
    # We will only use a fraction of fake data for distillation

    # define transform
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop((112, 112), 4)]
        + [transforms.Resize((resolution, resolution))]
        + [transforms.ToTensor()]
        + [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    # return the dataloader

    distill_dataset = ImageNetDS(root=datadir, transform=transform)
    n_distill = int(fraction * len(distill_dataset))
    return IdxSubset(distill_dataset, np.random.permutation(len(distill_dataset))[:n_distill], return_index=False)


# def record_net_data_stats(y_train, net_dataidx_map, logdir):

#     net_cls_counts = {}

#     for net_i, dataidx in net_dataidx_map.items():
#         unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
#         tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
#         net_cls_counts[net_i] = tmp

#     logger.info('Data statistics: %s' % str(net_cls_counts))

#     return net_cls_counts

def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts = {}
    # num_class = 100 if dataset == "cifar100" else 10
    num_class = np.unique(y_train).shape[0]
    # if dataset == "tiny":
    #     num_class = 20
    for net_i, dataidx in net_dataidx_map.items():  # label:sets
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)  # 去除数组中的重复数字，并进行排序之后输出。
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        for i in range(num_class):
            if i in tmp.keys():
                continue
            else:
                tmp[i] = 1  # 5

        net_cls_counts[net_i] = tmp

    print('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts

def partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.4):
    # np.random.seed(2020)
    # torch.manual_seed(2020)

    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    elif dataset == 'fmnist':
        X_train, y_train, X_test, y_test = load_fmnist_data(datadir)
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'svhn':
        X_train, y_train, X_test, y_test = load_svhn_data(datadir)
    elif dataset == 'celeba':
        X_train, y_train, X_test, y_test = load_celeba_data(datadir)
    elif dataset == 'femnist':
        X_train, y_train, u_train, X_test, y_test, u_test = load_femnist_data(
            datadir)
    elif dataset == 'generated':
        X_train, y_train = [], []
        for loc in range(4):
            for i in range(1000):
                p1 = random.random()
                p2 = random.random()
                p3 = random.random()
                if loc > 1:
                    p2 = -p2
                if loc % 2 == 1:
                    p3 = -p3
                if i % 2 == 0:
                    X_train.append([p1, p2, p3])
                    y_train.append(0)
                else:
                    X_train.append([-p1, -p2, -p3])
                    y_train.append(1)
        X_test, y_test = [], []
        for i in range(1000):
            p1 = random.random() * 2 - 1
            p2 = random.random() * 2 - 1
            p3 = random.random() * 2 - 1
            X_test.append([p1, p2, p3])
            if p1 > 0:
                y_test.append(0)
            else:
                y_test.append(1)
        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)
        y_test = np.array(y_test, dtype=np.int64)
        idxs = np.linspace(0, 3999, 4000, dtype=np.int64)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
        mkdirs("data/generated/")
        np.save("data/generated/X_train.npy", X_train)
        np.save("data/generated/X_test.npy", X_test)
        np.save("data/generated/y_train.npy", y_train)
        np.save("data/generated/y_test.npy", y_test)

    n_train = y_train.shape[0]

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    # Distribution-based label imbalance: according to Dirichlet distribution.
    elif partition == "noniid-labeldir":

        min_size = 0
        min_require_size = 10
        K = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            # min_require_size = 100

        N = y_train.shape[0]
        # np.random.seed(2020)
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                # Balance
                proportions = np.array(
                    [p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) *
                               len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,
                             idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        num = eval(partition[13:])
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            num = 1
            K = 2
        else:
            K = 10
        if num == 10:
            net_dataidx_map = {i: np.ndarray(
                0, dtype=np.int64) for i in range(n_parties)}
            for i in range(10):
                idx_k = np.where(y_train == i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, n_parties)
                for j in range(n_parties):
                    net_dataidx_map[j] = np.append(
                        net_dataidx_map[j], split[j])
        else:
            times = [0 for i in range(10)]
            contain = []
            for i in range(n_parties):
                current = [i % K]
                times[i % K] += 1
                j = 1
                while (j < num):
                    ind = random.randint(0, K-1)
                    if (ind not in current):
                        j = j+1
                        current.append(ind)
                        times[ind] += 1
                contain.append(current)
            net_dataidx_map = {i: np.ndarray(
                0, dtype=np.int64) for i in range(n_parties)}
            for i in range(K):
                idx_k = np.where(y_train == i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, times[i])
                ids = 0
                for j in range(n_parties):
                    if i in contain[j]:
                        net_dataidx_map[j] = np.append(
                            net_dataidx_map[j], split[ids])
                        ids += 1

    elif partition == "iid-diff-quantity":
        idxs = np.random.permutation(n_train)
        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = proportions/proportions.sum()
            min_size = np.min(proportions*len(idxs))
        proportions = (np.cumsum(proportions)*len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs, proportions)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "mixed":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            # min_require_size = 100

        N = y_train.shape[0]
        net_dataidx_map = {}

        times = [1 for i in range(10)]
        contain = []
        for i in range(n_parties):
            current = [i % K]
            j = 1
            while (j < 2):
                ind = random.randint(0, K-1)
                if (ind not in current and times[ind] < 2):
                    j = j+1
                    current.append(ind)
                    times[ind] += 1
            contain.append(current)
        net_dataidx_map = {i: np.ndarray(0, dtype=np.int64)
                           for i in range(n_parties)}

        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = proportions/proportions.sum()
            min_size = np.min(proportions*n_train)

        for i in range(K):
            idx_k = np.where(y_train == i)[0]
            np.random.shuffle(idx_k)

            proportions_k = np.random.dirichlet(np.repeat(beta, 2))
            #proportions_k = np.ndarray(0,dtype=np.float64)
            # for j in range(n_parties):
            #    if i in contain[j]:
            #        proportions_k=np.append(proportions_k ,proportions[j])

            proportions_k = (np.cumsum(proportions_k) *
                             len(idx_k)).astype(int)[:-1]

            split = np.split(idx_k, proportions_k)
            ids = 0
            for j in range(n_parties):
                if i in contain[j]:
                    net_dataidx_map[j] = np.append(
                        net_dataidx_map[j], split[ids])
                    ids += 1

    elif partition == "real" and dataset == "femnist":
        num_user = u_train.shape[0]
        user = np.zeros(num_user+1, dtype=np.int32)
        for i in range(1, num_user+1):
            user[i] = user[i-1] + u_train[i-1]
        no = np.random.permutation(num_user)
        batch_idxs = np.array_split(no, n_parties)
        net_dataidx_map = {i: np.zeros(0, dtype=np.int32)
                           for i in range(n_parties)}
        for i in range(n_parties):
            for j in batch_idxs[i]:
                net_dataidx_map[i] = np.append(
                    net_dataidx_map[i], np.arange(user[j], user[j+1]))

    elif partition == "real":
        stat = np.load("femnist-dis.npy")
        n_total = stat.shape[0]
        chosen = np.random.permutation(n_total)[:n_parties]
        stat = stat[chosen, :]

        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
        else:
            K = 10

        N = y_train.shape[0]
        # np.random.seed(2020)
        net_dataidx_map = {}

        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = stat[:, k]
            # logger.info("proportions2: ", proportions)
            proportions = proportions / proportions.sum()
            # logger.info("proportions3: ", proportions)
            proportions = (np.cumsum(proportions) *
                           len(idx_k)).astype(int)[:-1]
            # logger.info("proportions4: ", proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j,
                         idx in zip(idx_batch, np.split(idx_k, proportions))]

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(
        y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu"):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total = 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device, dtype=torch.int64)
                out = model(x)
                _, pred_label = torch.max(out.data, 1)

                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(
                        pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(
                        true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(
                        pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(
                        true_labels_list, target.data.cpu().numpy())

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct/float(total), conf_matrix

    return correct/float(total)


def compute_pi(datamap):
    # hard coding temporarily
    weight = [0]*10
    for k, v in datamap.items():
        weight[k] = v/sum(datamap.values())
    weight = np.array(weight) + 1e-7
    weight = torch.from_numpy(weight)
    return weight


def get_current_path(conf, rank):
    paths = conf.resume.split(",")
    splited_paths = map(lambda p: p.split("/")[-1].split("-")[:1], paths)
    splited_paths_dict = dict(
        [(path, paths[ind]) for ind, path in enumerate(splited_paths)]
    )
    return splited_paths_dict[str(rank)]


def build_dir(path, force):
    """build directory."""
    if os.path.exists(path) and force:
        shutil.rmtree(path)
        os.mkdir(path)
    elif not os.path.exists(path):
        os.mkdir(path)
    return path


def build_dirs(path):
    try:
        os.makedirs(path)
    except Exception as e:
        print(" encounter error: {}".format(e))


def remove_folder(path):
    try:
        shutil.rmtree(path)
    except Exception as e:
        print(" encounter error: {}".format(e))


def list_files(root_path):
    dirs = os.listdir(root_path)
    return [os.path.join(root_path, path) for path in dirs]
