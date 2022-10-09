"""
config file, details about yacs: https://github.com/rbgirshick/yacs
"""

from yacs.config import CfgNode as CN

_C = CN()
# set up random seed.
_C.SEED = 3047
# ["mnist", "fmnist", "femnist", "cifar10", "cifar100", "svhn"]
_C.DATASET = "mnist"
# ["simple-cnn", "moderate-cnn", "vggs", "resnet18","resnet50", "alexnet", "wideresnet", "small-cnn", "nin"]
_C.MODEL = "vgg16"
# data partition strategy, more details: https://github.com/Xtra-Computing/NIID-Bench
_C.PARTITION = "noniid-labeldir"
# concentration number for dirichlet distribution
_C.BETA = 0.5
# the frequency to save models
_C.CKPT = 100
# whether to assign helpers to clients during the training process. If assigned, you might want to tweak the loss both in Loss floder and local updates.
_C.HELP = False

# Data Path and Log Path
_C.PATH = CN()
_C.PATH.DATADIR = "../data/"
_C.PATH.DISTILLDATA = "../data/distillation_data"
_C.PATH.LOGDIR = "./logs/"

_C.SERVER = CN()
# Number of remote clients
_C.SERVER.CLIENTS = 10
# Communication rounds
_C.SERVER.COMMU_ROUND = 50
# Aggregation Algorithms: ["FedAvg", "FedProx", "FedNova", "SCAFFOLD", "FedWAvg"]
_C.SERVER.ALGO = "FedAvg"
# Fraction of clients used for updating each communication round(float, 1.0)
_C.SERVER.SAMPLE = 1.0
# Scale factor in FedWAvg algorithm
_C.SERVER.SCALE_FACTOR = 1.0
# the frequency to assign new helpers for each client
_C.SERVER.HELP_ROUND = 1
# learning rate for distillation process, we keep distillation optimizer the same as local trianing optimizer, neglect weight decay
_C.SERVER.LR = 0.0001


_C.USER = CN()
# Number of local training epochs
_C.USER.EPOCHS = 5
# Number of Training batch size
_C.USER.BATCH_SIZE = 128
# Number of learning rate
_C.USER.LR = 0.001
# optimizer, ["SGD", "Adam", "Adagrad"]
_C.USER.OPTIMIZER = "SGD"
# momentum in SGD
_C.USER.MOMENTUM = 0.9
# Whethter to perform adversarial training and test.
_C.USER.ADV = True
# weight decay in optimizer
_C.USER.WEIGHT_DECAY = 0.0
# fraction of local data to do adversarial training
_C.USER.RATIO = 1.0
# The proximal term parameter for FedProx
_C.USER.MU = 1.0
# the number of helpers (<= number of clients)
_C.USER.HELPERS = 5


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
