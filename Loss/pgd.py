import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def madry_loss(model,
               x_natural,
               y,
               optimizer,
               step_size=2/255,
               epsilon=8/255,
               perturb_steps=10,
               beta=1.0):
    """
    This is CE(f(x'), y) if beta = 1.0
    """
    # define KL-loss
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            logits = model(x_adv)
            loss_kl = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural -
                          epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    optimizer.zero_grad()

    adv_loss = F.cross_entropy(model(x_adv), y, reduction='none')
    nat_loss = F.cross_entropy(model(x_natural), y, reduction='none')
    two_loss = torch.stack([adv_loss, nat_loss], dim=1)
    large_loss, _ = torch.max(two_loss, dim=1)
    regu_loss = torch.mean(large_loss - nat_loss)
    nat_loss = torch.mean(nat_loss)

    return nat_loss + beta * regu_loss  # , nat_loss, regu_loss * beta, x_adv,0
