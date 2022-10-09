import torch
import torch.nn as nn
import torch.nn.functional as F


class PGDAttack:
    def __init__(self, model, epsilon, attack_steps, attack_lr, random_start=True):
        self.model = model
        self.epsilon = epsilon
        self.attack_steps = attack_steps
        self.attack_lr = attack_lr
        self.rand = random_start
        self.clamp = (0,1)

    def random_init(self, x):
        x = x + (torch.rand_like(x) * 2 * self.epsilon - self.epsilon)
        x = torch.clamp(x,*self.clamp)
        return x

    def perturb(self, x, y):
        x_adv = x.detach().clone()

        if self.rand:
            x_adv = self.random_init(x_adv)

        for i in range(self.attack_steps):
            x_adv.requires_grad = True
            logits = self.model(x_adv)
            self.model.zero_grad()
            
            loss = F.cross_entropy(logits, y,  reduction="sum")
            loss.backward()
            with torch.no_grad():                      
                grad = x_adv.grad
                grad = grad.sign()
                x_adv = x_adv + self.attack_lr * grad
                
                # Projection
                noise = torch.clamp(x_adv - x, min=-self.epsilon, max=self.epsilon)
                x_adv = torch.clamp(x + noise, min=0, max=1)

        return x_adv