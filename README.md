# Framework

General Code Structure:

[structure](./figures/structure.png)

# Attacks

To change the attack method, change the `torchattacks` method both in the mix_adversarial of server and users.

```python
attacker = torchattacks.PGD(self.net, eps=2/255,  alpha=2/255, steps=10, random_start=False)
```
