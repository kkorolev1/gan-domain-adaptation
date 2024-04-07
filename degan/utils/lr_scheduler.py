import torch
import math

    
class CosineAnnealingWithWarmupLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup_steps: int, max_steps: int):
        self.warmup = warmup_steps
        self.max_steps = max_steps
        super().__init__(optimizer)

    def get_lr(self):
        return self.get_last_lr()

    def get_last_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + math.cos(math.pi * epoch / self.max_steps))
        lr_factor *= min(epoch / self.warmup, 1.0)
        return lr_factor