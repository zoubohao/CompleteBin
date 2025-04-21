import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_warm_train_epoch, restart_epoch=None):
        self.multiplier = multiplier
        self.warm_epoch = warm_epoch
        if restart_epoch is None:
            if after_warm_train_epoch % 2 != 0:
                restart_epoch = after_warm_train_epoch // 2 + 1
            else:
                restart_epoch = after_warm_train_epoch // 2
        self.after_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=restart_epoch)
        self.finished = False
        self.last_epoch = None
        self.base_lrs = None
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.warm_epoch:
            if self.finished is False:
                self.after_scheduler.base_lrs = [
                    base_lr * self.multiplier for base_lr in self.base_lrs]
                self.after_scheduler.step()
                self.finished = True
            return self.after_scheduler.get_last_lr()
        return [base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.warm_epoch + 1.0) for base_lr in self.base_lrs]

    def step(self):
        if self.finished and self.after_scheduler:
            self.after_scheduler.step(None)
        else:
            return super(GradualWarmupScheduler, self).step(None)
