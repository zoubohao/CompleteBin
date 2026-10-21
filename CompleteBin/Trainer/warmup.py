import math

import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupSchedule:
    """warmup + 余弦退火，学习率是"第几个 epoch"的**纯函数**。

    lr(第 e 个 epoch，0 基) = base_lr × factor(e)，其中

        e < warmup_epochs:  factor = warmup_scale × (e + 1) / warmup_epochs
        e >= warmup_epochs: factor = warmup_scale × 0.5 × (1 + cos(π × progress))
                            progress = (e - warmup_epochs + 1) / (total_epochs - warmup_epochs)

    也就是：前 `warmup_epochs` 个 epoch 从基准值线性升到 `warmup_scale` 倍（峰值），
    之后按余弦一路降到 0，在最后一个 epoch 精确落到 0。

    为什么不用 `GradualWarmupScheduler` + `CosineAnnealingWarmRestarts`：
    那两个调度器把"进度"存在自己内部，续跑时要么丢掉进度、要么把整棵调度器对象
    pickle 回来——而 pickle 回来的对象里持有的是**反序列化出来的优化器副本**，
    之后 `step()` 全都在改那个副本，真实优化器的 lr 从此冻结。
    这个类里没有任何累计状态需要恢复：只要告诉它"总共几个 epoch、已经跑完几个"，
    算出来的 lr 一定和非中断跑完全一致。`state_dict()` 返回的是纯数字字典，
    存进 checkpoint 也不会带任何对象引用。
    """

    def __init__(self,
                 optimizer,
                 total_epochs: int,
                 warmup_epochs: int = 2,
                 warmup_scale: float = 8.0,
                 completed_epochs: int = 0,
                 base_lrs=None,
                 min_factor: float = 0.0):
        if int(total_epochs) < 1:
            raise ValueError(f"total_epochs must be >= 1, got {total_epochs}")
        self.optimizer = optimizer
        self.total_epochs = int(total_epochs)
        self.warmup_scale = float(warmup_scale)
        self.min_factor = float(min_factor)
        # 只有 1 个 epoch 时没有退火空间，直接用峰值，避免 lr 被算成 0。
        if self.total_epochs <= 1:
            self.warmup_epochs = 0
        else:
            self.warmup_epochs = min(int(warmup_epochs), self.total_epochs - 1)

        current_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        if base_lrs is None:
            self.base_lrs = current_lrs
        else:
            self.base_lrs = [float(value) for value in base_lrs]
            if len(self.base_lrs) != len(current_lrs):
                raise ValueError(
                    f"base_lrs has {len(self.base_lrs)} entries but the optimizer has "
                    f"{len(current_lrs)} parameter groups")
        self.epoch = int(completed_epochs)
        self.last_lrs = list(current_lrs)
        self._apply(self.epoch)

    def factor(self, epoch_index):
        """第 epoch_index 个 epoch（0 基）相对基准 lr 的倍数。"""
        if self.total_epochs <= 1:
            return self.warmup_scale
        index = max(0, min(int(epoch_index), self.total_epochs - 1))
        if index < self.warmup_epochs:
            return self.warmup_scale * (index + 1) / self.warmup_epochs
        span = max(1, self.total_epochs - self.warmup_epochs)
        progress = (index - self.warmup_epochs + 1) / span
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.warmup_scale * (self.min_factor + (1.0 - self.min_factor) * cosine)

    def _apply(self, epoch_index):
        factor = self.factor(epoch_index)
        self.last_lrs = []
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * factor
            self.last_lrs.append(group["lr"])

    def step(self, epoch=None):
        self.epoch = self.epoch + 1 if epoch is None else int(epoch) + 1
        self._apply(self.epoch)

    def get_last_lr(self):
        return list(self.last_lrs)

    def spec(self):
        """可 JSON 序列化的规格（不含进度），用来在续跑时原样重建同一条曲线。"""
        return {
            "kind": "cosine_warmup",
            "total_epochs": self.total_epochs,
            "warmup_epochs": self.warmup_epochs,
            "warmup_scale": self.warmup_scale,
            "min_factor": self.min_factor,
            "base_lrs": list(self.base_lrs),
        }

    def state_dict(self):
        state = self.spec()
        state["epoch"] = self.epoch
        return state

    def load_state_dict(self, state):
        """兼容接口。实际不使用：本类的 lr 由 (total_epochs, completed_epochs) 决定，
        恢复它的 state_dict 没有意义，反而会掩盖"进度传错"这类 bug。"""
        for key in ("total_epochs", "warmup_epochs", "warmup_scale", "min_factor"):
            if key in state:
                setattr(self, key, state[key])
        if "base_lrs" in state:
            self.base_lrs = [float(value) for value in state["base_lrs"]]
        self.epoch = int(state.get("epoch", self.epoch))
        self._apply(self.epoch)


def build_schedule(optimizer, spec, total_epochs, completed_epochs=0):
    """按 checkpoint 里存的 spec 重建调度器；spec 缺失/不匹配时退回当前 lr 当基准。"""
    if not spec or spec.get("kind") != "cosine_warmup":
        return CosineWarmupSchedule(optimizer, total_epochs=total_epochs,
                                    completed_epochs=completed_epochs)
    base_lrs = spec.get("base_lrs")
    if base_lrs is not None and len(base_lrs) != len(optimizer.param_groups):
        base_lrs = None
    return CosineWarmupSchedule(
        optimizer,
        total_epochs=total_epochs,
        warmup_epochs=spec.get("warmup_epochs", 2),
        warmup_scale=spec.get("warmup_scale", 8.0),
        completed_epochs=completed_epochs,
        base_lrs=base_lrs,
        min_factor=spec.get("min_factor", 0.0),
    )


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_warm_train_epoch, restart_epoch=None):
        self.multiplier = multiplier
        self.warm_epoch = warm_epoch
        if restart_epoch is None:
            restart_epoch = after_warm_train_epoch
            # if after_warm_train_epoch % 2 != 0:
            #     restart_epoch = after_warm_train_epoch // 2 + 1
            # else:
            #     restart_epoch = after_warm_train_epoch // 2
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
