""" Step Scheduler

Basic step LR schedule with warmup, noise.

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import torch

from .scheduler import Scheduler


'''以下修改是为了减少预热步数'''

# class StepLRScheduler(Scheduler):
#     def __init__(self,
#                  optimizer: torch.optim.Optimizer,
#                  decay_t: float,
#                  decay_rate: float = 1.,
#                  warmup_t=0,
#                  warmup_lr_init=0,
#                  max_lr=None,  # 添加 max_lr 参数
#                  t_in_epochs=True,
#                  noise_range_t=None,
#                  noise_pct=0.67,
#                  noise_std=1.0,
#                  noise_seed=42,
#                  initialize=True,
#                  ) -> None:
#         # 如果提供了 max_lr，则在调用父类初始化之前修改优化器的学习率
#         if max_lr is not None:
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = max_lr

#         super().__init__(
#             optimizer, param_group_field="lr",
#             noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
#             initialize=initialize)

#         self.decay_t = decay_t
#         self.decay_rate = decay_rate
#         self.warmup_t = warmup_t
#         self.warmup_lr_init = warmup_lr_init
#         self.t_in_epochs = t_in_epochs
class StepLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_iters: int,
                 max_lr=0.00006,  # 最大学习率
                 warmup_lr_init=0,
                 t_in_epochs=False,  # 使用迭代次数
                 decay_t=None,
                 decay_rate=1.,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.warmup_iters = warmup_iters
        self.max_lr = max_lr
        self.warmup_lr_init = warmup_lr_init
        self.lr_increment = (max_lr - warmup_lr_init) / warmup_iters  # 每次迭代增加的学习率
        self.t_in_epochs = t_in_epochs
        self.decay_t = decay_t
        self.decay_rate = decay_rate


        if self.warmup_t:
            # self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            # super().update_groups(self.warmup_lr_init)
            self.warmup_steps = [(max_lr - warmup_lr_init) / self.warmup_t for _ in self.base_values]#######################修改预热步数########################
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]


    # def _get_lr(self, t):
    #     if t < self.warmup_t:
    #         lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
    #     else:
    #         lrs = [v * (self.decay_rate ** (t // self.decay_t)) for v in self.base_values]
    #     return lrs
    def _get_lr(self, iter_count):
        if iter_count < self.warmup_iters:
            # 预热期内逐步增加学习率
            lrs = [self.warmup_lr_init + iter_count * self.lr_increment for _ in self.base_values]
        else:
            # 预热期结束后保持在 max_lr
            lrs = [self.max_lr for _ in self.base_values]
        return lrs


    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None
