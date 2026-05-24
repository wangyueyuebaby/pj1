from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        # 将 init_lr 改名为 lr，以匹配 scheduler 的调用
        self.lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    # 1. 权重衰减 (L2 正则化) [cite: 59-61]
                    if layer.weight_decay:
                        # 使用 *= 进行原地修改，确保 layer.W 也同步变化
                        layer.params[key] *= (1 - self.lr * layer.weight_decay_lambda)

                    # 2. 参数更新 [cite: 110-114]
                    # 重要：请使用 -= 而不是 = ... - ...
                    # 因为 layer.W 是对 layer.params['W'] 的引用
                    # 使用 -= 能保证修改直接作用于内存中的同一个数组
                    layer.params[key] -= self.lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = mu
        # 初始化动量缓存 v [cite: 54-56]
        self.v = {}

    def step(self):
        # 这是 Part C 的优化方向之一，你可以后续实现它 [cite: 54-56]
        pass