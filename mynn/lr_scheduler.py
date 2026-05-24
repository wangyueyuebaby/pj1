from abc import abstractmethod
import numpy as np


class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0

    @abstractmethod
    def step(self):
        pass


class StepLR(scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count % self.step_size == 0:
            # 修改优化器中的当前学习率
            self.optimizer.lr *= self.gamma


class MultiStepLR(scheduler):
    def __init__(self, optimizer, milestones, gamma=0.1) -> None:
        """
        milestones: 一个列表，包含触发学习率衰减的具体步数，例如 [800, 2400]
        """
        super().__init__(optimizer)
        self.milestones = milestones
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        # 如果当前步数在里程碑列表中，则进行衰减
        if self.step_count in self.milestones:
            self.optimizer.lr *= self.gamma


class ExponentialLR(scheduler):
    def __init__(self, optimizer, gamma=0.9) -> None:
        """
        每个 step 都会将学习率乘以 gamma
        """
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        self.optimizer.lr *= self.gamma