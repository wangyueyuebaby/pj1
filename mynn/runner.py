import numpy as np
import os
from tqdm import tqdm


class RunnerM():
    """
    This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """

    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0
        # 初始化一下防止第一步没走到 log_iters 报错
        dev_score = 0
        dev_loss = 0

        for epoch in range(num_epochs):
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]

            for iteration in range(int(X.shape[0] / self.batch_size) + 1):
                # 1. 记得在每一步清空梯度！(核心修改点，防止梯度累加导致训练失败)
                self.model.clear_grad()

                train_X = X[iteration * self.batch_size: (iteration + 1) * self.batch_size]
                train_y = y[iteration * self.batch_size: (iteration + 1) * self.batch_size]

                # 如果 batch 为空（比如跑到最后除不尽），就跳出
                if train_X.shape[0] == 0:
                    continue

                # 2. 前向传播与计算 Loss
                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                trn_score = self.metric(logits, train_y)

                # 3. 反向传播与参数更新
                self.loss_fn.backward()
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                # ==========================================
                # 核心修改点：大幅度降低验证频率！
                # 只有达到 log_iters 时才去验证集上做 evaluate，
                # 并且把训练集的 append 也放进来，保证画图时 x 和 y 维度一致！
                # ==========================================
                if iteration % log_iters == 0:
                    dev_score, dev_loss = self.evaluate(dev_set)

                    self.train_loss.append(trn_loss)
                    self.train_scores.append(trn_score)
                    self.dev_scores.append(dev_score)
                    self.dev_loss.append(dev_loss)

                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    print(f"[Dev] loss: {dev_loss}, score: {dev_score}")

            # 每一个 Epoch 跑完后，检查并保存目前最好的模型
            if dev_score > best_score:
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                print(f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score

        self.best_score = best_score

    def evaluate(self, data_set):
        X, y = data_set
        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        return score, loss

    def save_model(self, save_path):
        # 确保调用模型自身的 save_model 方法
        self.model.save_model(save_path)