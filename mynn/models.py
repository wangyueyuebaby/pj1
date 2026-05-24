from .op import *
import pickle


class Model_MLP(Layer):
    """
    A model with linear layers.
    """

    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def clear_grad(self):
        for layer in self.layers:
            if hasattr(layer, 'clear_grad'):
                layer.clear_grad()

    def load_model(self, param_path):
        with open(param_path, 'rb') as f:
            params = pickle.load(f)
        self.size_list = params[0]
        self.act_func = params[1]
        self.layers = []

        p_idx = 2
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = params[p_idx]['W']
            layer.b = params[p_idx]['b']
            layer.weight_decay = params[p_idx]['weight_decay']
            layer.weight_decay_lambda = params[p_idx]['lambda']
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(ReLU())
            p_idx += 1

    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W': layer.params['W'], 'b': layer.params['b'], 'weight_decay': layer.weight_decay,
                                   'lambda': layer.weight_decay_lambda})

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)


class Model_CNN(Layer):
    """
    A model with conv2D layers.
    """

    def __init__(self):
        super().__init__()
        self.layers = []

        # 1. 卷积层: 输入通道 1(黑白图像), 输出通道 8, 卷积核大小 3
        self.layers.append(conv2D(in_channels=1, out_channels=8, kernel_size=3))

        # 2. 激活层
        self.layers.append(ReLU())

        # 3. 展平层: 将 4D 张量变成 2D 张量 (Batch, 8 * 26 * 26)
        self.layers.append(Flatten())

        # 4. 线性层: 8 * 26 * 26 = 5408，输出 10 个类别
        self.layers.append(Linear(in_dim=5408, out_dim=10))

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        # 如果输入是 1D 图像，先将其 reshape 为 2D 格式 (Batch, Channel, H, W)
        if len(X.shape) == 2:
            X = X.reshape(-1, 1, 28, 28)

        outputs = X
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def clear_grad(self):
        for layer in self.layers:
            if hasattr(layer, 'clear_grad'):
                layer.clear_grad()

    # ==========================================
    # 核心修复点：为 CNN 加上保存和加载功能！
    # ==========================================
    def save_model(self, save_path):
        import pickle
        param_list = []
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

    def load_model(self, param_path):
        import pickle
        with open(param_path, 'rb') as f:
            param_list = pickle.load(f)

        p_idx = 0
        for layer in self.layers:
            if layer.optimizable:
                layer.W = param_list[p_idx]['W']
                layer.b = param_list[p_idx]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[p_idx]['weight_decay']
                layer.weight_decay_lambda = param_list[p_idx]['lambda']
                p_idx += 1