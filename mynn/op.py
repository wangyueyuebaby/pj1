from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True

    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass

class Linear(Layer):
    """
    The linear layer for a neural network.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False,
                 weight_decay_lambda=1e-8) -> None:
        super().__init__()
        # 缩小 scale 防止梯度爆炸
        self.W = initialize_method(scale=0.01, size=(in_dim, out_dim))
        self.b = np.zeros((1, out_dim))
        self.grads = {'W': None, 'b': None}
        self.input = None

        self.params = {'W': self.W, 'b': self.b}

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        self.input = X
        out = np.dot(X, self.W) + self.b
        return out

    def backward(self, grad: np.ndarray):
        dL_dX = np.dot(grad, self.W.T)
        dL_dW = np.dot(self.input.T, grad)
        dL_db = np.sum(grad, axis=0, keepdims=True)

        if self.weight_decay:
            dL_dW += self.weight_decay_lambda * self.W

        self.grads['W'] = dL_dW
        self.grads['b'] = dL_db
        return dL_dX

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}

class conv2D(Layer):
    """
    The 2D convolutional layer (NumPy TensorDot Accelerated).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal,
                 weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.W = initialize_method(scale=0.01, size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros((1, out_channels, 1, 1))

        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        self.input = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        self.input = X
        N, C, H, W = X.shape

        if self.padding > 0:
            X_pad = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        else:
            X_pad = X

        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        out = np.zeros((N, self.out_channels, out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                window = X_pad[:, :, h_start:h_end, w_start:w_end]
                out[:, :, i, j] = np.tensordot(window, self.W, axes=([1, 2, 3], [1, 2, 3]))

        out += self.b
        return out

    def backward(self, grads):
        N, C, H, W = self.input.shape
        out_h, out_w = grads.shape[2], grads.shape[3]

        if self.padding > 0:
            X_pad = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        else:
            X_pad = self.input

        dX_pad = np.zeros_like(X_pad)
        dW = np.zeros_like(self.W)

        db = np.sum(grads, axis=(0, 2, 3), keepdims=True)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                window = X_pad[:, :, h_start:h_end, w_start:w_end]
                grad_slice = grads[:, :, i, j]

                dW += np.tensordot(grad_slice, window, axes=([0], [0]))
                dX_pad[:, :, h_start:h_end, w_start:w_end] += np.tensordot(grad_slice, self.W, axes=([1], [0]))

        if self.padding > 0:
            dX = dX_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_pad

        if self.weight_decay:
            dW += self.weight_decay_lambda * self.W

        self.grads['W'] = dW
        self.grads['b'] = db

        return dX

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}

class Flatten(Layer):
    """
    用来将卷积层输出的 4D 张量展平为 2D 张量
    """
    def __init__(self) -> None:
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        return grads.reshape(self.input_shape)

class ReLU(Layer):
    """
    An activation layer. (修复了丢失的 ReLU)
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X < 0, 0, X)
        return output

    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    def __init__(self, model=None, max_classes=10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.optimizable = False
        self.predicts = None
        self.labels = None
        self.probs = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        self.predicts = predicts
        self.labels = labels
        batch_size = predicts.shape[0]

        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts

        correct_probs = self.probs[np.arange(batch_size), labels]
        loss = -np.sum(np.log(correct_probs + 1e-10)) / batch_size
        return loss

    def backward(self):
        batch_size = self.predicts.shape[0]

        if self.has_softmax:
            self.grads = self.probs.copy()
            self.grads[np.arange(batch_size), self.labels] -= 1
            self.grads = self.grads / batch_size
        else:
            self.grads = np.zeros_like(self.predicts)
            self.grads[np.arange(batch_size), self.labels] = -1.0 / (
                        self.probs[np.arange(batch_size), self.labels] + 1e-10)
            self.grads = self.grads / batch_size

        if self.model is not None:
            self.model.backward(self.grads)

        return self.grads

    def cancel_soft_max(self):
        self.has_softmax = False
        return self

def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition