import numpy as np

def relu(X):
    return np.maximum(X, 0)

def relu_deriv(Y):
    return (Y > 0) * 1.


def softmax(X):
    # Avoid Overflow and Underflow
    X = X - np.max(X, axis=1, keepdims=True)
    return np.exp(X) / np.exp(X).sum(axis=1, keepdims=True)

def cross_entropy(Y, T):
    log_likelihood = -np.log(Y[range(T.shape[0]), T.argmax(axis=1)])
    return log_likelihood.sum() / T.shape[0]


class Layer(object):

    def get_w(self):
        return []

    def get_b(self):
        return []

    def get_w_grad(self, X, next_grad):
        return []

    def get_b_grad(self, X, next_grad):
        return []

    def output(self, X):
        return []

    def calc_grad(self, Y, next_grad=None, T=None):
        return []


class FullyConnectedLayer(Layer):

    def __init__(self, n_in=None, n_out=None, random = True, W=None, B=None):
        if random:
            self.W = np.random.randn(n_in, n_out) * 0.4
            self.b = np.zeros(n_out)
        else:
            self.W = W
            self.b = B

    def get_w(self):
        return self.W

    def get_b(self):
        return self.b

    def get_w_grad(self, X, next_grad):
        return X.T.dot(next_grad)

    def get_b_grad(self, X, next_grad):
        return next_grad.sum(axis=0)

    def output(self, X):
        return X.dot(self.W) + self.b

    def calc_grad(self, Y, next_grad):
        return next_grad.dot(self.W.T)


class ReLULayer(Layer):
    def output(self, X):
        return relu(X)

    def calc_grad(self, Y, next_grad):
        return relu_deriv(Y) * next_grad


class Softmax_CrossEntropyLayer(Layer):

    def output(self, X):
        return softmax(X)

    def calc_grad(self, Y, T):
        return (Y - T) / T.shape[0]

    def calc_cost(self, Y, T):
        return cross_entropy(Y, T)