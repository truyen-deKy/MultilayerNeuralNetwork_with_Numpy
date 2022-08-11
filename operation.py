from helper import *


class Operation(object):

    def __init__(self):
        pass

    def forward(self, input_):
        self.input_ = input_
        self.output = self._output()
        return self.output

    def backward(self, output_grad):
        assert_same_shape(self.output, output_grad)
        self.input_grad = self._input_grad(output_grad)
        assert_same_shape(self.input_, self.input_grad)
        return self.input_grad

    def _output(self):
        raise NotImplementedError()

    def _input_grad(self, output_grad):
        raise NotImplementedError()


class ParamOperation(Operation):

    def __init__(self, param):
        super().__init__()
        self.param = param

    # compute gradient of input(non-parameter and parameter) of a operation
    # not return param_grad beacause there is no operation to pass backward
    def backward(self, output_grad):
        assert_same_shape(self.output, output_grad)
        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)
        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.param, self.param_grad)
        return self.input_grad

    def _param_grad(self, output_grad):
        raise NotImplementedError()


class WeightMultiply(ParamOperation):

    # parameter is weighted matrix
    def __init__(self, W):
        super().__init__(W)

    # output = input * parameter
    def _output(self):
        return np.dot(self.input_, self.param)

    # gradient of mutiply with respect to input is parameter
    def _input_grad(self, output_grad):
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    # gradient of mutiply with respect to parameter is input
    def _param_grad(self, output_grad):
        return np.dot(np.transpose(self.input_, (1, 0)), output_grad)


class BiasAdd(ParamOperation):

    # parameter is 1D vector of bias
    def __init__(self, B):
        assert B.shape[0] == 1
        super().__init__(B)

    # output = input + bias
    # bias is added to every column
    def _output(self):
        return self.input_ + self.param

    # gradient of addition with respect to a matrix is matrix of 1
    # amount of gradient of input is the same as output_grad
    def _input_grad(self, output_grad):
        return np.ones_like(self.input_) * output_grad

    # gradient of addition with respect to a 1D vector is a 1D vector
    # since bias is added for each element along axis 0...
    # ...we sum param_grad along axis 0 to get the total amount of gradient of each bias
    def _param_grad(self, output_grad):
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])


class Sigmoid(Operation):

    def __init__(self):
        super().__init__()

    def _output(self):
        return 1.0 / (1.0 + np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad):
        sigmoid_backward = self.output * (1.0 - self.output)
        return sigmoid_backward * output_grad


class Linear(Operation):

    def __init__(self):
        super().__init__()

    def _output(self):
        return self.input_

    def _input_grad(self, output_grad):
        return output_grad


class Softmax(Operation):
    def __init__(self):
        super().__init__()

    def _output(self):
        e_x = np.exp(self.input_ - np.max(self.input_, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def _input_grad(self, output_grad):
        softmax_backward = self.output * (1 - self.output)
        return softmax_backward * output_grad


class Tanh(Operation):
    def __init__(self):
        super().__init__()

    def _output(self):
        return 2 / (1 + np.exp(-2 * self.input_)) - 1

    def _input_grad(self, output_grad):
        Tanh_backward = 1 - np.power(self.output, 2)
        return Tanh_backward * output_grad

class LeakyReLU(Operation):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha

    def _output(self):
        return np.where(self.input_ >= 0, self.input_, self.alpha * self.input_)

    def _input_grad(self, output_grad):
        LeakyReLU_backward = np.where(self.input_ >= 0, 1, self.alpha)
        return LeakyReLU_backward * output_grad

class ReLU(Operation):
    def __init__(self):
        super().__init__()

    def _output(self):
        return np.where(self.input_ >= 0, self.input_, 0)

    def _input_grad(self, output_grad):
        ReLU_backward = np.where(self.input_ >= 0, 1, 0)
        return ReLU_backward * output_grad


class ReLU1(Operation):

    def __init__(self) -> None:
        super().__init__()

    def _output(self):
        return np.maximum(0, self.input_)

    def _input_grad(self, output_grad):
        _ReLU_backward = (self.output > 0) * 1
        input_grad = _ReLU_backward * output_grad
        return input_grad


class Tanh1(Operation):
    def __init__(self):
        super().__init__()

    def _output(self):
        return 2 / (1 + np.exp(-2 * self.input_)) - 1

    def _input_grad(self, output_grad):
        _Tanh_backward = 1 - np.power(self.output, 2)
        input_grad = _Tanh_backward * output_grad
        return input_grad
