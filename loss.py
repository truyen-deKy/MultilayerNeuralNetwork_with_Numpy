from helper import *


class Loss(object):

    def __init__(self):
        pass

    def forward(self, prediction, target):
        assert_same_shape(prediction, target)
        self.prediction = prediction
        self.target = target
        loss_value = self._output()
        return loss_value

    def backward(self):
        self.input_grad = self._input_grad()
        assert_same_shape(self.prediction, self.input_grad)
        return self.input_grad

    def _output(self):
        raise NotImplementedError()

    def _input_grad(self):
        raise NotImplementedError()


class MeanSquaredError(Loss):

    def __init__(self) -> None:
        super().__init__()

    def _output(self):
        loss = np.sum(np.power(self.prediction - self.target, 2)) / self.prediction.shape[0]
        return loss

    def _input_grad(self):
        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]
#--------------------------------------

def softmax(x,axis):
    max = np.max(x, axis=axis, keepdims=True)  # returns max of each row and keeps same dims
    e_x = np.exp(x - max)  # subtracts each row with its max value
    sum = np.sum(e_x, axis=axis, keepdims=True)  # returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x


class SoftmaxCrossEntropy(Loss):
    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps
        self.single_output = False

    def _output(self) -> float:
        softmax_preds = softmax(self.prediction,axis=1)
        self.softmax_preds = np.clip(softmax_preds, self.eps, 1 - self.eps)
        softmax_cross_entropy_loss = (-1.0 * self.target * np.log(self.softmax_preds) - (1.0 - self.target) * np.log(
            1 - self.softmax_preds))
        return np.sum(softmax_cross_entropy_loss)

    def _input_grad(self):
        return self.softmax_preds - self.target
