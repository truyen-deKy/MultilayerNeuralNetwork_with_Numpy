from typing import List
from operation import *


class Layer(object):

    def __init__(self, neurons: int):
        self.neurons = neurons
        self.first = True
        self.params: List[ndarray] = []
        self.param_grads: List[ndarray] = []
        self.operations: List[Operation] = []

    def _setup_layer(self, num_in: int) -> None:
        raise NotImplementedError()

    # pass forward every operation in network
    def forward(self, input_):
        if self.first:
            self._setup_layer(input_)
            self.first = False
        self.input_ = input_
        for operation in self.operations:
            input_ = operation.forward(input_)
        self.output = input_
        return self.output

    # in backward pass, gradient of parameters are computed and stored
    def backward(self, output_grad):
        assert_same_shape(self.output, output_grad)
        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)
        input_grad = output_grad
        self._param_grads()
        return input_grad

    def _param_grads(self):
        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self):

        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)


class Fully_connected(Layer):
    num_in = BATCH_SIZE

    def __init__(self,
                 neurons: int,
                 activation: Operation = Sigmoid(),
                 weight_init: str = None
                 ):
        super().__init__(neurons)
        self.activation = activation
        self.weight_init = weight_init

    def _setup_layer(self, input_):
        if self.seed:
            np.random.seed(self.seed)

        self.params = []
        if self.weight_init == 'glorot':
            scale = 2 / (self.num_in + self.neurons)
        elif self.weight_init == 'he':
            scale = 2 / self.num_in
        else:
            scale = 1
        # weights
        self.__class__.num_in = self.neurons
        self.params.append(np.random.normal(loc=0,
                                            scale=scale,
                                            size=(input_.shape[1], self.neurons)))

        # bias
        self.params.append(np.random.normal(loc=0,
                                            scale=scale,
                                            size=(1, self.neurons)))

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]
