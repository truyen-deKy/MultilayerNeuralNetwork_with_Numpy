from neuralnetwork import *


class Optimizer(object):
    def __init__(self,
                 learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def step(self) -> None:
        pass


class SGD(Optimizer):
    def __init__(self,
                 learning_rate: float = 0.01):
        super().__init__(learning_rate)

    # attribution net will be set in trainer.py
    def step(self):
        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            param -= self.learning_rate * param_grad


class SGDMomentum(Optimizer):
    def __init__(self,
                 learning_rate: float = 0.01,
                 momentum: float = 0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.is_first_call = True

    def step(self):
        if self.is_first_call:
            # now we will set up velocities on the first iteration
            self.velocities = [np.zeros_like(param)
                               for param in self.net.params()]
            self.is_first_call = False

        for (param, param_grad, velocity) in zip(self.net.params(),
                                                 self.net.param_grads(),
                                                 self.velocities):
            velocity *= self.momentum
            velocity += self.learning_rate * param_grad
            param -= velocity


class SGD_Momentum_Decay(Optimizer):
    def __init__(self,
                 learning_rate: float = 0.01,
                 momentum: float = 0.9,
                 final_learning_rate: float = 1e-5,
                 max_epochs: float = 100,
                 decay_type: str = 'exponential'):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.is_first_call = True
        self.final_learning_rate = final_learning_rate
        self.decay_type = decay_type
        self.max_epochs = max_epochs

    def step(self):
        if self.is_first_call:
            self.set_up_optimiser()
            self.is_first_call = False

        for (param, param_grad, velocity) in zip(self.net.params(),
                                                 self.net.param_grads(),
                                                 self.velocities):
            velocity *= self.momentum
            velocity += self.learning_rate * param_grad
            param -= velocity

    def _decay(self):
        if self.decay_type == 'exponential':
            self.learning_rate *= self.decay_per_epoch

        elif self.decay_type == 'linear':
            self.learning_rate -= self.decay_per_epoch

    def set_up_optimiser(self):
        # set up inital velocities as 0
        self.velocities = [np.zeros_like(param) for param in self.net.params()]

        # set up amount of learning rate decay every epoch
        if not self.decay_type:
            return
        elif self.decay_type == 'exponential':
            self.decay_per_epoch = np.power(self.final_learning_rate / self.learning_rate,
                                            1.0 / (self.max_epochs - 1))
        elif self.decay_type == 'linear':
            self.decay_per_epoch = (self.learning_rate - self.final_learning_rate) / (self.max_epochs - 1)
