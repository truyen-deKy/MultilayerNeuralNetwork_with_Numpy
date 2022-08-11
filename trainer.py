from copy import deepcopy
from optimiser import *


class Trainer(object):
    def __init__(self, net: NeuralNetwork, optim: Optimizer):
        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        setattr(self.optim, 'net', self.net)

    def generate_batches(self, X, y, batch_size=32):
        # Generate batches for training
        assert X.shape[0] == y.shape[0], f'feature has {X.shape[0]} rows but target has {y.shape[0]}'

        for i in range(0, X.shape[0], batch_size):
            X_batch, y_batch = X[i:i + batch_size], y[i:i + batch_size]
            yield X_batch, y_batch

    def fit(self, X_train, y_train, X_test, y_test,
            epochs: int = 100,
            eval_every: int = 10,
            batch_size: int = BATCH_SIZE,
            seed: int = 1):

        np.random.seed(seed)

        for e in range(epochs):
            # if this is time to evaluate on test_set, prepare a copy of current model
            # if this loss is greater than best loss
            if (e + 1) % eval_every == 0:
                last_model = deepcopy(self.net)

            X_train, y_train = permute_data(X_train, y_train)

            batch_generator = self.generate_batches(X_train, y_train, batch_size)
            # stochastic
            for n, (X_batch, y_batch) in enumerate(batch_generator):
                self.net.train_batch(X_batch, y_batch)
                self.optim.step()

            if (e + 1) % eval_every == 0:
                loss = self.net.train_batch(X_test, y_test)

                if loss < self.best_loss:
                    print(f"Validation loss after {e + 1} epochs is {loss:.3f}")
                    self.best_loss = loss
                else:
                    print(
                        f"Loss increased after epoch {e + 1}, final loss was {self.best_loss:.3f}, using the model from epoch {e + 1 - eval_every}")
                    self.net = last_model
                    # ensure self.optim is still updating self.net
                    setattr(self.optim, 'net', self.net)
                    break

            if self.optim.__class__ == SGD_Momentum_Decay:
                self.optim._decay()


def mae(y_true, y_pred):
    # mean absolute error
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    # root mean squared error
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))


def eval_regression_model(model: NeuralNetwork, X_test, y_test):
    preds = model.forward(X_test)
    preds = preds.reshape(-1, 1)
    print(f"Mean absolute error: {mae(preds, y_test):.2f}")
    print()
    print(f"Root mean squared error {rmse(preds, y_test):.2f}")

