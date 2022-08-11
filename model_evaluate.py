from trainer import *


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


linear_regression = NeuralNetwork(
    layers=[Fully_connected(neurons=1,
                            activation=Linear())],
    loss=MeanSquaredError(),
    seed=20194653)

neural_network = NeuralNetwork(
    layers=[Fully_connected(neurons=13,
                            activation=Sigmoid()),
            Fully_connected(neurons=1,
                            activation=Linear())],
    loss=MeanSquaredError(),
    seed=20194653)

deep_neural_network = NeuralNetwork(
    layers=[Fully_connected(neurons=13,
                            activation=Sigmoid()),
            Fully_connected(neurons=13,
                            activation=Sigmoid()),
            Fully_connected(neurons=1,
                            activation=Linear())],
    loss=MeanSquaredError(),
    seed=20194653)

deep_neural_network1 = NeuralNetwork(
    layers=[Fully_connected(neurons=13,
                            weight_init='glorot',
                            activation=Sigmoid()),
            Fully_connected(neurons=13,
                            weight_init='glorot',
                            activation=Sigmoid()),
            Fully_connected(neurons=1,
                            weight_init='glorot',
                            activation=Linear())],
    loss=MeanSquaredError(),
    seed=20194653)

deepnet1 = NeuralNetwork(layers=[
    Fully_connected(neurons=13,
                    weight_init='he',
                    activation=LeakyReLU()),
    Fully_connected(neurons=30,
                    weight_init='he',
                    activation=LeakyReLU()),
    Fully_connected(neurons=13,
                    weight_init='he',
                    activation=LeakyReLU()),
    Fully_connected(neurons=20,
                    weight_init='he',
                    activation=LeakyReLU()),
    Fully_connected(neurons=13,
                    weight_init='he',
                    activation=LeakyReLU()),
    Fully_connected(neurons=1,
                    weight_init='he',
                    activation=Linear())],
    loss=MeanSquaredError(),
    seed=20194653)

deepnet2 = NeuralNetwork(layers=[
    Fully_connected(neurons=20,
                    weight_init='glorot',
                    activation=Tanh()),
    Fully_connected(neurons=14,
                    weight_init='glorot',
                    activation=Tanh()),
    Fully_connected(neurons=7,
                    weight_init='glorot',
                    activation=Tanh()),
    Fully_connected(neurons=1,
                    weight_init='glorot',
                    activation=Linear())],
    loss=MeanSquaredError(),
    seed=20194653)
