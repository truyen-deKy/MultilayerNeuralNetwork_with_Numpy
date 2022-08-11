from model_evaluate import *
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

boston = load_boston()
data = boston.data
target = boston.target
features = boston.feature_names
s = StandardScaler()
data = s.fit_transform(data)


# Turns a 1D Tensor into 2D
def to_2d_np(a, type: str = "col"):
    assert a.ndim == 1, "Input tensors must be 1 dimensional"
    if type == "col":
        return a.reshape(-1, 1)
    elif type == "row":
        return a.reshape(1, -1)


X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size=0.3,
                                                    random_state=80718)

# make target 2d array
y_train, y_test = to_2d_np(y_train), to_2d_np(y_test)

model = deep_neural_network1

optimiser0 = SGD(learning_rate=0.005)

optimiser1 = SGDMomentum(
    learning_rate=0.001,
    momentum=0.8)

optimiser = SGD_Momentum_Decay(
    learning_rate=0.001,
    momentum=0.9,
    final_learning_rate=1e-6,
    max_epochs=NUMBER_OF_EPOCH,
    decay_type='exponential')

trainer = Trainer(model, optimiser)
trainer.fit(X_train, y_train, X_test, y_test,
            epochs=NUMBER_OF_EPOCH,
            eval_every=50,
            seed=20194653)

print()
eval_regression_model(model, X_test, y_test)
