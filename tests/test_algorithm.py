from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from geneticml.algorithms import BaseEstimator, EstimatorBuilder


PARAMETERS = {
    "hidden_layer_sizes": (10),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "batch_size": "auto",
    "learning_rate": "constant",
    "learning_rate_init": 0.001,
    "power_t": 0.5,
    "max_iter": 100,
    "shuffle": True,
    "random_state": 33311,
    "tol": 0.0001,
    "verbose": False,
    "warm_start": False,
    "momentum": 0.9,
    "nesterovs_momentum": True,
    "early_stopping": False,
    "validation_fraction": 0.1,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "epsilon": 1e-08,
    "n_iter_no_change": 10,
    "max_fun": 15000,
}


def fit(model, x, y):
    return model.fit(x, y)


def predict(model, x):
    return model.predict(x)


def test_estimator_builder():
    # Creates an estimator
    estimator = EstimatorBuilder()\
        .of(model_type=MLPClassifier)\
        .fit_with(func=fit)\
        .predict_with(func=predict)\
        .build()

    assert isinstance(estimator, BaseEstimator)


def test_estimator_fit():
    # Creates an estimator
    estimator = EstimatorBuilder()\
        .of(model_type=MLPClassifier)\
        .fit_with(func=fit)\
        .predict_with(func=predict)\
        .build()

    data = load_iris()
    estimator.initialize(parameters=PARAMETERS)
    estimator.fit(data.data, data.target)


def test_estimator_predict():
    # Creates an estimator
    estimator = EstimatorBuilder()\
        .of(model_type=MLPClassifier)\
        .fit_with(func=fit)\
        .predict_with(func=predict)\
        .build()

    data = load_iris()
    estimator.initialize(parameters=PARAMETERS)
    estimator.fit(data.data, data.target)
    y = estimator.predict(data.data).tolist()

    assert isinstance(y, list)
    assert len(y) == len(data.data)