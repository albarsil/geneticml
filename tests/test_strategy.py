from typing import List

from geneticml.algorithms import (BaseEstimator, EstimatorBuilder,
                                  EstimatorParameters)
from geneticml.strategy import (BaseStrategy, EvolutionaryStrategy,
                                StrategyParameters)
from sklearn.neural_network import MLPClassifier

# Define a set of parameters that could be tested
parameters = StrategyParameters(
    model_parameters={
        "hidden_layer_sizes": [(10), (10, 10,)],
        "activation": ["identity", "logistic", "tanh", "relu"],
        "solver": ["lbfgs", "sgd", "adam"],
        "alpha": [0.0001, 0.01],
        "batch_size": ["auto"],
        "learning_rate": ["constant", "invscaling", "adaptive",  "constant"],
        "learning_rate_init": [0.001, 0.1, 0.0001, 0.12, 0.112],
        "power_t": [0.5],
        "max_iter": [100, 20, 10, 50],
        "shuffle": [True, False],
        "random_state": [123123],
        "tol": [0.0001],
        "verbose": [False],
        "warm_start": [False],
        "momentum": [0.9, 1, 0.1, 0.88, 0.23],
        "nesterovs_momentum": [True, False],
        "early_stopping": [False, True],
        "validation_fraction": [0.1],
        "beta_1": [0.9],
        "beta_2": [0.999],
        "epsilon": [1e-08],
        "n_iter_no_change": [10],
        "max_fun": [15000],
    }
)


def fit(model, x, y):
    return model.fit(x, y)


def predict(model, x):
    return model.predict(x)


def test_strategy_evolutionary_initialization():
    # Creates an estimator
    estimator = EstimatorBuilder()\
        .of(model_type=MLPClassifier)\
        .fit_with(func=fit)\
        .predict_with(func=predict)\
        .build()

    # Defines a strategy for the optimization
    strategy = EvolutionaryStrategy(
        estimator=estimator,
        parameters=parameters,
        retain=0.4,
        random_select=0.1,
        mutate_chance=0.2,
        max_children=2,
        random_state=541342
    )

    assert isinstance(strategy, BaseStrategy)


def test_strategy_evolutionary_create_random_set():
    # Creates an estimator
    estimator = EstimatorBuilder()\
        .of(model_type=MLPClassifier)\
        .fit_with(func=fit)\
        .predict_with(func=predict)\
        .build()

    # Defines a strategy for the optimization
    strategy = EvolutionaryStrategy(
        estimator=estimator,
        parameters=parameters,
        retain=0.4,
        random_select=0.1,
        mutate_chance=0.2,
        max_children=2,
        random_state=541342
    )

    out = strategy.create_random_set()

    assert isinstance(out, EstimatorParameters)
    assert len(out.model_parameters.keys()) == len(parameters.model_parameters.keys())


def test_strategy_evolutionary_create_population():
    # Creates an estimator
    estimator = EstimatorBuilder()\
        .of(model_type=MLPClassifier)\
        .fit_with(func=fit)\
        .predict_with(func=predict)\
        .build()

    # Defines a strategy for the optimization
    strategy = EvolutionaryStrategy(
        estimator=estimator,
        parameters=parameters,
        retain=0.4,
        random_select=0.1,
        mutate_chance=0.2,
        max_children=2,
        random_state=541342
    )

    size = 5

    out = strategy.create_population(size=size)

    assert isinstance(out, list)
    assert isinstance(out[0], BaseEstimator)
    assert len(out) == size


def test_strategy_evolutionary_breed():
    # Creates an estimator
    estimator = EstimatorBuilder()\
        .of(model_type=MLPClassifier)\
        .fit_with(func=fit)\
        .predict_with(func=predict)\
        .build()

    # Defines a strategy for the optimization
    strategy = EvolutionaryStrategy(
        estimator=estimator,
        parameters=parameters,
        retain=0.4,
        random_select=0.1,
        mutate_chance=0.2,
        max_children=2,
        random_state=541342
    )

    out = strategy.create_population(size=2)
    out = strategy.breed(parent1=out[0], parent2=out[1])

    assert isinstance(out, list)
    assert isinstance(out[0], BaseEstimator)
    assert len(out) >= 1
    assert len(out) <= strategy._max_children


def test_strategy_evolutionary_mutate():
    # Creates an estimator
    estimator = EstimatorBuilder()\
        .of(model_type=MLPClassifier)\
        .fit_with(func=fit)\
        .predict_with(func=predict)\
        .build()

    # Defines a strategy for the optimization
    strategy = EvolutionaryStrategy(
        estimator=estimator,
        parameters=parameters,
        retain=0.4,
        random_select=0.1,
        mutate_chance=0.2,
        max_children=2,
        random_state=541342
    )

    out = strategy.create_population(size=1)
    out = strategy.mutate(out[0].parameters)

    assert isinstance(out, EstimatorParameters)
    assert len(out.model_parameters.keys()) == len(parameters.model_parameters.keys())
