from typing import List

import pytest
import sklearn.metrics as metrics
from geneticml.algorithms import BaseEstimator, DataLoader, EstimatorBuilder
from geneticml.optimizers import BaseOptimizer, GeneticOptimizer
from geneticml.strategy import (BaseStrategy, EvolutionaryStrategy,
                                StrategyParameters)
from sklearn.datasets import load_iris
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


def test_optimizer_genetic_initialization():
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

    # Creates the optimizer
    optimizer = GeneticOptimizer(strategy=strategy)

    assert isinstance(optimizer, BaseOptimizer)


def test_optimizer_genetic_simulate():
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

    # Creates the optimizer
    optimizer = GeneticOptimizer(strategy=strategy)

    data = load_iris()

    population = 5
    generation = 2

    # Create the simulation using the optimizer and the strategy
    out = optimizer.simulate(
        train_data=DataLoader(data=data.data, target=data.target),
        generations=generation,
        population=population,
        evaluation_function=metrics.accuracy_score,
        greater_is_better=True,
        verbose=False
    )

    assert len(out) == population
    assert isinstance(out[0], BaseEstimator)
