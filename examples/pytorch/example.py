
import warnings

import sklearn.metrics as metrics
from geneticml.algorithms import DataLoader, EstimatorBuilder
from geneticml.optimizers import GeneticOptimizer
from geneticml.strategy import EvolutionaryStrategy, StrategyParameters
from sklearn.datasets import load_iris
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

from model import SampleNeuralNetwork

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def fit(model, x, y):
    #TODO some training processing
    # train(model, epochs, ....)
    # ......
    return model


def predict(model, x):
    #TODO some processing over x to allow the inference through pytorch model
    # ......
    return model(x)


if __name__ == "__main__":

    seed = 12542
    generations = 5  # Number of times to evole the population.
    population = 2  # Number of networks in each generation.

    # Define a set of parameters that could be tested
    parameters = StrategyParameters(
        model_parameters={
            "n_classes": [4],
            "hidden_size": [[4], [4, 4, 2], [4, 4, 4]],
            "dropout_prob": [0.2, 0.11, 0.43, 0.55, 0.21, 0.1]
        }
    )

    # Creates an estimator
    estimator = EstimatorBuilder()\
        .of(model_type=SampleNeuralNetwork)\
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
        random_state=seed
    )

    # Creates the optimizer
    optimizer = GeneticOptimizer(strategy=strategy)

    # Loads the data
    data = load_iris()

    # Defines the metric
    metric = metric_accuracy
    greater_is_better = True

    # Create the simulation using the optimizer and the strategy
    models = optimizer.simulate(
        train_data=DataLoader(data=data.data, target=data.target),
        generations=generations,
        population=population,
        evaluation_function=metric,
        greater_is_better=greater_is_better,
        verbose=True
    )

    # Print the results
    print([
        1 - x.fitness if greater_is_better else x.fitness
        for x in models
    ])
