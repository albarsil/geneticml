
import warnings
from geneticml.optimizers import GeneticOptimizer
from geneticml.strategy import EvolutionaryStrategy
from geneticml.algorithms import EstimatorBuilder
from metrics import metric_accuracy
from model import SampleNeuralNetwork
from sklearn.datasets import load_iris
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def fit(model, x, y):
    #TODO some training processing
    # train(model, epochs, ....)
    ......
    return model


def predict(model, x):
    #TODO some processing over x to allow the inference through pytorch model
    ......
    return model(x)


if __name__ == "__main__":

    seed = 12542
    generations = 5  # Number of times to evole the population.
    population = 2  # Number of networks in each generation.

    # Define a set of parameters that could be tested
    parameters = {
        "n_classes": [4],
        "hidden_size": [[4], [4, 4, 2], [4, 4, 4]],
        "dropout_prob": [0.2, 0.11, 0.43, 0.55, 0.21, 0.1]
    }

    # Creates an estimator
    estimator = EstimatorBuilder()\
        .of(model_type=SampleNeuralNetwork)\
        .fit_with(func=fit)\
        .predict_with(func=predict)\
        .build()

    # Defines a strategy for the optimization
    strategy = EvolutionaryStrategy(
        estimator_type=estimator,
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
        data=data.data, 
        target=data.target,
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
