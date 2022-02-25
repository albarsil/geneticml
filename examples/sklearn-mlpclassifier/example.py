
import warnings

from geneticml.algorithms import DataLoader, DefaultEstimator, EstimatorBuilder
from geneticml.optimizers import GeneticOptimizer
from geneticml.strategy import EvolutionaryStrategy
from metrics import metric_accuracy
from sklearn.datasets import load_iris
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


if __name__ == "__main__":

    seed = 12542
    generations = 10  # Number of times to evole the population.
    population = 10  # Number of networks in each generation.

    # Define a set of parameters that could be tested
    parameters = {
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
        "random_state": [seed],
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

    # Creates an estimator
    estimator = EstimatorBuilder()\
        .of(model_type=MLPClassifier)\
        .fit_with(func=DefaultEstimator.fit)\
        .predict_with(func=DefaultEstimator.predict)\
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
