
import sys
sys.path.append('../geneticml/')

import warnings
from geneticml.optimizers import GeneticOptimizer
from geneticml.strategy import EvolutionaryOptimizer
from estimators import SklearnMLPEstimator
from metrics import metric_accuracy
from sklearn.datasets import load_iris
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


if __name__ == "__main__":

    generations = 10  # Number of times to evole the population.
    population = 10  # Number of networks in each generation.

    parameters = {
        "hidden_layer_sizes": [(10), (10, 10,)],
        "activation": ["relu"],
        "solver": ["adam"],
        "alpha": [0.0001, 0.01],
        "batch_size": ["auto"],
        "learning_rate": ["constant"],
        "learning_rate_init": [0.001, 0.1, 0.0001, 0.12, 0.112],
        "power_t": [0.5],
        "max_iter": [100, 20, 10, 50],
        "shuffle": [True, False],
        "random_state": [None],
        "tol": [0.0001],
        "verbose": [False],
        "warm_start": [False],
        "momentum": [0.9, 1, 0.1, 0.88, 0.23],
        "nesterovs_momentum": [True],
        "early_stopping": [False, True],
        "validation_fraction": [0.1],
        "beta_1": [0.9],
        "beta_2": [0.999],
        "epsilon": [1e-08],
        "n_iter_no_change": [10],
        "max_fun": [15000],
    }

    strategy = EvolutionaryOptimizer(member_class=SklearnMLPEstimator, parameters=parameters, retain=0.4, random_select=0.1, mutate_chance=0.2, max_children=2)

    optimizer = GeneticOptimizer(optimizer=strategy)

    data = load_iris()
    
    greater_is_better = True

    models = optimizer.simulate(
        data=data.data, 
        target=data.target,
        generations=generations,
        population=population,
        evaluation_function=metric_accuracy,
        greater_is_better=greater_is_better,
        verbose=True
    )

    print([
        1 - x.fitness if greater_is_better else x.fitness
        for x in models
    ])
