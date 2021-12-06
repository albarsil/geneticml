from geneticml.optimizers import GeneticOptimizer
from geneticml.strategy import EvolutionaryOptimizer


if __name__ == "__main__":

    generations = 10  # Number of times to evole the population.
    population = 30  # Number of networks in each generation.

    parameters = {
        "hidden_layer_sizes": [(100)],
        "activation": ["relu"],
        "solver": ["adam"],
        "alpha": [0.0001],
        "batch_size": ["auto"],
        "learning_rate": ["constant"],
        "learning_rate_init": [0.001],
        "power_t": [0.5],
        "max_iter": [200],
        "shuffle": [True],
        "random_state": [None],
        "tol": [0.0001],
        "verbose": [False],
        "warm_start": [False],
        "momentum": [0.9],
        "nesterovs_momentum": [True],
        "early_stopping": [False],
        "validation_fraction": [0.1],
        "beta_1": [0.9],
        "beta_2": [0.999],
        "epsilon": [1e-08],
        "n_iter_no_change": [10],
        "max_fun": [15000],
    }

    strategy = EvolutionaryOptimizer(parameters=parameters, retain=0.4, random_select=0.1, mutate_chance=0.2, max_children=2)

    optimizer = GeneticOptimizer(optimizer=strategy)

    optimizer.simulate(enerations=generations, population=population, verbose=True)
