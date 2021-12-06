"""
Class that holds a genetic algorithm for evolving a network.

Credit:
    A lot of those code was originally inspired by:
    https://github.com/mrpeel/genetic-keras
"""

from typing import List, Callable

from geneticml.strategy import BaseOptimizer
from geneticml.algorithms import BaseEstimator


class GeneticOptimizer:
    def __init__(self, optimizer: BaseOptimizer) -> None:
        """
        Create an optimizer.

        Parameters:
            optimizer (BaseOptimizer): The optimizer used for the optimization
        """
        self._optimizer = optimizer

    def simulate(self, data, target, generations: int, population: int, evaluation_function: Callable, greater_is_better: bool = False, verbose: bool = False) -> List[BaseEstimator]:
        """
        Generate a network with the genetic algorithm.

        Parameters:
            generations (int): Number of times to evole the population
            population (int): Number of estimators in each generation
            evaluation_function (Callable): The function that will calculate the metric
            greater_is_better (bool) Whether evaluation_function is a score function (default), meaning high is good, or a loss function, meaning low is good. In the latter case, the scorer object will sign-flip the outcome of the evaluation_function.
            verbose (bool): True if should verbose or False if not

        Returns:
            (List[BaseEstimator]): A list with the final population sorted by their loss

        """
        estimators = self._optimizer.create_population(population)

        # Evolve the generation.
        for i in range(generations):

            # Train and get loss for networks.
            losses = []
            for x in estimators:

                # Train the model
                x.fit(data, target)

                # Do the model inference
                y_pred = x.predict(data, target)

                # Peform the evaluation
                loss = evaluation_function(target, y_pred)

                # Workaround to make the lower losses better than the bigger ones
                if greater_is_better:
                    loss = 1 - loss

                # Assign the evaluation result to the estimator
                x.fitness = loss

                # Track the loss
                losses.append(loss)

            # Evolve, except on the last iteration.
            if i != generations - 1:
                # Do the evolution.
                estimators = self._optimizer.evolve(estimators)

        # Sort our final population.
        return sorted(estimators, key=lambda x: x.fitness, reverse=False)
