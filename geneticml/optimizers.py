"""
Class that holds a genetic algorithm for evolving a network.

Credit:
    A lot of those code was originally inspired by:
    https://github.com/mrpeel/genetic-keras
"""

from typing import List, Callable, Type, TypeVar
from abc import ABC, abstractmethod
from tqdm import tqdm
from geneticml.strategy import BaseStrategy
from geneticml.algorithms import BaseEstimator

T = TypeVar('T', bound=BaseEstimator)
E = TypeVar('E', bound=BaseStrategy)


class BaseOptimizer(ABC):
    def __init__(self, strategy: Type[BaseStrategy]) -> None:
        super().__init__()
        pass

    @abstractmethod
    def simulate(self, data, target, verbose: bool = True) -> List[T]:
        """
        Generate a network with the genetic algorithm.

        Parameters:
            data (?): The data used to train the algorithm
            target (?): The targets used to train the algorithm
            verbose (bool): True if should verbose or False if not

        Returns:
            (List[BaseEstimator]): A list with the final population sorted by their loss
        """
        pass


class GeneticOptimizer(BaseOptimizer):
    def __init__(self, strategy: Type[BaseStrategy]) -> None:
        """
        Create an optimizer.

        Parameters:
            strategy (Type[BaseStrategy]): Any strategy that inherits from the type defined. This strategy will be used for the optimization
        """
        super().__init__(strategy)
        self._strategy = strategy

    def simulate(self, data, target, generations: int, population: int, evaluation_function: Callable, greater_is_better: bool = False, verbose: bool = True) -> List[T]:
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

        estimators = self._strategy.create_population(population)

        if verbose:
            increment = 100 / generations
            pbar = tqdm(total=100)

        # Evolve the generation.
        for i in range(generations):

            # Train and get loss for networks.
            losses = []
            for x in estimators:

                # Train the model
                x.fit(data, target)

                # Do the model inference
                y_pred = x.predict(data)

                # Peform the evaluation
                loss = evaluation_function(target, y_pred)

                # Workaround to make the lower losses better than the bigger ones
                if greater_is_better:
                    loss = 1 - loss

                # Assign the evaluation result to the estimator
                x.fitness = loss

                # Track the loss
                losses.append(loss)

            avgloss = sum(losses) / len(losses)
            avgloss = avgloss if greater_is_better else 1 - avgloss

            # Evolve, except on the last iteration.
            if i != generations - 1:
                # Do the evolution.
                estimators = self._strategy.execute(estimators)
        
            if verbose:
                pbar.update(increment)
                pbar.set_postfix({'generation_loss': avgloss})
        
        if verbose:
            pbar.close()

        # Sort our final population.
        return sorted(estimators, key=lambda x: x.fitness, reverse=False)
