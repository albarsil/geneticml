"""
Class that holds a genetic algorithm for evolving a network.

Credit:
    A lot of those code was originally inspired by:
    https://github.com/mrpeel/genetic-keras
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Type, TypeVar

from tqdm import tqdm

from geneticml.algorithms import BaseEstimator, DataLoader
from geneticml.strategy import BaseStrategy

T = TypeVar('T', bound=BaseEstimator)
E = TypeVar('E', bound=BaseStrategy)


class BaseOptimizer(ABC):
    def __init__(self, strategy: Type[BaseStrategy]) -> None:
        super().__init__()
        pass

    @abstractmethod
    def simulate(self, train_data: DataLoader, test_data: DataLoader = None, verbose: bool = True) -> List[T]:
        """
        Generate a network with the genetic algorithm.

        Parameters:
            train_data (geneticml.algorithms.DataLoader): The train data loader
            test_data (geneticml.algorithms.DataLoader): The test data loader
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

    def simulate(self, train_data: DataLoader, generations: int, population: int, evaluation_function: Callable, test_data: DataLoader = None, greater_is_better: bool = False, verbose: bool = True, pbar: tqdm = None) -> List[T]:
        """
        Generate a network with the genetic algorithm.

        Parameters:
            train_data (geneticml.algorithms.DataLoader): The train data loader
            test_data (geneticml.algorithms.DataLoader): The test data loader
            generations (int): Number of times to evole the population
            population (int): Number of estimators in each generation
            evaluation_function (Callable): The function that will calculate the metric
            sampling_fuction (geneticml.algorithms.DefaultDataSampler): A function used to up/down sampling the trainset
            greater_is_better (bool) Whether evaluation_function is a score function (default), meaning high is good, or a loss function, meaning low is good. In the latter case, the scorer object will sign-flip the outcome of the evaluation_function.
            verbose (bool): True if should verbose or False if not

        Returns:
            (List[BaseEstimator]): A list with the final population sorted by their loss
        """

        estimators = self._strategy.create_population(population)

        if verbose:
            increment = 100 / generations
            pbar = pbar if pbar else tqdm(total=100)

        if test_data is None:
            test_data = train_data

        # Evolve the generation.
        for i in range(generations):

            # Train and get loss for networks.
            losses = []
            for x in estimators:

                # Do the data balancing if the estimator have it
                if x.has_data_balancing:
                    print(len(train_data.data))
                    xtrain, ytrain = x.data_balance(data=train_data.data, target=train_data.target)
                    train_data = DataLoader(data=xtrain, target=ytrain)
                    print(len(train_data.data))

                # Train the model
                x.fit(train_data.data, train_data.target)

                # Do the model inference
                y_pred = x.predict(test_data.data)

                # Peform the evaluation
                loss = evaluation_function(test_data.target, y_pred)

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
