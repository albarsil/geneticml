"""
Class that holds a genetic algorithm for evolving a estimator.

Credit:
    A lot of those code was originally inspired by:
    https://github.com/mrpeel/genetic-keras
"""

import random
from abc import ABC, abstractmethod
from functools import reduce
from operator import add
from typing import List, Type, TypeVar

from geneticml.algorithms import BaseEstimator, EstimatorParameters

T = TypeVar('T', bound=BaseEstimator)


class BaseStrategy(ABC):
    def __init__(self, estimator_type: Type[BaseEstimator]) -> None:
        super().__init__()
        pass

    @abstractmethod
    def execute(self, population: List[Type[T]]) -> List[T]:
        """
        Execute the strategy on a population

        Parameters:
            population (list): A list of estimator parameters

        Returns:
            (list): The population
        """
        pass

class StrategyParameters(object):
    """
    A class to wrap the strategy parameters
    """

    def __init__(self, model_parameters: dict, data_balancing_parameters: dict = None):
        """
        Create a class instance

        Parameters:
            model_parameters (?): The model parameters
            data_balancing_parameters (?): The data balancing parameters
        """
        self._model_parameters = model_parameters
        self._data_balancing_parameters = data_balancing_parameters
        self._balancing_key = 'DATA_BALACING'

    @property
    def model_parameters(self) -> dict:
        """
        Property to get the model parameters

        Returns:
            (dict): The model parameters
        """
        return self._model_parameters

    @property
    def data_balancing_parameters(self) -> dict:
        """
        Property to get the data balancing parameters

        Returns:
            (dict): The data balancing parameters
        """
        return self._data_balancing_parameters

class EvolutionaryStrategy(BaseStrategy):
    """Class that implements genetic algorithm for MLP optimization."""

    def __init__(self, estimator: Type[BaseEstimator], parameters: StrategyParameters, retain: float = 0.4, random_select: float = 0.1, mutate_chance: float = 0.2, max_children: int = 2, random_state: int = 1231) -> None:
        """
        Create an optimizer.

        Parameters:
            estimator (Type[BaseEstimator]): Any instance that inherits from the type defined
            parameters (strategy.StrategyParameters): Possible model paremters
            retain (float): Percentage of population to retain after each generation
            random_select (float): Probability of a rejected estimator remaining in the population
            mutate_chance (float): Probability a estimator will be randomly mutated
            max_children (int): The maximum size of babies that every family could have
            random_state (int): The random state used as seed for the algorithms
        """
        super().__init__(estimator)
        self._estimator = estimator
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self._parameters = parameters
        self._max_children = max_children
        self._random_state = random_state

        random.seed(self._random_state)

    def create_random_set(self) -> EstimatorParameters:
        """
        Generate a random set of model parameters

        Returns:
            (EstimatorParameters): An EstimatorParameters set of parameters
        """

        # Define the seed
        random.seed(self._random_state)

        if self._parameters.data_balancing_parameters is None:
            balance_params = None
        else:
            balance_params = {key:random.choice(self._parameters.data_balancing_parameters[key]) for key in self._parameters.data_balancing_parameters} 

        params = EstimatorParameters(
            model_parameters={key:random.choice(self._parameters.model_parameters[key]) for key in self._parameters.model_parameters},
            data_balancing_parameters=balance_params
        )

        return params

    def create_population(self, size: int) -> List[T]:
        """
        Create a population of random networks.

        Parameters:
            size (int): Number of networks to generate, aka the size of the population

        Returns:
            (list): Population of algorithms.BaseAlgorithm objects
        """

        return [self._estimator.initialize(self.create_random_set()) for val in range(0, size)]

    def grade(self, population: list) -> float:
        """
        Find average fitness for a population.

        Parameters:
            population (list): The population of networks

        Returns:
            (float): The average loss of the population
        """
        summed = reduce(add, (estimator.fitness for estimator in population))
        return summed / float((len(population)))

    def breed(self, parent1: T, parent2: T) -> List[T]:
        """
        Make children as parts of their parents.

        Parameters:
            parent1 (dict): The model parameters
            parent2 (dict): The model parameters

        Returns:
            (List[BaseEstimator]): Estimator objects
        """

        # Define the seed
        random.seed(self._random_state)

        children = random.randint(1, self._max_children)

        estimators = []
        for _ in range(0, children):

            if self._parameters.data_balancing_parameters is None:
                balance_params = None
            else:
                balance_params = {param:random.choice([parent1.parameters.data_balancing_parameters[param], parent2.parameters.data_balancing_parameters[param]]) for param in self._parameters.data_balancing_parameters}

            model_params = {param:random.choice([parent1.parameters.model_parameters[param], parent2.parameters.model_parameters[param]]) for param in self._parameters.model_parameters}

            estimators.append(self._estimator.initialize(EstimatorParameters(model_parameters=model_params, data_balancing_parameters=balance_params)))

        return estimators

    def mutate(self, estimator_parameters: EstimatorParameters) -> EstimatorParameters:
        """
        Randomly mutate one part of the parameters.

        Parameters:
            estimator_parameters (EstimatorParameters): The estimator parameters to mutate

        Returns:
            (EstimatorParameters): A randomly mutated parameters

        """

        # Define the seed
        random.seed(self._random_state)

        # Choose a random key.
        key = random.choice(list(estimator_parameters.model_parameters.keys()))
        estimator_parameters.model_parameters[key] = random.choice(self._parameters.model_parameters[key])

        if estimator_parameters.data_balancing_parameters is not None:
            key = random.choice(list(estimator_parameters.data_balancing_parameters.keys()))
            estimator_parameters.data_balancing_parameters[key] = random.choice(self._parameters.data_balancing_parameters[key])

        return estimator_parameters

    def execute(self, population: List[Type[T]]) -> List[T]:
        """
        Evolve a population of networks.

        Parameters:
            population (list): A list of estimator parameters

        Returns:
            (list): The evolved population of networks
        """

        # Define the seed
        random.seed(self._random_state)

        # Get scores for each estimator.
        graded = [(estimator.fitness, estimator) for estimator in population]

        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=False)]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded) * self.retain)

        # The parents are every estimator we want to keep.
        parents = graded[:retain_length]

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Randomly mutate some of the networks we're keeping.
        for i in range(len(parents)):
            if self.mutate_chance > random.random():
                parents[i] = self._estimator.initialize(parameters=self.mutate(parents[i].parameters))

        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(population) - parents_length
        children = []
        
        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            parent1 = 0 if parents_length == 0 else random.randint(0, parents_length - 1)
            parent2 = 0 if parents_length == 0 else random.randint(0, parents_length - 1)

            # Assuming they aren't the same estimator...
            if parent1 != parent2:
                parent1 = parents[parent1]
                parent2 = parents[parent2]

                # Breed them.
                babies = self.breed(parent1, parent2)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents
