"""
Class that holds a genetic algorithm for evolving a estimator.

Credit:
    A lot of those code was originally inspired by:
    https://github.com/mrpeel/genetic-keras
"""

from abc import ABC, abstractmethod
from typing import List, Type, TypeVar
from functools import reduce
from operator import add
import random
from geneticml.algorithms import BaseEstimator

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


class EvolutionaryStrategy(BaseStrategy):
    """Class that implements genetic algorithm for MLP optimization."""

    def __init__(self, estimator_type: Type[BaseEstimator], parameters: dict, retain: float = 0.4, random_select: float = 0.1, mutate_chance: float = 0.2, max_children: int = 2, random_state: int = 1231) -> None:
        """
        Create an optimizer.

        Parameters:
            estimator_type (Type[BaseEstimator]): Any instance that inherits from the type defined
            parameters (dict): Possible model paremters
            retain (float): Percentage of population to retain after each generation
            random_select (float): Probability of a rejected estimator remaining in the population
            mutate_chance (float): Probability a estimator will be randomly mutated
            max_children (int): The maximum size of babies that every family could have
            random_state (int): The random state used as seed for the algorithms
        """
        super().__init__(estimator_type)
        self._estimator_type = estimator_type
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self._parameters = parameters
        self._max_children = max_children
        self._random_state = random_state

        random.seed(self._random_state)

    def create_random_set(self) -> dict:
        """
        Generate a random set of model parameters

        Returns:
            (dict): A new set of parameters
        """

        # Define the seed
        random.seed(self._random_state)

        params = {}
        for key in self._parameters:
            params[key] = random.choice(self._parameters[key])
        return params

    def create_population(self, size: int) -> List[T]:
        """
        Create a population of random networks.

        Parameters:
            size (int): Number of networks to generate, aka the size of the population

        Returns:
            (list): Population of algorithms.BaseAlgorithm objects
        """

        return [self._estimator_type.initialize(self.create_random_set()) for val in range(0, size)]

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

        return [self._estimator_type.initialize(parameters={param: random.choice([parent1.parameters[param], parent2.parameters[param]]) for param in self._parameters}) for _ in range(0, children)]

    def mutate(self, parameters: dict) -> dict:
        """
        Randomly mutate one part of the parameters.

        Parameters:
            parameters (dict): The model parameters to mutate

        Returns:
            (dict): A randomly mutated parameters

        """

        # Define the seed
        random.seed(self._random_state)

        # Choose a random key.
        mutation = random.choice(list(self._parameters.keys()))

        # Mutate one of the params.
        parameters[mutation] = random.choice(self._parameters[mutation])

        return parameters

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
                parents[i] = self._estimator_type.initialize(parameters=self.mutate(parents[i].parameters))

        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(population) - parents_length
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            parent1 = random.randint(0, parents_length - 1)
            parent2 = random.randint(0, parents_length - 1)

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
