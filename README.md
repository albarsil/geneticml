# geneticml

[![Actions Status](https://github.com/albarsil/geneticml/workflows/Tests/badge.svg?branch=main)](https://github.com/albarsil/geneticml/actions/workflows/tests.yml)
[![CodeQL](https://github.com/albarsil/geneticml/workflows/CodeQL/badge.svg?branch=main)](https://github.com/albarsil/geneticml/actions?query=workflow%3ACodeQL)
[![PyPI](https://img.shields.io/pypi/v/geneticml?color=g)](https://pypi.org/project/geneticml/)
[![License](https://img.shields.io/badge/license-MIT-purple)](https://github.com/albarsil/geneticml/blob/master/LICENSE)

This package contains a simple and lightweight genetic algorithm for optimization of any machine learning model.

## Installation

Use pip to install the package from PyPI:

```bash
pip install geneticml
```

## Usage

This package provides a easy way to create estimators and perform the optimization with genetic algorithms. The example below describe in details how to create a simulation with genetic algorithms using evolutionary approach to train a `sklearn.neural_network.MLPClassifier`. A full list of examples could be found [here](https://github.com/albarsil/geneticml/tree/main/examples).


```python
from geneticml.algorithms import (DataLoader, DefaultEstimatorMethods,EstimatorBuilder)
from geneticml.optimizers import GeneticOptimizer
from geneticml.strategy import EvolutionaryStrategy, StrategyParameters
import sklearn.metrics as metrics
from sklearn.datasets import load_iris
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.neural_network import MLPClassifier

if __name__ == "__main__":

    seed = 11412

    # Creates an estimator
    estimator = EstimatorBuilder()\
        .of(model_type=MLPClassifier)\
        .fit_with(DefaultEstimatorMethods.fit)\
        .predict_with(DefaultEstimatorMethods.predict)\
        .build()

    # Defines a strategy for the optimization
    strategy = EvolutionaryStrategy(
        estimatort=estimator,
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
    greater_is_better = True

    # Create the simulation using the optimizer and the strategy
    models = optimizer.simulate(
        train_data=DataLoader(data=data.data, target=data.target),
        generations=generations,
        population=population,
        evaluation_function=metrics.accuracy_score,
        greater_is_better=greater_is_better,
        verbose=True
    )
```

The estimator is the way you define an algorithm or a class that will be used for model instantiation

```python
estimator = EstimatorBuilder().of(model_type=MLPClassifier).fit_with(func=fit).predict_with(func=predict).build()
```

You can use the default functions provided by the `geneticml.algorithms.DefaultEstimatorMethods` or you can speficy your own functions to perform the operations. If you want to create your own, please note that these functions need to use the same signature than the below ones. This happens because the algorithm is generic and needs to know how to perform the fit and predict functions for the models.

```python
# Creates a custom fit method
def fit(model, x, y):
    return model.fit(x, y)

# Creates a custom predict method
def predict(model, x):
    return model.predict(x)
```

## Custom strategy

You can create custom strategies for the optimizers by extending the `geneticml.strategy.BaseStrategy` and implementing the `execute(...)` function.

```python
class MyCustomStrategy(BaseStrategy):
    def __init__(self, estimator_type: Type[BaseEstimator]) -> None:
        super().__init__(estimator_type)

    def execute(self, population: List[Type[T]]) -> List[T]:
        return population
```

The custom strategies will allow you to create optimization strategies to archive your goals. We currently have the evolutionary strategy but you can define your own :)

## Custom optimizer

You can create custom optimizers by extending the `geneticml.optimizers.BaseOptimizer` and implementing the `simulate(...)` function.

```python
class MyCustomOptimizer(BaseOptimizer):
    def __init__(self, strategy: Type[BaseStrategy]) -> None:
        super().__init__(strategy)

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
        estimators = self._strategy.create_population()
        for x in estimators:
            x.fit(data, target)
            y_pred = x.predict(target)
        pass 
```

Custom optimizers will let you define how you want your algorithm to optimize the selected strategy. You can also combine custom strategies and optimizers to archive your desire objective.


## Testing

The following are the steps to create a virtual environment into a folder named "venv" and install the requirements.

```bash
# Create virtualenv
python3 -m venv venv
# activate virtualenv
source venv/bin/activate
# update packages
pip install --upgrade pip setuptools wheel
# install requirements
python setup.py install
```

Tests can be run with `python setup.py test` when the virtualenv is active.

# Contributing
All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.

A detailed overview on how to contribute can be found in the contributing guide. There is also an overview on GitHub.

If you are simply looking to start working with the geneticml codebase, navigate to the GitHub "issues" tab and start looking through interesting issues. Or maybe through using geneticml you have an idea of your own or are looking for something in the documentation and thinking ‘this can be improved’...you can do something about it!

Feel free to ask questions on the mailing the contributors.
