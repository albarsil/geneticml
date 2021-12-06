from sklearn.neural_network import MLPClassifier
from geneticml.algorithms import BaseEstimator
from typing import Callable
from metrics import metric_accuracy


class SklearnMLPEstimator(BaseEstimator):

    """
    A base class to be used for scikit-learn model optimization
    """

    def __init__(self, parameters: dict) -> None:
        """
        Create a class instance

        Parameters:
            parameters (dict): Possible model paremters
        """
        super.__init__(MLPClassifier, parameters)

    @property
    def parameters(self) -> dict:
        """
        Property to access the base model parameters

        Returns:
            (dict): The parameters
        """
        return self._parameters

    @property
    def model(self):
        """
        Property to access the model

        Returns:
            (?): The base model
        """

        return self._model


    def fit(self, x, y) -> None:
        """
        Method that performs model training. The inheriting class must implement this method.

        Parameters:
            x (?): The data set
            y (?): The correct label set
        """

        self._model.fit(x, y)

    def eval(self, x, y) -> None:
        """
        Method that performs model inference. The inheriting class must implement this method.

        Parameters:
            x (?): The data set
            y (?): The correct label set
        """
        return self._model.predict(x, y)
