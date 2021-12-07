from __future__ import annotations
from typing import Callable


class EstimatorBuilder(object):
    """
    Examples
    --------

    estimator = EstimatorBuilder.of(class).parameters(dict).fit_with(fit_func).predict_with(predict_func).build()
    
    """

    def of(self, model_type) -> EstimatorBuilder:
        """
        Assign a model type for the estimator

        Parameters:
            model_type (?): A model type

        Returns:
            (EstimatorBuilder): The current object
        """

        self._model_type = model_type
        return self

    def fit_with(self, func: Callable) -> EstimatorBuilder:
        """
        Define a function that will be used for the model training

        Parameters:
            func (Callable): A fit function used for model training

        Returns:
            (EstimatorBuilder): The current object
        """

        self._fit = func
        return self

    def predict_with(self, func: Callable) -> EstimatorBuilder:
        """
        Define a function that will be used for the model inference

        Parameters:
            func (Callable): A predict function used for model inference

        Returns:
            (EstimatorBuilder): The current object
        """

        self._predict = func
        return self

    def build(self) -> BaseEstimator:
        """
        Creates an instance of BaseEstimator

        Returns:
            (BaseEstimator): An instance of BaseEstimator that will be used for the optimization
        """

        return BaseEstimator(model_type=self._model_type, fit_func=self._fit, predict_func=self._predict)


class BaseEstimator(object):

    """
    A base class to be used for model optimization
    """

    def __init__(self, model_type, fit_func: Callable, predict_func: Callable) -> None:
        """
        Create a class instance

        Parameters:
            model_type (?): A model type
            parameters (dict): Possible model parameters
            fit_func (Callable): A fit function used for model training
            predict_func (Callable): A predict function used for model inference
        """
        
        self._parameters = None
        self._model = None
        self._model_type = model_type
        self._fit_func = fit_func
        self._predict_func = predict_func
        self._fitness = -1

    def initialize(self, parameters: dict) -> BaseEstimator:
        """
        Create a class instance

        Parameters:
            parameters (dict): Possible model parameters
        Returns:
            (BaseEstimator): The current object with the model initialized
        """

        self._parameters = parameters
        self._model = self._model_type(**self._parameters)
        return self

    @property
    def model_type(self):
        self._model_type

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
        Property to create a model from the model type and parameters

        Returns:
            (?): The base model
        """

        return self._model

    @property
    def fitness(self) -> float:
        """
        Property to access the base model metric value

        Returns:
            (float): The metric value
        """

        return self._fitness

    @fitness.setter
    def fitness(self, value) -> None:
        """
        Setter

        Parameters:
            value (float): The new fitness value
        """
        self._fitness = value

    def fit(self, x, y) -> None:
        """
        Method that performs model training. The inheriting class must implement this method.

        Parameters:
            x (?): The data set
            y (?): The correct label set
        """
        self._model = self._fit_func(self._model, x, y)

    def predict(self, x) -> list:
        """
        Method that performs model inference. The inheriting class must implement this method.

        Parameters:
            x (?): The data set

        Returns:
            (list): The predict output
        """
        return self._predict_func(self._model, x)
