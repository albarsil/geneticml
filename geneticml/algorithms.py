from typing import Callable, Tuple


class DataLoader(object):
    """
    A data loader object to create an abstraction for data points and targets
    """

    def __init__(self, data, target):
        """
        Create a class instance

        Parameters:
            data (?): The X data
            data (?): The target data
        """
        
        self._data = data
        self._target = target

    @property
    def data(self):
        return self._data

    @property
    def target(self):
        return self._target

class EstimatorParameters(object):
    """
    A class to wrap the estimator parameters
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

class DefaultEstimatorMethods(object):
    """
    A class with static methods and most common options to fill the estimators
    """

    @staticmethod
    def fit(model, data, target):
        """
        A simple fit function

        Parameters:
            model (?): A model instance
            data (?): The data that will be used to fit the algorithm
            target (?): The target that will be used to fit the algorithm

        Returns:
            (?): The model fitted
        """
        return model.fit(data, target)

    @staticmethod
    def predict(model, data):
        """
        A simple predict function

        Parameters:
            model (?): A model instance
            data (?): The data that will be used for predict

        Returns:
            (?): The model prediction
        """
        return model.predict(data)

    @staticmethod
    def data_balance(balancing_model, data, target) -> Tuple:
        """
        A simple fit function

        Parameters:
            balancing_model (?): A balancing model instance
            data (?): The data that will be used to fit the algorithm
            target (?): The target that will be used to fit the algorithm

        Returns:
            (tuple): A tuple containing the balanced data and targets
        """
        return balancing_model.fit_resample(data, target)

class EstimatorBuilder(object):
    """
    Examples
    --------

    estimator = EstimatorBuilder.of(class).parameters(dict).fit_with(fit_func).predict_with(predict_func).build()
    
    """

    def of(self, model_type) -> 'EstimatorBuilder':
        """
        Assign a model type for the estimator

        Parameters:
            model_type (?): A model type

        Returns:
            (EstimatorBuilder): The current object
        """

        self._model_type = model_type
        self._data_balance_model_type = None
        self._data_balance = None
        return self
    
    def data_balance_algorithm(self, data_balance_model_type) -> 'EstimatorBuilder':
        """
        Assign a data balance algorithm for the estimator

        Parameters:
            data_balance_model_type (?): A data balance model type

        Returns:
            (EstimatorBuilder): The current object
        """

        self._data_balance_model_type = data_balance_model_type
        return self

    def fit_with(self, func: Callable = DefaultEstimatorMethods.fit) -> 'EstimatorBuilder':
        """
        Define a function that will be used for the model training

        Parameters:
            func (Callable): A fit function used for model training

        Returns:
            (EstimatorBuilder): The current object
        """

        self._fit = func
        return self

    def predict_with(self, func: Callable = DefaultEstimatorMethods.predict) -> 'EstimatorBuilder':
        """
        Define a function that will be used for the model inference

        Parameters:
            func (Callable): A predict function used for model inference

        Returns:
            (EstimatorBuilder): The current object
        """

        self._predict = func
        return self

    def data_balance_with(self, func: Callable) -> 'EstimatorBuilder':
        """
        Define a function that will be used for the data balancing

        Parameters:
            func (Callable): A predict function used for data balancing

        Returns:
            (EstimatorBuilder): The current object
        """

        self._data_balance = func
        return self

    def build(self) -> 'BaseEstimator':
        """
        Creates an instance of BaseEstimator

        Returns:
            (BaseEstimator): An instance of BaseEstimator that will be used for the optimization
        """

        return BaseEstimator(model_type=self._model_type, fit_func=self._fit, predict_func=self._predict, data_balance_model_type=self._data_balance_model_type,balance_func=self._data_balance)

class BaseEstimator(object):

    """
    A base class to be used for model optimization
    """

    def __init__(self, model_type, fit_func: Callable, predict_func: Callable, data_balance_model_type = None, balance_func: Callable = None):
        """
        Create a class instance

        Parameters:
            model_type (?): A model type
            fit_func (Callable): A fit function used for model training
            predict_func (Callable): A predict function used for model inference
            data_balance_model_type (?): A data balancing model type
            balance_func (Callable): A data balancing function used for train data balancing
        """
        
        self._parameters = None
        self._model = None
        self._model_type = model_type
        self._data_balance_model = None
        self._data_balance_model_type = data_balance_model_type
        self._fit_func = fit_func
        self._predict_func = predict_func
        self._balance_func = balance_func
        self._fitness = -1
        

    def initialize(self, parameters: EstimatorParameters) -> 'BaseEstimator':
        """
        Create a class instance. It's used by the strategy and shouln't be called directly

        Parameters:
            parameters (algorithms.EstimatorParameters): Possible parameters of the estimator
        Returns:
            (BaseEstimator): The current object with the model initialized
        """

        self._parameters = parameters
        self._model = self._model_type(**parameters.model_parameters)

        if self._parameters.data_balancing_parameters is not None:
            self._data_balance_model = self._data_balance_model_type(**parameters.data_balancing_parameters)
        
        return self

    @property
    def has_data_balancing(self) -> bool:
        """
        Property to disovery if the estimator have a data balancing algorithm

        Returns:
            (dict): The model parameters
        """
        return self._data_balance_model is not None

    @property
    def model_type(self):
        self._model_type

    @property
    def parameters(self) -> dict:
        """
        Property to access the base model parameters

        Returns:
            (dict): The model parameters
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

    @property
    def parameters_balance(self) -> dict:
        """
        Property to access the base model data balance parameters

        Returns:
            (dict): The data balance parameters
        """

        return self._balance_parameters

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

    def data_balance(self, data, target) -> Tuple:
        """
        Create a class instance

        Parameters:
            data (?): The data that will be used to fit the algorithm
            target (?): The target that will be used to fit the algorithm
        Returns:
            (tuple): A tuple containing the balanced data and targets
        """
        if self._data_balance_model is None:
            raise ValueError('A data_balance_type was not specified on the estimator __init__ function')
        else:
            return self._balance_func(self._data_balance_model, data, target)
