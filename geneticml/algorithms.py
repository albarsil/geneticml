from abc import abstractmethod


class BaseEstimator(object):

    """
    A base class to be used for model optimization
    """

    def __init__(self, model, parameters: dict) -> None:
        """
        Create a class instance

        Parameters:
            instance (?): A model class instance
            parameters (dict): Possible model paremters
        """
        self._parameters = parameters
        self._model = model(**parameters)
        self._fitness = -1

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

    @abstractmethod
    def fit(self, x, y) -> None:
        """
        Method that performs model training. The inheriting class must implement this method.

        Parameters:
            x (?): The data set
            y (?): The correct label set
        """
        pass

    @abstractmethod
    def predict(self, x) -> None:
        """
        Method that performs model inference. The inheriting class must implement this method.

        Parameters:
            x (?): The data set
        """
        pass
