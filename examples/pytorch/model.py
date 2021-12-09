import torch
from torch import nn


class SampleNeuralNetwork(nn.Module):
    """
    A simple pytorch neural network
    """

    def __init__(self, n_classes: int, hidden_size: list, dropout_prob: float = 0.2) -> None:
        """
        Initialize the model

        Parameters:
            n_classes (int): The number os model classes
            hidden_size (list[int]): The model hidden layers
            dropout_prob (float): The dropout value between the layers

        For a detailed reference, please check: torch.nn.Module
        """

        super(SampleNeuralNetwork, self).__init__()

        self._drop_prob = dropout_prob
        self._n_classes = n_classes
        self._hidden_layers = hidden_size

        # Define the middle of the neural network. You can use as many layers as you want
        if len(self._hidden_size) == 0:
            raise ValueError("hidden_size value should be greater than 0")
        elif len(self._hidden_size) > 1:
            layers = [
                nn.Sequential(
                    nn.Linear(in_units, out_units),
                    nn.ReLU()
                )
                for in_units, out_units in zip(self._hidden_size, self._hidden_size[1:])
            ]

            self.fc = nn.Sequential(*layers)
        else:
            self.fc = None

        # Define the end of the neural network
        if dropout_prob is None:
            self.fc_out = nn.Sequential(nn.Linear(self._hidden_size[-1], n_classes))
        else:
            self.fc_out = nn.Sequential(
                nn.Dropout(self._drop_prob),
                nn.Linear(self._hidden_size[-1], n_classes)
            )

    def forward(self, inputs):

        if self.fc is not None:
            inputs = self.fc(inputs)

        return torch.sigmoid(self.fc_out(inputs))

    def arch(self) -> dict:
        """
        Get's the model architecture

        Returns:
            dict: The model architecture
        """

        return {
            "n_classes": self._n_classes,
            "dropout_prob": self._drop_prob,
            "hidden_size": ";".join([str(x) for x in self._hidden_layers])
        }
