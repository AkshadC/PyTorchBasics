import torch
from torch import nn as nn
import matplotlib.pyplot as plt

print(torch.__version__)


class CustomLinearRegressionModel(nn.Module):
    torch.manual_seed(
        42)  # Setting the seed value for the entire model so that the numbers created using random generators are

    # the same for the entirety

    def __init__(self):
        super().__init__()

        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None

        self.weight = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        One forward pass for the dataset on a linear regression model
        :param x: The train dataset or the X and must be a tensor
        :return: Returns the value of Y = wx + b for the given x
        """
        return self.weight * x + self.bias

    # Utility Functions

    def prepare_load_data(self, **kwargs):
        """
        Creates a train test split dataset for a particular instance of the linear model
        :return: returns a train test split for data generated using the given parameters
        """
        X = torch.arange(kwargs['start'], kwargs['end'], kwargs['step']).unsqueeze(dim=1)
        y = kwargs['random_weight'] * X + kwargs['random_bias']
        train_split = int(0.8 * len(X))
        X_train, y_train = X[:train_split], y[:train_split]
        X_test, y_test = X[train_split:], y[train_split:]

        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test

    def plot_dataset(self, predictions=None):
        """
        :return: Plots the dataset and predictions if not None
        """
        plt.figure(figsize=(10, 7))
        plt.scatter(self.X_train, self.y_train, c='b', s=4, label="Training Data")
        plt.scatter(self.X_test, self.y_test, c='r', s=4, label="Testing Data")

        if predictions is not None:
            plt.scatter(self.y_test, predictions, c="g", s=4, label="Predictions")

        plt.legend(prop={"size": 14})
        plt.show()


def main():
    linear_model_0 = CustomLinearRegressionModel()
    linear_model_0.prepare_load_data(start=0, end=1, step=0.02, random_weight=0.6, random_bias=0.9)
    linear_model_0.plot_dataset()
    print(f"Model Parameters: {linear_model_0.state_dict()}")

    with torch.inference_mode():
        y_predictions = linear_model_0(linear_model_0.X_test)

    linear_model_0.plot_dataset(predictions=y_predictions)


if __name__ == "__main__":
    main()
