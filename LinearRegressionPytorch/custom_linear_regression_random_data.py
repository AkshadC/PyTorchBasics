import os.path

import torch
from torch import nn as nn
import matplotlib.pyplot as plt

from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(torch.__version__)


class CustomLinearRegressionModel(nn.Module):
    torch.manual_seed(42)  # Setting the seed value for the entire model so that the numbers created using random

    # generators are

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

    def train_model(self, epochs):
        loss_function = nn.L1Loss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

        for epoch in range(epochs):
            self.train()
            y_predictions = self.forward(self.X_train)

            loss = loss_function(y_predictions, self.y_train)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # Testing mode
            self.eval()
            with torch.inference_mode():
                test_predictions = self(self.X_test)

                test_loss = loss_function(test_predictions, self.y_test)

            if epoch % 10 == 0:
                print(f"Epoch: {epoch}/{epochs}: Train Loss -> {loss}, Test Loss -> {test_loss}")
                print(f"Parameters after epoch: {epoch + 1}/{epochs} -> {self.state_dict()}")

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
    loaded_linear = CustomLinearRegressionModel()

    loaded_linear.load_state_dict(torch.load(f="Saved_Models/Linear_Regression_1_Random_data.pth"))

    print(loaded_linear.state_dict())


""" linear_model_0 = CustomLinearRegressionModel()

    linear_model_0.prepare_load_data(start=0, end=1, step=0.02, random_weight=0.7, random_bias=0.3)

    print(f"Model Parameters: {linear_model_0.state_dict()}")

    linear_model_0.train_model(200)
    with torch.inference_mode():
        y_predictions = linear_model_0(linear_model_0.X_test)

    linear_model_0.plot_dataset(y_predictions)

    torch.save(obj=linear_model_0.state_dict(), f=os.path.join("Saved_Models", "Linear_Regression_1_Random_data.pth"))"""

if __name__ == "__main__":
    main()
