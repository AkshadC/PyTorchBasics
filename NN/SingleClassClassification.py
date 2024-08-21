import sys
import torch
import matplotlib.pyplot as plt
from torch import nn
import sklearn
from sklearn.datasets import make_circles
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar
from utils import *
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def print_model_output_per_epoch(loss, train_acc, test_loss, test_accuracy, epoch):
    pass


def plot_dataset(X_train, X_test, y_train, y_test, predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(X_train, y_train, c='b', s=4, label="Training Data")
    plt.scatter(X_test, y_test, c='r', s=4, label="Testing Data")

    if predictions is not None:
        plt.scatter(y_test, predictions, c="g", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()


class CircleClassification(nn.Module):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=32)
        self.layer_2 = nn.Linear(in_features=32, out_features=32)
        self.layer_3 = nn.Linear(in_features=32, out_features=1)
        self.relu = nn.ReLU()
        self.optimizer = None

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

    def train_model(self, X_train, X_test, y_train, y_test, epochs):

        loss_function = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(params=self.parameters(), lr=0.1)

        epoch_till_no_improvement = 0
        previous_loss = float('inf')

        for epoch in range(epochs):
            current_lr = self.optimizer.param_groups[0]['lr']

            self.train()
            epoch_loss = 0
            correct_predictions = 0

            # Wrap the DataLoader with tqdm to track progress within the epoch

            for X_batch, y_batch in tqdm(zip(X_train, y_train), total=len(X_train), desc=f"Epoch {epoch + 1}/{epochs}"):
                y_logits = self.forward(X_batch.unsqueeze(0)).squeeze()
                y_preds = torch.round(torch.sigmoid(y_logits))

                loss = loss_function(y_logits, y_batch)
                epoch_loss += loss.item()

                correct_predictions += torch.eq(y_batch, y_preds).sum().item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_acc = (correct_predictions / len(y_train)) * 100

            self.eval()
            with torch.inference_mode():
                test_logits = self(X_test).squeeze()
                test_preds = torch.round(torch.sigmoid(test_logits))
                test_loss = loss_function(test_logits, y_test)
                test_accuracy = calculate_classification_accuracy(y_test, test_preds)

            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"Train Loss: {epoch_loss / len(X_train):.5f} | Train Accuracy: {train_acc:.2f}%, "
                  f"Test Loss: {test_loss:.5f} | Test Accuracy: {test_accuracy:.2f}% | "
                  f"Learning Rate -> {current_lr}")


def calculate_classification_accuracy(y_true, y_preds):
    correct = torch.eq(y_true, y_preds).sum().item()
    accuracy = (correct / len(y_preds)) * 100
    return accuracy


def make_circle_dataset(n_samples=100000, noise=0.03, random_state=42):
    X, y = make_circles(n_samples=n_samples, noise=noise, random_state=random_state)
    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

    return X_train, X_test, y_train, y_test


def main():
    X_train, X_test, y_train, y_test = make_circle_dataset()

    X_train = X_train.to(DEVICE)
    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)
    y_train = y_train.to(DEVICE)

    model_0 = CircleClassification().to(DEVICE)

    model_0.train_model(X_train, X_test, y_train, y_test, 10)


if __name__ == "__main__":
    main()
