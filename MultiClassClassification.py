import torch
import matplotlib.pyplot as plt
from torch import nn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy
import utils
from utils import *
import numpy as np
from sklearn.model_selection import train_test_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def plot_boundary(model):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model, model.X_train, model.y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model, model.X_test, model.y_test)

    plt.show()

class SpiralModelNN(nn.Module):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    def __init__(self):
        super().__init__()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=2, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=3),
        )

    def forward(self, x):
        return self.layer_stack(x)

    def make_dataset(self, N=1000, dimensionality=2, num_classes=3, plot=False):

        X = np.zeros((N * num_classes, dimensionality))
        y = np.zeros(N * num_classes, dtype='uint8')  # class labels

        for j in range(num_classes):
            ix = range(N * j, N * (j + 1))
            r = np.linspace(0.0, 1, N)
            t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            y[ix] = j

        if plot:
            plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
            plt.show()
        X = torch.from_numpy(X).type(torch.float)
        y = torch.from_numpy(y).type(torch.LongTensor)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = self.X_train.to(DEVICE), self.X_test.to(
            DEVICE), self.y_train.to(DEVICE), self.y_test.to(DEVICE)

    def train_model(self, epochs):

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=self.parameters(), lr=0.01)

        for epoch in range(epochs):
            self.train()

            train_logits = self(self.X_train)
            train_preds = torch.softmax(train_logits, dim=1).argmax(dim=1)

            train_loss = loss_function(train_logits, self.y_train)
            train_acc = accuracy_function(self.y_train, train_preds)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            self.eval()
            with torch.inference_mode():
                test_logits = self(self.X_test)
                test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)
                test_loss = loss_function(test_logits, self.y_test)

                test_acc = accuracy_function(self.y_test, test_preds)
            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}/{epochs}, Train Loss = {train_loss:.5f}, Train Accuracy = {train_acc:.2f}% || Test Loss = {test_loss:.5f}, Test Accuracy = {test_acc:.2f}")


class MultiClassNN(nn.Module):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    def __init__(self):
        super().__init__()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=2, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=4)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)

    def train_model(self, epochs, lr=0.1):

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params=self.parameters(), lr=lr)

        epochs_till_no_test_loss_change = 0
        prev_loss = 0

        for epoch in range(epochs):
            current_learning_rate = optimizer.param_groups[0]['lr']
            self.train()
            train_logits = self.forward(self.X_train)
            train_preds = torch.softmax(train_logits, dim=1).argmax(dim=1)

            loss = loss_function(train_logits, self.y_train)

            acc = utils.accuracy_function(self.y_train, train_preds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.eval()
            with torch.inference_mode():
                test_logits = self(self.X_test)
                test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)
                test_loss = loss_function(test_logits, self.y_test)
                test_acc = accuracy_function(self.y_test, test_preds)

            if test_loss < prev_loss:
                epochs_till_no_test_loss_change = 0
            else:
                epochs_till_no_test_loss_change += 1

            if epochs_till_no_test_loss_change == 3:
                print(f"Loss didnt improve from {prev_loss}, Changing the learning rate after epoch: {epoch}")
                optimizer = torch.optim.SGD(self.parameters(), lr=current_learning_rate / 2)
                epochs_till_no_test_loss_change = 0

            prev_loss = test_loss

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}/{epochs}, Train Loss = {loss:.5f}, Train Accuracy = {acc:.2f}% || Test Loss = {test_loss:.5f}, Test Accuracy = {test_acc:.2f}")

    def make_dataset(self, num_classes=4, num_features=2, num_samples=5000):
        X_blob, y_blob = make_blobs(n_samples=num_samples,
                                    n_features=num_features,
                                    centers=num_classes,
                                    cluster_std=1.5,
                                    random_state=42)

        X_blob = torch.from_numpy(X_blob).type(torch.float)
        y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_blob, y_blob, test_size=0.3,
                                                                                random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = self.X_train.to(DEVICE), self.X_test.to(
            DEVICE), self.y_train.to(DEVICE), self.y_test.to(DEVICE)
        print(f"Train Shape: {self.X_train.shape}, Test Shape: {self.X_test.shape}")
        print(f"Label Train Shape: {self.y_train.shape}, Label Test Shape: {self.y_test.shape}")

    def plot_dataset(self, predictions=None):
        plt.figure(figsize=(10, 7))
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c='b', label="Training Data")
        plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c='b', label="Test Data")

        if predictions is not None:
            plt.scatter(predictions[:, 0], predictions[:, 1], c='g', label="Test Data")
        plt.show()


def main():
    """    model_0 = MultiClassNN()
    model_0.to(DEVICE)
    model_0.make_dataset()
    model_0.train_model(100)



    model_0.eval()
    with torch.inference_mode():
        preds = model_0(model_0.X_test)
    preds = torch.softmax(preds, dim=1).argmax(dim=1)
    print(f"Test Accuracy: {accuracy_function(model_0.y_test, preds):.2f}%")

    torch_metric_acc = Accuracy(task="multiclass", num_classes=4).to(DEVICE)
    print(torch_metric_acc(preds, model_0.y_test))"""

    model_spiral = SpiralModelNN().to(DEVICE)
    model_spiral.make_dataset()
    model_spiral.train_model(900)
    plot_boundary(model_spiral)

if __name__ == "__main__":
    main()
