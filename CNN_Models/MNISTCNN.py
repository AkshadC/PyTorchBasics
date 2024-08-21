import random

import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
from utils import *

from torch import nn

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_data_loader, self.test_data_loader, self.train_data, self.test_data = None, None, None, None
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 10, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(10, 10, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(10, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fcnn_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 4096),
            nn.Linear(4096, out_features=10)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.fcnn_stack(x)
        # print(x.shape)
        return x

    def train_model(self, epochs):
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params=self.parameters(), lr=0.1)
        train_loss, train_acc, test_loss, test_acc = 0, 0, 0, 0
        epochs_till_test_loss_change = 0
        prev_loss = float('inf')

        for epoch in range(epochs):
            self.train()
            for X, y in tqdm(self.train_data_loader, total=len(self.train_data_loader), desc=f"Epoch {epoch}/{epochs}"):

                X, y, = X.to(DEVICE), y.to(DEVICE)
                pred = self(X)

                loss = loss_function(pred, y)
                train_loss += loss
                train_acc += accuracy_function(y, pred.argmax(dim=1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss /= len(self.train_data_loader)
            train_acc /= len(self.train_data_loader)

            self.eval()
            with torch.inference_mode():
                for X, y in self.test_data_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    pred = self(X)
                    loss = loss_function(pred, y)
                    test_loss += loss
                    test_acc += accuracy_function(y, pred.argmax(dim=1))
                test_loss /= len(self.test_data_loader)
                test_acc /= len(self.test_data_loader)
            if test_loss < prev_loss:
                prev_loss = test_loss
                epochs_till_test_loss_change = 0
            else:
                epochs_till_test_loss_change += 1

            if epochs_till_test_loss_change == 4:
                new_lr = optimizer.param_groups[0]['lr'] * 0.3
                print(f"Updating Learning Rate to {new_lr:.6f}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                epochs_till_test_loss_change = 0
            print(
                f"Train Loss: {train_loss:.2f}, Train Accuracy: {train_acc:.3f}% || Test Loss: {test_loss:.2f}, Test "
                f"Accuracy: {test_acc:.3f}%")

    def eval_model(self):
        loss, acc = 0, 0
        self.eval()
        loss_function = nn.CrossEntropyLoss()
        with torch.inference_mode():
            for X, y in self.test_data_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)

                y_pred = self(X)
                loss += loss_function(y_pred, y)
                acc += accuracy_function(y, y_pred.argmax(dim=1))
            loss /= len(self.test_data_loader)
            acc /= len(self.test_data_loader)

        return {"model_name": self.__class__.__name__,
                "model_loss": loss.item(),
                "model_acc": acc}

    def make_predictions(self):
        y_preds = []
        self.eval()
        for X, y in tqdm(self.test_data_loader, desc="Making Predictions"):
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_logits = self(X)
            y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
            y_preds.append(y_pred.cpu())
        return torch.cat(y_preds)

    def plot_confusionmatrix(self):
        prediction_tensor = self.make_predictions()
        confusion_matrix = ConfusionMatrix(num_classes=len(self.train_data.classes), task="multiclass")
        confusion_matrix_tensor = confusion_matrix(preds=prediction_tensor, target=self.test_data.targets)

        fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix_tensor.numpy(), class_names=self.train_data.classes,
                                        figsize=(10, 7))
        plt.show()

    def plot_random_images(self):
        plt.figure(figsize=(10, 7))

        for i in range(5):
            plt.subplot(1, 5, i + 1)
            image, label = self.train_data[torch.randint(0, len(self.train_data), size=[1]).item()]
            plt.imshow(torch.squeeze(image), cmap='gray')
            plt.title(self.train_data.classes[label])
            plt.axis(False)
        plt.tight_layout()
        plt.show()

    def plot_random_predictions(self):

        test_samples = []
        test_labels = []
        for sample, label in random.sample(list(self.test_data), 9):
            test_samples.append(sample)
            test_labels.append(label)
        pred_probs = []
        self.eval()
        with torch.inference_mode():
            for sample in test_samples:
                sample = torch.unsqueeze(sample, dim=0).to(DEVICE)

                pred_logits = self(sample)
                pred_probs.append(torch.softmax(pred_logits.squeeze(), dim=0).cpu())

        pred_probs = torch.stack(pred_probs)

        predictions = pred_probs.argmax(dim=1)

        plt.figure(figsize=(10, 7))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(test_samples[i].squeeze(), cmap='gray')
            if self.test_data.classes[predictions[i]] == self.test_data.classes[test_labels[i]]:
                plt.title(self.test_data.classes[predictions[i]], c='g')
            else:
                plt.title(self.test_data.classes[predictions[i]], c='r')

            plt.axis(False)
        plt.tight_layout()
        plt.show()


def main():
    train_data = datasets.MNIST("../TorchDatasets", train=True, transform=ToTensor(), download=True)
    test_data = datasets.MNIST("../TorchDatasets", train=False, transform=ToTensor(), download=True)

    model_1 = MNISTCNN().to(DEVICE)
    train_data_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model_1.train_data_loader, model_1.test_data_loader, model_1.train_data, model_1.test_data = train_data_loader, test_data_loader, train_data, test_data

    class_names = train_data.classes
    print(class_names)

    # model_1.plot_random_images()

    # random_tensor = torch.randn(1, 28, 28).to(DEVICE)
    # model_1(random_tensor)

    model_1.train_model(5)
    model_1.plot_confusionmatrix()
    model_1.plot_random_predictions()
    torch.save(model_1.state_dict(), f="Saved_Models/MnistCNNBest.pth")
    print("MODEL SAVED")


if __name__ == "__main__":
    main()
