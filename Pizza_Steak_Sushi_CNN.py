import os

from tqdm import tqdm

from utils import *

import torch
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from torch import nn

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from torchsummary import summary
import gc
import traceback

with torch.no_grad():
    torch.cuda.empty_cache()
gc.collect()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class FoodCNN3ClassVGG19(nn.Module):
    def __init__(self, train_dataset, test_dataset, batch_size, num_workers):
        super().__init__()
        self.train_data_loader, self.test_data_loader, self.train_dataset, self.test_dataset = None, None, train_dataset, test_dataset

        self.train_data_loader = DataLoader(self.train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            pin_memory=True
                                            )
        self.test_data_loader = DataLoader(self.test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=num_workers,
                                           pin_memory=True
                                           )
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fcnn_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, len(self.train_dataset.classes))

        )

    def forward(self, x):
        return self.fcnn_stack(
            self.conv_block_2(self.conv_block_1(x)))

    def train_model(self, epochs):
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=self.parameters(), lr=0.001)
        train_loss, train_acc, test_loss, test_acc = 0, 0, 0, 0
        epochs_till_test_loss_change, prev_loss = 0, float('inf')
        train_loss_list, train_acc_list, test_loss_list, test_acc_list = [], [], [], []
        for epoch in range(epochs):
            for X, y in tqdm(self.train_data_loader, total=len(self.train_data_loader),
                             desc=f"Epoch{epoch + 1}/{epochs}"):
                X, y = X.to(DEVICE), y.to(DEVICE)

                self.train()
                pred = self(X)
                loss = loss_function(pred, y)
                train_loss += loss
                train_acc += accuracy_function(y, torch.softmax(pred, dim=1).argmax(dim=1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_acc /= len(self.train_data_loader)
            train_loss /= len(self.train_data_loader)
            train_loss_list.append(train_loss.item())
            train_acc_list.append(train_acc)

            self.eval()
            with torch.inference_mode():
                for X, y in self.test_data_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    pred = self(X)
                    test_loss += loss_function(pred, y)
                    test_acc += accuracy_function(y, torch.softmax(pred, dim=1).argmax(dim=1))
                test_loss /= len(self.test_data_loader)
                test_acc /= len(self.test_data_loader)

            test_loss_list.append(test_loss.item())
            test_acc_list.append(test_acc)

            print(
                f"Train Loss: {train_loss:.2f}, Train Accuracy: {train_acc:.3f}% || Test Loss: {test_loss:.2f}, Test "
                f"Accuracy: {test_acc:.3f}%")

        plot_loss_curves(train_loss_list, train_acc_list, test_loss_list, test_acc_list)

    def plot_random_data(self):
        plt.figure(figsize=(10, 7))
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            image, label = self.train_dataset[torch.randint(0, len(self.train_dataset), size=[1]).item()]
            plt.imshow(image.permute(1, 2, 0), cmap='viridis')
            plt.title(self.train_dataset.classes[label])
            plt.axis(False)
        plt.tight_layout()
        plt.show()

    def eval_model(self):
        loss_function = nn.CrossEntropyLoss()
        loss, acc = 0, 0
        self.eval()
        with torch.inference_mode():
            for X, y in tqdm(self.test_data_loader, desc="Evaluating the Model Please wait!!"):
                X, y = X.to(DEVICE), y.to(DEVICE)

                pred = self(X)
                loss += loss_function(pred, y)
                acc += accuracy_function(y, torch.softmax(pred, dim=1).argmax(dim=1))
            loss /= len(self.test_data_loader)
            acc /= len(self.test_data_loader)

        return {"model_name": self.__class__.__name__,
                "model_loss": loss.item(),
                "model_acc": acc}

    def get_image_shape(self):
        image, label = self.train_dataset[0]
        return image.size()


def plot_loss_curves(train_loss_list, train_acc_list, test_loss_list, test_acc_list):
    epochs = len(train_acc_list)

    plt.figure(figsize=(10, 7))

    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_loss_list, label='train_loss')
    plt.plot(range(epochs), test_loss_list, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), test_loss_list, label='train_accuracy')
    plt.plot(range(epochs), test_acc_list, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


def main():
    try:

        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        train_dataset = ImageFolder(
            root='/home/thefilthysalad/PycharmProjects/PyTorchBasics/Datasets/pizza_steak_sushi/train',
            transform=data_transforms)
        test_dataset = ImageFolder(
            root='/home/thefilthysalad/PycharmProjects/PyTorchBasics/Datasets/pizza_steak_sushi/test',
            transform=data_transforms)

        model_1 = FoodCNN3ClassVGG19(train_dataset, test_dataset, 32, os.cpu_count()).to(DEVICE)
        # model_1.plot_random_data()
        summary(model_1, model_1.get_image_shape())

        model_1.train_model(10)
        print(model_1.eval_model())

    except:
        traceback.print_exc()

    finally:
        del model_1
        del train_dataset
        del test_dataset
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
