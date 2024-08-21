import random
import torch
from tqdm import tqdm
from utils import *
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fcnn_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32 * 7 * 7, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        # print(x.shape)
        x = self.conv_block2(x)
        # print(x.shape)
        x = self.fcnn_layers(x)
        # print(x.shape)
        return x

    def train_model(self, epochs, train_data, test_data):

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params=self.parameters(), lr=0.1)
        epochs_till_test_loss_change, prev_loss = 0, float('inf')
        train_loss, train_acc = 0, 0
        test_loss, test_acc = 0, 0
        for epoch in range(epochs):
            for X, y in tqdm(train_data, total=len(train_data), desc=f"Epoch {epoch + 1}/{epochs}"):
                X, y = X.to(DEVICE), y.to(DEVICE)
                pred = self(X)

                loss = loss_function(pred, y)
                train_loss += loss
                train_acc += accuracy_function(y, pred.argmax(dim=1))

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            train_loss /= len(train_data)
            train_acc /= len(train_data)

            self.eval()
            with torch.inference_mode():
                for X, y in test_data:
                    X, y = X.to(DEVICE), y.to(DEVICE)

                    test_pred = self(X)

                    test_loss += loss_function(test_pred, y)
                    test_acc += accuracy_function(y, test_pred.argmax(dim=1))

                test_loss /= len(test_data)
                test_acc /= len(test_data)

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
                f"Train Loss: {train_loss:.2f}, Train Accuracy: {train_acc:.3f}% || Test Loss: {test_loss:.2f}, Test Accuracy: {test_acc:.3f}%")


def plot_random_images(train_data, classes):
    plt.figure(figsize=(10, 7))
    for i in range(10):
        image, label = train_data[torch.randint(0, len(train_data), size=[1]).item()]
        plt.subplot(2, 5, i + 1)
        plt.imshow(torch.squeeze(image), cmap='gray')
        plt.title(classes[label])
        plt.axis(False)
    plt.tight_layout()
    plt.show()


def make_random_predictions(model: torch.nn.Module, data):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(DEVICE)

            pred_logits = model(sample)
            pred_probs.append(torch.softmax(pred_logits.squeeze(), dim=0).cpu())

    return torch.stack(pred_probs)


def make_predictions(model, test_data):
    y_preds = []
    model.eval()
    for X, y in tqdm(test_data, desc="Making Predictions"):
        X, y = X.to(DEVICE), y.to(DEVICE)
        y_logits = model(X)
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
        y_preds.append(y_pred.cpu())
    return torch.cat(y_preds)


def plot_confusionmatrix(prediction_tensor, test_data, class_names):
    confusion_matrix = ConfusionMatrix(num_classes=len(class_names), task="multiclass")
    confusion_matrix_tensor = confusion_matrix(preds=prediction_tensor, target=test_data.targets)

    fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix_tensor.numpy(), class_names=class_names, figsize=(10, 7))
    plt.show()


def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Make predictions with the model
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_pred = model(X)

            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                               y_pred=y_pred.argmax(
                                   dim=1))  # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)

        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__,  # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}


def main():
    # Loading the train and test data from torchvision datasets
    train_data = datasets.FashionMNIST(root="TorchDatasets", train=True, download=True,
                                       transform=ToTensor(),
                                       target_transform=None)

    test_data = datasets.FashionMNIST(root="TorchDatasets", train=False, download=True, transform=ToTensor(),
                                      target_transform=None)

    class_names = train_data.classes  # Class names present in the dataset
    # class_names_indexed = train_data.class_to_idx

    # print(class_names_indexed)
    # plot_random_images(train_data, class_names)

    BATCH_SIZE = 32

    # Making dataloaders for train and test
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    # print(len(train_data_loader))
    # print(len(test_data_loader))

    # Initializing the CNN model
    model_1 = FashionMNISTCNN().to(DEVICE)

    # Training the model
    model_1.train_model(5, train_data_loader, test_data_loader)

    # dummy_image = torch.randn(size=(1, 28, 28)).to(DEVICE)
    # model_1(dummy_image)

    # Evaluating the model and saving the results
    model_1_results = eval_model(model_1, test_data_loader, nn.CrossEntropyLoss(), accuracy_fn=accuracy_function)
    print(model_1_results)

    # Choosing 9 random samples and plotting the prediction
    test_samples = []
    test_labels = []
    for sample, label in random.sample(list(test_data), 9):
        test_samples.append(sample)
        test_labels.append(label)

    pred_probs = make_random_predictions(model_1, test_samples)
    predictions = pred_probs.argmax(dim=1)

    plt.figure(figsize=(10, 7))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(test_samples[i].squeeze(), cmap='gray')
        if class_names[predictions[i]] == class_names[test_labels[i]]:
            plt.title(class_names[predictions[i]], c='g')
        else:
            plt.title(class_names[predictions[i]], c='r')

        plt.axis(False)
    plt.tight_layout()
    plt.show()

    # Plotting the confusion matrix
    plot_confusionmatrix(make_predictions(model_1, test_data_loader), test_data, class_names)

    # Saving the current best model
    torch.save(model_1.state_dict(), f="Saved_Models/FashionMnistCNNBest.pth")
    print("MODEL SAVED")

    # Loading the saved model
    loaded_model = FashionMNISTCNN()
    loaded_model.load_state_dict(torch.load(f="Saved_Models/FashionMnistCNNBest.pth"))

    loaded_model = loaded_model.to(DEVICE)

    # Predicting on the test data using the loaded model to check if it's the same parameters.
    loaded_model_results = eval_model(loaded_model, test_data_loader, nn.CrossEntropyLoss(),
                                      accuracy_fn=accuracy_function)

    print(loaded_model_results)


if __name__ == "__main__":
    main()
