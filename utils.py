import sys
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from timeit import default_timer as timer


def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def accuracy_function(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100


def print_train_timer(start, end, device=None):
    total_time = end - start
    print(f"Train time on Device: {device}: {total_time:.3f} seconds")
    return total_time

def classifier_accuracy_function(y_true, y_pred):
    preds = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    correct = (preds == y_true).sum().item()
    total = y_true.size(0)  # This is the batch size
    return correct, total

from torch import nn
from tqdm import tqdm
from timeit import default_timer as timer


def train_engine(model, epochs, loss_function, optimizer, train_data_loader, test_data_loader, DEVICE):
    start_time = timer()
    results = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        train_correct = 0
        train_total = 0

        epoch_start_time = timer()

        pbar = tqdm(train_data_loader, total=len(train_data_loader), desc=f"Epoch {epoch + 1}/{epochs} - 0.0s elapsed")
        for X, y in pbar:
            X, y = X.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(X)
            loss = loss_function(pred, y)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            correct, total = classifier_accuracy_function(y, pred)
            train_correct += correct
            train_total += total

            # Update elapsed time in tqdm description
            elapsed = timer() - epoch_start_time
            pbar.set_description(f"Epoch {epoch + 1}/{epochs} - {elapsed:.1f}s elapsed")

        pbar.close()
        sys.stdout.flush()
        avg_train_loss = epoch_train_loss / len(train_data_loader)
        train_accuracy = (train_correct / train_total) * 100
        results['train_loss'].append(avg_train_loss)
        results['train_acc'].append(train_accuracy)

        # Evaluation
        model.eval()
        epoch_test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.inference_mode():
            for X, y in test_data_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                pred = model(X)
                loss = loss_function(pred, y)
                epoch_test_loss += loss.item()
                correct, total = classifier_accuracy_function(y, pred)
                test_correct += correct
                test_total += total

        avg_test_loss = epoch_test_loss / len(test_data_loader)
        test_accuracy = (test_correct / test_total) * 100
        results['test_loss'].append(avg_test_loss)
        results['test_acc'].append(test_accuracy)

        tqdm.write(
            f"Epoch {epoch + 1}: "
            f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% || "
            f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%"
        )
        time.sleep(1)
        sys.stdout.flush()

    end_time = timer()
    print(f"[INFO] Total training time: {end_time - epochs - start_time:.3f} seconds")
    print(results)
    return results

def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()