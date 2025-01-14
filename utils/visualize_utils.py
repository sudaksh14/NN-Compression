import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CyclicLR, CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import os
import pickle
import numpy as np



def plot_metrics(accuracy, f1_score, ticks, save_path="./plots/metrics_plot.png"):

    if len(accuracy) != len(f1_score) or len(accuracy) != len(ticks):
        raise ValueError("Accuracy, F1 score, and x_ticks must have the same length.")
    
    plt.figure(figsize=(8, 6))
    
    # Plot accuracy
    plt.plot(accuracy, label="Accuracy", marker='o', linestyle='-', color='b')
    
    # Plot F1 score
    plt.plot(f1_score, label="F1 Score", marker='x', linestyle='-', color='g')
    
    # Adding title and labels
    plt.title("Accuracy and F1 Score Over Pruning & Rebuilding", fontsize=14)
    plt.xlabel("% Pruning", fontsize=12)
    plt.ylabel("Metrics", fontsize=12)
    plt.legend(fontsize=10)
    
    # Set custom ticks
    plt.xticks(np.arange(len(accuracy)), labels=ticks, fontsize=10)
    
    plt.grid(alpha=0.4)
    
    # Save the plot to the specified file
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved as {save_path}")



# Plot the loss and accuracy curves
def plot_curves(num_epochs, train_losses, train_accuracies, prune_ratio, exp_name):
    
    # Plot the training and validation loss curves
    plt.figure(figsize=(12, 5))

    # Plot Loss vs. Epoch
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()

    # Plot Accuracy vs. Epoch
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    
    plt.savefig(f"./plots/{exp_name}_Learning_Curve_Fine-Tuning_pruning_{str(int(prune_ratio*100))}")
   
   
    
def plot_curves_with_val(num_epochs, train_losses, train_accuracies, val_losses, val_accuracies, lrs, prune_ratio, exp_name):
    
    # Plot the training and validation loss curves
    plt.figure(figsize=(16, 8))

    # Plot Loss vs. Epoch
    plt.subplot(2, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    plt.grid()

    # Plot Accuracy vs. Epoch
    plt.subplot(2, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 1, 2)
    plt.plot(range(1, num_epochs + 1), lrs, label='Learning Rate', color='purple', marker='o')
    plt.title('Learning Rate vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig(f"./plots/{exp_name}_Learning_Curve_Fine-Tuning_pruning_{str(int(prune_ratio*100))}")


# Plotting results
def plot_comparison(val_accuracy, val_f1_score, fine_tune_accuracy, fine_tune_f1_score, exp_name, prune_ratio):
    plt.figure(figsize=(8, 4))
    plt.bar(['Accuracy Before', 'F1 Score Before', 'Accuracy After', 'F1 Score After'],
            [val_accuracy, val_f1_score, fine_tune_accuracy, fine_tune_f1_score],
            color=['skyblue', 'salmon', 'skyblue', 'salmon'])
    plt.ylabel("Score")
    plt.title("Evaluation Metrics Before and After Fine-Tuning")
    plt.savefig(f"./plots/{exp_name}_Resnet20_Accuracy_Comparison_{str(int(prune_ratio*100))}")
    
    
def plot_comparison_3stage(accuracy_list, exp_name, prune_ratio):
    plt.figure(figsize=(10, 6))
    # Define labels and colors
    labels = ['Accuracy Original', 'F1 Score Original', 'Accuracy Pruned', 
              'F1 Score Pruned', 'Accuracy Rebuilt', 'F1 Score Rebuilt']
    colors = ['skyblue', 'salmon', 'skyblue', 'salmon', 'skyblue', 'salmon']
    
    # Create the bar plot
    bar_positions = range(len(accuracy_list))
    plt.bar(bar_positions, accuracy_list, color=colors)
    
    # Add metric values on top of the bars
    for i, value in enumerate(accuracy_list):
        plt.text(i, value + 0.01, f"{value:.2f}", ha='center', va='bottom', fontsize=10)
    
    # Add labels, title, and save the plot
    plt.xticks(bar_positions, labels, rotation=15, fontsize=10)
    plt.ylabel("Metrics")
    plt.title("Evaluation Metrics for Pruning and Rebuilding")
    plt.tight_layout()
    plt.savefig(f"./plots/{exp_name}_Resnet20_Accuracy_Comparison_{str(int(prune_ratio*100))}.png")


def plot_multilevel(accuracies, exp_name, labels=["Original", "Level-4", "Level-3", "Level-2", "Level-1"]):
    if len(accuracies) != len(labels):
        raise ValueError("accuracies and labels must have the same length.")

    plt.figure(figsize=(10, 6))

    # Plot Accuracy
    plt.plot(
        labels,
        accuracies,
        marker="o",
        label="Accuracy",
        color="blue")

    # Annotate Accuracy Points
    for i, (acc, label) in enumerate(zip(accuracies, labels)):
        plt.text(i, acc + 0.02, f"{acc:.2f}", fontsize=9, ha="center", color="black")

    # Configure Plot
    plt.title("Accuracy vs Resource Tradeoff", fontsize=16)
    plt.xlabel("Model Size", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.ylim(min(accuracies) - 0.05, max(accuracies) + 0.05)  # Adjust y-limits
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Save Plot
    plt.savefig(f"./plots/{exp_name}_Resnet20_Rebuilt_performance.png")

    

def plot_profiling(params, macs, sizes, exp_name, prune_ratio, labels = ["Original", "Pruned", "Rebuilt"]):

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot Parameters
    axes[0].plot(labels, params, marker='o', color='blue', linestyle='-')
    axes[0].set_ylabel("# Parameters")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Plot Model Size
    axes[1].plot(labels, sizes, marker='d', color='red', linestyle='-')
    axes[1].set_ylabel("Model Size (MB)")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Plot MACs
    axes[2].plot(labels, macs, marker='s', color='cyan', linestyle='--')
    axes[2].set_ylabel("MACs")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    # Adjust layout
    plt.tight_layout()
    fig.suptitle("Profiling Metrics", fontsize=16)

    # Save and show the plot
    plt.savefig(f"./plots/{exp_name}_Profiling_Metrics_{str(int(prune_ratio*100))}")


def visualize_weights(weights, title="Pruned Weights in Conv1", save_path="./plots", name="conv1_pruned_weights.png"):
    """
    Visualizes and saves the weights of a convolutional layer as a series of 2D images, 
    where each image corresponds to one filter in the layer.
    
    Parameters:
        weights (torch.Tensor): The weight tensor of shape (out_channels, in_channels, height, width).
        title (str): Title for the plot.
        save_path (str): Path where the image will be saved.
    """
    out_channels, in_channels, height, width = weights.shape

    # Set up subplots to display each out_channel
    n_cols = min(8, out_channels)  # Set the number of columns
    n_rows = (out_channels + n_cols - 1) // n_cols  # Calculate rows based on columns

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    fig.suptitle(title, fontsize=16)

    # Loop through each filter (output channel) and plot it
    for i in range(out_channels):
        # Selecting the weights for the first input channel of each filter for visualization
        kernel = weights[i, 0].cpu().detach().numpy()

        # Determine the plot position
        ax = axes[i // n_cols, i % n_cols] if n_rows > 1 else axes[i % n_cols]
        ax.imshow(kernel, cmap="gray")
        ax.axis("off")
        ax.set_title(f"Filter {i}")

    # Hide any remaining empty subplots
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes.flatten()[j])

    # Adjust layout and save the figure
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(f"{save_path}/{name}")
    print(f"Pruned weights visualization saved")
    
    
def visualize_4d_tensor(tensor, title="Pruned Weights in Conv1", save_path="./plots", name="conv1_pruned_weights.png"):
    
    dim0, dim1, dim2, dim3 = tensor.shape
    
    # Create a figure with 'dim1' rows and 'dim0' columns
    fig, axs = plt.subplots(dim1, dim0, figsize=(dim0*3, dim1*3))
    fig.suptitle(title, fontsize=16)
    
    for i in range(dim1):
        for j in range(dim0):
            ax = axs[i, j] if dim1 > 1 and dim0 > 1 else axs[max(i, j)]
            matrix = tensor[j, i].cpu().numpy()  # Swap i and j to reflect dim0 as columns
            
            # Plot the matrix as an image
            im = ax.imshow(matrix, cmap='viridis', aspect='auto')
            ax.set_title(f"Matrix [{j}, {i}]")
            ax.axis('off')
            
            # Add color bar
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(f"{save_path}/{name}")
    print(f"Pruned weights visualization saved")