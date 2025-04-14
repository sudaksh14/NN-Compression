import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torchvision.models import vit_b_16
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np


def plot_power(x, y1, y2):
    fig, ax1 = plt.subplots()

    # Plot data on the primary axis
    ax1.plot(x, y1, 'b-', label='# Params (Millions)')
    ax1.set_xlabel('Models')
    ax1.set_xscale('linear')
    ax1.set_yscale('log') 
    ax1.set_ylabel('# Params (Millions)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([val for val in x], rotation=30, fontsize=10)

    # Create a secondary y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(x, y2, 'r--', label='Energy per Inference (Joules)')
    ax2.set_ylabel('Energy per Inference (Joules)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Show the plot
    plt.title('Increment in Power Consumption')
    plt.tight_layout() 
    plt.savefig("./power_plot")


def plot_rebuilt_accuracy(accuracy1, accuracy2, accuracy3, accuracy4, labels = ["Original", "Core Model", "Rebuilt"]):

    model_size = [58.244937896728516, 8.150760650634766, 15.178897857666016, 29.423160552978516, 58.244937896728516]

    labels = [f"{lab} ({round(model_size[i], 2)})" for i,lab in enumerate(labels)]
    plt.figure(figsize=(10, 6))

    # Plot Accuracy
    plt.plot(labels, accuracy1, marker="o", label="20 Epochs", color="blue")
    plt.plot(labels, accuracy2, marker="^", label="30 Epochs_BN", color="red")
    plt.plot(labels, accuracy3, marker="s", label="Replace_BN", color="green")
    plt.plot(labels, accuracy4, marker="d", label="Replace&Update_BN", color="cyan")

    # Annotate Accuracy Points
    # for i, acc in enumerate(accuracy1):
    #     plt.text(i+0.05, acc + 1, f"{acc:.2f}", fontsize=9, ha="center", color="black")

    # for i, acc in enumerate(accuracy2):
    #     plt.text(i+0.05, acc - 1, f"{acc:.2f}", fontsize=9, ha="center", color="black")
    
    # Configure Plot
    plt.title("Accuracy vs Resource Tradeoff", fontsize=16)
    plt.legend()
    plt.xlabel("Model Size (MB)", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.ylim(min(accuracy1) - 5, max(accuracy1) + 5)  # Adjust y-limits
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save Plot
    plt.savefig(f"./BN_Rebuilt_performance.png")


def plot_resource_tradeoff_curves(num_params, orig_acc, accuracy_classifier, accuracy_1, accuracy_2, accuracy_3, accuracy_4, model="Resnet18"):
    plt.figure(figsize=(10, 6))  # Set figure size

    # Plot each accuracy curve with markers
    plt.axhline(y = orig_acc, color = 'darkorange', label="Original Accuracy", linestyle="--", linewidth=2) # Solid black line
    plt.plot(num_params, accuracy_classifier, 'k-', marker='d', label="Classification", linewidth=2, markersize=6)
    plt.plot(num_params, accuracy_1, 'b--', marker='s', label="Aplha = 0", linewidth=2, markersize=6)  
    plt.plot(num_params, accuracy_2, 'g-.', marker='^', label="Aplha = 0.25", linewidth=2, markersize=6)  
    plt.plot(num_params, accuracy_3, 'r:', marker='s', label="Aplha = 0.5", linewidth=2, markersize=6)  
    plt.plot(num_params, accuracy_4, 'c-.', marker='d', label="Aplha = 0.75", linewidth=2, markersize=6)  

    # Labels, title, and legend
    plt.xlabel("MAC's(M)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Resource-Accuracy Tradeoff ({model})")
    plt.legend()

    # Set x-axis labels manually
    plt.xticks(num_params, labels=[p for p in num_params])  

    # Reduce the gap between min and max values on y-axis
    min_acc = 70
    max_acc = 100
    plt.ylim(min_acc - 1, max_acc + 1)  

    plt.grid(True, linestyle='--', alpha=0.7)  # Add grid for better visualization

    # Save and show the figure
    plt.tight_layout()
    plt.savefig(f"./Resource_Accuracy_Tradeoff_{model}.png", dpi=300)


if __name__ == '__main__':

    orig_acc = 92.60
    accuracy_classifier = [76.03, 79.50, 82.09, 83.96]
    accuracy_alpha_0 = [71.30, 76.24, 77.94, 81.12]
    accuracy_alpha_25 = [70.29, 79.11, 82.09, 77.91]
    accuracy_alpha_50 = [74.1, 78.93, 80.71, 84.28]
    accuracy_alpha_75 = [75.04, 76.88, 75.11, 80.39]
    raw_macs = [37601510, 72368448, 139407810, 276742784]

    plot_rebuilt_accuracy(accuracy1=[92.60, 76.03, 79.50, 82.09, 83.96], accuracy2=[92.60, 77.58, 80.22, 82.59, 83.88], accuracy3=[92.60, 77.91, 78.09, 82.01, 83.21], accuracy4=[92.60, 78.12, 78.72, 81.77, 83.35], labels=['Original', 'Leve-4', 'Level-3', 'Level-2', 'Level-1'])
    exit()

    macs = [f"{macs / 1e6:.3f}" for macs in raw_macs]
    plot_resource_tradeoff_curves(macs, orig_acc, accuracy_classifier, accuracy_alpha_0, accuracy_alpha_25, accuracy_alpha_50, accuracy_alpha_75)
    exit()
    
    model_names = ["MobileNet-v2", "EfficientNet-B0", "ResNet-150", "VGG-16", "ViT-B/16", "BERT Base", "GPT-2 Medium", "GPT-3"]
    params = [3.4,5.3,60,138,86,110,355,175000]
    power = [0.1,0.2,0.6,1.5,1.2,2,6,30]
    plot_power(model_names, params, power)
