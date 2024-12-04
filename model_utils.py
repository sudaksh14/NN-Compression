import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import os
import pickle
import numpy as np
from ptflops import get_model_complexity_info
from calflops import calculate_flops
from thop import profile
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CyclicLR, CosineAnnealingLR



# Set device for training and evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

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
    plt.bar(['Accuracy Original', 'F1 Score Original', 'Accuracy Pruned', 'F1 Score Pruned', 'Accuracy Rebuilt', 'F1 Score Rebuilt'],
            [*accuracy_list],
            color=['skyblue', 'salmon', 'skyblue', 'salmon', 'skyblue', 'salmon'])
    plt.ylabel("Metrics")
    plt.title("Evaluation Metrics for Pruning and Rebuilding")
    plt.savefig(f"./plots/{exp_name}_Resnet20_Accuracy_Comparison_{str(int(prune_ratio*100))}")
    

def plot_profiling(params, macs, sizes, exp_name, prune_ratio):
    
    labels = ["Original", "Pruned", "Rebuilt"]

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot Parameters
    axes[0].plot(labels, params, marker='o', color='blue', linestyle='-')
    axes[0].set_title("Performance Parameters")
    axes[0].set_ylabel("Parameters")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Plot Model Size
    axes[1].plot(labels, sizes, marker='d', color='red', linestyle='-')
    axes[1].set_title("Model Size (MB)")
    axes[1].set_ylabel("Size (MB)")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Plot MACs
    axes[2].plot(labels, macs, marker='s', color='cyan', linestyle='--')
    axes[2].set_title("MACs")
    axes[2].set_ylabel("MACs")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    # Adjust layout
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(f"./plots/{exp_name}_Performance_Metrics_{str(int(prune_ratio*100))}")
    
    
def evaluate(model, data_loader, plot_title="Test Accuracy & F1-Score", device=device, plot=False, profiling=False):
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        
    # Plot accuracy and F1 score
    if plot:
        plt.figure(figsize=(8, 4))
        plt.bar(['Accuracy', 'F1 Score'], [accuracy, f1], color=['skyblue', 'salmon'])
        plt.ylabel("Score")
        plt.title("Evaluation Metrics")
        plt.savefig(f"./plots/{plot_title}")
        
    if profiling:
        params, mac, size_mb = profile_model_during_training(model, (3,32,32))
        return accuracy, f1, params, mac, size_mb
        
    return accuracy, f1


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
    
    
    
def print_pruned_weights(layer, pruned_indices, dim=0):
    """
    Prints the weights of the layer that were pruned based on the pruned indices.

    Parameters:
        layer (torch.nn.Module): The layer from which weights were pruned.
        pruned_indices (torch.Tensor): Indices of the pruned channels.
        dim (int): Dimension along which pruning was applied (0 for output channels, 1 for input channels).
    """
    # Access the weight data
    weights = layer.weight.data

    if dim == 0:
        # Pruned output channels
        pruned_weights = weights[pruned_indices]
    elif dim == 1:
        # Pruned input channels
        pruned_weights = weights[:, pruned_indices]
    else:
        raise ValueError("Dimension for pruning must be 0 or 1.")
    
    print(f"Pruned Weights (dim={dim}) for indices {pruned_indices.tolist()}:\n", pruned_weights)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0])
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc = nn.Linear(64, 10)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out,(1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResNet_prune(nn.Module):
    def __init__(self, block, num_blocks, prune_ratio, first_channel):
        super(ResNet_prune, self).__init__()
        
        
        self.in_channels = round((first_channel)*(1-prune_ratio))
        self.conv1 = nn.Conv2d(3, round((first_channel)*(1-prune_ratio)), kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(round((first_channel)*(1-prune_ratio)))
        self.layer1 = self._make_layer(block, round((first_channel)*(1-prune_ratio)), num_blocks[0])
        self.layer2 = self._make_layer(block, round((2*first_channel)*(1-prune_ratio)), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, round((4*first_channel)*(1-prune_ratio)), num_blocks[2], stride=2)
        self.fc = nn.Linear(round((4*first_channel)*(1-prune_ratio)), 10)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out,(1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

"""
channel_dict = {'conv1': (13, 3), 'layer1.0.conv1': (13, 13), 'layer1.0.conv2': (13, 13), 'layer1.1.conv1': (13, 13), 'layer1.1.conv2': (13, 13), 'layer1.2.conv1': (13, 13), 
 'layer1.2.conv2': (13, 13), 'layer2.0.conv1': (26, 13), 'layer2.0.conv2': (26, 26), 'layer2.0.downsample.0': (26, 13), 'layer2.1.conv1': (26, 26), 
 'layer2.1.conv2': (26, 26), 'layer2.2.conv1': (26, 26), 'layer2.2.conv2': (26, 26), 'layer3.0.conv1': (52, 26), 'layer3.0.conv2': (52, 52), 
 'layer3.0.downsample.0': (52, 26), 'layer3.1.conv1': (52, 52), 'layer3.1.conv2': (52, 52), 'layer3.2.conv1': (52, 52), 'layer3.2.conv2': (52, 52)}
 """

class ResNet_general(nn.Module):
    def __init__(self, block, num_blocks, channel_dict):
        super(ResNet_general, self).__init__()
        
        self.in_channels = channel_dict['conv1'][0]
        self.conv1 = nn.Conv2d(channel_dict['conv1'][1], channel_dict['conv1'][0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_dict['conv1'][0])
        self.layer1 = self._make_layer(block, channel_dict['layer1.0.conv1'][0], num_blocks[0])
        self.layer2 = self._make_layer(block, channel_dict['layer2.0.conv1'][0], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, channel_dict['layer3.0.conv1'][0], num_blocks[2], stride=2)
        self.fc = nn.Linear(channel_dict['layer3.2.conv2'][0], 10)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out,(1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def Resnet_General(channel_dict):
    return ResNet_general(BasicBlock, [3,3,3], channel_dict)

    
def prune_resnet(model, amount=0.2):
    # Function to apply structural pruning
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0) # Prune along the channel dimension


def rebuild_resnet(model, p, first_channel=16):
    new_model = ResNet_prune(BasicBlock, [3,3,3], prune_ratio=p, first_channel=first_channel)
    
    # Create a list to hold the layers and corresponding weights
    layer_weights = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Get the mask used for pruning
            if hasattr(module, 'weight_mask'):
                mask = module.weight_mask.bool()  # The mask indicates the remaining channels
                remaining_weights = module.weight.data[mask].clone()
                out_channels, in_channels, kernel_height, kernel_width = module.weight.shape
                remaining_weights = remaining_weights.view(-1, in_channels, kernel_height, kernel_width)
                
                # print(name, module.weight.data.shape, mask.shape, remaining_weights.shape)
                
                # If it's the first layer, directly assign to new model's first conv layer
                if name == 'conv1':
                    new_model.conv1.weight.data = remaining_weights
                else:
                    # Update other layers based on their names
                    layer_weights.append((name, remaining_weights))

    # Now we have the remaining weights for each layer, let's set them
    for name, weights in layer_weights:
        layer_name = name.replace('.', '_')  # Change '.' to '_' for attribute access
        new_model_state_dict = new_model.state_dict()
        new_model_state_dict[layer_name + '.weight'] = weights
        new_model.load_state_dict(new_model_state_dict, strict=False)

    return new_model



def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`torch.nn.Linear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer
        



class ChannelPruning(prune.BasePruningMethod):
    """
    Custom pruning method that prunes entire channels (filters) along a specified dimension.
    """
    PRUNING_TYPE = "structured"
    
    def __init__(self, indices, dim):
        
        self.indices = indices  # Indices of channels to prune
        self.dim = dim          # Dimension along which to prune
        

    def compute_mask(self, t, default_mask):
        """
        Computes a mask that removes the specified channels.
        """
        # mask = torch.ones_like(t, dtype=bool)
        mask = default_mask.clone()
        if self.dim == 0:
            mask[self.indices, :, :, :] = False
        elif self.dim == 1:
            mask[:, self.indices, :, :] = False
        else:
            raise ValueError("Dimension must be 0 or 1 for channel pruning.")
        return mask
    
    @classmethod
    def apply(cls, module, name, indices, dim, importance_scores=None):
        return super().apply(
            module,
            name,
            indices=indices,
            dim=dim,
            importance_scores=importance_scores,
        )



def prune_bias(layer, indices):
    """
    Prunes the bias of a layer by zeroing out values at specified indices.
    """
    if layer.bias is not None:
        mask = torch.ones_like(layer.bias, dtype=bool)
        mask[indices] = False  # Set the mask to zero for the pruned indices
        layer.bias.data[~mask] = 0  # Apply the mask to the bias to prune it
        print(f"Pruned bias for channels {indices.tolist()}.")

def get_pruned_channel_indices(tensor, amount, norm_type="L1"):
    """
    Calculate channel indices to prune based on smallest norms across channels.
    """
    num_channels = tensor.size(0)  # Number of output channels in Conv2D
    num_prune = int(num_channels * amount)

    if norm_type == "L1":
        norms = torch.abs(tensor).view(num_channels, -1).sum(dim=1)
    elif norm_type == "L2":
        norms = torch.sqrt((tensor ** 2).view(num_channels, -1).sum(dim=1))
    else:
        raise ValueError("norm_type must be 'L1' or 'L2'.")

    # Get indices with the smallest norms
    pruned_indices = torch.topk(norms, num_prune, largest=False).indices
    return pruned_indices

def prune_conv_layer_dim0(model, layer_name, amount=0.2, norm_type="L1"):
    """
    Prunes the channels in a convolutional layer based on norms and returns pruned indices.
    """
    layer = dict(model.named_modules())[layer_name]
    pruned_indices = get_pruned_channel_indices(layer.weight, amount, norm_type=norm_type)

    # Apply channel pruning
    pruner = ChannelPruning(pruned_indices, dim=0)  # Pruning output channels
    pruner.apply(layer, name="weight", indices=pruned_indices, dim=0)
    print(f"Pruned channels in {layer_name} with indices {pruned_indices.tolist()} using {norm_type} norm.")
    
    # Prune the bias associated with the pruned channels
    prune_bias(layer, pruned_indices)

    return pruned_indices

def prune_conv_layer_dim1(model, layer_name, pruned_indices):
    """
    Prunes the input channels in the next convolutional layer based on the pruned indices from the previous layer.
    """
    layer = dict(model.named_modules())[layer_name]

    # Apply channel pruning on the input channels using the same indices
    pruner = ChannelPruning(pruned_indices, dim=1)  # Pruning input channels
    pruner.apply(layer, name="weight", indices=pruned_indices, dim=1)
    print(f"Pruned input channels in {layer_name} using the same indices: {pruned_indices.tolist()}.")
    
    # Prune the bias associated with the pruned channels
    prune_bias(layer, pruned_indices)
    

# Example CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


def Index_Aware_pruner(model, amount, res_blocks=3, plot=True):
    
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if name == "conv1":
                print(name,  type(module))
                pruned_indices_0 = get_pruned_channel_indices(module.weight, amount)
                pruner = ChannelPruning(pruned_indices_0, dim=0)
                pruner.apply(module, name="weight", indices=pruned_indices_0, dim=0) # Prune along the output channel dimension
                prune.remove(module, name="weight")
 
            else:
                
                print(name,  type(module))
                if 'downsample' in name:
                    
                    pruner_down_1 = ChannelPruning(indices_down_1, dim=1)
                    pruner_down_0 = ChannelPruning(indices_last_layer, dim=0)
                    pruner_down_1.apply(module, name="weight", indices=indices_down_1, dim=1) # Prune along the input channel dimension 
                    pruner_down_0.apply(module, name="weight", indices=indices_last_layer, dim=0) # Prune along the output channel dimension
                    prune.remove(module, name="weight")
                    
                else:
  
                    pruned_indices_0 = get_pruned_channel_indices(module.weight, amount)
                    pruner_dim_1 = ChannelPruning(indices_last_layer, dim=1)
                    pruner_dim_0 = ChannelPruning(pruned_indices_0, dim=0)
                    pruner_dim_1.apply(module, name="weight", indices=indices_last_layer, dim=1) # Prune along the input channel dimension
                    pruner_dim_0.apply(module, name="weight", indices=pruned_indices_0, dim=0) # Prune along the output channel dimension
                    prune.remove(module, name="weight")

                    count += 1
                
                
            indices_last_layer = pruned_indices_0
            if count != 0 and count % (res_blocks*2) == 0:
                indices_down_1 = indices_last_layer

            print("Pruned at output channel Indices: ", indices_last_layer.tolist())
            
            if plot:
                visualize_4d_tensor(module.weight.data, title=f"Pruned weights in {name}", name=f"Resnet_{name}_pruned_weights")
            
        
        
def get_nonzero_indices(model):
    non_zero_indices = {}
    new_channels = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            
            weights = module.weight.data
            indices = []

            # Iterate over both dimensions (dim 0: output channels, dim 1: input channels)
            for out_idx in range(weights.shape[0]):  # dim 0 (output channels)
                for in_idx in range(weights.shape[1]):  # dim 1 (input channels)
                    
                    if weights[out_idx, in_idx].abs().sum() > 0:  # Check if the (out_idx, in_idx) slice is non-zero
                        indices.append((out_idx, in_idx))
        
            non_zero_indices[name] = indices
            out_channels = len(set(t[0] for t in indices))
            in_channels = len(set(t[1] for t in indices))
            new_channels[name] = (out_channels, in_channels)
                            
                        
    return non_zero_indices, new_channels


# Function to copy weights from pruned model to reduced model
def copy_weights(pruned_model, rebuilt_model, nonzero_channels):
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            
            n=0
        
            pruned_layer = module
            reduced_layer = dict(rebuilt_model.named_modules())[name]
            
            out_channels, in_channels,_,_ = reduced_layer.weight.shape
        
            with torch.no_grad():
                for i in range(out_channels):
                    for j in range(in_channels):
                        dim0, dim1 = nonzero_channels[name][n]
                        reduced_layer.weight.data[i,j] = pruned_layer.weight.data[dim0, dim1]
                        n += 1
     
     
                
def copy_weights_from_dict(pruned_model, unpruned_weights):
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            
            module.weight.data = unpruned_weights[name]


def rebuilt_bigger(channel_dict_path, state_dict_path, device):
    
    with open(channel_dict_path, "rb") as f:
        channel_dict = pickle.load(f)
    
    
    model = Resnet_General(channel_dict)
    model.load_state_dict(torch.load(state_dict_path, map_location=device, weights_only=True))
    return model




def get_pruned_indices_and_counts(model):
    """
    Analyze a structurally pruned model to determine:
    - The indices of pruned channels (dim 0 and dim 1).
    - The number of pruned channels in each layer.
    
    Parameters:
    - model: The PyTorch model.

    Returns:
    - pruned_info: A dictionary containing pruned indices and counts for each layer.
    """
    pruned_info = {}
    pruned_weights = {}
    num_pruned_channels = {}
    
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):  # Check for Conv2d layers
            layer_info = {}
            
            # Get the weight tensor
            weights = layer.weight.detach()
            
            # Identify pruned filters along dim=0 (output channels)
            pruned_dim0 = torch.nonzero(weights.abs().sum(dim=(1, 2, 3)) == 0).squeeze().tolist()
            if isinstance(pruned_dim0, int):  # Handle single index case
                pruned_dim0 = [pruned_dim0]
            layer_info['pruned_dim0'] = pruned_dim0
            
            
            # Identify pruned filters along dim=1 (input channels)
            pruned_dim1 = torch.nonzero(weights.abs().sum(dim=(0, 2, 3)) == 0).squeeze().tolist()
            if isinstance(pruned_dim1, int):  # Handle single index case
                pruned_dim1 = [pruned_dim1]
            layer_info['pruned_dim1'] = pruned_dim1
            
            # Add layer information to pruned_info
            pruned_info[name] = layer_info
            num_pruned_channels[name] = (len(pruned_dim0), len(pruned_dim1))
            
            pruned_weights[name] = weights[pruned_dim0][:,pruned_dim1,:,:]
    
    return pruned_info, num_pruned_channels, pruned_weights
            
            
def get_unpruned_indices_and_counts(model):
    non_pruned_info = {}
    num_unpruned_channels = {}
    unpruned_weights = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            
            layer_info = {}
            
            # Get the weight tensor
            weights = module.weight.detach()
            
            # Identify pruned filters along dim=0 (output channels)
            pruned_dim0 = torch.nonzero(weights.abs().sum(dim=(1, 2, 3)) > 0).squeeze().tolist()
            if isinstance(pruned_dim0, int):  # Handle single index case
                pruned_dim0 = [pruned_dim0]
            layer_info['pruned_dim0'] = pruned_dim0
            
            
            # Identify pruned filters along dim=1 (input channels)
            pruned_dim1 = torch.nonzero(weights.abs().sum(dim=(0, 2, 3)) > 0).squeeze().tolist()
            if isinstance(pruned_dim1, int):  # Handle single index case
                pruned_dim1 = [pruned_dim1]
            layer_info['pruned_dim1'] = pruned_dim1
        
            non_pruned_info[name] = layer_info
            num_unpruned_channels[name] = (len(pruned_dim0), len(pruned_dim1))
            unpruned_weights[name] = weights[pruned_dim0][:,pruned_dim1,:,:]
                            
                        
    return non_pruned_info, num_unpruned_channels, unpruned_weights

            

def extend_channels(model, pruned_dict):
    
    new_channel_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            
            new_in_channel = int(module.weight.data.shape[0] + pruned_dict[name][0])
            new_out_channel = int(module.weight.data.shape[1] + pruned_dict[name][1])
            
            new_channel_dict[name] = (new_in_channel, new_out_channel)
            
    return new_channel_dict



def reconstruct_weights_from_dicts(model, pruned_indices, pruned_weights, unpruned_indices, unpruned_weights):
    """
    Reconstruct weights for a model using pruned and unpruned indices and tensors.

    Parameters:
    - pruned_indices: dict, mapping layer names to (dim0_indices, dim1_indices) of pruned weights.
    - pruned_weights: dict, mapping layer names to tensors of pruned weights.
    - unpruned_indices: dict, mapping layer names to (dim0_indices, dim1_indices) of unpruned weights.
    - unpruned_weights: dict, mapping layer names to tensors of unpruned weights.
    - model: torch.nn.Module, the model to reconstruct weights for.

    Returns:
    - reconstructed_model: torch.nn.Module, the model with reconstructed weights.
    """
    # Iterate through the model's state_dict to dynamically fetch layer shapes
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            
            new_device = layer.weight.device

            # Retrieve pruned and unpruned indices
            pruned_dim0, pruned_dim1 = pruned_indices[name].values()
            unpruned_dim0, unpruned_dim1 = unpruned_indices[name].values()

            # Assign pruned weights
            layer.weight.data[pruned_dim0][:,pruned_dim1,:,:] = pruned_weights[name].to(new_device)

            # Assign unpruned weights
            layer.weight.data[unpruned_dim0][:,unpruned_dim1,:,:] = unpruned_weights[name].to(new_device)
            
    return model
                       
                       

def freeze_channels(model, channel_dict):
    """
    Freeze specific channels in convolutional layers based on the provided channel dictionary.

    Parameters:
    - model: The PyTorch model (e.g., ResNet-20).
    - channel_dict: A dictionary where keys are layer names, and values are dicts specifying
                    channels to freeze for dim 0 (output channels) and dim 1 (input channels).
                    Example:
                        {
                            "layer1.0.conv1": {"dim0": [0, 1], "dim1": [2, 3]},
                            "layer2.0.conv1": {"dim0": [4], "dim1": [1, 6]}
                        }

    Returns:
    - model: The modified model with specified channels frozen.
    """
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) and name in channel_dict:
            freeze_info = channel_dict[name]
            dim0_channels = freeze_info.get("pruned_dim0", [])
            dim1_channels = freeze_info.get("pruned_dim1", [])
            
            with torch.no_grad():
                weight = layer.weight
                if weight.requires_grad:
                    # Create masks for the dimensions
                    mask_dim0 = torch.zeros_like(weight, dtype=torch.bool)
                    mask_dim0[dim0_channels, :, :, :] = True
                    
                    mask_dim1 = torch.zeros_like(weight, dtype=torch.bool)
                    mask_dim1[:, dim1_channels, :, :] = True
                                        
                    # Combine masks to handle both dim 0 and dim 1
                    combined_mask = mask_dim0 | mask_dim1
                    print("layer_name:", name, combined_mask.shape)
                    
                    # Register a backward hook to ensure frozen channels' gradients are zeroed out
                    def freeze_grad_hook(grad):
                        grad[combined_mask] = 0
                        return grad

                    weight.register_hook(freeze_grad_hook)
                    
    return model
       
 
 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
 
def compute_flops_and_macs(model, input_size):
    macs, params = get_model_complexity_info(model, input_size, as_strings=False, verbose=True)
    # macs, params = profile(model, inputs=input_size)
    
    return macs, params
       
def model_size_in_mb(model):
    torch.save(model.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p") / (1024 * 1024)  # Convert bytes to MB
    os.remove("temp.p")
    return size_mb
       
# Function to profile model
def profile_model_during_training(model, input_size):
    macs, params = compute_flops_and_macs(model, input_size)
    size_mb = model_size_in_mb(model)

    print(f"Params: {params}, MACs: {macs}, Size: {size_mb:.2f} MB")
    return params, macs, size_mb                
          
          
def fine_tuner(model, train_loader, val_loader, num_epochs, scheduler_type, prune_ratio, exp_name, LR=1e-3):

    model = model.to(device)
    model.train()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Define LR scheduler
    if scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    elif scheduler_type == 'exponential':
        scheduler = ExponentialLR(optimizer, gamma=0.95)
    elif scheduler_type == 'cyclic':
        scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=0.01, step_size_up=20, mode='triangular2')
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
        

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    lrs = []  # To track learning rates

    # Fine-tuning loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # Step the scheduler and track learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        lrs.append(current_lr)
        
        # Validation loop
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Accumulate validation metrics
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= total
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}, Learning Rate: {current_lr:.6f}")
    
    plot_curves_with_val(num_epochs, train_losses, train_accuracies, val_losses, val_accuracies, lrs, prune_ratio, exp_name)
           
    print("Fine-Tuning Complete")          
             

    
if __name__ == '__main__':

    model = SimpleCNN()
    Index_Aware_pruner(model, amount=0.2, plot=False)
    non_zero_weights, new_channels = get_nonzero_indices(model)
    print(new_channels)
    exit()
    
    # Example usage
    conv_layer = torch.nn.Conv2d(3, 6, kernel_size=3, bias=True)

    # Example pruning (randomly zero out some channels for demonstration)
    with torch.no_grad():
        conv_layer.weight[0] = 0  # Set all weights in channel 0 to zero
        conv_layer.weight[2] = 0  # Set all weights in channel 2 to zero


    # Prune conv1's output channels and use the same indices to prune conv2's input channels
    pruned_indices_conv1 = prune_conv_layer_dim0(model, 'conv1', amount=0.2, norm_type="L1")
    prune_conv_layer_dim1(model, 'conv2', pruned_indices_conv1)

    # Prune conv2's output channels and use the same indices to prune conv3's input channels
    pruned_indices_conv2 = prune_conv_layer_dim0(model, 'conv2', amount=0.2, norm_type="L1")
    prune_conv_layer_dim1(model, 'conv3', pruned_indices_conv2)
    
    pruned_indices_conv3 = prune_conv_layer_dim0(model, 'conv3', amount=0.2, norm_type="L1")









