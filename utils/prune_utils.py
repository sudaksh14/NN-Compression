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
from utils.eval_utils import *
from utils.visualize_utils import *
from models.Resnet import *



# Set device for training and evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)
    
    
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

    
def prune_resnet_torch(model, amount=0.2):
    # Function to apply structural pruning
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0) # Prune along the channel dimension



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

def get_pruned_channel_indices(tensor, amount, norm_type="L1", prune_step=0):
    """
    Calculate channel indices to prune based on smallest norms across channels.
    """
    num_channels = tensor.size(0)  # Number of output channels in Conv2D
    num_prune = int(num_channels * amount)

    if norm_type == "L1":
        norms = torch.abs(tensor).view(num_channels, -1).sum(dim=1)
        # norms = torch.sum(torch.abs(tensor), dim=(1, 2, 3))
    elif norm_type == "L2":
        norms = torch.sqrt((tensor ** 2).view(num_channels, -1).sum(dim=1))
    else:
        raise ValueError("norm_type must be 'L1' or 'L2'.")
    
    non_zero_channels = torch.sum(norms != 0).item()  # Number of channels with non-zero L1 norms
    zero_channels = torch.sum(norms == 0).item()      # Number of channels with zero L1 norms
    total_channels = norms.size(0)                   # Total number of channels (out_channels)

    # Output the results
    # print(f"Number of channels with non-zero L1 norms: {non_zero_channels}")
    # print(f"Number of channels with zero L1 norms: {zero_channels}")
    # print(f"Total number of channels: {total_channels}")
    # print(total_channels == non_zero_channels + zero_channels)
    
    if prune_step > 0:
        norms[norms == 0] = float('inf') # Set zero norms to infinity to avoid pruning them
        num_prune = int((total_channels - zero_channels) * amount)  # Prune ratio normalized based on non-zero norms

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


def Index_Aware_pruner(model, amount, prune_step=0, res_blocks=3, plot=True, BN=True):
    
    pruning_index = {}
    count = 0
    for i, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, nn.Conv2d):
            if name == "conv1" or name == "features.0":
                # print(name,  type(module))
                pruned_indices_0 = get_pruned_channel_indices(module.weight, amount, prune_step=prune_step)
                pruner = ChannelPruning(pruned_indices_0, dim=0)
                pruner.apply(module, name="weight", indices=pruned_indices_0, dim=0) # Prune along the output channel dimension
                prune.remove(module, name="weight")
                pruning_index[name] = {"pruned_dim0": pruned_indices_0.tolist(), "pruned_dim1": []}
 
            else:
                
                # print(name,  type(module))
                if 'downsample' in name:
                    
                    pruner_down_1 = ChannelPruning(indices_down_1, dim=1)
                    pruner_down_0 = ChannelPruning(indices_last_layer, dim=0)
                    pruner_down_1.apply(module, name="weight", indices=indices_down_1, dim=1) # Prune along the input channel dimension 
                    pruner_down_0.apply(module, name="weight", indices=indices_last_layer, dim=0) # Prune along the output channel dimension
                    prune.remove(module, name="weight")
                    pruning_index[name] = {"pruned_dim0": indices_last_layer.tolist(), "pruned_dim1": indices_down_1.tolist()}
                    
                else:
  
                    pruned_indices_0 = get_pruned_channel_indices(module.weight, amount, prune_step=prune_step)
                    pruner_dim_1 = ChannelPruning(indices_last_layer, dim=1)
                    pruner_dim_0 = ChannelPruning(pruned_indices_0, dim=0)
                    pruner_dim_1.apply(module, name="weight", indices=indices_last_layer, dim=1) # Prune along the input channel dimension
                    pruner_dim_0.apply(module, name="weight", indices=pruned_indices_0, dim=0) # Prune along the output channel dimension
                    prune.remove(module, name="weight")
                    pruning_index[name] = {"pruned_dim0": pruned_indices_0.tolist(), "pruned_dim1": indices_last_layer.tolist()}

                    count += 1
                
                
            indices_last_layer = pruned_indices_0
            if count != 0 and count % (res_blocks*2) == 0:
                indices_down_1 = indices_last_layer

            # print("Pruned at output channel Indices: ", indices_last_layer.tolist())
            
            if plot:
                visualize_4d_tensor(module.weight.data, title=f"Pruned weights in {name}", name=f"Resnet_{name}_pruned_weights")
    
    return model, pruning_index        
        

def prune_batchnorm(model, unpruned_indices):
    """
    Prune the BatchNorm layers based on provided indices.

    Args:
        model (torch.nn.Module): The PyTorch model (ResNet18 or any CNN).
        bn_prune_indices (dict): A dictionary where keys are BN layer names 
                                 and values are lists of channel indices to be restored.

    Returns:
        model (torch.nn.Module): The list with unpruned BN indices and weights.
    """

    bn_indices = {}
    bn_weights = {}

    dict_iter = iter(unpruned_indices.items())

    for name, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            indices_to_keep = next(dict_iter)[1].get("unpruned_dim0", [])
            bn_indices[name] = indices_to_keep

            # Extract and update only the remaining channels
            bn_weights[name+"_weights"] = layer.weight.data[indices_to_keep]
            bn_weights[name+"_bias"] = layer.bias.data[indices_to_keep]
            bn_weights[name+"_runnin_mean"] = layer.running_mean[indices_to_keep]
            bn_weights[name+"_running_var"] = layer.running_var[indices_to_keep]

    return bn_indices, bn_weights
        
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
     
                
def copy_weights_from_dict(pruned_model, unpruned_weights, BN_params, BN=False):
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            module.weight.data = unpruned_weights[name]

    if BN:
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.weight.data = BN_params[name+"_weights"]
                module.bias.data = BN_params[name+"_bias"]
                module.running_mean = BN_params[name+"_runnin_mean"]
                module.running_var = BN_params[name+"_running_var"]



def rebuilt_bigger(channel_dict_path, state_dict_path, device):
    
    with open(channel_dict_path, "rb") as f:
        channel_dict = pickle.load(f)
    
    model = Resnet_General(channel_dict)
    model.load_state_dict(torch.load(state_dict_path, map_location=device, weights_only=True))
    return model


# NOT IN USE    
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
            
            # Save pruned weights
            if pruned_dim0 and pruned_dim1:
                pruned_weights[name] = weights[pruned_dim0, :, :, :][:, pruned_dim1, :, :]
            else:
                pruned_weights[name] = weights[pruned_dim0, :, :, :]
    
    return pruned_info, num_pruned_channels, pruned_weights


def get_pruned_channels_weights(model, pruned_index):
    """
    Analyze a structurally pruned model to determine:
    - The indices of pruned channels (dim 0 and dim 1).
    - The number of pruned channels in each layer.
    
    Parameters:
    - model: The Original PyTorch model.

    Returns:
    - pruned_info: A dictionary containing pruned indices, counts and weights for each layer.
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
            pruned_dim0 = pruned_index[name]["pruned_dim0"]
            if isinstance(pruned_dim0, int):  # Handle single index case
                pruned_dim0 = [pruned_dim0]
            layer_info['pruned_dim0'] = sorted(pruned_dim0)
            # layer_info['pruned_dim0'] = pruned_dim0
            
            
            # Identify pruned filters along dim=1 (input channels)
            pruned_dim1 = pruned_index[name]["pruned_dim1"]
            if isinstance(pruned_dim1, int):  # Handle single index case
                pruned_dim1 = [pruned_dim1]
            layer_info['pruned_dim1'] = sorted(pruned_dim1)
            # layer_info['pruned_dim1'] = pruned_dim1
            
            # Add layer information to pruned_info
            pruned_info[name] = layer_info
            num_pruned_channels[name] = (len(pruned_dim0), len(pruned_dim1))

            # Save pruned weights
            if pruned_dim0 and pruned_dim1:
                pruned_weights[name] = weights[pruned_dim0, :, :, :][:, pruned_dim1, :, :]
            else:
                pruned_weights[name] = weights[pruned_dim0, :, :, :]
    
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
            layer_info['unpruned_dim0'] = pruned_dim0
            
            
            # Identify pruned filters along dim=1 (input channels)
            pruned_dim1 = torch.nonzero(weights.abs().sum(dim=(0, 2, 3)) > 0).squeeze().tolist()
            if isinstance(pruned_dim1, int):  # Handle single index case
                pruned_dim1 = [pruned_dim1]
            layer_info['unpruned_dim1'] = pruned_dim1
        
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


def get_rebuild_channels(unpruned_channels, pruned_channels):
    new_channels_dict = {}
    for name, weight in pruned_channels.items():
        new_in_channels = int(unpruned_channels[name][0] + pruned_channels[name][0])
        new_out_channels = int(unpruned_channels[name][1] + pruned_channels[name][1])
        
        new_channels_dict[name] = (new_in_channels, new_out_channels)
        
    return new_channels_dict


def get_core_weights(pruned_model, unpruned_weights, BN_params, restore_BN=False):
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            unpruned_weights[name] = module.weight.data

        if restore_BN:
            if isinstance(module, nn.BatchNorm2d):
                BN_params[name+"_weights"] = module.weight.data
                BN_params[name+"_bias"] = module.bias.data
                BN_params[name+"_runnin_mean"] = module.running_mean
                BN_params[name+"_running_var"] = module.running_var




# Function to replace weights from Lower model to Higher model
def replace_previous_weights(model, pruned_indices, unpruned_indices, unpruned_weights):
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            new_device = layer.weight.device

            # Retrieve pruned and unpruned indices
            pruned_dim0, pruned_dim1 = pruned_indices[name].values()
            unpruned_dim0, unpruned_dim1 = unpruned_indices[name].values()

            # Combine and sort
            combined_dim0 = sorted(pruned_dim0 + unpruned_dim0)
            combined_dim1 = sorted(pruned_dim1 + unpruned_dim1)

            # Find new indices for list1 and list2
            new_unpruned_dim0 = [combined_dim0.index(x) for x in unpruned_dim0]
            new_unpruned_dim1 = [combined_dim1.index(x) for x in unpruned_dim1]

            for i in range(len(new_unpruned_dim0)):
                out_idx = new_unpruned_dim0[i]  # Output channel index
                for j in range(len(new_unpruned_dim1)):
                    in_idx = new_unpruned_dim1[j]   # Input channel index
                    layer.weight.data[out_idx, in_idx, :, :] = unpruned_weights[name][i, j].to(new_device)

    return model

def restore_BN_weights(model, BN_params, restore_indices, restore_params, update_weights=False):

    dict_iter = iter(restore_indices.items())
    for name, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.weight.data = BN_params[name+"_weights"]
            layer.bias.data = BN_params[name+"_bias"]
            layer.running_mean = BN_params[name+"_runnin_mean"]
            layer.running_var = BN_params[name+"_running_var"]

            # Update the BN params after Fine-Tuning
            if update_weights:
                indices_to_replace = next(dict_iter)[1]

                layer.weight.data[indices_to_replace] = restore_params[name+"_weights"]
                layer.bias.data[indices_to_replace] = restore_params[name+"_bias"]
                layer.running_mean[indices_to_replace] = restore_params[name+"_runnin_mean"]
                layer.running_var[indices_to_replace] = restore_params[name+"_running_var"]
            # check_overlap(layer.weight.data, restore_params[name+"_weights"])

    return model


def reconstruct_Global_weights_from_dicts(model, pruned_indices, pruned_weights, unpruned_indices, unpruned_weights, freezing=False):
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
    new_unpruned_dim0_freeze_list = {}
    new_unpruned_dim1_freeze_list = {}
    # Iterate through the model's state_dict to dynamically fetch layer shapes
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            new_device = layer.weight.device

            # Retrieve pruned and unpruned indices
            pruned_dim0, pruned_dim1 = pruned_indices[name].values()
            unpruned_dim0, unpruned_dim1 = unpruned_indices[name].values()

            # Combine and sort
            combined_dim0 = sorted(pruned_dim0 + unpruned_dim0)
            combined_dim1 = sorted(pruned_dim1 + unpruned_dim1)

            # Find new indices for list1 and list2
            new_pruned_dim0 = [combined_dim0.index(x) for x in pruned_dim0]
            new_unpruned_dim0 = [combined_dim0.index(x) for x in unpruned_dim0]

            new_pruned_dim1 = [combined_dim1.index(x) for x in pruned_dim1]
            new_unpruned_dim1 = [combined_dim1.index(x) for x in unpruned_dim1]

            new_unpruned_dim0_freeze_list[name] = new_unpruned_dim0
            new_unpruned_dim1_freeze_list[name] = new_unpruned_dim1

            # Assign pruned weights
            for i in range(len(new_pruned_dim0)):
                    out_idx = new_pruned_dim0[i]  # Output channel index
                    for j in range(len(new_pruned_dim1)):
                        in_idx = new_pruned_dim1[j]   # Input channel index
                        layer.weight.data[out_idx, in_idx, :, :] = pruned_weights[name][i, j].to(new_device)

            # Assign unpruned weights
            for i in range(len(new_unpruned_dim0)):
                    out_idx = new_unpruned_dim0[i]  # Output channel index
                    for j in range(len(new_unpruned_dim1)):
                        in_idx = new_unpruned_dim1[j]   # Input channel index
                        layer.weight.data[out_idx, in_idx, :, :] = unpruned_weights[name][i, j].to(new_device)

            # Channel Freezing --> NOT WORKING
            if freezing:
                for i in range(len(new_unpruned_dim0)):
                        out_idx = new_unpruned_dim0[i]  # Output channel index
                        for j in range(len(new_unpruned_dim1)):
                            in_idx = new_unpruned_dim1[j]   # Input channel index
                            layer.weight.data[out_idx, in_idx, :, :].requires_grad = False

                print(name, layer.weight.requires_grad)
                
    return model, new_unpruned_dim0_freeze_list, new_unpruned_dim1_freeze_list


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
            if pruned_dim0 and pruned_dim1:
                for i in range(len(pruned_dim0)):
                    out_idx = pruned_dim0[i]  # Output channel index
                    for j in range(len(pruned_dim1)):
                        in_idx = pruned_dim1[j]   # Input channel index
                        layer.weight.data[out_idx, in_idx, :, :] = pruned_weights[name][i, j].to(new_device)
            else:
                for i in range(len(pruned_dim0)):
                    out_idx = pruned_dim0[i]  # Output channel index
                    layer.weight.data[out_idx, :, :, :] = pruned_weights[name][i].to(new_device)
            

            # Assign unpruned weights
            for i in range(len(unpruned_dim0)):
                    out_idx = unpruned_dim0[i]  # Output channel index
                    for j in range(len(unpruned_dim1)):
                        in_idx = unpruned_dim1[j]   # Input channel index
                        layer.weight.data[out_idx, in_idx, :, :] = unpruned_weights[name][i, j].to(new_device)
            
    return model
                       
"""
# Function to create masks for each parameter
def create_mask(param, dim0_indices, dim1_indices):
    mask = torch.ones_like(param, dtype=torch.bool)  # Initialize all False
    for i in dim0_indices:
        for j in dim1_indices:
            if i <= param.shape[0] and j <= param.shape[1]:  # Ensure indices are in range
                mask[i, j, :, :] = False # Mark these indices to be zeroed
    return mask

# Hook function to apply the mask
def mask_gradients(grad, mask):
    grad[mask] = 0  # Zero out specific indices
    return grad

# NOT IN USE
# Approach1: Use of hooks to freeze gradients at once
def freeze_grads(model, new_unpruned_dim0, new_unpruned_dim1):
 
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):

            # Create and store masks for each parameter
            mask = create_mask(layer.weight, new_unpruned_dim0[name], new_unpruned_dim1[name])

            # Register hooks for each parameter
            layer.weight.register_hook(lambda grad, mask=mask: mask_gradients(grad, mask))

    return model
"""

# Approach2: Converts gradients to zero in each training epoch
def zero_out_gradients_v1(model, dim0_indices, dim1_indices):
    with torch.no_grad():
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d):
                if layer.weight.grad is not None:
                    for i in dim0_indices[name]:
                        for j in dim1_indices[name]:
                            if i <= layer.weight.shape[0] and j <= layer.weight.shape[1]:  # Ensure indices are in range
                                layer.weight.grad[i, j, :, :] = 0  # Zero out specific gradient indices


# More efficient in use of memory
def zero_out_gradients_v2(model, dim0_indices, dim1_indices):
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) and layer.weight.grad is not None:
            dim0_idx = torch.tensor(dim0_indices[name], dtype=torch.long, device=layer.weight.grad.device)
            dim1_idx = torch.tensor(dim1_indices[name], dtype=torch.long, device=layer.weight.grad.device)

            # Ensure indices are within valid range
            dim0_idx = dim0_idx[dim0_idx < layer.weight.grad.shape[0]]
            dim1_idx = dim1_idx[dim1_idx < layer.weight.grad.shape[1]]

            # Create a grid of (dim0, dim1) combinations
            grid_dim0, grid_dim1 = torch.meshgrid(dim0_idx, dim1_idx, indexing='ij')

            # Zero out gradients at specified indices
            with torch.no_grad():
                layer.weight.grad[grid_dim0, grid_dim1, :, :] = 0


# NOT IN USE
def freeze_channels_2d(model, channel_indices):
    """
    Freezes specific channels in both `dim 0` (output channels) and `dim 1` (input channels) for the specified layers.

    Args:
        model: The model (e.g., ResNet18).
        channel_indices: A dictionary with layer names as keys and a dictionary with 'dim0' and 'dim1' indices to freeze.
    """
    def freeze_grad_hook(grad, frozen_dim0, frozen_dim1):
        """
        Zero out gradients for the specified indices in both dimensions.
        """
        grad[frozen_dim0, :, :, :] = 0  # Freeze dim 0
        grad[:, frozen_dim1, :, :] = 0  # Freeze dim 1
        return grad

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) and name in channel_indices:
            indices = channel_indices[name]
            frozen_dim0 = indices.get('unpruned_dim0', [])
            frozen_dim1 = indices.get('unpruned_dim1', [])

            print(layer.weight.requires_grad)
            print(layer.weight.grad[frozen_dim0, :, :, :])
            print(layer.weight.grad.shape)
            print(len(frozen_dim0))
            print(layer.weight.grad[frozen_dim0, :, :, :].shape)
            
            if layer.weight.requires_grad:
                # Register hook to zero gradients in both dimensions
                print("inside")
                layer.weight.register_hook(
                    lambda grad, frozen_dim0=frozen_dim0, frozen_dim1=frozen_dim1: freeze_grad_hook(grad, frozen_dim0, frozen_dim1)
                )
            print(layer.weight.grad[frozen_dim0, :, :, :])
            exit()
            
            # If the layer has bias, freeze only corresponding to `dim0` since biases don't have `dim1`
            if layer.bias is not None and layer.bias.requires_grad:
                layer.bias.register_hook(lambda grad, frozen_dim0=frozen_dim0: grad.index_fill_(0, torch.tensor(frozen_dim0, device=grad.device), 0))

    return model

# NOT IN USE
def validate_frozen_channels(model, channel_indices):
    """
    Validate that the specified channels in the model are frozen.

    Args:
        model: The model with frozen channels.
        channel_indices: A dictionary with layer names and indices for `dim0` and `dim1`.

    Returns:
        A boolean indicating whether the specified channels are frozen.
    """

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) and name in channel_indices:
            indices = channel_indices[name]
            frozen_dim0 = indices.get('unpruned_dim0', [])
            frozen_dim1 = indices.get('unpruned_dim1', [])

            if layer.weight.grad is None:
                print(name)
            
            if layer.weight.grad is not None:
                grad = layer.weight.grad
                
                # Validate dim0 gradients
                dim0_frozen = grad[frozen_dim0, :, :, :].abs().sum().item() == 0
                if not dim0_frozen:
                    print(f"Dim 0 not frozen for {name}: Indices {frozen_dim0}")
                
                # Validate dim1 gradients
                dim1_frozen = grad[:, frozen_dim1, :, :].abs().sum().item() == 0
                if not dim1_frozen:
                    print(f"Dim 1 not frozen for {name}: Indices {frozen_dim1}")

                # all_frozen = all_frozen and dim0_frozen and dim1_frozen
                all_frozen = dim0_frozen and dim1_frozen

            # Validate bias if it exists
            if layer.bias is not None and layer.bias.grad is not None:
                bias_grad = layer.bias.grad
                dim0_bias_frozen = bias_grad[frozen_dim0].abs().sum().item() == 0
                if not dim0_bias_frozen:
                    print(f"Bias not frozen for {name}: Indices {frozen_dim0}")
                all_frozen = all_frozen and dim0_bias_frozen

    return all_frozen


def check_overlap(big_tensor, small_tensor):
    """
    Check if there's any overlap between two tensors.

    Args:
        big_tensor: The larger tensor.
        small_tensor: The smaller tensor.

    Returns:
        A boolean indicating whether there's overlap.
    """

    print(f"Small Tensor Shape: {small_tensor.shape}")
    print(f"Big Tensor Shape: {big_tensor.shape}")

    overlap = torch.isin(small_tensor, big_tensor)

    print(f"Overlap found: {overlap.any().item()}")  # True if there's at least one overlap
    print(f"Matching values: {small_tensor[overlap].shape}")  # Prints matching values
    print(f"All values Nested: {small_tensor.flatten().shape == small_tensor[overlap].shape}")


    
if __name__ == '__main__':

    # Example usage

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














