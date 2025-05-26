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
import time
import numpy as np
from ptflops import get_model_complexity_info
from calflops import calculate_flops
from thop import profile
from deepspeed.profiling.flops_profiler import get_model_profile, FlopsProfiler
from torchsummary import summary

# Set device for training and evaluation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Running on device:", device)

def evaluate(model, data_loader, plot_title="Test Accuracy & F1-Score", device=device, plot=False, profiling=False, KD=False, layerwise=False):
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []

    if layerwise:
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _, _, _  = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    elif KD:
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _  = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    else:
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
        params, mac, size_mb = profile_model(model, (1,3,32,32))
        return accuracy, f1, params, mac, size_mb
        
    return accuracy, f1

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
def profile_model(model, input_size):
    # macs, params = compute_flops_and_macs(model, input_size)
    # flops, macs, params = get_model_profile(model=model, input_shape=input_size, print_profile=True, as_string=False)
    flops, macs, params = get_model_profile(model=model, input_shape=input_size, print_profile=False, as_string=False)
    size_mb = model_size_in_mb(model)

    print(f"Params: {params / 1e6:.3f} M, MACs: {macs / 1e9:.3f} GMACs, FLOPs: {flops / 1e9:.3f} GFLOPs, Size: {size_mb:.2f} MB")
    return params, macs, size_mb


def model_summary_with_frozen_counts(model, input_size=(3, 32, 32), summary=False):
    """
    Generate a summary of the model, including counts of frozen and unfrozen parameters.
    
    Args:
    - model: PyTorch model instance.
    - input_size: Tuple specifying the input size (e.g., (3, 224, 224) for an image).
    - device: Device for the model summary ("cuda" or "cpu").
    """
    # Use torchsummary for the general model summary
    if summary:
        print(summary(model, input_size))
    
    frozen_params_count = 0
    unfrozen_params_count = 0

    for param in model.parameters():
        if param.requires_grad:
            unfrozen_params_count += param.numel()
        else:
            frozen_params_count += param.numel()

    print("\nParameter Statistics:")
    print(f"  Total parameters: {frozen_params_count + unfrozen_params_count:,}")
    print(f"  Trainable parameters (unfrozen): {unfrozen_params_count:,}")
    print(f"  Non-trainable parameters (frozen): {frozen_params_count:,}")


def measure_latency(model, data_loader, device=device, warmup=5, iterations=100):
    latencies = []
    
    # Warm-Up
    for _ in range(warmup):
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
            break  # Warm-up for only one batch
    
    # Measure Latency
    with torch.no_grad():
        for i, (inputs, _) in enumerate(data_loader):
            if i >= iterations:
                break
            inputs = inputs.to(device)
            
            start_time = time.perf_counter()
            _ = model(inputs)
            end_time = time.perf_counter()
            
            latencies.append(end_time - start_time)
    
    avg_latency = sum(latencies) / len(latencies)
    print(f"Average Latency per Batch: {avg_latency:.6f} seconds")
    print(f"Latency per Image: {avg_latency / inputs.size(0):.6f} seconds")
    return avg_latency

def conv2d_macs(layer, input_shape):
        """
        Calculate MACs for a Conv2D layer.
        """
        # Get layer parameters
        in_channels = layer.in_channels
        out_channels = layer.out_channels
        kernel_size = layer.kernel_size[0] * layer.kernel_size[1]
        output_height = input_shape[2] // layer.stride[0]
        output_width = input_shape[3] // layer.stride[1]
        
        # MACs for Conv2D = Out_H * Out_W * (In_Ch * K * K + 1) * Out_Ch
        macs = output_height * output_width * (in_channels * kernel_size + 1) * out_channels
        return macs, (1, out_channels, output_height, output_width)

def fc_macs(layer, input_shape):
        """
        Calculate MACs for a Fully Connected (Linear) layer.
        """
        in_features = layer.in_features
        out_features = layer.out_features
        # MACs = In_Features * Out_Features
        macs = in_features * out_features
        return macs, (1, out_features)

def calculate_total_macs(model, input_size=(1, 3, 32, 32)):
    """
    Calculate Multiply-Accumulate Operations (MACs) for a given model manually.
    
    Args:
        model (torch.nn.Module): The PyTorch model to analyze.
        input_size (tuple): Input size (batch_size, channels, height, width).
        
    Returns:
        macs (int): Total MACs for the model.
    """

    total_macs = 0
    input_shape = input_size
    
    # Iterate over layers
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            macs, output_shape = conv2d_macs(layer, input_shape)
            total_macs += macs
            input_shape = output_shape
        elif isinstance(layer, nn.Linear):
            macs, output_shape = fc_macs(layer, input_shape)
            total_macs += macs
            input_shape = output_shape
        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            # AdaptiveAvgPool2d doesn't add MACs, but it changes the shape
            output_height, output_width = layer.output_size
            input_shape = (1, input_shape[1], output_height, output_width)

    print(f"Total MACs: {total_macs/1e9:.3f} GMACs")
