import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model_utils import *


# Set device for training and evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

full_dataset = datasets.CIFAR10(root='./data/Cifar-10', train=True, download=False, transform=transform)
# Load CIFAR-10 datasets
# train_dataset = datasets.CIFAR10(root='./data/Cifar-10', train=True, download=False, transform=transform)
test_dataset = datasets.CIFAR10(root='./data/Cifar-10', train=False, download=False, transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Split into training and validation sets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)



print(f"Number of Train samples: {len(train_dataset)}")
print(f"Number of Val samples: {len(val_dataset)}")
print(f"Number of Test samples: {len(test_dataset)}")


def adaptivity_CNN(original_model, scheduler, prune_ratio=0.5, exp_name="ToyExp_Adaptivity_CNN"):
    
    print(f"{"-"*20}Starting with Experiment {exp_name}{"-"*20}")
    acc_original, f1_original, params_original, mac_original, size_mb_original = evaluate(original_model, test_loader, profiling=True)
            
    # Apply pruning
    Index_Aware_pruner(original_model, amount=prune_ratio, plot=False)

    zero_channels_info, num_pruned_channels, pruned_weights = get_pruned_indices_and_counts(original_model)
    non_zero_channels_info, num_unpruned_channels, unpruned_weights = get_unpruned_indices_and_counts(original_model)

    core_model = Resnet_General(num_unpruned_channels).to(device)
    copy_weights_from_dict(core_model, unpruned_weights)
    print("DONE REBUILDING")

    fine_tuner(core_model, train_loader, val_loader, num_epochs=200, scheduler_type=scheduler, prune_ratio=0.5, exp_name=f"{exp_name}_Training_core_model", LR=1e-2)
    acc_pruned, f1_pruned, params_pruned, mac_pruned, size_mb_pruned = evaluate(core_model, test_loader, profiling=True)

    new_channels = extend_channels(core_model, num_pruned_channels)
    rebuilt_model = Resnet_General(new_channels).to(device)
    del core_model

    #TODO : Replace the unpruned weights with fine-tuned core model weights
    rebuilt_model = reconstruct_weights_from_dicts(rebuilt_model, pruned_indices=zero_channels_info, pruned_weights=pruned_weights, unpruned_indices=non_zero_channels_info, unpruned_weights=unpruned_weights)
    rebuilt_model = freeze_channels(rebuilt_model, non_zero_channels_info)

    fine_tuner(rebuilt_model, train_loader, val_loader, num_epochs=200, scheduler_type=scheduler, prune_ratio=0.5, exp_name=f"{exp_name}_Training_full_model", LR=1e-2)
    acc_rebuilt, f1_rebuilt, params_rebuilt, mac_rebuilt, size_mb_rebuilt = evaluate(rebuilt_model, test_loader, profiling=True)

    plot_comparison_3stage([acc_original, f1_original, acc_pruned, f1_pruned, acc_rebuilt, f1_rebuilt], exp_name, prune_ratio)
    plot_profiling([params_original, params_pruned, params_rebuilt], [mac_original, mac_pruned, mac_rebuilt], [size_mb_original, size_mb_pruned, size_mb_rebuilt], exp_name, prune_ratio)



if __name__ == '__main__':
    
    
    schedulers = ['cosine', 'step', 'exponential', 'cyclic']
    for sch in schedulers:
        # Load a pre-trained model
        orig_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True).to(device)
        adaptivity_CNN(orig_model, scheduler=sch, exp_name=f"Exp_adaptivity_{sch}")

    print(f"{"-"*20}EXPERIMENT FINISHED{"-"*20}")
        
    
    




