import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CyclicLR, CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import os
import pickle
import numpy as np
from utils.eval_utils import *
from utils.visualize_utils import *
from utils.prune_utils import *
from utils.train_utils import *
from models.Resnet import *
from models.VGG import *
import onnx
import pandas as pd
import random
from copy import deepcopy
from openpyxl import load_workbook

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

torch.manual_seed(42)

train_transform = transforms.Compose([
    ImgAugTransform_Large(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

full_dataset = datasets.CIFAR10(root='./data/Cifar-10', train=True, download=False, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data/Cifar-10', train=False, download=False, transform=test_transform)

train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

val_dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Number of Train samples: {len(train_dataset)}")
print(f"Number of Val samples: {len(val_dataset)}")
print(f"Number of Test samples: {len(test_dataset)}")

def iterative_filter_pruning(scheduler='cosine', prune_steps=3, prune_ratio=[0.3, 0.3, 0.3], epochs=20, lr=1e-3, optim="adam", exp_name="ToyExp_Global_Iter_Adaptivity_Resnet20", rebuild=True):

    assert prune_steps == len(prune_ratio), f"Expected Prune steps ({prune_steps}) to match the length of prune ratios ({len(prune_ratio)})."

    original_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True).to(device)

    accuracy_list = []
    profiling = {"params":[], "MAC's":[], "model_size":[]}#, "latency":[]}

    print(f"{'-'*20}Starting with Experiment {exp_name}{'-'*20}")
    acc_original, f1_original, params_original, mac_original, size_mb_original = evaluate(original_model, test_loader, profiling=True)
    accuracy_list.append(acc_original)
    profiling["params"].append(params_original)
    profiling["MAC's"].append(mac_original)
    profiling["model_size"].append(size_mb_original)
    #profiling["latency"].append(latency_original)

    orig_copy = deepcopy(original_model)
    unprune_meta_data = []

    # Iterative Pruning of Original Model
    for i in range(prune_steps):
        original_model, pruned_index = Index_Aware_pruner(original_model, amount=prune_ratio[i], prune_step=i, plot=False)
        non_zero_channels_info, num_unpruned_channels, unpruned_weights = get_unpruned_indices_and_counts(original_model)
        unprune_meta_data.append({"non_zero_channels_info" : non_zero_channels_info, "num_unpruned_channels" : num_unpruned_channels, "unpruned_weights" : unpruned_weights})

    with open(f"./saves/pruning_metadata/{exp_name}_unprune_data.txt", "w") as file:
        for index, dictionary in enumerate(unprune_meta_data):
            file.write(f"Entry {index + 1}:\n")
            for key, value in dictionary.items():
                file.write(f"  {key}: {value}\n")
            file.write("\n")  # Add a blank line between entries

    # Core Model Training
    core_model = Resnet_General(unprune_meta_data[-1]["num_unpruned_channels"]).to(device)

    copy_weights_from_dict(core_model, unprune_meta_data[-1]["unpruned_weights"])

    fine_tuner(core_model, train_loader, val_loader, num_epochs=epochs, optim_type=optim, scheduler_type=scheduler, exp_name=f"{exp_name}_Training_core_model_Level-{str(prune_steps+1)}", LR=lr)
    # KD_trainer_zerograd(teacher=orig_copy, student=core_model, train_loader=train_loader, val_loader=val_loader, lr=lr, alpha=0.75, beta=3, freeze_dim0=None, freeze_dim1=None, num_epochs=epochs, exp_name=f"{exp_name}_Training_model_Level-{i+1}")
    # fine_tuner_BN(core_model, train_loader, val_loader, num_epochs=epochs, num_epochs_BN=10, optim_type=optim, scheduler_type=scheduler, exp_name=f"{exp_name}_Training_core_model_Level-{str(prune_steps+1)}", LR=lr)

    acc_pruned, f1_pruned, params_core, mac_core, size_mb_core = evaluate(core_model, test_loader, profiling=True)

    save_statedict(core_model, unprune_meta_data[-1]["num_unpruned_channels"], path=f"./saves/state_dicts/{exp_name}_{model_name}_Core_Model.pth")
    accuracy_list.append(acc_pruned)
    profiling["params"].append(params_core)
    profiling["MAC's"].append(mac_core)
    profiling["model_size"].append(size_mb_core)
    #profiling["latency"].append(latency_core)
    model_summary_with_frozen_counts(core_model) 