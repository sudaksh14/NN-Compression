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



# Set device for training and evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

Data_Augmentation = False


if Data_Augmentation:
    train_transform = transforms.Compose([
        ImgAugTransform(),  # Apply ImgAug augmentations
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    
    full_dataset = datasets.CIFAR10(root='./data/Cifar-10', train=True, download=False, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data/Cifar-10', train=False, download=False, transform=test_transform)
    
    
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    
    full_dataset = datasets.CIFAR10(root='./data/Cifar-10', train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data/Cifar-10', train=False, download=False, transform=transform)


# Load CIFAR-10 datasets

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Split into training and validation sets
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Number of Train samples: {len(train_dataset)}")
print(f"Number of Val samples: {len(val_dataset)}")
print(f"Number of Test samples: {len(test_dataset)}")

# original_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True).to(device)
# torch.onnx.export(original_model, (torch.randn(1, 3, 32, 32).to(device),), f=f"./saves/onnx/VGG16_Original_Model.onnx")
# print(dir(original_model))

def iterative_filter_pruning(scheduler='cosine', prune_steps=3, prune_ratio=[0.3, 0.3, 0.3], model_name="Resnet18", epochs=20, lr=1e-3, exp_name="ToyExp_Iter_Adaptivity_Resnet20", rebuild=True):
    
    assert prune_steps == len(prune_ratio), f"Expected Prune steps ({prune_steps}) to match the length of prune ratios ({len(prune_ratio)})."

    accuracy_list = []
    profiling = {"params":[], "MAC's":[], "model_size":[], "latency":[]}
    
    print(f"{"-"*20}Starting with Experiment {exp_name}{"-"*20}")
    acc_original, f1_original, params_original, mac_original, size_mb_original, latency_original = evaluate(original_model, test_loader, profiling=True)
    accuracy_list.append(acc_original)
    profiling["params"].append(params_original)
    profiling["MAC's"].append(mac_original)
    profiling["model_size"].append(size_mb_original)
    profiling["latency"].append(latency_original)
                       
    prune_meta_data = []
    unprune_meta_data = []
    
    for i in range(prune_steps):
        if i == 0:
            Index_Aware_pruner(original_model, amount=prune_ratio[i], plot=False)
            zero_channels_info, num_pruned_channels, pruned_weights = get_pruned_indices_and_counts(original_model)
            non_zero_channels_info, num_unpruned_channels, unpruned_weights = get_unpruned_indices_and_counts(original_model)
        else:
            Index_Aware_pruner(core_model, amount=prune_ratio[i], plot=False)
            zero_channels_info, num_pruned_channels, pruned_weights = get_pruned_indices_and_counts(core_model)
            non_zero_channels_info, num_unpruned_channels, unpruned_weights = get_unpruned_indices_and_counts(core_model)
            
        prune_meta_data.append({"zero_channels_info" : zero_channels_info, "num_pruned_channels" : num_pruned_channels, "pruned_weights" : pruned_weights}) 
        unprune_meta_data.append({"non_zero_channels_info" : non_zero_channels_info, "num_unpruned_channels" : num_unpruned_channels, "unpruned_weights" : unpruned_weights})

        if "Resnet" in model_name:
            core_model = Resnet_General(unprune_meta_data[-1]["num_unpruned_channels"]).to(device)
        elif "VGG" in model_name:
            core_model = VGG_General(unprune_meta_data[-1]["num_unpruned_channels"]).to(device)
        copy_weights_from_dict(core_model, unprune_meta_data[-1]["unpruned_weights"])
    
    fine_tuner(core_model, train_loader, val_loader, num_epochs=epochs, optim_type="SGD", scheduler_type=scheduler, prune_ratio=0.3, exp_name=f"{exp_name}_Training_core_model_Level-4", LR=lr)
    acc_pruned, f1_pruned, params_core, mac_core, size_mb_core, latency_core = evaluate(core_model, test_loader, profiling=True)
    # torch.onnx.export(core_model, (torch.randn(1, 3, 32, 32).to(device),), f=f"./saves/onnx/{model_name}_Core_Model.onnx")
    # save_statedict(core_model, unprune_meta_data[-1]["num_unpruned_channels"], path=f"./saves/state_dicts/{model_name}_Core_Model.pth")
    accuracy_list.append(acc_pruned)
    profiling["params"].append(params_core)
    profiling["MAC's"].append(mac_core)
    profiling["model_size"].append(size_mb_core)
    profiling["latency"].append(latency_core)
    model_summary_with_frozen_counts(core_model, input_size=(3, 32, 32)) 

    if rebuild:
        for i in range(prune_steps-1, -1, -1):
            new_channels = get_rebuild_channels(unprune_meta_data[i]["num_unpruned_channels"], prune_meta_data[i]["num_pruned_channels"])
            if "Resnet" in model_name:
                rebuilt_model = Resnet_General(new_channels).to(device)
            elif "VGG" in model_name:
                rebuilt_model = VGG_General(new_channels).to(device)
            get_core_weights(core_model, unprune_meta_data[i]["unpruned_weights"])
            del core_model

            rebuilt_model = reconstruct_weights_from_dicts(rebuilt_model, pruned_indices=prune_meta_data[i]["zero_channels_info"], pruned_weights=prune_meta_data[i]["pruned_weights"], unpruned_indices=unprune_meta_data[i]["non_zero_channels_info"], unpruned_weights=unprune_meta_data[i]["unpruned_weights"])
            rebuilt_model = freeze_channels_2d(rebuilt_model, unprune_meta_data[i]["non_zero_channels_info"])
            model_summary_with_frozen_counts(rebuilt_model, input_size=(3, 32, 32))
            
            fine_tuner(rebuilt_model, train_loader, val_loader, num_epochs=epochs, optim_type="SGD", scheduler_type=scheduler, prune_ratio=0.3, exp_name=f"{exp_name}_Training_model_Level-{i+1}", LR=lr)
            check = validate_frozen_channels(rebuilt_model, unprune_meta_data[i]["non_zero_channels_info"])
            print(f"Channel Freezing applied correctly: {check}")
            acc_rebuilt, f1_rebuilt, params_rebuilt, mac_rebuilt, size_mb_rebuilt, latency_rebuilt = evaluate(rebuilt_model, test_loader, profiling=True)
            # torch.onnx.export(rebuilt_model, (torch.randn(1, 3, 32, 32).to(device),), f=f"./saves/onnx/{model_name}_Descendant_Model_Level-{i+1}.onnx")
            # save_statedict(rebuilt_model, new_channels, path=f"./saves/state_dicts/{model_name}_Descendant_Model_Level-{i+1}.pth")
            core_model = rebuilt_model

            accuracy_list.append(acc_rebuilt)
            profiling["params"].append(params_rebuilt)
            profiling["MAC's"].append(mac_rebuilt)
            profiling["model_size"].append(size_mb_rebuilt)
            profiling["latency"].append(latency_rebuilt)
            

        plot_multilevel(accuracy_list, exp_name)
        plot_profiling(profiling["params"], profiling["MAC's"], profiling["model_size"], exp_name, prune_ratio=0.3, labels=["Original", "Level-4", "Level-3", "Level-2", "Level-1"])

        return accuracy_list, profiling
        
        
        
        
  

if __name__ == '__main__':

    schedulers = ['cosine', 'step', 'exponential', 'cyclic', 'Default']
    lrs = [1e-2, 1e-3, 1e-4, 1e-5]
    epochs = [20, 50, 100]

    results = []
    trial = 0

    for sch in schedulers:
        for lr in lrs:
            for epoch in epochs:
                original_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True).to(device)
                test_acc, profiling = iterative_filter_pruning(model_name="Resnet18", epochs=epoch, lr=lr, scheduler=sch, exp_name=f"Exp_{trial+1}_Iter_Adaptivity_Resnet18_{sch.upper()}_{str(lr)}_{str(epoch)}")
                
                results.append({
                        "Trial": trial + 1,
                        "Learning Rate": lr,
                        "Epochs": epoch,
                        "Scheduler": sch.upper(),
                        "Test Accuracy (Original)": test_acc[0],
                        "Test Accuracy (Core Model)": test_acc[1],
                        "Test Accuracy (Level-3)": test_acc[2],
                        "Test Accuracy (Level-2)": test_acc[3],
                        "Test Accuracy (Level-1)": test_acc[4],
                        "Num_Params": profiling["params"],
                        "MAC's": profiling["MAC's"],
                        "Model Size (MB)": profiling["model_size"],
                        "Latency": profiling["latency"]
                        
                    })
                trial += 1
    
    results_df = pd.DataFrame(results)
    results_df.to_excel(f"./saves/Results/HPO_Resnet18_results.xlsx")

    print(f"{"-"*20}EXPERIMENT FINISHED{"-"*20}")

    
    # input = (1,3,32,32)
    # path = "saves/onnx/Resnet20_Core_Model.onnx"
    # path = "saves/onnx/Resnet20_Descendant_Model_Level-1.onnx"
    # model = ConvertModel(onnx.load(path))
    # model.eval()
    # profile_model(model, input)

    # path_core = "saves/state_dicts/VGG16_Core_Model.pth"
    # path_level3 = "saves/state_dicts/VGG16_Descendant_Model_Level-3.pth"


    # loaded_data_core = torch.load(path_core, map_location=device)
    # loaded_data_level3 = torch.load(path_level3, map_location=device)

    # state_dict_core = loaded_data_core["state_dict"]
    # channel_dict_core = loaded_data_core["metadata"]
    # model_core = VGG_General(channel_dict_core).to(device)
    # model_core.load_state_dict(state_dict_core)

    # state_dict_level3 = loaded_data_level3["state_dict"]
    # channel_dict_level3 = loaded_data_level3["metadata"]
    # model_level3 = VGG_General(channel_dict_level3).to(device)
    # model_level3.load_state_dict(state_dict_level3)

    # model_core.eval()
    # model_level3.eval()

    # layer_name = "features.0"
    # # channel_index = random.randint(1, 23)
    # # print(channel_index)

    # layer_core = dict(model_core.named_modules()).get(layer_name)
    # layer_level3 = dict(model_level3.named_modules()).get(layer_name)
    
    # for channel_index in range(23):
    #     tensor1 = layer_core.weight.data[channel_index,2]
    #     tensor2 = layer_level3.weight.data[channel_index,2]

    #     print(torch.equal(tensor1, tensor2))