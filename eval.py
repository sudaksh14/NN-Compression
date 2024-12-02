import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from torchsummary import summary
from model_utils import *
from pytorch_modelsize import SizeEstimator
from tqdm import tqdm


# Set device for training and evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

data_dir = "./data/tiny-imagenet/tiny-imagenet-200"

# print(torch.hub.list("chenyaofo/pytorch-cifar-models", force_reload=True))a

# Define transformations for data augmentation
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# Load the Tiny-ImageNet datasets
# train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
# test_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load CIFAR-10 datasets
train_dataset = datasets.CIFAR10(root='./data/Cifar-10', train=True, download=False, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data/Cifar-10', train=False, download=False, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

print(f"Number of Train samples: {len(train_dataset)}")
print(f"Number of Test samples: {len(test_dataset)}")

# model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn", pretrained=True)

# Modify the last layer for 10 classes (CIFAR-10)
# num_classes = 10
# Modify the last layer for 200 classes (Tiny Imagenet)
# num_classes = 200
# alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, num_classes)
# model.fc = nn.Linear(model.fc.in_features, num_classes)


# print(summary(model, input_size=(3, 32, 32)))

# for name,layer in rebuilt_model.named_children():
#     print(f"Layer Name: {name} | Layer Type: {layer}")

# se = SizeEstimator(rebuilt_model, input_size=(1, 3, 32, 32))
# print(se.estimate_size())
# print(se.param_bits) # bits taken up by parameters
# print(se.forward_backward_bits) # bits stored for forward and backward
# print(se.input_bits) # bits for input


def fine_tuner(model, data_loader, num_epochs, prune_ratio, exp_name, LR=1e-3):

    model = model.to(device)
    model.train()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    train_losses = []
    train_accuracies = []

    # Fine-tuning loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in data_loader:
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

        scheduler.step()
            

        
        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}")
    
    plot_curves(num_epochs, train_losses, train_accuracies, prune_ratio, exp_name)
           
    print("Fine-Tuning Complete")
    

def pruner(exp_name, prune_amount=[0.1,0.3,0.5,0.7,0.9]):
    for prune_ratio in prune_amount:
        p = str(int(prune_ratio*100))
        print(f"Starting with Pruning ratio of {p}%")
        
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True).to(device)
        
        with open("resnet18_orig_state_dict.txt", "w") as f:
            for param_tensor in model.state_dict():
                f.write(f"{param_tensor}:\n{model.state_dict()[param_tensor]}\n\n")
        
        # Evaluate the model before Pruning
        orig_accuracy, orig_f1_score = evaluate(model, test_loader, f"{exp_name}_Resnet_Before_Pruning_{p}%")
        
        # Apply pruning
        prune_resnet(model, amount=prune_ratio)
        
        with open("resnet20_pruned_state_dict.txt", "w") as f:
            for param_tensor in model.state_dict():
                f.write(f"{param_tensor}:\n{model.state_dict()[param_tensor]}\n\n")
            
        
        # Rebuilding Model
        rebuilt_model = rebuild_resnet(model, prune_ratio).to(device)
        print("DONE REBUILDING")
        
        # Check the number of parameters in the original and pruned models
        original_params = sum(p.numel() for p in model.parameters())
        final_params = sum(p.numel() for p in rebuilt_model.parameters())
        print(f'Original Model Parameters: {original_params}')
        print(f'Pruned Model Parameters: {final_params}')
        
        # Fine-Tune the rebuilt model
        fine_tuner(rebuilt_model, train_loader, num_epochs=100, prune_ratio=p, exp_name=exp_name)
        fine_tune_accuracy, fine_tune_f1_score = evaluate(rebuilt_model, test_loader, f"{exp_name}_Resnet_After_Fine-tune_{p}%")
        
        # Plot the results
        plot_comparison(orig_accuracy, orig_f1_score, fine_tune_accuracy, fine_tune_f1_score, exp_name, prune_ratio)
        

def iterative_pruner(train_loader, test_loader, num_epochs, prune_ratio, exp_name):
    # Set model to training mode
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True).to(device)
    
    # Evaluate the model before Pruning
    orig_accuracy, orig_f1_score = evaluate(model, test_loader, f"{exp_name}_Resnet_Before_Pruning")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Cosine Annealing Warm Restarts scheduler with Adam
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Iterative pruning and fine-tuning loop
    num_pruning_cycles = 3
    
    for pruning_cycle in range(num_pruning_cycles):
        print(f"\nPruning Cycle {pruning_cycle + 1}/{num_pruning_cycles}")
        
        prune_resnet(model, amount=prune_ratio)
        
        if pruning_cycle == 0:
            model = rebuild_resnet(model, prune_ratio)
            print("DONE REBUILDING")
              
            
        else:
            model = rebuild_resnet(model, prune_ratio, first_channel)
            print("DONE REBUILDING")
            
        

        train_losses = []
        train_accuracies = []

        # Fine-tuning loop
        for epoch in range(num_epochs):
            model.to(device)
            model.train()
            
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
            # Step the scheduler to apply warm restarts
            scheduler.step(epoch + pruning_cycle * num_epochs)

            
            # Calculate average loss and accuracy for the epoch
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_accuracy = correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)
            
            print(f"Epoch [{epoch+1}/{num_epochs}], "f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}")
        
        plot_curves(num_epochs, train_losses, train_accuracies, prune_ratio, f"{exp_name}_{pruning_cycle}")
        
        fine_tune_accuracy, fine_tune_f1_score = evaluate(model, test_loader, f"{exp_name}_Resnet_After_Fine-tune")
        print(f"Fine-Tuning Complete for Pruning Cycle {pruning_cycle + 1}/{num_pruning_cycles}")
        
        # Plot the results
        plot_comparison(orig_accuracy, orig_f1_score, fine_tune_accuracy, fine_tune_f1_score, f"{exp_name}_{pruning_cycle}", prune_ratio)
        
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Conv2d) and name == "conv1":
                first_channel = layer.weight.shape[0]
                
                
                
def filter_pruning(exp_name, prune_amount=[0.1,0.3,0.5,0.7,0.9]):
    for prune_ratio in prune_amount:
        p = str(int(prune_ratio*100))
        print(f"Starting with Pruning ratio of {p}%")
        
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True).to(device)
        
        
        # Evaluate the model before Pruning
        orig_accuracy, orig_f1_score = evaluate(model, test_loader, f"{exp_name}_Resnet_Before_Pruning_{p}%")
        
        # Apply pruning
        Index_Aware_pruner(model, amount=prune_ratio, plot=False)
        non_zero_weights, new_channels = get_nonzero_indices(model)

        # Rebuilding Model
        rebuilt_model = Resnet_General(new_channels).to(device)
        copy_weights(model, rebuilt_model, non_zero_weights)
        print("DONE REBUILDING")
        
        # Check the number of parameters in the original and pruned models
        original_params = sum(p.numel() for p in model.parameters())
        final_params = sum(p.numel() for p in rebuilt_model.parameters())
        print(f'Original Model Parameters: {original_params}')
        print(f'Pruned Model Parameters: {final_params}')
        
        # Fine-Tune the rebuilt model
        fine_tuner(rebuilt_model, train_loader, num_epochs=100, prune_ratio=p, exp_name=exp_name, LR=1e-2)
        fine_tune_accuracy, fine_tune_f1_score = evaluate(rebuilt_model, test_loader, f"{exp_name}_Resnet_After_Fine-tune_{p}%")
        
        torch.onnx.export(rebuilt_model, (torch.randn(1, 3, 32, 32).to(device),), f=f"./saves/onnx/rebuilt_model_{p}.onnx")
        
        # Plot the results
        plot_comparison(orig_accuracy, orig_f1_score, fine_tune_accuracy, fine_tune_f1_score, exp_name, p)
        

def self_adaptive_CNN(exp_name, pruning_ratios=[0.3,0.5,0.7], path="./saves/state_dicts/", rebuilding=True):
    
    orig_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True).to(device)
    torch.save(orig_model.state_dict(), os.path.join(path, f"Resnet_model_original"))
    moving_acc = []
    moving_f1 = []
    
    # Evaluate the model before Pruning
    orig_accuracy, orig_f1_score = evaluate(orig_model, test_loader, "Resnet_Orig_metrics_cifar10", device=device)
    moving_acc.append(orig_accuracy)
    moving_f1.append(orig_f1_score)
    
    
    for prune_ratio in pruning_ratios: 
        p = str(int(prune_ratio*100))
        print(f"Starting with Pruning ratio of {p}%")
        
        orig_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True).to(device)
        
        # Apply pruning
        Index_Aware_pruner(orig_model, amount=prune_ratio, plot=False)
        non_zero_weights, new_channels = get_nonzero_indices(orig_model)
        
        # Saving the channel dicts
        channel_dict_path = os.path.join("./saves/channel_dicts", f"channel_dict_pruned_{p}")
        with open(channel_dict_path, "wb") as f:
            pickle.dump(new_channels, f)

        # Rebuilding the pruned Model
        pruned_model = Resnet_General(new_channels).to(device)
        copy_weights(orig_model, pruned_model, non_zero_weights)
        print("DONE REBUILDING")
        
        # Check the number of parameters in the original and pruned models
        original_params = sum(p.numel() for p in orig_model.parameters())
        final_params = sum(p.numel() for p in pruned_model.parameters())
        print(f'Original Model Parameters: {original_params}')
        print(f'Pruned Model Parameters: {final_params}')
        
        # Fine-Tune the pruned model
        fine_tuner(pruned_model, train_loader, num_epochs=200, prune_ratio=p, exp_name=exp_name, LR=1e-2)
        fine_tune_accuracy, fine_tune_f1_score = evaluate(pruned_model, test_loader, f"{exp_name}_Resnet_After_Fine-tune_{p}%", device=device)
        moving_acc.append(fine_tune_accuracy)
        moving_f1.append(fine_tune_f1_score)
        
        # torch.onnx.export(pruned_model, (torch.randn(1, 3, 32, 32).to(device),), f=f"./saves/onnx/pruned_model_{p}.onnx")
        torch.save(pruned_model.state_dict(), os.path.join("./saves/state_dicts", f"Resnet_model_pruned_{p}"))
        
        # Plot the results
        plot_comparison(orig_accuracy, orig_f1_score, fine_tune_accuracy, fine_tune_f1_score, exp_name, p)
    
    if rebuilding:
        
        for prune_ratio in pruning_ratios[-2::-1]:
            p = str(int(prune_ratio*100))
            
            # Rebuilding a bigger model
            channel_dict_path = os.path.join("./saves/channel_dicts", f"channel_dict_pruned_{p}")
            state_dict_path = os.path.join("./saves/state_dicts", f"Resnet_model_pruned_{p}")
            rebuilt_model = rebuilt_bigger(channel_dict_path, state_dict_path, device=device)
            rebuilt_accuracy, rebuilt_f1_score = evaluate(rebuilt_model, test_loader, f"{exp_name}_Resnet_After_Fine-tune_{p}%", device=device)
            moving_acc.append(rebuilt_accuracy)
            moving_f1.append(rebuilt_f1_score)

        plot_metrics(moving_acc, moving_f1, ticks=["Orig", "30%", "50%", "70%", "50%", "30%"], save_path=f"./plots/{exp_name}_metrics_plot.png")
            
            
            
    
    
    

    
        
if __name__ == '__main__':
    
    # model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True).to(device)
    # pruner('FE_finetune')
    # iterative_pruner(train_loader, test_loader, 100, 0.2, "Iter_Prune")
    # Index_Aware_pruner(model, amount=0.2, plot=False)
    # non_zero_weights, new_channels = get_nonzero_indices(model)
    # filter_pruning("HighLR_Index_Aware_Pruning")
    self_adaptive_CNN("Exp_Resnet_pruning_200ep")
    
        
    
    
    
    

