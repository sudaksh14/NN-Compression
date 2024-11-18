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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm


# Set device for training and evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

data_dir = "./data/tiny-imagenet/tiny-imagenet-200"

# print(torch.hub.list("chenyaofo/pytorch-cifar-models", force_reload=True))

# Define transformations for validation set
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the Tiny-ImageNet datasets
# train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
# test_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load CIFAR-10 datasets
train_dataset = datasets.CIFAR10(root='./data/Cifar-10', train=True, download=False, transform=transform)
test_dataset = datasets.CIFAR10(root='./data/Cifar-10', train=False, download=False, transform=transform)

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


def evaluate(model, data_loader, plot_title, plot=False):
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
    
    return accuracy, f1



def fine_tuner(model, data_loader, num_epochs, prune_ratio, exp_name, LR=1e-3):
    # Set model to training mode
    model = model.to(device)
    model.train()

    # Only fine-tune the classifier layer
    # for param in model.parameters():
    #     param.requires_grad = False
        
    # for param in model.fc.parameters():  # For models with `fc` as the classifier layer, like ResNet
    #     param.requires_grad = True

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    train_losses = []
    train_accuracies = []

    # Fine-tuning loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(data_loader, desc="Training Batches", leave=True):
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
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}")
    
    plot_curves(num_epochs, train_losses, train_accuracies, prune_ratio, exp_name)
           
    print("Fine-Tuning Complete")
    

# Plot the loss and accuracy curves
def plot_curves(num_epochs, train_losses, train_accuracies, prune_ratio, exp_name):
    
    # Plot the training and validation loss curves
    plt.figure(figsize=(12, 5))

    # Plot Loss vs. Epoch
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    # plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()

    # Plot Accuracy vs. Epoch
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    # plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    
    plt.savefig(f"./plots/{exp_name}_Learning_Curve_Fine-Tuning_pruning_{prune_ratio}")


# Plotting results
def plot_comparison(val_accuracy, val_f1_score, fine_tune_accuracy, fine_tune_f1_score, exp_name, prune_ratio):
    plt.figure(figsize=(8, 4))
    plt.bar(['Accuracy Before', 'F1 Score Before', 'Accuracy After', 'F1 Score After'],
            [val_accuracy, val_f1_score, fine_tune_accuracy, fine_tune_f1_score],
            color=['skyblue', 'salmon', 'skyblue', 'salmon'])
    plt.ylabel("Score")
    plt.title("Evaluation Metrics Before and After Fine-Tuning")
    plt.savefig(f"./plots/{exp_name}_Resnet20_Accuracy_Comparison_{prune_ratio}")



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
        plot_comparison(orig_accuracy, orig_f1_score, fine_tune_accuracy, fine_tune_f1_score, exp_name, p)
        

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
                
                
                
def pruner_new(exp_name, prune_amount=[0.1,0.3,0.5,0.7,0.9]):
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

      
        
if __name__ == '__main__':
    
    # model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True).to(device)
    # pruner('FE_finetune')
    # iterative_pruner(train_loader, test_loader, 100, 0.2, "Iter_Prune")
    # Index_Aware_pruner(model, amount=0.2, plot=False)
    # non_zero_weights, new_channels = get_nonzero_indices(model)
    pruner_new("HighLR_Index_Aware_Pruning")
    
    
    
        
    
    
    
    

