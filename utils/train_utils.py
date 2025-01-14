import torch
import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CyclicLR, CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.visualize_utils import plot_curves_with_val



# Define a custom ImgAugTransform
class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.CropAndPad(px=(0, 4)),  # Random Crop with padding
            iaa.Fliplr(0.5),  # Random Horizontal Flip
            iaa.Affine(rotate=(-15, 15)),  # Random Rotation
            iaa.AddToHueAndSaturation((-20, 20)),  # Random Color Jitter
            iaa.Cutout(nb_iterations=1, size=0.2, squared=True),  # Cutout with 20% size
        ])

    def __call__(self, img):
        img = np.array(img)  # Convert PIL image to numpy array
        img = self.aug(image=img)  # Apply augmentation
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]




def save_image_augment(train_loader):
    # Function to show images
    def imshow(img):
        img = img.numpy().transpose((1, 2, 0))  # Convert to HWC format
        img = img * np.array([0.247, 0.243, 0.261]).reshape(1, 1, 3) + np.array([0.4914, 0.4822, 0.4465]).reshape(1, 1, 3)  # Unnormalize
        plt.imshow(np.clip(img, 0, 1))
        plt.axis('off')

    # Plot a batch of images
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    plt.figure(figsize=(12, 6))
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        imshow(images[i])
        plt.title(f"Label: {labels[i]}")
    plt.axis("off")
    plt.savefig("./data_aug_image")



def fine_tuner(model, train_loader, val_loader, num_epochs, scheduler_type, prune_ratio, exp_name, device=device, LR=1e-3):

    model = model.to(device)

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
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # Calculate average loss and accuracy for the epoch
        epoch_loss = train_loss / len(train_loader.dataset)
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
    del optimizer
           
    print("Fine-Tuning Complete")          
             

def fine_tuner_tqdm(model, train_loader, val_loader, num_epochs, scheduler_type, prune_ratio, exp_name, device=device, LR=1e-3):

    model = model.to(device)

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
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # Calculate average loss and accuracy for the epoch
        epoch_loss = train_loss / len(train_loader.dataset)
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
    del optimizer
           
    print("Fine-Tuning Complete")