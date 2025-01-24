import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CyclicLR, CosineAnnealingLR, ReduceLROnPlateau
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.visualize_utils import plot_curves_with_val, plot_KD_metrics
from tqdm import tqdm

# Set device for training and evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

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

def imgaug_dolly():
    aug = iaa.Sequential(
            [
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5), # add gaussian noise
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.1),  # vertically flip 20% of all images
                sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, per axis
                    translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},  # translate by -20 to +20 percent
                    rotate=(-5, 5),  # rotate by -45 to +45 degrees
                    shear=(-5, 5),  # shear by -16 to +16 degrees
                    # order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    # cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 3),
                           [
                               # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                               iaa.OneOf([
                                   iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5), # add gaussian noise
                                   #iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means (kernel sizes between 2 and 7)
                                   iaa.MedianBlur(k=(3, 11)),
                                   # blur image using local medians (kernel sizes between 2 and 7)
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                               iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                               # search either for all edges or for directed edges
                               # sometimes(iaa.OneOf([
                               #    iaa.EdgeDetect(alpha=(0, 0.7)),
                               #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                               # ])),
                               
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),
                               iaa.Invert(0.05, per_channel=True), # invert color channels
                               iaa.Add((-10, 10), per_channel=0.5),  # change brightness of images
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),  # change brightness of images
                               iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )
    return aug


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


def save_statedict(model, channel_dict, path="./saves/state_dicts"):
    state_dict = model.state_dict()

    save_dict = {
        "state_dict": state_dict,
        "metadata": channel_dict,
    }

    torch.save(save_dict, path)
    print(f"Model state and metadata saved to {path}")



def fine_tuner(model, train_loader, val_loader, num_epochs, optim_type, scheduler_type, prune_ratio, exp_name, device=device, LR=1e-3):

    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    if optim_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Define LR scheduler
    if scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
    elif scheduler_type == 'exponential':
        scheduler = ExponentialLR(optimizer, gamma=0.95)
    elif scheduler_type == 'cyclic':
        scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=0.01, step_size_up=20, mode='triangular2')
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        

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

        # Step the scheduler and track learning rate
        if scheduler_type == 'Default':
            scheduler.step(val_loss)
        else:    
            scheduler.step()
        
        current_lr = scheduler.get_last_lr()[0]
        lrs.append(current_lr)
        
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
    
    # plot_curves_with_val(num_epochs, train_losses, train_accuracies, val_losses, val_accuracies, lrs, prune_ratio, exp_name)
    del optimizer
           
    print("Fine-Tuning Complete")



class HintLoss(nn.Module):
    def __init__(self):
        super(HintLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, teacher_features, transformed_student_features):
        return self.mse_loss(teacher_features, transformed_student_features)

class TransformationLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        A transformation layer to align the channel dimensions of teacher and student feature maps.

        Args:
            in_channels (int): Number of input channels from the student model.
            out_channels (int): Number of output channels matching the teacher model.
        """
        super(TransformationLayer, self).__init__()
        self.transform = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.transform(x)

def validate(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update metrics
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy

def train_accuracy(model, train_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def train_student_with_distillation(student, teacher, train_loader, val_loader, hint_layer_name, guided_layer_name, teacher_channels, student_channels, lr=0.001, alpha=0.75, beta=3, device=device):
    teacher = teacher.to(device).eval()
    student = student.to(device).train()

    # Initialize the transformation layer
    transformation_layer = TransformationLayer(student_channels, teacher_channels).to(device)
    hint_loss_fn = HintLoss()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(list(student.parameters()) + list(transformation_layer.parameters()), lr=lr)

    train_total_loss = 0.0
    train_total_correct = 0
    train_total_samples = 0
    train_total_hint_loss = 0.0
    train_total_distillation_loss = 0.0

    # Hook dictionaries to store intermediate features
    student_features = {}
    teacher_features = {}

    # Hook functions to capture intermediate features
    def hook_student(_, __, output):
        student_features[hint_layer_name] = output

    def hook_teacher(_, __, output):
        teacher_features[guided_layer_name] = output

    # Register hooks
    getattr(student, hint_layer_name).register_forward_hook(hook_student)
    getattr(teacher, guided_layer_name).register_forward_hook(hook_teacher)

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        student_outputs = student(inputs)
        with torch.no_grad():
            teacher_outputs = teacher(inputs)

        # Compute losses
        classification_loss = criterion(student_outputs, labels)
        hint_loss = hint_loss_fn(transformation_layer(student_features[hint_layer_name]), teacher_features[guided_layer_name])
        distillation_loss = F.kl_div(F.log_softmax(student_outputs / beta, dim=1),
                                     F.softmax(teacher_outputs / beta, dim=1),
                                     reduction="batchmean") * (beta ** 2)

        loss = alpha * classification_loss + (1 - alpha) * distillation_loss + hint_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update train metrics
        train_total_loss += loss.item() * inputs.size(0)
        train_total_hint_loss += hint_loss.item() * inputs.size(0)
        train_total_distillation_loss += distillation_loss.item() * inputs.size(0)
        _, predicted = torch.max(student_outputs, dim=1)
        train_total_correct += (predicted == labels).sum().item()
        train_total_samples += inputs.size(0)

    train_avg_loss = train_total_loss / train_total_samples
    train_avg_hint_loss = train_total_hint_loss / train_total_samples
    train_avg_distillation_loss = train_total_distillation_loss / train_total_samples
    train_accuracy = train_total_correct / train_total_samples

    # Validation loop
    student.eval()

    val_total_loss = 0.0
    val_total_correct = 0
    val_total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = student(inputs)
            loss = criterion(outputs, labels)

            # Update validation metrics
            val_total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            val_total_correct += (predicted == labels).sum().item()
            val_total_samples += inputs.size(0)

    val_avg_loss = val_total_loss / val_total_samples
    val_accuracy = val_total_correct / val_total_samples

    return train_avg_loss, train_avg_hint_loss, train_avg_distillation_loss, train_accuracy, val_avg_loss, val_accuracy


def KD_trainer(student, teacher, train_loader, val_loader, optimizer, criterion, hint_loss_fn, transformation_layer, alpha, beta, device, epochs=20, exp_name="KD"):
    train_losses = []
    train_hint_losses = []
    train_distillation_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        train_avg_loss, train_avg_hint_loss, train_avg_distillation_loss, train_accuracy, val_avg_loss, val_accuracy = train_student_with_distillation(
            student, teacher, train_loader, val_loader, optimizer, criterion, hint_loss_fn, transformation_layer, alpha, beta, device
        )

        train_losses.append(train_avg_loss)
        train_hint_losses.append(train_avg_hint_loss)
        train_distillation_losses.append(train_avg_distillation_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_avg_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch + 1}/{epochs}]: Train Loss: {train_avg_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_avg_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    plot_KD_metrics(train_losses, train_hint_losses, train_distillation_losses, train_accuracies, val_losses, val_accuracies, epochs, exp_name)
    print("Knowledge Distillation Training Complete")


if __name__ == '__main__':
    teacher = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True).to(device)
    student = "Pruned model"
    KD_trainer()
    


