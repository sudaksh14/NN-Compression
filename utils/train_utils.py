import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CyclicLR, CosineAnnealingLR, ReduceLROnPlateau
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.visualize_utils import plot_curves_with_val, plot_KD_metrics
from tqdm import tqdm
from utils.prune_utils import replace_previous_weights, zero_out_gradients_v2

# Set device for training and evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

ia.seed(1)  # Set seed for reproducibility

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

# Define a custom ImgAugTransform for large augmentations
class ImgAugTransform_Large():
    def __init__(self):
        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
        # image.
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        # Define our sequence of augmentation steps that will be applied to every image.
        seq = iaa.Sequential(
            [
            #
            # Apply the following augmenters to most images.
            #
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.2), # vertically flip 20% of all images

            # crop some of the images by 0-10% of their height/width
            sometimes(iaa.Crop(percent=(0, 0.1))),

            # Apply affine transformations to some of the images
            # - scale to 80-120% of image height/width (each axis independently)
            # - translate by -20 to +20 relative to height/width (per axis)
            # - rotate by -45 to +45 degrees
            # - shear by -16 to +16 degrees
            # - order: use nearest neighbour or bilinear interpolation (fast)
            # - mode: use any available mode to fill newly created pixels
            #         see API or scikit-image for which modes are available
            # - cval: if the mode is constant, then use a random brightness
            #         for the newly created pixels (e.g. sometimes black,
            #         sometimes white)
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),
                shear=(-16, 16),
                order=[0, 1],
                cval=(0, 255),
                mode=ia.ALL
            )),

            #
            # Execute 0 to 5 of the following (less important) augmenters per
            # image. Don't execute all of them, as that would often be way too
            # strong.
            #
            iaa.SomeOf((0, 5),
                [
                    # Convert some images into their superpixel representation,
                    # sample between 20 and 200 superpixels per image, but do
                    # not replace all superpixels with their average, only
                    # some of them (p_replace).
                    sometimes(
                        iaa.Superpixels(
                            p_replace=(0, 1.0),
                            n_segments=(20, 200)
                        )
                    ),

                    # Blur each image with varying strength using
                    # gaussian blur (sigma between 0 and 3.0),
                    # average/uniform blur (kernel size between 2x2 and 7x7)
                    # median blur (kernel size between 3x3 and 11x11).
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)),
                        iaa.AverageBlur(k=(2, 7)),
                        iaa.MedianBlur(k=(3, 11)),
                    ]),

                    # Sharpen each image, overlay the result with the original
                    # image using an alpha between 0 (no sharpening) and 1
                    # (full sharpening effect).
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                    # Same as sharpen, but for an embossing effect.
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                    # Search in some images either for all edges or for
                    # directed edges. These edges are then marked in a black
                    # and white image and overlayed with the original image
                    # using an alpha of 0 to 0.7.
                    sometimes(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0, 0.7)),
                        iaa.DirectedEdgeDetect(
                            alpha=(0, 0.7), direction=(0.0, 1.0)
                        ),
                    ])),

                    # Add gaussian noise to some images.
                    # In 50% of these cases, the noise is randomly sampled per
                    # channel and pixel.
                    # In the other 50% of all cases it is sampled once per
                    # pixel (i.e. brightness change).
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                    ),

                    # Either drop randomly 1 to 10% of all pixels (i.e. set
                    # them to black) or drop them on an image with 2-5% percent
                    # of the original size, leading to large dropped
                    # rectangles.
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5),
                        iaa.CoarseDropout(
                            (0.03, 0.15), size_percent=(0.02, 0.05),
                            per_channel=0.2
                        ),
                    ]),

                    # Invert each image's channel with 5% probability.
                    # This sets each pixel value v to 255-v.
                    iaa.Invert(0.05, per_channel=True), # invert color channels

                    # Add a value of -10 to 10 to each pixel.
                    iaa.Add((-10, 10), per_channel=0.5),

                    # Change brightness of images (50-150% of original value).
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),

                    # Improve or worsen the contrast of images.
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5),

                    # Convert each image to grayscale and then overlay the
                    # result with the original with random alpha. I.e. remove
                    # colors with varying strengths.
                    iaa.Grayscale(alpha=(0.0, 1.0)),

                    # In some images move pixels locally around (with random
                    # strengths).
                    sometimes(
                        iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                    ),

                    # In some images distort local areas with varying strength.
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                ],
                # do all of the above augmentations in random order
                random_order=True
            )
        ],
        # do all of the above augmentations in random order
        random_order=True
        )
        
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
    plt.savefig("./data_aug_image_large")


def save_statedict(model, channel_dict, path="./saves/state_dicts"):
    state_dict = model.state_dict()

    save_dict = {
        "state_dict": state_dict,
        "metadata": channel_dict,
    }

    torch.save(save_dict, path)
    print(f"Model state and metadata saved to {path}")



def fine_tuner(model, train_loader, val_loader, num_epochs, optim_type, scheduler_type, exp_name, LR=1e-3):

    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    if optim_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Define LR scheduler
    if scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=int(num_epochs*0.25), gamma=0.1)
    elif scheduler_type == 'exponential':
        scheduler = ExponentialLR(optimizer, gamma=0.95)
    elif scheduler_type == 'cyclic':
        scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=0.01, step_size_up=int(num_epochs*0.1), mode='triangular2')
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=int(num_epochs*0.2))
        

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
    
    plot_curves_with_val(num_epochs, train_losses, train_accuracies, val_losses, val_accuracies, lrs, exp_name)
    del optimizer
           
    print("Fine-Tuning Complete")        


def fine_tuner_zerograd(model, train_loader, val_loader, freeze_dim0, freeze_dim1, num_epochs, optim_type, scheduler_type, exp_name, device=device, LR=1e-3):

    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    if optim_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Define LR scheduler
    if scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=int(num_epochs*0.25), gamma=0.1)
    elif scheduler_type == 'exponential':
        scheduler = ExponentialLR(optimizer, gamma=0.95)
    elif scheduler_type == 'cyclic':
        scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=0.01, step_size_up=int(num_epochs*0.1), mode='triangular2')
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=int(num_epochs*0.2))
        

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
            # difference 
            zero_out_gradients_v2(model, freeze_dim0, freeze_dim1)
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
    
    plot_curves_with_val(num_epochs, train_losses, train_accuracies, val_losses, val_accuracies, lrs, exp_name)
    del optimizer
           
    print("Fine-Tuning Complete")        


def fine_tuner_BN(model, train_loader, val_loader, num_epochs, num_epochs_BN, optim_type, scheduler_type, exp_name, pruned_indices, unpruned_indices, unpruned_weights, device=device, LR=1e-3, replace_core=False):

    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    if optim_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Define LR scheduler
    if scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=int((num_epochs+num_epochs_BN)*0.25), gamma=0.1)
    elif scheduler_type == 'exponential':
        scheduler = ExponentialLR(optimizer, gamma=0.95)
    elif scheduler_type == 'cyclic':
        scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=0.01, step_size_up=int((num_epochs+num_epochs_BN)*0.1), mode='triangular2')
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=(num_epochs+num_epochs_BN), eta_min=0)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=int((num_epochs+num_epochs_BN)*0.2))
        

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    lrs = []  # To track learning rates

    # Fine-tuning loop
    for epoch in range(num_epochs+num_epochs_BN):

        if epoch == num_epochs:
            if replace_core:
                model = replace_previous_weights(model, pruned_indices, unpruned_indices, unpruned_weights)
            for name, layer in model.named_modules():
                if isinstance(layer, nn.Conv2d):
                    layer.weight.requires_grad = False # Only Freeze Conv2D layers

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
        
        print(f"Epoch [{epoch+1}/{(num_epochs+num_epochs_BN)}], "f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}, Learning Rate: {current_lr:.6f}")
    
    plot_curves_with_val((num_epochs+num_epochs_BN), train_losses, train_accuracies, val_losses, val_accuracies, lrs, exp_name)
           
    print("Fine-Tuning Complete")     


class LayerLoss(nn.Module):
    def __init__(self):
        super(LayerLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, teacher_features, transformed_student_features):
        return self.mse_loss(teacher_features, transformed_student_features)


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


def train_student_with_hints(student, teacher, train_loader, val_loader, teacher_channels, student_channels, lr=0.001, alpha=0.75):
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

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        student_outputs, inter_student = student(inputs)
        with torch.no_grad():
            _, inter_teacher = teacher(inputs)

        # Compute losses
        classification_loss = criterion(student_outputs, labels)
        hint_loss = hint_loss_fn(transformation_layer(inter_student), inter_teacher)

        loss = alpha * classification_loss + (1 - alpha) * hint_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update train metrics
        train_total_loss += loss.item() * inputs.size(0)
        train_total_hint_loss += hint_loss.item() * inputs.size(0)
        _, predicted = torch.max(student_outputs, dim=1)
        train_total_correct += (predicted == labels).sum().item()
        train_total_samples += inputs.size(0)

    train_avg_loss = train_total_loss / train_total_samples
    train_avg_hint_loss = train_total_hint_loss / train_total_samples
    train_accuracy = train_total_correct / train_total_samples

    # Validation looph
    student.eval()

    val_total_loss = 0.0
    val_total_correct = 0
    val_total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs, _ = student(inputs)
            loss = criterion(outputs, labels)

            # Update validation metrics
            val_total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            val_total_correct += (predicted == labels).sum().item()
            val_total_samples += inputs.size(0)

    val_avg_loss = val_total_loss / val_total_samples
    val_accuracy = val_total_correct / val_total_samples

    return train_avg_loss, train_avg_hint_loss, train_accuracy, val_avg_loss, val_accuracy



def train_student_with_distillation(student, teacher, train_loader, val_loader, lr=0.001, alpha=0.75, beta=3):
    teacher = teacher.to(device).eval()
    student = student.to(device).train()

    # Initialize the transformation layer
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(student.parameters(), lr=lr)

    train_total_loss = 0.0
    train_total_correct = 0
    train_total_samples = 0
    train_total_distillation_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        student_outputs = student(inputs)
        with torch.no_grad():
            teacher_outputs = teacher(inputs)

        # Compute losses
        classification_loss = criterion(student_outputs, labels)
        distillation_loss = F.kl_div(F.log_softmax(student_outputs / beta, dim=1),
                                     F.softmax(teacher_outputs / beta, dim=1),
                                     reduction="batchmean") * (beta ** 2)

        loss = alpha * classification_loss + (1 - alpha) * distillation_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update train metrics
        train_total_loss += loss.item() * inputs.size(0)
        train_total_distillation_loss += distillation_loss.item() * inputs.size(0)
        _, predicted = torch.max(student_outputs, dim=1)
        train_total_correct += (predicted == labels).sum().item()
        train_total_samples += inputs.size(0)

    train_avg_loss = train_total_loss / train_total_samples
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

    return train_avg_loss, train_avg_distillation_loss, train_accuracy, val_avg_loss, val_accuracy


def train_student_with_layerwise(
        student, 
        teacher, 
        train_loader, 
        val_loader, 
        teacher_channels,
        student_channels,
        n_epochs,
        lr=0.001, 
        alpha=0.75, 
        beta=3, 
        omega=0.6, 
    ):
    teacher = teacher.to(device).eval()
    student = student.to(device).train()

    # Initialize the transformation layer

    transformation_layers = [
        TransformationLayer(student_ch, teacher_ch).to(device) for student_ch, teacher_ch in zip(student_channels, teacher_channels)
    ]

    all_params = list(student.parameters()) + [
        param for layer in transformation_layers for param in layer.parameters()
    ]

    optimizer = optim.Adam(all_params, lr=lr)
    layer_loss_fn = LayerLoss()
    criterion = nn.CrossEntropyLoss()

    train_total_loss = 0.0
    train_total_correct = 0
    train_total_samples = 0
    train_total_layerwise_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        student_outputs, inter_student1, inter_student2, inter_student3 = student(inputs)
        with torch.no_grad():
            teacher_outputs, inter_teacher1, inter_teacher2, inter_teacher3 = teacher(inputs)

        classification_loss = criterion(student_outputs, labels)
        distillation_loss = F.kl_div(
            F.log_softmax(student_outputs / beta, dim=1),
            F.softmax(teacher_outputs / beta, dim=1),
            reduction="batchmean",
        ) * (beta**2)
        layer_loss = (
            layer_loss_fn(transformation_layers[0](inter_student1), inter_teacher1)
            + layer_loss_fn(transformation_layers[1](inter_student2), inter_teacher2)
            + layer_loss_fn(transformation_layers[2](inter_student3), inter_teacher3)
        )

        # Experimentation will need to be done to find the optimal alpha and omega
        loss = alpha * classification_loss + (1 - alpha) * (
            omega * distillation_loss + (1 - omega) * layer_loss
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update train metrics
        train_total_loss += loss.item() * inputs.size(0)
        train_total_layerwise_loss += layer_loss.item() * inputs.size(0)
        _, predicted = torch.max(student_outputs, dim=1)
        train_total_correct += (predicted == labels).sum().item()
        train_total_samples += inputs.size(0)

    train_avg_loss = train_total_loss / train_total_samples
    train_avg_layer_loss = train_total_layerwise_loss / train_total_samples
    train_accuracy = train_total_correct / train_total_samples

    # Validation looph
    student.eval()

    val_total_loss = 0.0
    val_total_correct = 0
    val_total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs, _, _, _ = student(inputs)
            loss = criterion(outputs, labels)

            # Update validation metrics
            val_total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            val_total_correct += (predicted == labels).sum().item()
            val_total_samples += inputs.size(0)

    val_avg_loss = val_total_loss / val_total_samples
    val_accuracy = val_total_correct / val_total_samples

    return train_avg_loss, train_avg_layer_loss, train_accuracy, val_avg_loss, val_accuracy


def KD_trainer(student, teacher, train_loader, val_loader, teacher_channels, student_channels, lr, alpha, beta, omega, num_epochs=20, exp_name="KD", KD_variant="hints", device=device):
    train_losses = []
    if KD_variant == "hints":
        train_hint_losses = []
    elif KD_variant == "layerwise":
        train_layerwise_losses = []
    else:
        train_distillation_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        if KD_variant == "hints":
            train_avg_loss, train_avg_hint_loss, train_accuracy, val_avg_loss, val_accuracy = train_student_with_hints(
                student, teacher, train_loader, val_loader, teacher_channels, student_channels, lr, alpha)
        elif KD_variant == "layerwise":
            train_avg_loss, train_avg_layerwise_loss, train_accuracy, val_avg_loss, val_accuracy = train_student_with_layerwise(
                student, teacher, train_loader, val_loader, teacher_channels, student_channels, num_epochs, lr, alpha, beta, omega)
        else:
            train_avg_loss, train_avg_distillation_loss, train_accuracy, val_avg_loss, val_accuracy = train_student_with_distillation(
                student, teacher, train_loader, val_loader, lr, alpha, beta)

        train_losses.append(train_avg_loss)
        if KD_variant == "hints":
            train_hint_losses.append(train_avg_hint_loss)
        elif KD_variant == "layerwise":
            train_layerwise_losses.append(train_avg_layerwise_loss)
        else:
            train_distillation_losses.append(train_avg_distillation_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_avg_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}]: Train Loss: {train_avg_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_avg_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    if KD_variant == "hints":
        plot_KD_metrics(train_losses, train_hint_losses, train_accuracies, val_losses, val_accuracies, num_epochs, exp_name, KD_variant)
    elif KD_variant == "layerwise":
        plot_KD_metrics(train_losses, train_layerwise_losses, train_accuracies, val_losses, val_accuracies, num_epochs, exp_name, KD_variant)
    else:
        plot_KD_metrics(train_losses, train_distillation_losses, train_accuracies, val_losses, val_accuracies, num_epochs, exp_name, KD_variant)
    print("Knowledge Distillation Training Complete")

