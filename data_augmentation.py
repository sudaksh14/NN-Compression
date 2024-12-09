import torch
import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt



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