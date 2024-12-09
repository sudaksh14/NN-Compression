import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from model_utils import *
from data_augmentation import ImgAugTransform

# Define the Basic Layer
class BasicLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicLayer, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm2d(out_channels * 3)
        self.activation = nn.ReLU()

    def forward(self, x):
        out3 = self.conv3x3(x)
        out5 = self.conv5x5(x)
        out7 = self.conv7x7(x)
        out = torch.cat([out3, out5, out7], dim=1)
        out = self.bn(out)
        out = self.activation(out)
        return out

# Define the Custom Model
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.block1 = BasicLayer(in_channels, out_channels[0])
        self.block2 = BasicLayer(out_channels[0] * 3, out_channels[1])
        self.block3 = BasicLayer(out_channels[1] * 3, out_channels[2])
        self.block4 = BasicLayer(out_channels[2] * 3, out_channels[3])

        self.residual1 = nn.Conv2d(in_channels, out_channels[1] * 3, kernel_size=1)
        self.residual2 = nn.Conv2d(out_channels[1] * 3, out_channels[3] * 3, kernel_size=1)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        residual = self.residual1(x)
        out = self.block1(x)
        out = self.block2(out)
        out = out + residual

        residual = self.residual2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = out + residual

        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        return out

# Define the Model with Multiple Custom Blocks
class ParallelBlockModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, num_classes):
        super(ParallelBlockModel, self).__init__()
        # Create multiple custom models based on user input
        self.blocks = nn.ModuleList([
            BasicBlock(in_channels, out_channels) for _ in range(num_blocks)
        ])

        # Final classifier
        self.fc = nn.Linear(out_channels[-1] * 3 * num_blocks, num_classes)

    def forward(self, x):
        # Process input through all submodels
        outputs = [block(x) for block in self.blocks]

        # Concatenate all outputs
        concatenated = torch.cat(outputs, dim=1)

        # Final classification
        out = self.fc(concatenated)
        return out
    
    
    


if __name__ == "__main__":

    num_blocks = 3  # Number of Custom/Parallel blocks
    in_channels = 3  # Input image channels (e.g., RGB)
    out_channels = [16,32,64,128]  # Number of output channels in each custom block
    num_classes = 10  # Number of classes (e.g., for classification)

    # Instantiate the multi-model
    model = ParallelBlockModel(in_channels, out_channels, num_blocks, num_classes)
    
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
    
    # Load CIFAR-10 datasets

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
    
    fine_tuner(model, train_loader, val_loader, num_epochs=200, scheduler_type="cosine", prune_ratio=0.5, exp_name="PBM_CIFAR-10_FROM_SCRATCH", LR=0.1)
    torch.save(model.state_dict(), "./saves/state_dicts/PBM_cifar10.pth")
    torch.onnx.export(model, (torch.randn(1, in_channels, 32, 32).to(device),), f=f"./saves/onnx/PBM_cifar10.onnx")
