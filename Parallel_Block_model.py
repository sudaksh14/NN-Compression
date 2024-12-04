import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Basic Layer (same as before)
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

# Define the Custom Model (same as before)
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.block1 = BasicLayer(in_channels, out_channels)
        self.block2 = BasicLayer(out_channels * 3, out_channels)
        self.block3 = BasicLayer(out_channels * 3, out_channels)
        self.block4 = BasicLayer(out_channels * 3, out_channels)

        self.residual1 = nn.Conv2d(in_channels, out_channels * 3, kernel_size=1)
        self.residual2 = nn.Conv2d(out_channels * 3, out_channels * 3, kernel_size=1)

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

# Define the Model with Multiple Custom Submodels
class ParallelBlockModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_models, num_classes):
        super(ParallelBlockModel, self).__init__()
        # Create multiple custom models based on user input
        self.models = nn.ModuleList([
            BasicBlock(in_channels, out_channels) for _ in range(num_models)
        ])

        # Final classifier
        self.fc = nn.Linear(out_channels * 3 * num_models, num_classes)

    def forward(self, x):
        # Process input through all submodels
        outputs = [model(x) for model in self.models]

        # Concatenate all outputs
        concatenated = torch.cat(outputs, dim=1)

        # Final classification
        out = self.fc(concatenated)
        return out

# Instantiate and Test the Model
if __name__ == "__main__":
    # User-defined parameters
    num_models = 3  # Number of custom models
    in_channels = 3  # Input image channels (e.g., RGB)
    out_channels = 16  # Number of output channels in each custom model
    num_classes = 10  # Number of classes (e.g., for classification)

    # Instantiate the multi-model
    model = ParallelBlockModel(in_channels, out_channels, num_models, num_classes)
    print(model)

    # Test with a random input tensor
    x = torch.randn(1, in_channels, 224, 224)  # Batch size=1, 3 channels, 224x224 image
    y = model(x)
    print(f"Output shape: {y.shape}")  # Should output shape: [1, num_classes]
