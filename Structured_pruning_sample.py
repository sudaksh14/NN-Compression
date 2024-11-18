import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load a pre-trained AlexNet model
alexnet = models.alexnet(pretrained=True)

# Define the pruning function for a specific layer
def prune_layer_l2_norm(model, layer_name, amount=0.3):
    layer = dict(model.named_modules())[layer_name]
    if isinstance(layer, nn.Conv2d):
        # Prune channels with lowest L2 norm
        prune.ln_structured(layer, name="weight", amount=amount, n=2, dim=0)  # Prune out_channels
        print(f"Pruned layer {layer_name} by {amount * 100}%")
    else:
        print(f"{layer_name} is not a Conv2D layer.")

# Apply pruning to specific convolutional layers in AlexNet
prune_layer_l2_norm(alexnet, 'features.0', amount=0.3)  # Prune first conv layer
prune_layer_l2_norm(alexnet, 'features.3', amount=0.3)  # Prune second conv layer

def get_remaining_channels(layer):
    # Count non-zeroed channels (out_channels) after pruning
    return int((layer.weight != 0).sum(dim=(1, 2, 3)).bool().sum().item())

# Get remaining channels in each pruned layer
remaining_channels_0 = get_remaining_channels(alexnet.features[0])
remaining_channels_3 = get_remaining_channels(alexnet.features[3])

# Define a new AlexNet with pruned layers' new channel counts
class PrunedAlexNet(nn.Module):
    def __init__(self):
        super(PrunedAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, remaining_channels_0, kernel_size=11, stride=4, padding=2),  # Adjusted layer 0
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(remaining_channels_0, remaining_channels_3, kernel_size=5, padding=2),  # Adjusted layer 3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Remaining layers follow the original structure
            nn.Conv2d(remaining_channels_3, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = alexnet.classifier  # Reuse original classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Initialize the new pruned model
pruned_alexnet = PrunedAlexNet()

def transfer_weights(original_model, pruned_model):
    # Copy weights from original pruned model to new pruned model
    pruned_model.features[0].weight.data = original_model.features[0].weight[original_model.features[0].weight != 0].view_as(pruned_model.features[0].weight)
    pruned_model.features[3].weight.data = original_model.features[3].weight[original_model.features[3].weight != 0].view_as(pruned_model.features[3].weight)

# Apply weight transfer
transfer_weights(alexnet, pruned_alexnet)

