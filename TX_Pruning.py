import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torchvision.models import vit_b_16
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

vit_model = vit_b_16(pretrained=True)

total_params = sum(p.numel() for p in vit_model.parameters())
print(f"Number of parameters: {total_params}")    

# onnx_program = torch.onnx.export(vit_model, (torch.randn(1, 3, 224, 224),), f="original_vit.onnx")

print(vit_model.state_dict().keys())

exit()

# Structured Pruning Function using L2-Norm
def str_pruning(model, layer_name, prune_ratio=0.5):
    
    layer = dict(model.named_modules())[layer_name]
    if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d):
        # Prune channels with the smallest L2 norm using structured pruning
        prune.ln_structured(layer, name='weight', amount=prune_ratio, n=2, dim=0)  # Prunes entire rows (out_channels)
        print(f"Pruned layer {layer_name} with amount {prune_ratio}")
    else:
        print(f"Layer {layer_name} is not a Linear or Conv2D layer and cannot be pruned.")
        

# Pruning

str_pruning(vit_model, 'encoder.layers.0.attn.proj')

# for name, layer in vit_model.named_modules():
#     print(name)
#     str_pruning(vit_model, name)
    

class PrunedViT(nn.Module):
    def __init__(self, pruned_model, pruned_count):
        super(PrunedViT, self).__init__()
        
        # Create a new model structure
        # Copy layers from the original model, excluding pruned channels
        
        # For simplicity, let's assume we are copying just the modified layers
        # Here, add logic to skip zeroed-out channels in `attn.proj`
        
        # Example copying only the first encoder layer and adjusting for reduced channels
        self.encoder = nn.Sequential(
            *[
                nn.Linear(768, 768 - pruned_count)  # Hypothetically adjusting channels
                if i == 0 else layer
                for i, layer in enumerate(pruned_model.encoder.layers)
            ]
        )
        # Other layers can be copied directly or pruned similarly
        # Example: self.head = pruned_model.head
    
    def forward(self, x):
        # Define forward pass for the pruned architecture
        x = self.encoder(x)
        # Other model passes would go here
        return x

# Step 3: Initialize new model with fewer channels and transfer remaining weights
# You may need to carefully map layers based on the pruning pattern
pruned_model = PrunedViT(vit_model, 1)


total_params = sum(p.numel() for p in vit_model.parameters())
print(f"Number of parameters: {total_params}")   





