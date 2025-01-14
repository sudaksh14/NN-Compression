import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0])
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc = nn.Linear(64, 10)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out,(1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResNet_prune(nn.Module):
    def __init__(self, block, num_blocks, prune_ratio, first_channel):
        super(ResNet_prune, self).__init__()
        
        
        self.in_channels = round((first_channel)*(1-prune_ratio))
        self.conv1 = nn.Conv2d(3, round((first_channel)*(1-prune_ratio)), kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(round((first_channel)*(1-prune_ratio)))
        self.layer1 = self._make_layer(block, round((first_channel)*(1-prune_ratio)), num_blocks[0])
        self.layer2 = self._make_layer(block, round((2*first_channel)*(1-prune_ratio)), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, round((4*first_channel)*(1-prune_ratio)), num_blocks[2], stride=2)
        self.fc = nn.Linear(round((4*first_channel)*(1-prune_ratio)), 10)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out,(1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

"""
channel_dict = {'conv1': (13, 3), 'layer1.0.conv1': (13, 13), 'layer1.0.conv2': (13, 13), 'layer1.1.conv1': (13, 13), 'layer1.1.conv2': (13, 13), 'layer1.2.conv1': (13, 13), 
 'layer1.2.conv2': (13, 13), 'layer2.0.conv1': (26, 13), 'layer2.0.conv2': (26, 26), 'layer2.0.downsample.0': (26, 13), 'layer2.1.conv1': (26, 26), 
 'layer2.1.conv2': (26, 26), 'layer2.2.conv1': (26, 26), 'layer2.2.conv2': (26, 26), 'layer3.0.conv1': (52, 26), 'layer3.0.conv2': (52, 52), 
 'layer3.0.downsample.0': (52, 26), 'layer3.1.conv1': (52, 52), 'layer3.1.conv2': (52, 52), 'layer3.2.conv1': (52, 52), 'layer3.2.conv2': (52, 52)}
 """

class ResNet_general(nn.Module):
    def __init__(self, block, num_blocks, channel_dict):
        super(ResNet_general, self).__init__()
        
        self.in_channels = channel_dict['conv1'][0]
        self.conv1 = nn.Conv2d(channel_dict['conv1'][1], channel_dict['conv1'][0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_dict['conv1'][0])
        self.layer1 = self._make_layer(block, channel_dict['layer1.0.conv1'][0], num_blocks[0])
        self.layer2 = self._make_layer(block, channel_dict['layer2.0.conv1'][0], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, channel_dict['layer3.0.conv1'][0], num_blocks[2], stride=2)
        self.fc = nn.Linear(channel_dict['layer3.2.conv2'][0], 10)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out,(1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def Resnet_General(channel_dict):
    return ResNet_general(BasicBlock, [3,3,3], channel_dict)



def rebuild_resnet(model, p, first_channel=16):
    new_model = ResNet_prune(BasicBlock, [3,3,3], prune_ratio=p, first_channel=first_channel)
    
    # Create a list to hold the layers and corresponding weights
    layer_weights = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Get the mask used for pruning
            if hasattr(module, 'weight_mask'):
                mask = module.weight_mask.bool()  # The mask indicates the remaining channels
                remaining_weights = module.weight.data[mask].clone()
                out_channels, in_channels, kernel_height, kernel_width = module.weight.shape
                remaining_weights = remaining_weights.view(-1, in_channels, kernel_height, kernel_width)
                
                # print(name, module.weight.data.shape, mask.shape, remaining_weights.shape)
                
                # If it's the first layer, directly assign to new model's first conv layer
                if name == 'conv1':
                    new_model.conv1.weight.data = remaining_weights
                else:
                    # Update other layers based on their names
                    layer_weights.append((name, remaining_weights))

    # Now we have the remaining weights for each layer, let's set them
    for name, weights in layer_weights:
        layer_name = name.replace('.', '_')  # Change '.' to '_' for attribute access
        new_model_state_dict = new_model.state_dict()
        new_model_state_dict[layer_name + '.weight'] = weights
        new_model.load_state_dict(new_model_state_dict, strict=False)

    return new_model

    