import torch
import torch.nn as nn
import math

def Normalize(in_channels):
    """
    Adjusted normalization function to handle varying channel counts
    """
    if in_channels >= 32:
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    else:
        # For smaller channel counts, use a smaller number of groups
        num_groups = max(1, in_channels // 4)  # Ensure at least 1 group
        return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

#nonlinearity(x): x * torch.sigmoid(x)
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super(ConvBlock, self).__init__()
        
        if padding == 'same':
            padding = kernel_size // 2
            
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = Normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm2 = Normalize(out_channels)
   
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = x*torch.sigmoid(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x*torch.sigmoid(x)

class BranchBlock(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, stride=1):
        super(BranchBlock, self).__init__()
        
        # Ensure out_channels is divisible by 32 if it's large enough
        if out_channels >= 32:
            out_channels = ((out_channels + 31) // 32) * 32
            
        self.block1 = ConvBlock(in_channels, out_channels, kernel_size, stride)
        self.block2 = ConvBlock(out_channels, out_channels, kernel_size, 1)
        
    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        return out2 + out1

class MultiStageEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_branches=3, stride=1):
        super(MultiStageEncoder, self).__init__()
        
        # Calculate channels per branch, ensuring divisibility by 32 if possible
        min_channels_per_branch = 32 if out_channels >= 96 else max(1, out_channels // num_branches)
        channels_per_branch = ((min_channels_per_branch + 31) // 32) * 32 if min_channels_per_branch >= 32 else min_channels_per_branch
        
        # Create branches
        self.branches = nn.ModuleList([
            BranchBlock(
                kernel_size=3+2*i,
                in_channels=in_channels,
                out_channels=channels_per_branch,
                stride=stride
            ) for i in range(num_branches)
        ])
        
        total_branch_channels = channels_per_branch * num_branches
        
        # Add projection if needed to match desired output channels
        self.proj = None

        if total_branch_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(total_branch_channels, out_channels, 1, 1, 0),
                Normalize(out_channels),
                Swish()
            )
            
    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outputs, dim=1)
        
        if self.proj is not None:
            out = self.proj(out)
        return out
