from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class Reshape(nn.Module):
    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.reshape(-1, *self.shape)


class NeutralNet(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=3, padding=1, batch_norm=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=kernel_size, padding=padding)
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(hidden_channels)
            self.bn2 = nn.BatchNorm2d(in_channels)
            self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, input: Tensor) -> Tensor:
        x = self.conv1(input)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.relu(input + x)
        x = self.conv_out(x)
        if self.batch_norm:
            x = self.bn_out(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, batch_norm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=kernel_size, 
                              stride=2, padding=1)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.batch_norm = batch_norm

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.relu(x)
        return x
    
    
class ResidualNet(nn.Module):
    def __init__(self, out_channels=200, residual=True,
                 batch_norm=True, pool='avg', linear_output=4096,
                 pool_stride=2, pool_kernel_size=2, conv_kernel_size=2):
        assert pool in ('avg', 'max')
        assert pool_kernel_size in (3, 2)
        pools_class = {'avg': nn.AvgPool2d, 'max': nn.MaxPool2d}
        
        
        super().__init__()
        self.relu = nn.ReLU()
        self.pool1 = pools_class[pool](kernel_size=pool_kernel_size, stride=pool_stride)
        self.pool2 = pools_class[pool](kernel_size=pool_kernel_size, stride=pool_stride)
        if batch_norm:
            self.norm = nn.BatchNorm2d(32)
        self.batch_norm = batch_norm
        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)

        if residual:
            self.res1 = ResidualBlock(32, 128, 64, batch_norm=batch_norm)
            self.res2 = ResidualBlock(64, 256, 128, batch_norm=batch_norm)
        self.residual = residual

        self.layer1 = ConvBlock(32, 64, kernel_size=conv_kernel_size, batch_norm=batch_norm)
        self.layer2 = ConvBlock(128 if residual else 64, 128, kernel_size=conv_kernel_size, batch_norm=batch_norm)

        self.linear_output = linear_output
        self.fc = nn.Linear(self.linear_output, out_channels)
        self.reshape = Reshape(self.linear_output)
        # out_volume = [(in_volume−kernel_size+2*padding)/stride]+1
  
    def forward(self, x: Tensor) -> Tensor: # [B, 3, 64, 64]
        x = self.conv(x) 
        if self.batch_norm:
            x = self.norm(x)
        x = self.relu(x)
        x = self.pool1(x)

        # residual blocks
        if self.residual:
            x = self.res1(x)
            x = self.res2(x)
        else:
            x = self.layer1(x)
        
        # downsapling blocks
        x = self.layer2(x)
        x = self.pool2(x)

        x = self.reshape(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)