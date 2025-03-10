import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, height * width)
        v = self.value(x).view(batch_size, -1, height * width)
        
        attention = torch.bmm(q, k)
        attention = torch.softmax(attention, dim=-1)
        
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        return self.gamma * out + x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class EnhancedDenoisingAutoencoder(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(EnhancedDenoisingAutoencoder, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2)
        
        # Residual blocks
        self.res1 = ResidualBlock(32)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(128)
        
        # Attention modules
        self.attn1 = SelfAttention(32)
        self.attn2 = SelfAttention(64)
        self.attn3 = SelfAttention(128)
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid()  # Ensure output is in [0,1]
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e1 = self.res1(e1)
        e1 = self.attn1(e1)
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)
        e2 = self.res2(e2)
        e2 = self.attn2(e2)
        p2 = self.pool(e2)
        
        e3 = self.enc3(p2)
        e3 = self.res3(e3)
        e3 = self.attn3(e3)
        p3 = self.pool(e3)
        
        # Decoder with skip connections
        d1 = self.dec1(p3)
        d1 = torch.add(d1, e2)  # Skip connection
        
        d2 = self.dec2(d1)
        d2 = torch.add(d2, e1)  # Skip connection
        
        d3 = self.dec3(d2)
        
        return d3
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class LightResidualBlock(nn.Module):
    def __init__(self, channels):
        super(LightResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class LightDenoisingAutoencoder(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(LightDenoisingAutoencoder, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2)
        
        # Residual blocks
        self.res1 = LightResidualBlock(16)
        self.res2 = LightResidualBlock(32)
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid()  # Ensure output is in [0,1]
        )
        
    def forward(self, x):
        # Encoder with intermediate outputs for skip connections
        e1 = self.enc1(x)
        e1 = self.res1(e1)
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)
        e2 = self.res2(e2)
        p2 = self.pool(e2)
        
        # Decoder with skip connections
        d1 = self.dec1(p2)
        d1 = torch.add(d1, p1)  # Skip connection with pooled features
        
        d2 = self.dec2(d1)
        
        return d2
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 