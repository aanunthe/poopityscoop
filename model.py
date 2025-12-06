import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class UNetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.block(x)

class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3, feature_channels=[64, 128, 256, 512], latent_dim=512):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        prev_channels = in_channels
        for out_channels in feature_channels:
            self.blocks.append(UNetEncoderBlock(prev_channels, out_channels))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = out_channels
        
        # Keep spatial information instead of global pooling
        self.bottleneck = nn.Sequential(
            nn.Conv2d(prev_channels, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skip_connections = []
        
        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            skip_connections.append(x)  # Save for skip connections
            x = pool(x)
        
        x = self.bottleneck(x)
        return x, skip_connections

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Compute spatial attention map
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention

class UnetDecoder(nn.Module):
    def __init__(self, hdim=768, out_channels=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hdim, 512 * 9 * 9),
            nn.ReLU()
        )

        # Decoder with attention
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.att1 = SpatialAttention(256)
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.att2 = SpatialAttention(128)
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.att3 = SpatialAttention(64)
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        out = self.proj(x)
        out = out.view(-1, 512, 9, 9)
        
        # Apply decoder with attention at each level
        out = self.dec1(out)
        out = self.att1(out)
        
        out = self.dec2(out)
        out = self.att2(out)
        
        out = self.dec3(out)
        out = self.att3(out)
        
        out = self.dec4(out)
        out = self.final(out)
        
        return out


class UNetSkipDecoder(nn.Module):
    def __init__(self, latent_dim=512, out_channels=2):
        super().__init__()
        # Decoder blocks with skip connections
        self.up1 = nn.ConvTranspose2d(latent_dim, 256, kernel_size=2, stride=2)
        self.dec1 = UNetEncoderBlock(256 + 512, 256)  # 256 + skip connection
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = UNetEncoderBlock(128 + 256, 128)  # 128 + skip connection
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = UNetEncoderBlock(64 + 128, 64)   # 64 + skip connection
        
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec4 = UNetEncoderBlock(32 + 64, 32)    # 32 + skip connection
        
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        
        # Spatial attention for each level
        self.att1 = SpatialAttention(256)
        self.att2 = SpatialAttention(128)
        self.att3 = SpatialAttention(64)
        self.att4 = SpatialAttention(32)
    
    def forward(self, x, skip_connections):
        # Reverse skip connections (from deepest to shallowest)
        skips = skip_connections[::-1]
        
        # Decoder with skip connections
        x = self.up1(x)
        x = torch.cat([x, skips[0]], dim=1)  # Concatenate skip connection
        x = self.dec1(x)
        x = self.att1(x)
        
        x = self.up2(x)
        x = torch.cat([x, skips[1]], dim=1)
        x = self.dec2(x)
        x = self.att2(x)
        
        x = self.up3(x)
        x = torch.cat([x, skips[2]], dim=1)
        x = self.dec3(x)
        x = self.att3(x)
        
        x = self.up4(x)
        x = torch.cat([x, skips[3]], dim=1)
        x = self.dec4(x)
        x = self.att4(x)
        
        x = self.final(x)
        return torch.tanh(x)

class EncDec(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.enc = UNetEncoder(3, latent_dim=latent_dim)
        self.dec = UNetSkipDecoder(latent_dim, 2)

    def forward(self, x):
        encoded, skip_connections = self.enc(x)
        return self.dec(encoded, skip_connections)

# 🔍 Example usage
if __name__ == "__main__":
    model = UNetEncoder(in_channels=3, latent_dim=1024)
    x = torch.randn(1, 3, 288, 288)
    out = model(x)
    print(f"Output shape: {out.shape}")  # Expected: (1, 512)
