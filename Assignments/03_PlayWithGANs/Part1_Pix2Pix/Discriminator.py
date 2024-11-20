import torch.nn as nn
import torch
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(128, 256, 3)
        self.bn = nn.BatchNorm2d(256)
        self.last = nn.Conv2d(256, 1, 3)
 
    def forward(self, anno, img):
        x = torch.cat([anno, img], dim=1)  # (batch, 6, H, W)  PS:这里需要将anno和img进行合并
        x = self.down1(x)
        x = self.down2(x)
        x = nn.functional.dropout2d(self.bn(nn.functional.leaky_relu_(self.conv1(x))))
        x = torch.sigmoid(self.last(x))  # (batch, 1, 60, 60)
        return x
