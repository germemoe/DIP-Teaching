import torch.nn as nn
import torch
class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        self.dconv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        ### FILL: add more CONV Layers
        self.dconv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dconv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dconv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.down = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.5)
        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function
        
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        self.uconv1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.uconv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.uconv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.uconv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1, stride=1),  
            nn.BatchNorm2d(3),
            nn.Tanh()
        )


    def forward(self, x):
        # Encoder forward pass
        x1 = self.dconv1(x)
        x2 = self.down(x1)
        x2 = self.dconv2(x2)
        x3 = self.down(x2)
        x3 = self.dconv3(x3)
        x3 = self.drop(x3)
        x4 = self.down(x3)
        x4 = self.dconv4(x4)
        x4 = self.drop(x4)
        x5 = self.down(x4)
        x5 = self.conv5(x5)
        ### FILL: encoder-decoder forward pass
        x6 = self.up1(x5)
        x6 = torch.cat([x4,x6],dim=1)
        x6 = self.uconv1(x6)
        x6 = self.up2(x6)
        x6 = torch.cat([x3,x6],dim=1)
        x6 = self.uconv2(x6)
        x6 = self.up3(x6)
        x6 = torch.cat([x2,x6],dim=1)
        x6 = self.uconv3(x6)
        x6 = self.up4(x6)
        x6 = torch.cat([x1,x6],dim=1)
        x6 = self.uconv4(x6)
        
        return x6