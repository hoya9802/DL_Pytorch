import torch
import torch.nn as nn
import numpy as np

class FullyConnected(torch.nn.Module):
    def __init__(self, input_size, output_size, activation_fn='linear'):
        super(FullyConnected, self).__init__()

        self.act_fn = activation_fn
        self.relu = torch.nn.ReLU()
        self.lrelu = torch.nn.LeakyReLU()

        self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        if self.act_fn == 'relu':
            x = self.relu(x)
        elif self.act_fn == 'lrelu':
            x = self.lrelu(x)
        elif self.act_fn == 'sigmoid':
            x = torch.sigmoid(x)

        return x
    

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, bdown=False):
        super(ResBlock, self).__init__()
        stride = 2 if bdown else 1
        
        self.c_in = c_in
        self.c_out = c_out

        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=(3, 3), padding=1, stride=(stride, stride))
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(c_out)

        if c_in != c_out:
            self.identity = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(stride, stride))

    def forward(self, x):
        y = x      # [8, 512, 64, 64]
        x = self.conv1(x) # [8, 512, 32, 32]
        x = self.bn1(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.c_in != self.c_out:
            y = self.identity(y) # [8, 512, 32, 32]

        x = x + y

        x = self.ReLU(x)
        return x
    
class ResBlock_Sq(nn.Module):
    def __init__(self, c_in, c_out, bdown=False):
        super(ResBlock_Sq, self).__init__()
        stride = 2 if bdown else 1
        
        self.c_in = c_in
        self.c_out = c_out

        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=(3, 3), padding=1, stride=(stride, stride))
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(c_out)

        if c_in != c_out:
            self.identity = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(stride, stride))

        self.sq = nn.Sequential(
            FullyConnected(c_out, c_out // 16, activation_fn='relu'),
            FullyConnected(c_out // 16, c_out, activation_fn='sigmoid')
            
        )

    def forward(self, x):
        y = x      # [8, 512, 64, 64]
        x = self.conv1(x) # [8, 512, 32, 32]
        x = self.bn1(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # Sq Attention
        x_sq = torch.mean(x, dim=(2, 3)) # [B, C, H, W] -> [B, C]
        x_sq = self.sq(x_sq) # [B, C]
        x_sq = x_sq[:, :, np.newaxis, np.newaxis] # [B, C, 1, 1]
        x = x * x_sq

        if self.c_in != self.c_out:
            y = self.identity(y) # [8, 512, 32, 32]

        x = x + y

        x = self.ReLU(x)
        return x

class ResBlock_BAM(nn.Module):
    def __init__(self, c_in, c_out, bdown=False, r = 16, d = 4):
        super(ResBlock_BAM, self).__init__()
        stride = 2 if bdown else 1
        
        self.c_in = c_in
        self.c_out = c_out

        self.bn1 = nn.BatchNorm2d(c_in)
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=(3, 3), padding=1, stride=(stride, stride))

        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=(3, 3), padding=1)

        if c_in != c_out:
            self.identity = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(stride, stride))

        self.ch_at = nn.Sequential(
            FullyConnected(c_out, c_out // r, activation_fn='relu'),
            FullyConnected(c_out // r, c_out, activation_fn='linear'),
        )

        self.bn_chat = nn.BatchNorm2d(c_out)

        self.sp_at = nn.Sequential(
            nn.Conv2d(c_out, c_out // r, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(c_out // r, c_out // r, kernel_size=(3, 3), padding=4 ,dilation=4),
            nn.ReLU(),
            nn.Conv2d(c_out // r, c_out // r, kernel_size=(3, 3), padding=4 ,dilation=4),
            nn.ReLU(),
            nn.Conv2d(c_out // r, 1, kernel_size=(1, 1)),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        y = x      # [8, 512, 64, 64]
        x = self.bn1(x)
        x = self.ReLU(x)
        x = self.conv1(x) 

        x = self.bn2(x)
        x = self.ReLU(x)
        x = self.conv2(x) 

        if self.c_in != self.c_out:
            y = self.identity(y) # [8, 512, 32, 32]

        x = x + y

        # --- BAM
        x_ch = torch.mean(x, dim=(2, 3)) # [B, C]
        x_ch = self.ch_at(x_ch) 
        x_ch = x_ch[:, :, np.newaxis, np.newaxis]
        x_ch = self.bn_chat(x_ch)
        x_sp = self.sp_at(x)
        x_at = torch.sigmoid(x_ch + x_sp)

        x = x * x_at + x

        return x


class ResNet18_sq(nn.Module):
    def __init__(self, num_class):
        super(ResNet18_sq, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=(7, 7), padding=3, stride=(2, 2))
        self.bn = nn.BatchNorm2d(64)
        self.ReLU = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.blocks = nn.Sequential(
            ResBlock_Sq(64, 64),
            ResBlock_Sq(64, 64),

            ResBlock_Sq(64, 128, bdown=True),
            ResBlock_Sq(128, 128),

            ResBlock_Sq(128, 256, bdown=True),
            ResBlock_Sq(256, 256),

            ResBlock_Sq(256, 512, bdown=True),
            ResBlock_Sq(512, 512),
        )

        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.maxp(x)

        x = self.blocks(x)
        x = torch.mean(x, dim=(2, 3)) # [B, C, H, W] = [0, 1, 2, 3]
        x = self.fc(x)

        return x
    
class ResNet18_BAM(nn.Module):
    def __init__(self, num_class):
        super(ResNet18_BAM, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=(7, 7), padding=3, stride=(2, 2))
        # self.bn = nn.BatchNorm2d(64)
        # self.ReLU = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.blocks = nn.Sequential(
            ResBlock_BAM(64, 64),
            ResBlock_BAM(64, 64),

            ResBlock_BAM(64, 128, bdown=True),
            ResBlock_BAM(128, 128),

            ResBlock_BAM(128, 256, bdown=True),
            ResBlock_BAM(256, 256),

            ResBlock_BAM(256, 512, bdown=True),
            ResBlock_BAM(512, 512),
        )

        self.fc = nn.Linear(512, num_class)
        self.bn_last = torch.nn.BatchNorm2d(512)
        self.relu_last = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.maxp(x)

        x = self.blocks(x)
        x = self.bn_last(x)
        x = self.relu_last(x)

        x = torch.mean(x, dim=(2, 3)) # [B, C, H, W] = [0, 1, 2, 3]
        x = self.fc(x)

        return x

class ResNet34(nn.Module):
    def __init__(self, num_class):
        super(ResNet34, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=(7, 7), padding=3, stride=(2, 2))
        self.bn = nn.BatchNorm2d(64)
        self.ReLU = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.blocks = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),

            ResBlock(64, 128, bdown=True),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),

            ResBlock(128, 256, bdown=True),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),

            ResBlock(256, 512, bdown=True),
            ResBlock(512, 512),
            ResBlock(512, 512),
        )

        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.maxp(x)

        x = self.blocks(x)
        x = torch.mean(x, dim=(2, 3)) # [B, C, H, W] = [0, 1, 2, 3]
        x = self.fc(x)

        return x

class ResBlock_bottle(nn.Module):
    def __init__(self, c_in, c_out, bdown=False):
        super(ResBlock_bottle, self).__init__()
        stride = 2 if bdown else 1
        
        self.c_in = c_in
        self.c_out = c_out

        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(c_in, c_out // 4, kernel_size=(1, 1), stride=(stride, stride))
        self.bn1 = nn.BatchNorm2d(c_out // 4)
        self.conv2 = nn.Conv2d(c_out // 4, c_out // 4, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(c_out // 4)
        self.conv3 = nn.Conv2d(c_out // 4, c_out, kernel_size=(1, 1))
        self.bn3 = nn.BatchNorm2d(c_out)


        if c_in != c_out:
            self.identity = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(stride, stride))

    def forward(self, x):
        y = x      # [8, 512, 64, 64]
        x = self.conv1(x) # [8, 512, 32, 32]
        x = self.bn1(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.ReLU(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.c_in != self.c_out:
            y = self.identity(y) # [8, 512, 32, 32]

        x = x + y

        x = self.ReLU(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, num_class):
        super(ResNet50, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=(7, 7), padding=3, stride=(2, 2))
        self.bn = nn.BatchNorm2d(64)
        self.ReLU = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.blocks = nn.Sequential(
            ResBlock_bottle(64, 256),
            ResBlock_bottle(256, 256),
            ResBlock_bottle(256, 256),
            
            ResBlock_bottle(256, 512, bdown=True),
            ResBlock_bottle(512, 512),
            ResBlock_bottle(512, 512),
            ResBlock_bottle(512, 512),

            ResBlock_bottle(512, 1024, bdown=True),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            
            ResBlock_bottle(1024, 2048, bdown=True),
            ResBlock_bottle(2048, 2048),
            ResBlock_bottle(2048, 2048),
        )

        self.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.maxp(x)

        x = self.blocks(x)
        x = torch.mean(x, dim=(2, 3)) # [B, C, H, W] = [0, 1, 2, 3]
        x = self.fc(x)

        return x
    
class ResNet101(nn.Module):
    def __init__(self, num_class):
        super(ResNet101, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=(7, 7), padding=3, stride=(2, 2))
        self.bn = nn.BatchNorm2d(64)
        self.ReLU = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.blocks = nn.Sequential(
            ResBlock_bottle(64, 256),
            ResBlock_bottle(256, 256),
            ResBlock_bottle(256, 256),
            
            ResBlock_bottle(256, 512, bdown=True),
            ResBlock_bottle(512, 512),
            ResBlock_bottle(512, 512),
            ResBlock_bottle(512, 512),

            ResBlock_bottle(512, 1024, bdown=True),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            
            ResBlock_bottle(1024, 2048, bdown=True),
            ResBlock_bottle(2048, 2048),
            ResBlock_bottle(2048, 2048),
        )

        self.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.maxp(x)

        x = self.blocks(x)
        x = torch.mean(x, dim=(2, 3)) # [B, C, H, W] = [0, 1, 2, 3]
        x = self.fc(x)

        return x
    
class ResNet152(nn.Module):
    def __init__(self, num_class):
        super(ResNet152, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=(7, 7), padding=3, stride=(2, 2))
        self.bn = nn.BatchNorm2d(64)
        self.ReLU = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.blocks = nn.Sequential(
            ResBlock_bottle(64, 256),
            ResBlock_bottle(256, 256),
            ResBlock_bottle(256, 256),
            
            ResBlock_bottle(256, 512, bdown=True),
            ResBlock_bottle(512, 512),
            ResBlock_bottle(512, 512),
            ResBlock_bottle(512, 512),
            ResBlock_bottle(512, 512),
            ResBlock_bottle(512, 512),
            ResBlock_bottle(512, 512),
            ResBlock_bottle(512, 512),

            ResBlock_bottle(512, 1024, bdown=True),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            ResBlock_bottle(1024, 1024),
            
            ResBlock_bottle(1024, 2048, bdown=True),
            ResBlock_bottle(2048, 2048),
            ResBlock_bottle(2048, 2048),
        )

        self.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.maxp(x)

        x = self.blocks(x)
        x = torch.mean(x, dim=(2, 3)) # [B, C, H, W] = [0, 1, 2, 3]
        x = self.fc(x)

        return x