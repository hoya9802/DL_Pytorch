import torch
import torch.nn as nn

class conv_bn_relu(nn.Module):
    def __init__(self, c_in, c_out):
        super(conv_bn_relu, self).__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_bn_relu(x)
        return x

class VGG16(nn.Module):
    def __init__(self, num_class):
        super(VGG16, self).__init__()
        # self.conv_ch = conv_ch
        # [B, C, H, W]
        self.vgg16 = nn.Sequential( # img: [B, 3, 128, 128]
            conv_bn_relu(3, 64),    # [B, 64, 128, 128]
            conv_bn_relu(64, 64),   # [B, 64, 128, 128]
            nn.MaxPool2d(kernel_size=2, stride=2), # [B, 64, 64, 64]

            conv_bn_relu(64, 128),  # [B, 128, 64, 64]
            conv_bn_relu(128, 128),  # [B, 128, 64, 64]
            nn.MaxPool2d(kernel_size=2, stride=2), # [B, 128, 32, 32]

            conv_bn_relu(128, 256),  # [B, 256, 32, 32]
            conv_bn_relu(256, 256),  # [B, 256, 32, 32]
            conv_bn_relu(256, 256),  # [B, 256, 32, 32]
            nn.MaxPool2d(kernel_size=2, stride=2), # [B, 256, 16, 16]

            conv_bn_relu(256, 512),  # [B, 512, 16, 16]
            conv_bn_relu(512, 512),  # [B, 512, 16, 16]
            conv_bn_relu(512, 512),  # [B, 512, 16, 16]
            nn.MaxPool2d(kernel_size=2, stride=2), # [B, 512, 8, 8]

            conv_bn_relu(512, 512),  # [B, 512, 8, 8]
            conv_bn_relu(512, 512),  # [B, 512, 8, 8]
            conv_bn_relu(512, 512),  # [B, 512, 8, 8]
            nn.MaxPool2d(kernel_size=2, stride=2), # [B, 512, 4, 4]
        )

        self.fcs = nn.Sequential(nn.Linear(512*4*4, 4096),
                                 nn.Dropout(p=0.5),
                                 nn.ReLU(),
                                 nn.Linear(4096, 4096),
                                 nn.Dropout(p=0.5),
                                 nn.ReLU(),
                                 nn.Linear(4096, num_class))

    def forward(self, x):
        x = self.vgg16(x) # [8, 512, 4, 4]
        x = torch.reshape(x, (-1, 4 * 4 * 512)) # [8, 512 * 4 * 4]
        x = self.fcs(x)
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


class ResNet18(nn.Module):
    def __init__(self, num_class):
        super(ResNet18, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=(7, 7), padding=3, stride=(2, 2))
        self.bn = nn.BatchNorm2d(64)
        self.ReLU = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.blocks = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),

            ResBlock(64, 128, bdown=True),
            ResBlock(128, 128),

            ResBlock(128, 256, bdown=True),
            ResBlock(256, 256),

            ResBlock(256, 512, bdown=True),
            ResBlock(512, 512),
        )

        self.dr = nn.Dropout(p=0.2)
        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.maxp(x)

        x = self.blocks(x)
        x = torch.mean(x, dim=(2, 3)) # [B, C, H, W] = [0, 1, 2, 3]
        x = self.dr(x)
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

        self.dr = nn.Dropout(p=0.2)
        self.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.maxp(x)

        x = self.blocks(x)
        x = torch.mean(x, dim=(2, 3)) # [B, C, H, W] = [0, 1, 2, 3]
        x = self.dr(x)
        x = self.fc(x)

        return x