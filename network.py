import torch
import torch.nn as nn

class bn_relu_conv(nn.Module):
    def __init__(self, c_in, c_out, kernel=1, st=1, pad=0):
        super(bn_relu_conv, self).__init__()
        self.bn_relu_conv = nn.Sequential(
            nn.BatchNorm2d(c_in),
            nn.ReLU(),
            nn.Conv2d(c_in, c_out, kernel_size=(kernel, kernel), stride=st, padding=pad)
        )

    def forward(self, x):
        x = self.bn_relu_conv(x)

        return x

class Transition(nn.Module):
    def __init__(self, c_in, c_out):
        super(Transition, self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.bn = nn.BatchNorm2d(c_in)
        self.conv = nn.Conv2d(self.c_in, self.c_out, kernel_size=(1,1))
        self.avg = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        x = self.avg(x)
        
        return x
class DenseBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(DenseBlock. self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.bn_relu_conv1 = bn_relu_conv(self.c_in, self.c_out * 4,)
        self.bn_relu_conv2 = bn_relu_conv(self.c_out * 4, self.c_out, kernel=3, pad=1)

    def forward(self, x):
        x = self.bn_relu_conv1(x)
        out = self.bn_relu_conv2(x)


        return torch.cat([x, out], 1)


class DenseNet121(nn.Module):
    def __init__(self, num_class, k):
        super(DenseNet121, self).__init__()
        
        self.conv = nn.Conv2d(3, k * 2, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.maxp = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)




    def forward(self, x):
        x = self.conv(x)
        x = self.maxp(x)

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