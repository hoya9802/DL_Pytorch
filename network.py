import torch
import torch.nn as nn

class bn_relu_conv(nn.Module):
    def __init__(self, c_in, c_out, kernel=1, st=1, pad=0):
        super(bn_relu_conv, self).__init__()
        self.bn_relu_conv = nn.Sequential(
            nn.BatchNorm2d(c_in),
            nn.ReLU(),
            nn.Conv2d(c_in, c_out, kernel_size=(kernel, kernel), stride=st, padding=pad, bias=False)
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
    def __init__(self, c_in, c_out, r):
        super(DenseBlock, self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.r = r
        self.bn_relu_conv1 = bn_relu_conv(self.c_in, self.c_out * 4,)
        self.bn_relu_conv2 = bn_relu_conv(self.c_out * 4, self.c_out, kernel=3, pad=1)

    def forward(self, x):
        for _ in range(self.r):
            x = self.bn_relu_conv1(x)
            out = self.bn_relu_conv2(x)
            x = torch.cat([x, out], 1)
        return x


class DenseNet121(nn.Module):
    def __init__(self, num_class, k):
        super(DenseNet121, self).__init__()
        
        self.k = k
        self.conv = nn.Conv2d(3, k * 2, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.maxp = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxp(x)

        x = DenseBlock(self.k * 2, self.k, 6)
        x = Transition(self.k, self.k)
        x = DenseBlock(self.k, self.k, 12)
        x = Transition(self.k, self.k)
        x = DenseBlock(self.k, self.k, 24)
        x = Transition(self.k, self.k)
        x = DenseBlock(self.k, self.k, 16)

        x = torch.mean(x, dim=(2, 3))
        x = self.fc(x)

        return x