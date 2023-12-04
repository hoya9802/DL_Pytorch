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

        self.bn = nn.BatchNorm2d(c_in)
        self.conv = nn.Conv2d(c_in, c_out//2, kernel_size=(1,1))
        self.avg = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        x = self.avg(x)
        
        return x
    
class DenseBlock(nn.Module):
    def __init__(self, c_in, k):
        super(DenseBlock, self).__init__()
        
        self.bn_relu_conv1 = bn_relu_conv(c_in, k * 4)   
        self.bn_relu_conv2 = bn_relu_conv(k * 4, k, kernel=3, pad=1)

    def forward(self, x):
        out = self.bn_relu_conv1(x)
        out = self.bn_relu_conv2(out)
        out = torch.cat((x, out), 1) 
        return out


class DenseNet121(nn.Module):
    def __init__(self, num_class, k):
        super(DenseNet121, self).__init__()

        self.num_class = num_class
        self.k = k
        self.conv = nn.Conv2d(3, k*2, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)  # [B, 3]
        self.maxp = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)                   # [B, 64]

        self.densenet = nn.Sequential(
                DenseBlock(64, self.k),
                DenseBlock(96, self.k),
                DenseBlock(128, self.k),
                DenseBlock(160, self.k),
                DenseBlock(192, self.k),
                DenseBlock(224, self.k),

                Transition(256, 128),

                DenseBlock(128, self.k),
                DenseBlock(160, self.k),
                DenseBlock(192, self.k),
                DenseBlock(224, self.k),
                DenseBlock(256, self.k),
                DenseBlock(288, self.k),
                DenseBlock(320, self.k),
                DenseBlock(352, self.k),
                DenseBlock(384, self.k),
                DenseBlock(416, self.k),
                DenseBlock(448, self.k),
                DenseBlock(480, self.k)
        )

        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxp(x)
        x = self.densenet(x)
        x = torch.mean(x, dim=(2, 3))
        x = self.fc(x)

        return x