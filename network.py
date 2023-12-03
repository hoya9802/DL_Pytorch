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
    
class Bottleneck(nn.Module):
    def __init__(self, c_in, c_out):
        super(Bottleneck, self).__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.bn_relu_conv1 = bn_relu_conv(self.c_in, self.c_out * 4,)
        self.bn_relu_conv2 = bn_relu_conv(self.c_out * 4, self.c_out, kernel=3, pad=1)

    def forward(self, x):
        out = self.bn_relu_conv1(x)
        out = self.bn_relu_conv2(out)
        out = torch.cat((x, out), 1)
        return out


class DenseNet121(nn.Module):
    def __init__(self, num_class):
        super(DenseNet121, self).__init__()

        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.maxp = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.fc = nn.Linear(, num_class)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxp(x)

        x = DenseBlock(64, , 6)
        x = Transition(, )
        x = DenseBlock(, , 12)
        x = Transition(, )
        x = DenseBlock(, , 24)
        x = Transition(, )
        x = DenseBlock(, , 16)

        x = torch.mean(x, dim=(2, 3))
        x = self.fc(x)

        return x