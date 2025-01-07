import torch
import torch.nn as nn

class conv_bn_relu(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, pad=1):
        super(conv_bn_relu, self).__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=(k_size, k_size), padding=pad),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_bn_relu(x)

        return x

class ConvDown(nn.Module):
    def __init__(self, c_in, c_out):
        super(ConvDown, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(c_out)
        self.bn2 = nn.BatchNorm2d(c_out)

        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        y = self.avgpool(x)

        return x, y

class ConvUp(nn.Module):
    def __init__(self, c_in, c_out):
        super(ConvUp, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1)


        self.bn1 = nn.BatchNorm2d(c_out)
        self.bn2 = nn.BatchNorm2d(c_out)

        self.relu = nn.ReLU()
        
    def forward(self, x, y):
        x = torch.nn.functional.upsample_bilinear(x, size=(2 * x.shape[2], 2 * x.shape[3]))
        x = torch.cat([x, y], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        # x.shape = [B, C, H, W]

        return x


class UNet(nn.Module):
    def __init__(self, num_class):
        super(UNet, self).__init__()
        # 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.down1 = ConvDown(3, 64)     # [B, 64, 112, 112]
        self.down2 = ConvDown(64, 128)   # [B, 128, 56, 56]
        self.down3 = ConvDown(128, 256)   # [B, 256, 28, 28]
        self.down4 = ConvDown(256, 512)   # [B, 512, 14, 14]
        self.down5 = ConvDown(512, 512)   # [B, 512, 7, 7]

        self.conv1 = conv_bn_relu(512, 512)
        self.conv2 = conv_bn_relu(512, 512)

        self.up1 = ConvUp(1024, 512)
        self.up2 = ConvUp(1024, 512)
        self.up3 = ConvUp(768, 256)
        self.up4 = ConvUp(384, 128)
        self.up5 = ConvUp(192, 64)

        self.pred = nn.Conv2d(64, num_class, kernel_size=1, padding=0 )

    def forward(self, x):
        x224, x = self.down1(x)  # x224 = 64ch
        x112, x = self.down2(x)  # x112 = 128ch
        x56, x = self.down3(x)   # x56 = 256ch
        x28, x = self.down4(x)   # x28 = 512ch
        x14, x = self.down5(x)   # x14 = 512ch

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.up1(x, x14)
        x = self.up2(x, x28)
        x = self.up3(x, x56)
        x = self.up4(x, x112)
        x = self.up5(x, x224)
    
        x = self.pred(x)

        return x