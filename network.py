import torch
import torch.nn as nn
import pretrained_network as pn

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

class PSPModule(nn.Module):
    def __init__(self, c_in):
        super(PSPModule, self).__init__()
        self.avgp1 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv1 = nn.Conv2d(c_in, c_in // 4, kernel_size=1)

        self.avgp2 = nn.AdaptiveAvgPool2d(output_size=(2, 2))
        self.conv2 = nn.Conv2d(c_in, c_in // 4, kernel_size=1)

        self.avgp3 = nn.AdaptiveAvgPool2d(output_size=(3, 3))
        self.conv3 = nn.Conv2d(c_in, c_in // 4, kernel_size=1)

        self.avgp6 = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.conv6 = nn.Conv2d(c_in, c_in // 4, kernel_size=1)

        self.conv_psp = nn.Conv2d(2 * c_in, c_in, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.avgp1(x)
        x1 = self.conv1(x1)
        x1 = torch.nn.functional.upsample_bilinear(x1, size=(x.shape[2], x.shape[3]))
        
        x2 = self.avgp2(x)
        x2 = self.conv2(x2)
        x2 = torch.nn.functional.upsample_bilinear(x2, size=(x.shape[2], x.shape[3]))

        x3 = self.avgp3(x)
        x3 = self.conv3(x3)
        x3 = torch.nn.functional.upsample_bilinear(x3, size=(x.shape[2], x.shape[3]))

        x6 = self.avgp6(x)
        x6 = self.conv6(x6)
        x6 = torch.nn.functional.upsample_bilinear(x6, size=(x.shape[2], x.shape[3]))

        x = torch.cat([x, x1, x2, x3, x6], dim=1)
        x = self.conv_psp(x)
        x = self.relu(x)

        return x

class PSPNet(nn.Module):
    def __init__(self, num_class):
        super(PSPNet, self).__init__()
        self.feats = pn.resnet50(pretrained=True)
        self.psp = PSPModule(2048)
        self.pred = nn.Conv2d(64, num_class, kernel_size=1, padding=0)

        self.conv1 = conv_bn_relu(2048, 256)
        self.conv2 = conv_bn_relu(256, 128)
        self.conv3 = conv_bn_relu(128, 64)


    def forward(self, x):
        x = self.feats(x) # [B, 2048, 28, 28]
        x = self.psp(x) # [B, 2048, 28, 28]

        x = torch.nn.functional.upsample_bilinear(x, size=(2 * x.shape[2], 2 * x.shape[3]))
        x = self.conv1(x) # [B, 256, 56, 56]

        x = torch.nn.functional.upsample_bilinear(x, size=(2 * x.shape[2], 2 * x.shape[3]))
        x = self.conv2(x) # [B, 256, 112, 112]

        x = torch.nn.functional.upsample_bilinear(x, size=(2 * x.shape[2], 2 * x.shape[3]))
        x = self.conv3(x) # [B, 256, 56, 56]

        x = self.pred(x)


        return x