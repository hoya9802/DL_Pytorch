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
