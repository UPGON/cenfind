import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels=in_ch,
                                             out_channels=out_ch,
                                             kernel_size=k_size,
                                             stride=stride,
                                             padding=padding),
                                   nn.ReLU())

    def forward(self, x):
        return self.block(x)


class FociDetector(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, k_size, padding):
        super(FociDetector, self).__init__()
        self.net = nn.Sequential(
            ConvBlock(in_ch, mid_ch, k_size, stride=2, padding=padding),
            ConvBlock(mid_ch, mid_ch, k_size, stride=2, padding=padding),
            ConvBlock(mid_ch, mid_ch, k_size, stride=2, padding=padding),
            ConvBlock(mid_ch, out_ch, k_size, stride=2, padding=padding)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    x = torch.rand((1, 1, 2048, 2048))
    model = FociDetector(in_ch=1, mid_ch=10, out_ch=1, k_size=5, padding=2)
    y = model(x)
    print(y.shape)
