import torch
from torch import nn
from torch.nn import functional as F


class FociDetector(nn.Module):
    def __init__(self, input_channels=3, ksize=5, hidden_channels=10):
        super(FociDetector, self).__init__()
        self.conv_padding = int((ksize - 1) / 2)
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, ksize, stride=2, padding=self.conv_padding),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, ksize, stride=2, padding=self.conv_padding),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, ksize, stride=2, padding=self.conv_padding),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, ksize, padding=self.conv_padding),
            nn.ReLU(),
        )

    def forward(self, x):
        output = self.model(x)
        return output


class MultiChannelCombinedScorer(nn.Module):
    def __init__(self, k_size=5, hidden_channels=10):
        super(MultiChannelCombinedScorer, self).__init__()
        self.channel1 = FociDetector(input_channels=1,
                                     ksize=k_size,
                                     hidden_channels=hidden_channels)
        self.channel2 = FociDetector(input_channels=1,
                                     ksize=k_size,
                                     hidden_channels=hidden_channels)

    def forward(self, x):
        output1 = torch.sigmoid(F.interpolate(self.channel1.double()(x[:, [0], :, :]),
                                              size=x.shape[-2:]))
        output2 = torch.sigmoid(F.interpolate(self.channel2.double()(x[:, [2], :, :]),
                                              size=x.shape[-2:]))
        output3 = torch.sigmoid(x[:, [0], :, :])
        output4 = torch.sigmoid(x[:, [2], :, :])
        return output1 * output2 * output3 * output4
