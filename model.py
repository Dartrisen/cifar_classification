import torch
import torch.nn as nn
import torch.nn.functional as F


# Define a convolution neural network
class Network(nn.Module):
    def __init__(self, num_classes: int):
        super(Network, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Linear(24 * 10 * 10, num_classes)  # Adjusting input size to match resized images

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = x.view(-1, 24 * 10 * 10)
        x = self.classifier(x)
        return x
