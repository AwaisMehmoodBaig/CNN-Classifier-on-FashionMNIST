# cnn_model.py

import torch
import torch.nn as nn  # nn = neural network module

# Define our CNN model class
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()  # Call the parent class (nn.Module) constructor

        # 1st convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=1,      # Grayscale image (1 channel)
            out_channels=16,    # Number of filters
            kernel_size=3,      # 3x3 filter
            padding=1           # Keeps output size same as input
        )

        self.relu1 = nn.ReLU()  # Add non-linearity

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,      # Downsample by 2x2
            stride=2            # Move 2 pixels per step
        )

        # 2nd convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=16,     # Input from previous conv layer
            out_channels=32,    # Increase number of filters
            kernel_size=3,
            padding=1
        )

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # After 2 poolings: 28 → 14 → 7
        # So feature map is 32 x 7 x 7
        self.fc = nn.Linear(32 * 7 * 7, 10)  # 10 classes for output

    def forward(self, x):
        # Input shape: [batch_size, 1, 28, 28]
        x = self.conv1(x)     # → [batch, 16, 28, 28]
        x = self.relu1(x)
        x = self.pool1(x)     # → [batch, 16, 14, 14]

        x = self.conv2(x)     # → [batch, 32, 14, 14]
        x = self.relu2(x)
        x = self.pool2(x)     # → [batch, 32, 7, 7]

        x = x.view(x.size(0), -1)  # Flatten: [batch, 32*7*7]
        x = self.fc(x)        # → [batch, 10]

        return x
