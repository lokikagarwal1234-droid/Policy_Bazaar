"""
Neural network architecture for the DQN agent.
A standard CNN that takes stacked frames and outputs
Q-values for each action.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    Convolutional Q-Network.

    Input:  (batch, n_frames, 84, 84)
    Output: (batch, n_actions)
    """

    def __init__(self, n_frames, n_actions):
        """
        Define the layers:
          - 3 convolutional layers (with ReLU)
          - 2 fully connected layers
        Args:
            n_frames: int, number of stacked frames (input channels)
            n_actions: int, number of possible actions
        """
        super().__init__()
        
        # 1. Convolutional Layers
        # The input channels match the number of stacked frames (default is 4)
        self.conv1 = nn.Conv2d(in_channels=n_frames, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # 2. Fully Connected Layers
        # After passing an 84x84 image through the above 3 layers, 
        # the spatial dimensions are reduced to 7x7.
        # Flattened dimension = 64 (channels) * 7 * 7 = 3136
        self.fc1 = nn.Linear(in_features=3136, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=n_actions)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: torch.Tensor of shape (batch, n_frames, 84, 84)
        Returns:
            torch.Tensor of shape (batch, n_actions)
        """
        # Pass through convolutions with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the tensor starting from the channel dimension (dim 1)
        x = torch.flatten(x, start_dim=1)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # No activation on the final layer (outputs raw Q-values)
        
        return x