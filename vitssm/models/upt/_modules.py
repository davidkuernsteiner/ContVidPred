import torch
import torch.nn as nn

class AffineTransformation(nn.Module):
    def __init__(self, num_features):
        """
        A PyTorch module that performs an affine transformation.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(AffineTransformation, self).__init__()
        # Learnable parameters: gamma (scale) and beta (shift)
        self.gamma = nn.Parameter(torch.ones(num_features))  # Initialized to 1
        self.beta = nn.Parameter(torch.zeros(num_features))  # Initialized to 0

    def forward(self, x):
        """
        Apply the affine transformation.

        Args:
            x (Tensor): Input tensor of shape (..., num_features).

        Returns:
            Tensor: Affine-transformed tensor of the same shape as input.
        """
        return self.gamma * x + self.beta
