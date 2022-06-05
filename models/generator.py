"""
Generator models for GANs.
"""

from typing import List
import torch
from torch import nn


class Generator(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128, 64]):
        super().__init__()
        conv_modules = []
        hidden_dims = [input_dim] + hidden_dims + [3] # 3 channels output
        conv_modules.append(nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[0], hidden_dims[1], kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.ReLU(inplace=True)
        ))
        for i in range(1, len(hidden_dims) - 2):
            conv_modules.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dims[i+1]),
                nn.ReLU(inplace=True)
            ))
        conv_modules.append(nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-2], hidden_dims[-1], kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        ))
        self.conv_modules = nn.Sequential(*conv_modules)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generates an image from a noise vector.
        """
        return self.conv_modules(z.view(z.size(0), z.size(1), 1, 1))
