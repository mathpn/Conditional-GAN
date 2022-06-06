"""
Discriminator models for GANs.
"""

from typing import List

import torch
from torch import nn


class Discriminator(nn.Module):

    def __init__(self, hidden_dims: List[int] = [64, 128, 256, 512]):
        super().__init__()
        conv_modules = []
        conv_modules.append(nn.Sequential(
            nn.Conv2d(3, hidden_dims[0], kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        for i in range(len(hidden_dims) - 1):
            conv_modules.append(nn.Sequential(
                nn.Conv2d(hidden_dims[i], hidden_dims[i+1], kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dims[i+1]),
                nn.LeakyReLU(0.2)
            ))
        self.conv_modules = nn.Sequential(*conv_modules)
        self.final_layer = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes an image and returns the probability of being real.
        """
        x = self.conv_modules(x)
        x = self.final_layer(x)
        return x


class CondDiscriminator(nn.Module):
    """ Conditional discriminator for CondGAN model. """

    def __init__(self, img_size: int, embedding_size: int, embedding_dim: int, hidden_dims: List[int] = [64, 128, 256, 512]):
        super().__init__()
        conv_modules = []
        conv_modules.append(nn.Sequential(
            nn.Conv2d(3, hidden_dims[0], kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        for i in range(len(hidden_dims) - 1):
            conv_modules.append(nn.Sequential(
                nn.Conv2d(hidden_dims[i], hidden_dims[i+1], kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dims[i+1]),
                nn.LeakyReLU(0.2)
            ))
        self.conv_modules = nn.Sequential(*conv_modules)
        # calculate the shape of the output of the last conv layer
        # img_size // 2 ** (len(hidden_dims) - 1) * hidden_dims[-1] but since we have one extra conv
        # layer at the start, we need to add 1 to the size
        final_tensor_shape = (img_size // (2 ** len(hidden_dims))) ** 2 * hidden_dims[-1]
        self.fc = nn.Sequential(
            nn.Linear(final_tensor_shape + embedding_dim, 1),
            nn.Sigmoid()
        )
        self.embedding = nn.Embedding(embedding_size, embedding_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Takes an image and category label and returns the probability of being real.
        """
        x = self.conv_modules(x)
        x = x.view(x.size(0), -1)
        embedding = self.embedding(labels)
        x = torch.cat([x, embedding], dim=1)
        x = self.fc(x)
        return x