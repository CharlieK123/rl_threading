import torch
import torch.nn as nn
import math

class FFN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_hidden, actor=True, gain=math.sqrt(2)):
        super().__init__()

        layers = []
        self.gain = gain

        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # final layer has no non-linear act
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.network = nn.Sequential(*layers)

        # initialise weights and biases correctly, apply necessary gain
        self.apply(self.init_layer)

        # adjust final layer gain to match recommended actor critic gains
        out_gain = 0.01 if actor else 1
        output_layer = self.network[-1]
        nn.init.orthogonal_(output_layer.weight, gain=out_gain)
        nn.init.zeros_(output_layer.bias)


    def init_layer(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=self.gain)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)
