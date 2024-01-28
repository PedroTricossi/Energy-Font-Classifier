import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from attention_models import CrossViT, ViT


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ShallowCNN(nn.Module):
    def __init__(self, hidden_features=32, num_classes=42, **kwargs):
        super().__init__()
        self.c_hid1 = hidden_features
        self.c_hid2 = hidden_features * 2
        self.c_hid3 = hidden_features * 4

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, self.c_hid1, kernel_size=5, stride=2, padding=4),
            Swish(),
            nn.Conv2d(self.c_hid1, self.c_hid2, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(self.c_hid2, self.c_hid3, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(self.c_hid3, self.c_hid3, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(self.c_hid3, self.c_hid3, kernel_size=3, stride=2, padding=1),
            Swish()
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.c_hid3, num_classes)
        )

    def get_logits(self, x):
        # TODO (3.2): Implement classification procedure that outputs the logits across the classes
        pass

    def forward(self, x, y=None) -> torch.Tensor:
        # TODO (3.2): Implement forward function for (1) EBM, (2) Unconditional JEM, (3) Conditional JEM.
        #  Consider using F.adaptive_avg_pool2d to convert between the 2D features and a linear representation.
        #  (You can also reuse your implementation of 'self.get_logits(x)' if this helps you.)
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        return x


class Unconditional_JEM(nn.Module):
    def __init__(self, hidden_features=32, num_classes=42, **kwargs):
        super(Unconditional_JEM, self).__init__()
        # self.f = ShallowCNN(hidden_features, num_classes, **kwargs)
        dim = 224
        num_classes = 42

        self.f = ViT(image_size=56, patch_size=28, num_classes=42, dim=224,
                     dim_head=28,
                     depth=6, heads=8, mlp_dim=2048, dropout=0.1,
                     emb_dropout=0.1, channels=1)

        self.mlp_energy_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )


    def forward(self, x, y=None):
        x = self.f(x)
        return self.mlp_energy_head(x).squeeze()

    def classify(self, x):
        x = self.f(x)
        return self.mlp_head(x).squeeze()


class Conditional_JEM(Unconditional_JEM):
    def __init__(self, hidden_features=32, num_classes=42, **kwargs):
        super(Conditional_JEM, self).__init__(hidden_features, num_classes, **kwargs)

    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(-1)
        else:
            return torch.gather(logits, 1, y[:, None])
