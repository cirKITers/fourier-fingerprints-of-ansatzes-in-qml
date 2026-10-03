import torch.nn as nn
import torch


def set_torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


class HEPRegressor(nn.Module):
    def __init__(self, n_features, width=128, depth=4):
        super().__init__()
        self.degree = width * depth  # TODO: remove
        self.n_input_feat = n_features

        layers = [nn.Linear(n_features, width), nn.LeakyReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.LeakyReLU()]
        layers += [nn.Linear(width, 1)]
        self.model = nn.Sequential(*layers)

        # self.model.apply(self.init_weights)

    def forward(self, x):
        output = self.model(x)
        return output.squeeze()

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 1)
