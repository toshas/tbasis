import torch
from torch.nn.modules.flatten import Flatten


class ModelNetLenet5(torch.nn.Module):
    """
    A Wide Compression flavor of Lenet5 used for benchmarking model compression
    """
    def __init__(self, cfg):
        super().__init__()
        assert cfg.dataset == 'mnist'
        self.model = torch.nn.Sequential(
            # 1x28x28
            torch.nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2, bias=True),
            torch.nn.ReLU(),
            # 20x28x28
            torch.nn.MaxPool2d(kernel_size=2),
            # 20x14x14
            torch.nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=0, bias=True),
            torch.nn.ReLU(),
            # 50x10x10
            torch.nn.MaxPool2d(kernel_size=2),
            # 50x5x5
            Flatten(),
            # 1250
            torch.nn.Linear(50 * 5 * 5, 320),
            torch.nn.ReLU(),
            # 320
            torch.nn.Linear(320, 10)
        )

    def forward(self, input):
        assert input.dim() == 4 and input.shape[1:] == (1, 28, 28)
        return self.model(input)
