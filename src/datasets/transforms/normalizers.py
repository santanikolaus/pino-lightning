import torch


class UnitGaussianNormalizer(torch.nn.Module):

    def __init__(self, mean=None, std=None, eps=1e-7, dim=None):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.eps = eps
        if isinstance(dim, int):
            dim = [dim]
        self.dim = dim

    def fit(self, data):
        self.mean = torch.mean(data, dim=self.dim, keepdim=True)
        self.std = torch.std(data, dim=self.dim, keepdim=True)

    def forward(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x):
        return x * (self.std + self.eps) + self.mean
