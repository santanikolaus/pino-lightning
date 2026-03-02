import torch

from src.datasets.transforms.base_transforms import Transform
from src.utils.utils import count_tensor_params


class UnitGaussianNormalizer(Transform):

    def __init__(self, mean=None, std=None, eps=1e-7, dim=None, mask=None):
        super().__init__()

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("mask", mask)
        self.register_buffer("_n_elements_t", torch.tensor(0.0))

        # TODO: If we want checkpoint/resume, ensure all stats needed to resume
        # are registered buffers and always present (e.g. squared_mean too).
        # Otherwise load_state_dict(strict=True) can fail or resume can be wrong.

        self.eps = eps
        if mean is not None:
            self.ndim = mean.ndim
        if isinstance(dim, int):
            dim = [dim]
        self.dim = dim
        self.n_elements = 0

    def fit(self, data_batch):
        self.update_mean_std(data_batch)

    def partial_fit(self, data_batch, batch_size=1):
        if 0 in list(data_batch.shape):
            return
        count = 0
        n_samples = len(data_batch)
        while count < n_samples:
            samples = data_batch[count : count + batch_size]
            if self.n_elements != 0:
                self.incremental_update_mean_std(samples)
            else:
                self.update_mean_std(samples)
            count += batch_size

    def update_mean_std(self, data_batch):
        self.ndim = data_batch.ndim  # Note this includes batch-size
        if self.mask is None:
            self.n_elements = count_tensor_params(data_batch, self.dim)
            self.mean = torch.mean(data_batch, dim=self.dim, keepdim=True)
            self.squared_mean = torch.mean(data_batch**2, dim=self.dim, keepdim=True)
            self.std = torch.std(data_batch, dim=self.dim, keepdim=True)
            self._n_elements_t = data_batch.new_tensor(float(self.n_elements))
        else:
            mask = self.mask.unsqueeze(0).to(dtype=data_batch.dtype)
            wx = data_batch * mask

            n_t = torch.sum(mask, dim=self.dim, keepdim=True)
            if 0 in self.dim:  # account for batch > 1
                n_t = n_t * data_batch.shape[0]

            n_safe = n_t.clamp_min(1.0)

            self._n_elements_t = n_t
            self.n_elements = int(n_t.sum().item())

            self.mean = torch.sum(wx, dim=self.dim, keepdim=True) / n_safe
            self.squared_mean = torch.sum(wx**2, dim=self.dim, keepdim=True) / n_safe

            var = (self.squared_mean - self.mean**2).clamp_min(0.0)
            corr = n_t / (n_t - 1.0).clamp_min(1.0)
            self.std = torch.sqrt(var * corr)  # (eps handled in transform)

    def incremental_update_mean_std(self, data_batch):
        if self.mask is None:
            n_t = data_batch.new_tensor(float(count_tensor_params(data_batch, self.dim)))
            wx = data_batch
        else:
            mask = self.mask.unsqueeze(0).to(dtype=data_batch.dtype)
            wx = data_batch * mask

            n_t = torch.sum(mask, dim=self.dim, keepdim=True)
            if 0 in self.dim:  # account for batch > 1
                n_t = n_t * data_batch.shape[0]

        sum_x = torch.sum(wx, dim=self.dim, keepdim=True)
        sum_x2 = torch.sum(wx**2, dim=self.dim, keepdim=True)

        old_n_t = self._n_elements_t
        new_n_t = old_n_t + n_t
        new_n_safe = new_n_t.clamp_min(1.0)

        self.mean = (old_n_t * self.mean + sum_x) / new_n_safe
        self.squared_mean = (old_n_t * self.squared_mean + sum_x2) / new_n_safe

        self._n_elements_t = new_n_t
        self.n_elements = int(new_n_t.sum().item())

        var = (self.squared_mean - self.mean**2).clamp_min(0.0)
        corr = new_n_t / (new_n_t - 1.0).clamp_min(1.0)
        self.std = torch.sqrt(var * corr)  # (eps handled in transform)

    def transform(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x):
        return x * (self.std + self.eps) + self.mean

    def forward(self, x):
        return self.transform(x)

    def cuda(self):
        self.mean = self.mean.cuda() if self.mean is not None else None
        self.std = self.std.cuda() if self.std is not None else None
        self.mask = self.mask.cuda() if self.mask is not None else None
        self._n_elements_t = self._n_elements_t.cuda()
        return self

    def cpu(self):
        self.mean = self.mean.cpu() if self.mean is not None else None
        self.std = self.std.cpu() if self.std is not None else None
        self.mask = self.mask.cpu() if self.mask is not None else None
        self._n_elements_t = self._n_elements_t.cpu()
        return self

    def to(self, device):
        self.mean = self.mean.to(device) if self.mean is not None else None
        self.std = self.std.to(device) if self.std is not None else None
        self.mask = self.mask.to(device) if self.mask is not None else None
        self._n_elements_t = self._n_elements_t.to(device)
        return self

    @classmethod
    def from_dataset(cls, dataset, dim=None, keys=None, mask=None):
        for i, data_dict in enumerate(dataset):
            if not i and not keys:
                keys = data_dict.keys()

        instances = {key: cls(dim=dim, mask=mask) for key in keys}
        for data_dict in dataset:
            for key, sample in data_dict.items():
                if key in keys:
                    instances[key].partial_fit(sample.unsqueeze(0))
        return instances
