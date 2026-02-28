from torch.utils.data.dataset import Dataset


class TensorDataset(Dataset):
    def __init__(self, x, y):
        assert x.size(0) == y.size(0), "Size mismatch between tensors"
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        return {"x": x, "y": y}

    def __len__(self):
        return self.x.size(0)
