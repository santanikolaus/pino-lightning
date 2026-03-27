import torch


class DefaultDataProcessor(torch.nn.Module):

    def __init__(self, in_normalizer=None, out_normalizer=None):
        super().__init__()
        self.in_normalizer = in_normalizer
        self.out_normalizer = out_normalizer
        self.device = "cpu"

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def preprocess(self, data_dict):
        x = data_dict["x"].to(self.device)
        y = data_dict["y"].to(self.device)

        if self.in_normalizer is not None:
            x = self.in_normalizer.transform(x)
        if self.out_normalizer is not None and self.training:
            y = self.out_normalizer.transform(y)

        return {**data_dict, "x": x, "y": y}

    def postprocess(self, output):
        if self.out_normalizer and not self.training:
            output = self.out_normalizer.inverse_transform(output)
        return output
