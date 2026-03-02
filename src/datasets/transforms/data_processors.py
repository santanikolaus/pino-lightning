from abc import ABCMeta, abstractmethod
import torch


class DataProcessor(torch.nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def to(self, device):
        pass

    @abstractmethod
    def preprocess(self, x):
        pass

    @abstractmethod
    def postprocess(self, x):
        pass

    # default wrap method
    def wrap(self, model):
        self.model = model
        return self

    # default train and eval methods
    def train(self, val: bool = True):
        super().train(val)
        if self.model is not None:
            self.model.train()

    def eval(self):
        super().eval()
        if self.model is not None:
            self.model.eval()

    @abstractmethod
    def forward(self, x):
        pass


class DefaultDataProcessor(DataProcessor):
    def __init__(self, in_normalizer=None, out_normalizer=None):
        super().__init__()
        self.in_normalizer = in_normalizer
        self.out_normalizer = out_normalizer
        self.device = "cpu"
        self.model = None

    def to(self, device):
        if self.in_normalizer is not None:
            self.in_normalizer = self.in_normalizer.to(device)
        if self.out_normalizer is not None:
            self.out_normalizer = self.out_normalizer.to(device)
        self.device = device
        return self

    def preprocess(self, data_dict, batched=True):
        x = data_dict["x"].to(self.device)
        y = data_dict["y"].to(self.device)

        if self.in_normalizer is not None:
            x = self.in_normalizer.transform(x)
        if self.out_normalizer is not None and self.training:
            y = self.out_normalizer.transform(y)

        data_dict["x"] = x
        data_dict["y"] = y

        return data_dict

    def postprocess(self, output, data_dict):
        if self.out_normalizer and not self.training:
            output = self.out_normalizer.inverse_transform(output)
        return output, data_dict

    def forward(self, **data_dict):
        data_dict = self.preprocess(data_dict)
        output = self.model(data_dict["x"])
        output, data_dict = self.postprocess(output, data_dict)
        return output, data_dict
