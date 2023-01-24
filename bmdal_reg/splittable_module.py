import collections.abc

from torch import nn


class SplittableModule(nn.Module, collections.abc.Sequence):
    """
    Base class for vectorized (multiple NNs in parallel) training modules that can be split into non-vectorized modules
    """
    def __init__(self, n_models: int):
        super().__init__()
        self.n_models = n_models

    def __getitem__(self, index: int):
        return self.get_single_model(index)

    def __len__(self) -> int:
        return self.n_models

    def get_single_model(self, i: int) -> nn.Module:
        raise NotImplementedError()
