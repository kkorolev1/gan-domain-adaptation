from abc import abstractmethod
from typing import Union

import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from degan.utils import nan_hook

class BaseModel(nn.Module):
    """
    Base class for all models
    """

    def __init__(self):
        super().__init__()

        # for submodule in self.modules():
        #     submodule.register_forward_hook(nan_hook)

    @abstractmethod
    def forward(self, **batch) -> Union[Tensor, dict]:
        """
        Forward pass logic.
        Can return a torch.Tensor (it will be interpreted as logits) or a dict.

        :return: Model output
        """
        raise NotImplementedError()

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

 