from typing import List

import torch
import torch.nn as nn

from esidlm.modules.sidlm import DeepDense


class EntityDenseNet(nn.Module):

    def __init__(self, n_cont: int, n_cates: List[int], d_embed: int, d_model: int, 
                 n_layers: int, p_drop: float, act_fn: str):
        super().__init__()

        self.deepdense = DeepDense(n_cont, n_cates, d_embed, d_model, n_layers, p_drop, act_fn)

    def forward(self, x_cont: torch.Tensor, x_cate: torch.Tensor):
        x = self.deepdense(x_cont, x_cate)
        return x.squeeze(1)
