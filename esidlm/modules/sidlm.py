from typing import List

import torch
import torch.nn as nn


_ACT_LAYER_MAP = {
    "sigmoid": nn.Sigmoid(),
    "relu": nn.ReLU(),
    "leakyrelu": nn.LeakyReLU(),
    "elu": nn.ELU()
}


class MLPBlock(nn.Module):

    def __init__(self, d_inp: int, d_out: int, p_drop: float, act_fn: str):
        super().__init__()

        if act_fn not in _ACT_LAYER_MAP:
            raise ValueError(f"{act_fn} is not supported.")
        act_layer = _ACT_LAYER_MAP[act_fn]

        self.layers = nn.Sequential(
            nn.Linear(d_inp, d_out),
            act_layer,
            nn.BatchNorm1d(d_out),
            nn.Dropout(p_drop)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class CategoricalEmbedding(nn.Module):

    def __init__(self, n_cates: List[int], d_embed: int):
        super().__init__()

        category_offsets = torch.tensor([0] + n_cates[:-1]).cumsum(0)
        self.register_buffer("category_offsets", category_offsets, persistent=False)

        self.embedding = nn.Embedding(sum(n_cates), d_embed)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x + self.category_offsets.unsqueeze(0))
        return torch.flatten(x, start_dim=1)


class DeepDense(nn.Module):

    def __init__(self, n_cont: int, n_cates: List[int], d_embed: int, d_model: int, 
                 n_layers: int, p_drop: float, act_fn: str):
        super().__init__() 

        self.embed_layer = CategoricalEmbedding(n_cates, d_embed)

        d_inp = len(n_cates) * d_embed + n_cont
        self.input_layer = MLPBlock(d_inp, d_model, p_drop, act_fn)
        self.hidden_layers = nn.Sequential(
            *[MLPBlock(d_model, d_model, p_drop, act_fn) for _ in range(n_layers)]
        )
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x_cont: torch.Tensor, x_cate: torch.Tensor):
        x_embed = self.embed_layer(x_cate)
        x = torch.cat([x_cont, x_embed], dim=1)
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


class Wide(nn.Module):

    def __init__(self, n_wide: int):
        super().__init__()
        self.linear = nn.Linear(n_wide, 1, bias=False)

    def forward(self, x: torch.Tensor):
        return self.linear(x)


class SIDLM(nn.Module):

    def __init__(self, n_wide: int, n_cont: int, n_cates: List[int], d_embed: int,
                 d_model: int, n_layers: int, p_drop: float, act_fn: str):
        super().__init__()

        self.wide = Wide(n_wide)
        self.deepdense = DeepDense(n_cont, n_cates, d_embed, d_model, n_layers, p_drop, act_fn)

    def forward(self, x_wide: torch.Tensor, x_cont: torch.Tensor, x_cate: torch.Tensor):
        x = self.wide(x_wide) + self.deepdense(x_cont, x_cate)
        return x.squeeze(1)
