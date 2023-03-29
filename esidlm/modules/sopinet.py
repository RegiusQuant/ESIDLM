from typing import List

import torch
import torch.nn as nn

from esidlm.modules.sidlm import CategoricalEmbedding, MLPBlock


class DeepDenseEncoder(nn.Module):

    def __init__(self, n_cont: int, n_cates: List[int], d_embed: int, d_model: int, 
                 n_layers: int, p_drop: float, act_fn: str):
        super().__init__() 

        self.embed_layer = CategoricalEmbedding(n_cates, d_embed)

        d_inp = len(n_cates) * d_embed + n_cont
        self.input_layer = MLPBlock(d_inp, d_model, p_drop, act_fn)
        self.hidden_layers = nn.Sequential(
            *[MLPBlock(d_model, d_model, p_drop, act_fn) for _ in range(n_layers)]
        )

    def forward(self, x_cont: torch.Tensor, x_cate: torch.Tensor):
        x_embed = self.embed_layer(x_cate)
        x = torch.cat([x_cont, x_embed], dim=1)
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        return x


class TimeSeriesEncoder(nn.Module):

    def __init__(self, n_time: int, d_model: int, n_head: int, p_drop: float):
        super().__init__()

        self.proj = nn.Linear(n_time, d_model)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 2,
            dropout=p_drop,
            batch_first=True
        )

    def forward(self, x_time: torch.Tensor):
        x = self.proj(x_time)
        x = self.encoder(x)
        x, _ = torch.max(x, dim=1)
        return x


class SOPiNet(nn.Module):

    def __init__(self, n_cont: int, n_cates: List[int], n_time: int, n_out: int, d_embed: int,
                 d_model: int, n_layers: int, n_head: int, p_drop: float, act_fn: str):
        super().__init__()

        self.deepdense = DeepDenseEncoder(n_cont, n_cates, d_embed, d_model, n_layers, p_drop, act_fn)
        self.timeseries = TimeSeriesEncoder(n_time, d_model, n_head, p_drop)
        self.decoder = nn.Linear(d_model * 2, n_out)

    def forward(self, x_cont: torch.Tensor, x_cate: torch.Tensor, x_time: torch.Tensor):
        x_deep = self.deepdense(x_cont, x_cate)
        x_time = self.timeseries(x_time)
        x = torch.cat([x_deep, x_time], dim=1)
        return self.decoder(x)
