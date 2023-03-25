from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from esidlm.modules.sidlm import SIDLM


class LitSIDLMModel(pl.LightningModule):

    def __init__(self, n_wide: int, n_cont: int, n_cates: List[int], model_config: Dict):
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config

        self.net = SIDLM(n_wide, n_cont, n_cates, **model_config["net"])
        self.criterion = nn.MSELoss()
        self.metric_r2 = torchmetrics.R2Score()

    def forward(self, batch):
        return self.net(batch["X_WIDE"], batch["X_CONT"], batch["X_CATE"])

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs, batch["Y"])
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs, batch["Y"])

        self.metric_r2(outputs, batch["Y"])
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid_r2", self.metric_r2, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **self.model_config["optimizer"])