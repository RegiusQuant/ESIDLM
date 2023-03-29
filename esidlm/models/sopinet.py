from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from esidlm.modules.sopinet import SOPiNet


class LitSOPiNetModel(pl.LightningModule):

    def __init__(self, n_cont: int, n_cates: List[int], n_time: int, n_out: int, model_config: Dict):
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config

        self.net = SOPiNet(n_cont, n_cates, n_time, n_out, **model_config["net"])
        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, batch):
        return self.net(batch["X_CONT"], batch["X_CATE"], batch["X_TIME"])
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs, batch["Y"])
        loss = (loss * batch["MASK"]).sum() / (batch["MASK"].sum() + 1e-6)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs, batch["Y"])
        loss = (loss * batch["MASK"]).sum() / (batch["MASK"].sum() + 1e-6)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        return self(batch)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **self.model_config["optimizer"])
