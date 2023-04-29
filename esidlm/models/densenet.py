from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from esidlm.modules.densenet import EntityDenseNet
from esidlm.modules.loss.rel import RelativeErrorLoss


class LitEntityDenseNetModel(pl.LightningModule):

    def __init__(self, n_cont: int, n_cates: List[int], model_config: Dict):
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config

        self.net = EntityDenseNet(n_cont, n_cates, **model_config["net"])
        
        if "loss" not in model_config or model_config["loss"]["name"] == "mse":
            self.criterion = nn.MSELoss()
        elif model_config["loss"]["name"] == "rel":
            self.criterion = RelativeErrorLoss(model_config["loss"]["a"], model_config["loss"]["b"])
        else:
            raise ValueError("Loss not Supported!")

        self.metric_r2 = torchmetrics.R2Score()

    def forward(self, batch):
        return self.net(batch["X_CONT"], batch["X_CATE"])

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs, batch["Y"])
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
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
