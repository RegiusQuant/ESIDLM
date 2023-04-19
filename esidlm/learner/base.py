import os
import shutil
from abc import ABC, abstractmethod
from typing import Dict

import torch
import pytorch_lightning as pl


class BaseLearner(ABC):

    def __init__(self, running_config: Dict):
        self.running_config = running_config
        self.global_config = running_config.get("global")
        self.data_config = running_config.get("data")
        self.dataloader_config = running_config.get("dataloader")
        self.model_config = running_config.get("model")
        self.callback_config = running_config.get("callback")
        self.trainer_config = running_config.get("trainer")

        torch.set_float32_matmul_precision("high")
        pl.seed_everything(self.global_config.get("seed", 42))
        if not os.path.exists(self.global_config["output_folder"]):
            os.makedirs(self.global_config["output_folder"])

        self.preprocess_folder = os.path.join(self.global_config["output_folder"], "preprocessing")
        if not os.path.exists(self.preprocess_folder):
            os.makedirs(self.preprocess_folder)

        self.inference_folder = os.path.join(self.global_config["output_folder"], "inference")
        if not os.path.exists(self.inference_folder):
            os.makedirs(self.inference_folder)
            
        self.interpretation_folder = os.path.join(self.global_config["output_folder"], "interpretation")
        if os.path.exists(self.interpretation_folder):
            shutil.rmtree(self.interpretation_folder)
        os.makedirs(self.interpretation_folder)

    @abstractmethod
    def run_model_training(self):
        raise NotImplementedError
    
    @abstractmethod
    def run_model_inference(self):
        raise NotImplementedError
    
    @abstractmethod
    def run_model_interpretation(self):
        raise NotImplementedError
