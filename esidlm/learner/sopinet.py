import os
from typing import Dict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from esidlm.dataset.sopinet import SOPiNetDataset
from esidlm.learner.base import BaseLearner
from esidlm.preprocessing import preprocess_cont_data, preprocess_cate_data


class SOPiNetLearner(BaseLearner):

    def __init__(self, running_config: Dict):
        super().__init__(running_config)


    def _preprocess_input_data(self, input_data: pd.DataFrame, is_train: bool):
        cont_data = preprocess_cont_data(
            input_data=input_data[self.data_config["cont_cols"]].copy(),
            file_path=os.path.join(self.preprocess_folder, "standard_scaler.pkl"),
            is_train=is_train
        )
        cate_data = preprocess_cate_data(
            input_data=input_data[self.data_config["cate_cols"]].copy(),
            file_path=os.path.join(self.preprocess_folder, "label_encoders.pkl"),
            is_train=is_train
        )
        time_data = []
        for i, c in enumerate(self.data_config["time_cols"]):
            temp_data = preprocess_cont_data(
                input_data=input_data[c].copy(),
                file_path=os.path.join(self.preprocess_folder, f"standard_scaler_{i}.pkl"),
                is_train=is_train
            )
            temp_data = np.expand_dims(temp_data, axis=1)
            time_data.append(temp_data)
        time_data = np.concatenate(time_data, axis=1)
        return cont_data, cate_data, time_data
        

    def run_model_training(self):
        train_data = pd.read_csv(self.data_config["train_data"])
        valid_data = pd.read_csv(self.data_config["valid_data"])

        x_cont_train, x_cate_train, x_time_train = self._preprocess_input_data(
            input_data=train_data,
            is_train=True
        )
        x_cont_valid, x_cate_valid, x_time_valid = self._preprocess_input_data(
            input_data=valid_data,
            is_train=False
        )

        train_set = SOPiNetDataset(x_cont_train, x_cate_train, x_time_train,
                                    train_data[self.data_config["target_cols"]].values,
                                    train_data[self.data_config["mask_cols"]].values)
        valid_set = SOPiNetDataset(x_cont_valid, x_cate_valid, x_time_valid,
                                    valid_data[self.data_config["target_cols"]].values,
                                    valid_data[self.data_config["mask_cols"]].values)
        train_loader = DataLoader(train_set, shuffle=True, **self.dataloader_config)
        valid_loader = DataLoader(valid_set, shuffle=False, **self.dataloader_config)


    def run_model_inference(self):
        pass
