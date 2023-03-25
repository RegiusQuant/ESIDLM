import os
import pickle
from typing import Dict

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from torch.utils.data import DataLoader

from esidlm.dataset.sidlm import SIDLMDataset
from esidlm.metrics import calc_regression_metric
from esidlm.models.sidlm import LitSIDLMModel


class SIDLMLearner:

    def __init__(self, running_config: Dict):
        self.running_config = running_config
        self.global_config = running_config.get("global")
        self.data_config = running_config.get("data")
        self.dataloader_config = running_config.get("dataloader")
        self.model_config = running_config.get("model")
        self.callback_config = running_config.get("callback")
        self.trainer_config = running_config.get("trainer")

        pl.seed_everything(self.global_config["seed"])
        if not os.path.exists(self.global_config["output_folder"]):
            os.makedirs(self.global_config["output_folder"])

    def _preprocess_wide_data(self, wide_data: pd.DataFrame, is_train: bool):
        onehot_encoder_path = os.path.join(self.global_config["output_folder"], "onehot_encoder.pkl")
        
        if is_train:
            onehot_encoder = OneHotEncoder(sparse_output=False)
            wide_data = onehot_encoder.fit_transform(wide_data)
            with open(onehot_encoder_path, "wb") as f:
                pickle.dump(onehot_encoder, f)
        else:
            with open(onehot_encoder_path, "rb") as f:
                onehot_encoder = pickle.load(f)
            wide_data = onehot_encoder.transform(wide_data)

        return wide_data
    
    def _preprocess_cont_data(self, cont_data: pd.DataFrame, is_train: bool):
        standard_scaler_path = os.path.join(self.global_config["output_folder"], "standard_scaler.pkl")

        if is_train:
            standard_scaler = StandardScaler()
            cont_data = standard_scaler.fit_transform(cont_data)
            with open(standard_scaler_path, "wb") as f:
                pickle.dump(standard_scaler, f)
        else:
            with open(standard_scaler_path, "rb") as f:
                standard_scaler = pickle.load(f)
            cont_data = standard_scaler.transform(cont_data)

        return cont_data
    
    def _preprocess_cate_data(self, cate_data: pd.DataFrame, is_train: bool):
        label_encoders_path = os.path.join(self.global_config["output_folder"], "label_encoders.pkl")

        cate_cols = cate_data.columns
        if is_train:
            label_encoders = {}
            for c in cate_cols:
                label_encoder = LabelEncoder()
                cate_data[c] = label_encoder.fit_transform(cate_data[c])
                label_encoders[c] = label_encoder
            with open(label_encoders_path, "wb") as f:
                pickle.dump(label_encoders, f)
        else:
            with open(label_encoders_path, "rb") as f:
                label_encoders = pickle.load(f)
            for c in cate_cols:
                cate_data[c] = label_encoders[c].transform(cate_data[c])

        cate_data = cate_data.values
        return cate_data
    
    def _preprocess_input_data(self, wide_data: pd.DataFrame, cont_data: pd.DataFrame,
                               cate_data: pd.DataFrame, is_train: bool):
        wide_data = self._preprocess_wide_data(wide_data, is_train)
        cont_data = self._preprocess_cont_data(cont_data, is_train)
        cate_data = self._preprocess_cate_data(cate_data, is_train)
        return wide_data, cont_data, cate_data

    def run_model_training(self):
        train_data = pd.read_csv(self.data_config["train_data"])
        valid_data = pd.read_csv(self.data_config["valid_data"])

        x_wide_train, x_cont_train, x_cate_train = self._preprocess_input_data(
            wide_data=train_data[self.data_config["wide_cols"]].copy(),
            cont_data=train_data[self.data_config["cont_cols"]].copy(),
            cate_data=train_data[self.data_config["cate_cols"]].copy(),
            is_train=True
        )
        x_wide_valid, x_cont_valid, x_cate_valid = self._preprocess_input_data(
            wide_data=valid_data[self.data_config["wide_cols"]].copy(),
            cont_data=valid_data[self.data_config["cont_cols"]].copy(),
            cate_data=valid_data[self.data_config["cate_cols"]].copy(),
            is_train=False
        )

        train_set = SIDLMDataset(x_wide_train, x_cont_train, x_cate_train, 
                                    train_data[self.data_config["target_col"]])
        valid_set = SIDLMDataset(x_wide_valid, x_cont_valid, x_cate_valid,
                                    valid_data[self.data_config["target_col"]])
        train_loader = DataLoader(train_set, shuffle=True, **self.dataloader_config)
        valid_loader = DataLoader(valid_set, shuffle=False, **self.dataloader_config)

        model = LitSIDLMModel(
            n_wide=x_wide_train.shape[1],
            n_cont=x_cont_train.shape[1],
            n_cates=[train_data[c].nunique() for c in self.data_config["cate_cols"]],
            model_config=self.model_config
        )
        
        model_checkpoint = ModelCheckpoint(**self.callback_config["model_checkpoint"])
        early_stopping = EarlyStopping(**self.callback_config["early_stopping"])
        trainer = pl.Trainer(
            default_root_dir=self.global_config["output_folder"],
            callbacks=[model_checkpoint, early_stopping],
            **self.trainer_config
        )
        trainer.fit(model, train_loader, valid_loader)

        test_data = pd.read_csv(self.data_config["test_data"])
        x_wide_test, x_cont_test, x_cate_test = self._preprocess_input_data(
            wide_data=test_data[self.data_config["wide_cols"]].copy(),
            cont_data=test_data[self.data_config["cont_cols"]].copy(),
            cate_data=test_data[self.data_config["cate_cols"]].copy(),
            is_train=False
        )
        test_set = SIDLMDataset(x_wide_test, x_cont_test, x_cate_test, 
                                   test_data[self.data_config["target_col"]])
        test_loader = DataLoader(test_set, shuffle=False, **self.dataloader_config)

        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            print("Best Checkpoint not Found! Using Current Weights for Prediction ...")
            ckpt_path = None
        y_pred = trainer.predict(model, dataloaders=test_loader, ckpt_path=ckpt_path)
        y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
        y_true = test_data[self.data_config["target_col"]].values
        print(calc_regression_metric(y_true, y_pred))

        test_data[self.data_config["target_col"] + "_PRED"] = y_pred
        test_name = os.path.basename(self.data_config["test_data"]).split(".")[0]
        test_data.to_csv(os.path.join(self.global_config["output_folder"], f"{test_name}_pred.csv"), index=False)
