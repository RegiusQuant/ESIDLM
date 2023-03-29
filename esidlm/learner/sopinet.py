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
from esidlm.metrics import calc_regression_metric
from esidlm.models.sopinet import LitSOPiNetModel
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

        model = LitSOPiNetModel(
            n_cont=x_cont_train.shape[1],
            n_cates=[train_data[c].nunique() for c in self.data_config["cate_cols"]],
            n_time=x_time_train.shape[2],
            n_out=len(self.data_config["target_cols"]),
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
        x_cont_test, x_cate_test, x_time_test = self._preprocess_input_data(
            input_data=test_data,
            is_train=False
        )
        test_set = SOPiNetDataset(x_cont_test, x_cate_test, x_time_test,
                                  test_data[self.data_config["target_cols"]].values,
                                  test_data[self.data_config["mask_cols"]].values)
        test_loader = DataLoader(test_set, shuffle=False, **self.dataloader_config)

        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            print("Best Checkpoint not Found! Using Current Weights for Prediction ...")
            ckpt_path = None
        y_preds = trainer.predict(model, dataloaders=test_loader, ckpt_path=ckpt_path)
        y_preds = torch.cat(y_preds, dim=0).cpu().numpy()
        y_trues = test_data[self.data_config["target_cols"]].values
        y_masks = test_data[self.data_config["mask_cols"]].values

        for target_idx, target_col in enumerate(self.data_config["target_cols"]):
            y_mask = y_masks[:, target_idx]
            y_pred = y_preds[:, target_idx]
            y_true = y_trues[:, target_idx]
            test_data[target_col + "_PRED"] = y_pred

            y_pred = y_pred[y_mask == 1]
            y_true = y_true[y_mask == 1]
            print(target_col)
            print(calc_regression_metric(y_true, y_pred))
    
        test_name = os.path.basename(self.data_config["test_data"]).split(".")[0]
        test_folder = os.path.join(self.global_config["output_folder"], "test")
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
        test_data.to_csv(os.path.join(test_folder, f"{test_name}_pred.csv"), index=False)

    def run_model_inference(self):
        test_data_paths = os.listdir(self.data_config["inference_folder"])
        test_data_paths = [os.path.join(self.data_config["inference_folder"], p) for p in test_data_paths]
        
        model = LitSOPiNetModel.load_from_checkpoint(self.model_config["model_checkpoint_path"])
        trainer = pl.Trainer(default_root_dir=self.global_config["output_folder"], **self.trainer_config)

        for test_data_path in test_data_paths:
            test_data = pd.read_csv(test_data_path)
            x_cont_test, x_cate_test, x_time_test = self._preprocess_input_data(
                input_data=test_data,
                is_train=False
            )

            test_set = SOPiNetDataset(x_cont_test, x_cate_test, x_time_test)
            test_loader = DataLoader(test_set, shuffle=False, **self.dataloader_config)

            y_preds = trainer.predict(model, dataloaders=test_loader)
            y_preds = torch.cat(y_preds, dim=0).cpu().numpy()

            for target_idx, target_col in enumerate(self.data_config["target_cols"]):
                y_pred = y_preds[:, target_idx]
                test_data[target_col + "_PRED"] = y_pred

            test_name = os.path.basename(test_data_path).split(".")[0]
            test_data.to_csv(os.path.join(self.inference_folder, f"{test_name}_pred.csv"), index=False)
            print(f"Inference {test_name} Finish.")
