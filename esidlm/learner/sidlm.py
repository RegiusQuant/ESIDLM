import os
import pickle
import shutil
from typing import Dict

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from esidlm.dataset.sidlm import SIDLMDataset
from esidlm.learner.base import BaseLearner
from esidlm.metrics import calc_regression_metric
from esidlm.models.sidlm import LitSIDLMModel
from esidlm.preprocessing import preprocess_onehot_data, preprocess_cont_data, preprocess_cate_data


class SIDLMLearner(BaseLearner):

    def __init__(self, running_config: Dict):
        super().__init__(running_config)
    
    def _preprocess_input_data(self, input_data: pd.DataFrame, is_train: bool):
        wide_data = preprocess_onehot_data(
            input_data=input_data[self.data_config["wide_cols"]].copy(),
            file_path=os.path.join(self.preprocess_folder, "onehot_encoder.pkl"),
            is_train=is_train
        )
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
        return wide_data, cont_data, cate_data

    def run_model_training(self):
        train_data = pd.read_csv(self.data_config["train_data"])
        valid_data = pd.read_csv(self.data_config["valid_data"])

        x_wide_train, x_cont_train, x_cate_train = self._preprocess_input_data(
            input_data=train_data,
            is_train=True
        )
        x_wide_valid, x_cont_valid, x_cate_valid = self._preprocess_input_data(
            input_data=valid_data,
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
            input_data=test_data,
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
        test_folder = os.path.join(self.global_config["output_folder"], "test")
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
        test_data.to_csv(os.path.join(test_folder, f"{test_name}_pred.csv"), index=False)

    def run_model_inference(self):
        test_data_paths = os.listdir(self.data_config["inference_folder"])
        test_data_paths = [os.path.join(self.data_config["inference_folder"], p) for p in test_data_paths]
        
        model = LitSIDLMModel.load_from_checkpoint(self.model_config["model_checkpoint_path"])
        trainer = pl.Trainer(default_root_dir=self.global_config["output_folder"], **self.trainer_config)

        for test_data_path in test_data_paths:
            test_data = pd.read_csv(test_data_path)
            x_wide_test, x_cont_test, x_cate_test = self._preprocess_input_data(
                input_data=test_data,
                is_train=False
            )

            test_set = SIDLMDataset(x_wide_test, x_cont_test, x_cate_test)
            test_loader = DataLoader(test_set, shuffle=False, **self.dataloader_config)

            y_pred = trainer.predict(model, dataloaders=test_loader)
            y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
            test_data[self.data_config["target_col"] + "_PRED"] = y_pred
            test_name = os.path.basename(test_data_path).split(".")[0]
            test_data.to_csv(os.path.join(self.inference_folder, f"{test_name}_pred.csv"), index=False)
            print(f"Inference {test_name} Finish.")

    def run_model_interpretation(self):
        model = LitSIDLMModel.load_from_checkpoint(self.model_config["model_checkpoint_path"])

        with open(os.path.join(self.preprocess_folder, "onehot_encoder.pkl"), "rb") as f:
            onehot_encoder = pickle.load(f)
        wide_features = []
        for i, feature_name in enumerate(onehot_encoder.feature_names_in_):
            for v in onehot_encoder.categories_[i]:
                wide_features.append(f"{feature_name}_{v}")
        wide_weights = model.net.wide.linear.weight.squeeze().detach().numpy()
        wide_data = pd.DataFrame({
            "WIDE_FEATURE": wide_features,
            "WIDE_WEIGHT": wide_weights
        })
        wide_data.to_csv(os.path.join(self.interpretation_folder, "wide_data.csv"), index=False)

        writer = SummaryWriter(self.interpretation_folder)
        cate_embedding = model.net.deepdense.embed_layer.embedding.weight
        category_offsets = model.net.deepdense.embed_layer.category_offsets.detach().numpy()
        with open(os.path.join(self.preprocess_folder, "label_encoders.pkl"), "rb") as f:
            label_encoders = pickle.load(f)
        for i, c in enumerate(self.data_config["cate_cols"]):
            label_encoder = label_encoders[c]
            writer.add_embedding(
                cate_embedding[category_offsets[i]: category_offsets[i] + len(label_encoder.classes_)],
                metadata=label_encoder.classes_,
                tag=f"Categorical Embedding {c}",
                global_step=0
            )
        writer.close()
