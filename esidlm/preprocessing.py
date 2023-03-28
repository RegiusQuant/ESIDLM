import pickle

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


def preprocess_onehot_data(input_data: pd.DataFrame, file_path: str, is_train: bool):
    if is_train:
        onehot_encoder = OneHotEncoder(sparse_output=False)
        input_data = onehot_encoder.fit_transform(input_data)
        with open(file_path, "wb") as f:
            pickle.dump(onehot_encoder, f)
    else:
        with open(file_path, "rb") as f:
            onehot_encoder = pickle.load(f)
        input_data = onehot_encoder.transform(input_data)
    return input_data


def preprocess_cont_data(input_data: pd.DataFrame, file_path: str, is_train: bool):
    if is_train:
        standard_scaler = StandardScaler()
        input_data = standard_scaler.fit_transform(input_data)
        with open(file_path, "wb") as f:
            pickle.dump(standard_scaler, f)
    else:
        with open(file_path, "rb") as f:
            standard_scaler = pickle.load(f)
        input_data = standard_scaler.transform(input_data)
    return input_data


def preprocess_cate_data(input_data: pd.DataFrame, file_path: str, is_train: bool):
    cate_cols = input_data.columns
    if is_train:
        label_encoders = {}
        for c in cate_cols:
            label_encoder = LabelEncoder()
            input_data[c] = label_encoder.fit_transform(input_data[c])
            label_encoders[c] = label_encoder
        with open(file_path, "wb") as f:
            pickle.dump(label_encoders, f)
    else:
        with open(file_path, "rb") as f:
            label_encoders = pickle.load(f)
        for c in cate_cols:
            input_data[c] = label_encoders[c].transform(input_data[c])
    input_data = input_data.values
    return input_data