from esidlm.learner.widedeep import WideDeepLearner


WIDEDEEP_TRAINING_CONFIG = {
    "global": {
        "seed": 42,
        "output_folder": "outputs/widedeep",
    },

    "data": {
        "train_data": "data/train.csv",
        "valid_data": "data/valid.csv",
        "test_data": "data/test.csv",

        "wide_cols": ['C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'C_6', 'C_7', 'C_8', 'C_9', 'C_10', 
                      'C_11', 'C_12', 'C_13', 'C_14'],
        "cont_cols": ['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9', 'X_10', 
                      'X_11', 'X_12', 'X_13', 'X_14', 'X_15', 'X_16', 'X_17', 'X_18', 'X_19', 
                      'X_20', 'X_21', 'X_22', 'X_23', 'X_24', 'X_25', 'X_26', 'X_27', 'X_28', 
                      'X_29', 'X_30', 'X_31', 'X_32'],
        "cate_cols": ['C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'C_6', 'C_7', 'C_8', 'C_9', 'C_10', 
                      'C_11', 'C_12', 'C_13', 'C_14'],
        "target_col": "Y",
    },

    "dataloader": {
        "batch_size": 1024,
        "num_workers": 4,
    },

    "model": {
        "net": {
            "d_embed": 32,
            "d_model": 128,
            "n_layers": 2,
            "p_drop": 0.3,
            "act_fn": "relu"
        },
        "optimizer": {
            "lr": 3e-4,
            "weight_decay": 1e-5,
        },
    },

    "callback": {
        "model_checkpoint": {
            "save_top_k": 1,
            "monitor": "valid_loss",
            "mode": "min",
            "verbose": True
        },
        "early_stopping": {
            "monitor": "valid_loss",
            "mode": "min",
            "patience": 5,
            "verbose": True
        }
    },

    "trainer": {
        "max_epochs": 20,
        "accelerator": "gpu",
        "devices": 1,
        "deterministic": True
    }
}



if __name__ == "__main__":
    learner = WideDeepLearner(WIDEDEEP_TRAINING_CONFIG)
    learner.run_model_training()