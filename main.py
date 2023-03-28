from esidlm.learner.sidlm import SIDLMLearner
from esidlm.learner.sopinet import SOPiNetLearner


SIDLM_TRAINING_CONFIG = {
    "global": {
        "seed": 42,
        "output_folder": "outputs/sidlm",
    },

    "data": {
        "train_data": "data/sidlm/train.csv",
        "valid_data": "data/sidlm/valid.csv",
        "test_data": "data/sidlm/test.csv",

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

SIDLM_INFERENCE_CONFIG = {
    "global": {
        "seed": 42,
        "output_folder": "outputs/sidlm",
    },

    "data": {
        "inference_folder": "data/sidlm/test",

        "wide_cols": ['C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'C_6', 'C_7', 'C_8', 'C_9', 'C_10', 
                      'C_11', 'C_12', 'C_13', 'C_14'],
        "cont_cols": ['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9', 'X_10', 
                      'X_11', 'X_12', 'X_13', 'X_14', 'X_15', 'X_16', 'X_17', 'X_18', 'X_19', 
                      'X_20', 'X_21', 'X_22', 'X_23', 'X_24', 'X_25', 'X_26', 'X_27', 'X_28', 
                      'X_29', 'X_30', 'X_31', 'X_32'],
        "cate_cols": ['C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'C_6', 'C_7', 'C_8', 'C_9', 'C_10', 
                      'C_11', 'C_12', 'C_13', 'C_14'],
        "target_col": "Y"
    },

    "dataloader": {
        "batch_size": 1024,
        "num_workers": 4,
    },

    "model": {
        "model_checkpoint_path": "outputs/sidlm/lightning_logs/version_4/checkpoints/epoch=19-step=4320.ckpt"
    },

    "trainer": {
        "accelerator": "gpu",
        "devices": 1,
    },

}

SOPINET_TRAINING_CONFIG = {
    "global": {
        "seed": 42,
        "output_folder": "outputs/sopinet",
    },

    "data": {
        "train_data": "data/sopinet/train.csv",
        "valid_data": "data/sopinet/valid.csv",
        "test_data": "data/sopinet/test.csv",

        "cont_cols": ['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9', 'X_10', 
                      'X_11', 'X_12', 'X_13', 'X_14', 'X_15', 'X_16', 'X_17', 'X_18', 'X_19', 'X_20', 
                      'X_21', 'X_22', 'X_23', 'X_24', 'X_25', 'X_26', 'X_27', 'X_28', 'X_29', 'X_30', 
                      'X_31', 'X_32', 'X_33', 'X_34', 'X_35', 'X_36', 'X_37', 'X_38', 'X_39', 'X_40', 
                      'X_41', 'X_42', 'X_43', 'X_44', 'X_45', 'X_46', 'X_47', 'X_48', 'X_49', 'X_50', 
                      'X_51', 'X_52', 'X_53', 'X_54', 'X_55', 'X_56'],
        "cate_cols": ['C_1', 'C_2', 'C_3', 'C_4', 'C_5'],
        "time_cols": [
            ['X_57', 'X_77', 'X_97', 'X_117', 'X_137', 'X_157', 'X_177', 'X_197', 'X_217', 'X_237', 'X_257', 'X_277', 'X_297', 'X_317'], 
            ['X_58', 'X_78', 'X_98', 'X_118', 'X_138', 'X_158', 'X_178', 'X_198', 'X_218', 'X_238', 'X_258', 'X_278', 'X_298', 'X_318'], 
            ['X_59', 'X_79', 'X_99', 'X_119', 'X_139', 'X_159', 'X_179', 'X_199', 'X_219', 'X_239', 'X_259', 'X_279', 'X_299', 'X_319'], 
            ['X_60', 'X_80', 'X_100', 'X_120', 'X_140', 'X_160', 'X_180', 'X_200', 'X_220', 'X_240', 'X_260', 'X_280', 'X_300', 'X_320'], 
            ['X_61', 'X_81', 'X_101', 'X_121', 'X_141', 'X_161', 'X_181', 'X_201', 'X_221', 'X_241', 'X_261', 'X_281', 'X_301', 'X_321'], 
            ['X_62', 'X_82', 'X_102', 'X_122', 'X_142', 'X_162', 'X_182', 'X_202', 'X_222', 'X_242', 'X_262', 'X_282', 'X_302', 'X_322'], 
            ['X_63', 'X_83', 'X_103', 'X_123', 'X_143', 'X_163', 'X_183', 'X_203', 'X_223', 'X_243', 'X_263', 'X_283', 'X_303', 'X_323'], 
            ['X_64', 'X_84', 'X_104', 'X_124', 'X_144', 'X_164', 'X_184', 'X_204', 'X_224', 'X_244', 'X_264', 'X_284', 'X_304', 'X_324'], 
            ['X_65', 'X_85', 'X_105', 'X_125', 'X_145', 'X_165', 'X_185', 'X_205', 'X_225', 'X_245', 'X_265', 'X_285', 'X_305', 'X_325'], 
            ['X_66', 'X_86', 'X_106', 'X_126', 'X_146', 'X_166', 'X_186', 'X_206', 'X_226', 'X_246', 'X_266', 'X_286', 'X_306', 'X_326'], 
            ['X_67', 'X_87', 'X_107', 'X_127', 'X_147', 'X_167', 'X_187', 'X_207', 'X_227', 'X_247', 'X_267', 'X_287', 'X_307', 'X_327'], 
            ['X_68', 'X_88', 'X_108', 'X_128', 'X_148', 'X_168', 'X_188', 'X_208', 'X_228', 'X_248', 'X_268', 'X_288', 'X_308', 'X_328'], 
            ['X_69', 'X_89', 'X_109', 'X_129', 'X_149', 'X_169', 'X_189', 'X_209', 'X_229', 'X_249', 'X_269', 'X_289', 'X_309', 'X_329'], 
            ['X_70', 'X_90', 'X_110', 'X_130', 'X_150', 'X_170', 'X_190', 'X_210', 'X_230', 'X_250', 'X_270', 'X_290', 'X_310', 'X_330'], 
            ['X_71', 'X_91', 'X_111', 'X_131', 'X_151', 'X_171', 'X_191', 'X_211', 'X_231', 'X_251', 'X_271', 'X_291', 'X_311', 'X_331'], 
            ['X_72', 'X_92', 'X_112', 'X_132', 'X_152', 'X_172', 'X_192', 'X_212', 'X_232', 'X_252', 'X_272', 'X_292', 'X_312', 'X_332'], 
            ['X_73', 'X_93', 'X_113', 'X_133', 'X_153', 'X_173', 'X_193', 'X_213', 'X_233', 'X_253', 'X_273', 'X_293', 'X_313', 'X_333'], 
            ['X_74', 'X_94', 'X_114', 'X_134', 'X_154', 'X_174', 'X_194', 'X_214', 'X_234', 'X_254', 'X_274', 'X_294', 'X_314', 'X_334'], 
            ['X_75', 'X_95', 'X_115', 'X_135', 'X_155', 'X_175', 'X_195', 'X_215', 'X_235', 'X_255', 'X_275', 'X_295', 'X_315', 'X_335'], 
            ['X_76', 'X_96', 'X_116', 'X_136', 'X_156', 'X_176', 'X_196', 'X_216', 'X_236', 'X_256', 'X_276', 'X_296', 'X_316', 'X_336']
        ],
        "target_cols": ["Y_1", "Y_2"],
        "mask_cols": ["Y_1_masked", "Y_2_masked"]
    },

    "dataloader": {
        "batch_size": 256,
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
    # learner = SIDLMLearner(SIDLM_TRAINING_CONFIG)
    # learner.run_model_training()

    # learner = SIDLMLearner(SIDLM_INFERENCE_CONFIG)
    # learner.run_model_inference()

    learner = SOPiNetLearner(SOPINET_TRAINING_CONFIG)
    learner.run_model_training()
