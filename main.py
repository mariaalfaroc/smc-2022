# -*- coding: utf-8 -*-

import os

import tensorflow as tf

import config
from experimentation import k_fold_experiment, k_fold_transfer_learning_experiment, k_fold_test_experiment

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
tf.config.list_physical_devices("GPU")

if __name__ == "__main__":
    epochs = 100
    r = [50, 75, 100, 250, 500, 1000]

    # TASK SPECIFIC EXPERIMENTS
    # Using the AMT model architecture, we train two models: one for solving the OMR task, and another for solving the AMT task
    # OMR
    config.set_task(value="omr")
    config.set_data_globals()
    config.set_arch_globals()
    k_fold_experiment(epochs)
    # Incremental experiment
    for i in r:
        print(f"Incremental experiment: using {i} samples")
        folder_name=str(i)
        k_fold_experiment(epochs, folder_name=folder_name, num_train_samples=i)
    # AMT
    config.set_task(value="amt")
    config.set_data_globals()
    config.set_arch_globals()
    # k_fold_experiment(epochs)
    # Incremental experiment
    for i in r:
        print(f"Incremental experiment: using {i} samples")
        folder_name=str(i)
        k_fold_experiment(epochs, folder_name=folder_name, num_train_samples=i)

    # TRANSFER LEARNING EXPERIMENTS
    # OMR -> AMT Transfer learning
    config.set_task(value="amt")
    config.set_data_globals()
    config.set_arch_globals()
    k_fold_transfer_learning_experiment(epochs, folder_name="NoFrozen", frozen_layers=[])
    k_fold_transfer_learning_experiment(epochs, folder_name="FrozenDense", frozen_layers=["Dense"])
    k_fold_transfer_learning_experiment(epochs, folder_name="FrozenDenseBLSTM2", frozen_layers=["Dense", "Bidirectional_2", "LSTM_2"])
    k_fold_transfer_learning_experiment(epochs, folder_name="FrozenDenseBLSTM12", frozen_layers=["Dense", "Bidirectional_2", "LSTM_2", "Bidirectional_1", "LSTM_1"])
    k_fold_transfer_learning_experiment(epochs, folder_name="FrozenDenseBLSTM12CNN2", 
    frozen_layers=["Dense", "Bidirectional_2", "LSTM_2", "Bidirectional_1", "LSTM_1", "Reshape", "Permute", "MaxPool2D_2", "LeakyReLU_2", "BatchNorm_2", "Conv2D_2"])
    # Incremental experiment with the best transfer learning model
    for i in r:
        print(f"Incremental experiment: using {i} samples")
        folder_name=str(i) + "_FrozenDense"
        k_fold_transfer_learning_experiment(epochs, folder_name=folder_name, frozen_layers=["Dense"], num_train_samples=i)

    # AMT -> OMR Transfer learning
    config.set_task(value="omr")
    config.set_data_globals()
    config.set_arch_globals()
    k_fold_transfer_learning_experiment(epochs, folder_name="NoFrozen", frozen_layers=[], pretrained="amt")
    k_fold_transfer_learning_experiment(epochs, folder_name="FrozenDense", frozen_layers=["Dense"], pretrained="amt")
    k_fold_transfer_learning_experiment(epochs, folder_name="FrozenDenseBLSTM2", frozen_layers=["Dense", "Bidirectional_2", "LSTM_2"], pretrained="amt")
    k_fold_transfer_learning_experiment(epochs, folder_name="FrozenDenseBLSTM12", frozen_layers=["Dense", "Bidirectional_2", "LSTM_2", "Bidirectional_1", "LSTM_1"], pretrained="amt")
    k_fold_transfer_learning_experiment(epochs, folder_name="FrozenDenseBLSTM12CNN2", 
    frozen_layers=["Dense", "Bidirectional_2", "LSTM_2", "Bidirectional_1", "LSTM_1", "Reshape", "Permute", "MaxPool2D_2", "LeakyReLU_2", "BatchNorm_2", "Conv2D_2"], pretrained="amt")
    # Incremental experiment with the best transfer learning model
    for i in r:
        print(f"Incremental experiment: using {i} samples")
        folder_name=str(i) + "_NoFrozen"
        k_fold_transfer_learning_experiment(epochs, folder_name=folder_name, frozen_layers=[], num_train_samples=i, pretrained="amt")
    
    
    # TEST EXPERIMENTS 
    # To evaluate models on the counterpart modalility that the one they were trained on and see the corresponding degradation
    omr_baseline_folder = config.base_dir / "SMC-2022" / "Experiments" / "TaskSpecific" / "omr" 
    amt_baseline_folder = config.base_dir / "SMC-2022" / "Experiments" / "TaskSpecific" / "amt"
    k_fold_test_experiment(test_type="amt", model_folder_name=omr_baseline_folder)
    k_fold_test_experiment(test_type="omr", model_folder_name=amt_baseline_folder) 
    omr2amt_trasfer_folder = config.base_dir / "SMC-2022" / "Experiments" / "TransferLearning" / "amt" 
    amt2omr_trasfer_folder = config.base_dir / "SMC-2022" / "Experiments" / "TransferLearning" / "omr" 
    schemes = ["NoFrozen", "FrozenDense", "FrozenDenseBLSTM2", "FrozenDenseBLSTM12", "FrozenDenseBLSTM12CNN2"]
    for i in schemes:
        k_fold_test_experiment(test_type="omr", model_folder_name=omr2amt_trasfer_folder / i)
        k_fold_test_experiment(test_type="amt", model_folder_name=amt2omr_trasfer_folder / i)