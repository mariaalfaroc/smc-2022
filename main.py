# -*- coding: utf-8 -*-

import os

import tensorflow as tf

import config
from experimentation import k_fold_experiment, k_fold_transfer_learning_experiment

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
tf.config.list_physical_devices("GPU")

if __name__ == "__main__":
    epochs = 100
    num_samples = 22285

    # TASK SPECIFIC EXPERIMENTS
    # Using the AMT model architecture, we train two models: one for solving the OMR task, and another for solving the AMT task
    # OMR
    config.set_task(value="omr")
    config.set_data_globals()
    config.set_arch_globals()
    k_fold_experiment(epochs)
    # AMT
    config.set_task(value="amt")
    config.set_data_globals()
    config.set_arch_globals()
    k_fold_experiment(epochs)
    # Incremental experiment
    for i in range(5000, num_samples, 5000):
        print(f"Incremental experiment: using {i} samples")
        folder_name=str(i)
        k_fold_experiment(epochs, folder_name=folder_name, num_samples=i)

    # TRANSFER LEARNING EXPERIMENTS
    # OMR -> AMT Transfer learning
    config.set_task(value="amt")
    config.set_data_globals()
    config.set_arch_globals()
    k_fold_transfer_learning_experiment(epochs, folder_name="NoFrozen", frozen_layers=[])
    k_fold_transfer_learning_experiment(epochs, folder_name="FrozenDense", frozen_layers=["Dense"])
    k_fold_transfer_learning_experiment(epochs, folder_name="FrozenDenseBLSTM2", frozen_layers=["Dense", "Bidirectional_2", "LSTM_2"])
    k_fold_transfer_learning_experiment(epochs, folder_name="FrozenDenseBLSTM12", frozen_layers=["Dense", "Bidirectional_2", "LSTM_2", "Bidirectional_1", "LSTM_1"])
    k_fold_transfer_learning_experiment(epochs, folder_name="FrozenDenseBLSTM12CNN_2", 
    frozen_layers=["Dense", "Bidirectional_2", "LSTM_2", "Bidirectional_1", "LSTM_1", "Reshape", "Permute", "MaxPool2D_2", "LeakyReLU_2", "BatchNorm_2", "Conv2D_2"])
    k_fold_transfer_learning_experiment(epochs, folder_name="FrozenDenseBLSTM12CNN_1", 
    frozen_layers=["Dense", "Bidirectional_2", "LSTM_2", "Bidirectional_1", "LSTM_1", "Reshape", "Permute", "MaxPool2D_1", "LeakyReLU_1", "BatchNorm_1", "Conv2D_1"])
    # Incremental experiment with the two best transfer learning models
    for i in range(5000, num_samples, 5000):
        print(f"Incremental experiment: using {i} samples")
        folder_name=str(i) + "_FrozenDense"
        k_fold_transfer_learning_experiment(epochs, folder_name=folder_name, frozen_layers=["Dense"], num_samples=i)
    for i in range(5000, num_samples, 5000):
        print(f"Incremental experiment: using {i} samples")
        folder_name=str(i) + "_NoFrozen"
        k_fold_transfer_learning_experiment(epochs, folder_name=folder_name, frozen_layers=[], num_samples=i)