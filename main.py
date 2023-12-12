import os
import random
from itertools import chain

import numpy as np
import tensorflow as tf

from experimentation import k_fold_experiment, k_fold_transfer_learning_experiment

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
tf.config.list_physical_devices("GPU")

# Seed
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

if __name__ == "__main__":
    EPOCHS = 100
    BATCH_SIZE = 4
    SYMER_THRESHOLD = 30

    ##################################### SCENARIO A:
    # Use the entire training partition
    # Compare performance with and without transfer learning

    # Get baselines
    for task in ["omr", "amt"]:
        k_fold_experiment(
            task=task,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
        )

    # Perform transfer learning considering different frozen layers
    MODEL_BLOCKS = [
        ["Dense"],
        ["Bidirectional_2"],
        ["Bidirectional_1"],
        ["Conv2D_2", "BN_2"],
        ["Conv2D_1", "BN_1"],
    ]
    for task in ["omr", "amt"]:
        for i in range(1, len(MODEL_BLOCKS)):
            frozen_layers = list(chain(*MODEL_BLOCKS[:-i]))
            k_fold_transfer_learning_experiment(
                task=task,
                frozen_layers_names=frozen_layers,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
            )

    ##################################### SCENARIO B:
    # Assess the influence of the number of training samples on the performance

    # Get baselines
    for task in ["omr", "amt"]:
        for num_train_samples in [50, 75, 100, 250, 500, 1000]:
            k_fold_experiment(
                task=task,
                num_train_samples=num_train_samples,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
            )

    # Perform transfer learning considering the best transfer learning model
    for task in ["omr", "amt"]:
        if task == "omr":
            # Update all layers when going from AMT to OMR
            frozen_layers = []
        else:
            # Freeze only the classification layers when going from OMR to AMT
            frozen_layers = ["Dense"]
        for num_train_samples in [50, 75, 100, 250, 500, 1000]:
            k_fold_transfer_learning_experiment(
                task=task,
                frozen_layers_names=frozen_layers,
                num_train_samples=num_train_samples,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
            )
