import os
import gc
from typing import List

import pandas as pd
from tensorflow import keras

from my_utils.loader import get_folds_filenames, get_datafold_filenames
from my_utils.vocabulary import check_and_retrive_vocabulary
from networks.models import build_models, build_models_from_pretrained
from networks.train import train_and_test_model
from networks.test import evaluate_model


######################################################################## STAND-ALONE EVALUATION:


# Utility function for performing a k-fold cross-validation
# experiment on a single dataset
def k_fold_experiment(
    *,
    task: str,
    num_train_samples: int = -1,
    epochs: int = 150,
    batch_size: int = 16,
):
    keras.backend.clear_session()
    gc.collect()

    # ---------- PRINT EXPERIMENT DETAILS

    print(f"5-fold cross-validation experiment")
    print(f"\tTask: {task}")
    print(f"\tNumber of train samples: {num_train_samples}")
    print(f"\tEpochs: {epochs}")
    print(f"\tBatch size: {batch_size}")

    # ---------- FOLDS COLLECTION

    folds = get_folds_filenames()

    # ---------- 5-FOLD EVALUATION

    for id, (train_fold, val_fold, test_fold) in enumerate(
        zip(folds["train"], folds["val"], folds["test"])
    ):
        # With 'clear_session()' called at the beginning,
        # Keras starts with a blank state at each iteration
        # and memory consumption is constant over time
        keras.backend.clear_session()
        gc.collect()

        print(f"Fold {id}")

        # Get the current fold data
        train_images, train_labels = get_datafold_filenames(
            task=task, fold_filename=train_fold
        )
        train_images = train_images[:num_train_samples]
        train_labels = train_labels[:num_train_samples]
        val_images, val_labels = get_datafold_filenames(
            task=task, fold_filename=val_fold
        )
        test_images, test_labels = get_datafold_filenames(
            task=task, fold_filename=test_fold
        )
        print(f"Train size: {len(train_images)}")
        print(f"Validation size: {len(val_images)}")
        print(f"Test size: {len(test_images)}")

        # Check and retrieve vocabulary
        w2i, i2w = check_and_retrive_vocabulary(fold_id=id)

        # Build the models
        model, prediction_model = build_models(num_labels=len(w2i))

        # Set filepaths outputs
        output_dir = (
            f"TrainSize_{num_train_samples}"
            if num_train_samples != -1
            else "TrainSize_All"
        )
        output_dir = os.path.join("results", "standard", output_dir, f"fold{id}")
        os.makedirs(output_dir, exist_ok=True)
        pred_model_filepath = os.path.join(output_dir, f"best_{task}_model.keras")
        log_path = os.path.join(output_dir, f"{task}_logs.csv")

        # Train, validate, and test models
        # Save logs in CSV file
        train_and_test_model(
            task=task,
            data=(
                train_images,
                train_labels,
                val_images,
                val_labels,
                test_images,
                test_labels,
            ),
            vocabularies=(w2i, i2w),
            epochs=epochs,
            batch_size=batch_size,
            model=model,
            prediction_model=prediction_model,
            pred_model_filepath=pred_model_filepath,
            log_path=log_path,
        )

        # Clear memory
        del train_images, train_labels, val_images, val_labels, test_images, test_labels
        del model, prediction_model

    # Test on the opposite task
    k_fold_test_experiment(
        task="omr" if task == "amt" else "amt",
        pred_model_filepath=pred_model_filepath,
    )


# Utility function for performing a k-fold test experiment
def k_fold_test_experiment(
    *,
    task: str,
    pred_model_filepath: str,
):
    keras.backend.clear_session()
    gc.collect()

    # ---------- PRINT EXPERIMENT DETAILS
    print(f"5-fold cross-validation test experiment")
    print(f"\tTask: {task}")
    print(f"\tModel: {pred_model_filepath}")

    # ---------- FOLDS COLLECTION

    folds = get_folds_filenames()

    # ---------- 5-FOLD EVALUATION

    for id, test_fold in enumerate(folds["test"]):
        # With 'clear_session()' called at the beginning,
        # Keras starts with a blank state at each iteration
        # and memory consumption is constant over time
        keras.backend.clear_session()
        gc.collect()

        print(f"Fold {id}")

        # Get the current fold data
        test_images, test_labels = get_datafold_filenames(
            task=task, fold_filename=test_fold
        )
        print(f"Test size: {len(test_images)}")

        # Check and retrieve vocabulary
        _, i2w = check_and_retrive_vocabulary(fold_id=id)

        # Load prediction model
        assert os.path.exists(pred_model_filepath), "Model not found!"
        prediction_model = keras.models.load_model(pred_model_filepath)

        # Train, validate, and test models
        test_symer, test_seqer = evaluate_model(
            task=task,
            model=prediction_model,
            images_files=test_images,
            labels_files=test_labels,
            i2w=i2w,
        )

        # Save fold logs
        log_path = pred_model_filepath.replace(".keras", f"_test_on_{task}_logs.csv")
        logs = {"test_symer": [test_symer], "test_seqer": [test_seqer]}
        logs = pd.DataFrame.from_dict(logs)
        logs.to_csv(log_path, index=False)

        # Clear memory
        del test_images, test_labels
        del prediction_model


######################################################################## TRANSFER LEARNING EVALUATION:


# Utility function for performing a k-fold cross-validation transfer learning
# experiment on a single dataset
def k_fold_transfer_learning_experiment(
    *,
    task: str,
    frozen_layers_names: List[str],
    num_train_samples: int = -1,
    epochs: int = 150,
    batch_size: int = 16,
):
    keras.backend.clear_session()
    gc.collect()

    # ---------- PRINT EXPERIMENT DETAILS

    print(f"5-fold cross-validation transfer learning experiment")
    print(f"\tTask: {task}")
    print(f"\tFrozen layers: {frozen_layers_names}")
    print(f"\tNumber of train samples: {num_train_samples}")
    print(f"\tEpochs: {epochs}")
    print(f"\tBatch size: {batch_size}")

    # ---------- FOLDS COLLECTION

    folds = get_folds_filenames()

    # ---------- 5-FOLD EVALUATION

    for id, (train_fold, val_fold, test_fold) in enumerate(
        zip(folds["train"], folds["val"], folds["test"])
    ):
        # With 'clear_session()' called at the beginning,
        # Keras starts with a blank state at each iteration
        # and memory consumption is constant over time
        keras.backend.clear_session()
        gc.collect()

        print(f"Fold {id}")

        # Get the current fold data
        train_images, train_labels = get_datafold_filenames(
            task=task, fold_filename=train_fold
        )
        train_images = train_images[:num_train_samples]
        train_labels = train_labels[:num_train_samples]
        val_images, val_labels = get_datafold_filenames(
            task=task, fold_filename=val_fold
        )
        test_images, test_labels = get_datafold_filenames(
            task=task, fold_filename=test_fold
        )
        print(f"Train size: {len(train_images)}")
        print(f"Validation size: {len(val_images)}")
        print(f"Test size: {len(test_images)}")

        # Check and retrieve vocabulary
        w2i, i2w = check_and_retrive_vocabulary(fold_id=id)

        # Build the models
        pred_model_filepath = os.path.join(
            "results", "standard", "TrainSize_All", f"fold{id}"
        )
        pred_model_filepath = os.path.join(
            pred_model_filepath, f"best_{'omr' if task == 'amt' else 'amt'}_model.keras"
        )
        model, prediction_model = build_models_from_pretrained(
            num_labels=len(w2i),
            pretrained_model_filepath=pred_model_filepath,
            frozen_layers_names=frozen_layers_names,
        )

        # Set filepaths outputs
        size_dir_name = (
            f"TrainSize_{num_train_samples}"
            if num_train_samples != -1
            else "TrainSize_All"
        )
        output_dir = os.path.join(
            "results",
            "tl",
            "-".join(frozen_layers_names),
            size_dir_name,
            f"fold{id}",
        )
        os.makedirs(output_dir, exist_ok=True)
        new_pred_model_filepath = os.path.join(output_dir, f"best_{task}_model.keras")
        log_path = os.path.join(output_dir, f"{task}_logs.csv")

        # Train, validate, and test models
        # Save logs in CSV file
        train_and_test_model(
            task=task,
            data=(
                train_images,
                train_labels,
                val_images,
                val_labels,
                test_images,
                test_labels,
            ),
            vocabularies=(w2i, i2w),
            epochs=epochs,
            batch_size=batch_size,
            model=model,
            prediction_model=prediction_model,
            pred_model_filepath=new_pred_model_filepath,
            log_path=log_path,
        )

        # Clear memory
        del train_images, train_labels, val_images, val_labels, test_images, test_labels
        del model, prediction_model

    # Test on the opposite task
    k_fold_test_experiment(
        task="omr" if task == "amt" else "amt",
        pred_model_filepath=new_pred_model_filepath,
    )
