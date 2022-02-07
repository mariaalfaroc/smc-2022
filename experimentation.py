# -*- coding: utf-8 -*-

import os, gc

import pandas as pd
import numpy as np
from tensorflow import keras

import config
from data_processing import get_folds_filenames, get_datafolds_filenames, get_fold_vocabularies, save_w2i_dictionary, load_dictionaries, train_data_generator
from models import build_models
from evaluation import evaluate_model

# Utility function for training, validating, and testing a model and saving the logs in a CSV file
def train_and_test_model(data, vocabularies, epochs, model, prediction_model, pred_model_filepath, log_path):
    train_images, train_labels, val_images, val_labels, test_images, test_labels = data
    w2i, i2w = vocabularies

    # Instantiate logs variables
    loss_acc = []
    val_symer_acc = []
    val_seqer_acc = []

    # Train and validate
    best_symer = np.Inf
    best_epoch = 0
    for epoch in range(epochs):
        print(f"--Epoch {epoch + 1}--")
        print("Training:")
        history = model.fit(
            train_data_generator(train_images, train_labels, w2i),
            epochs=1, 
            verbose=2,
            steps_per_epoch=len(train_images) // config.batch_size
        )
        loss_acc.extend(history.history["loss"])
        print("Validating:")
        val_symer, val_seqer = evaluate_model(prediction_model, val_images, val_labels, i2w)[0:2]
        val_symer_acc.append(val_symer)
        val_seqer_acc.append(val_seqer)
        if val_symer < best_symer:
            best_symer = val_symer
            best_epoch = epoch
            print(f"Saving new best prediction model to file {pred_model_filepath}")
            prediction_model.save(filepath=pred_model_filepath)
    print(f"Best validation SymER (%): {best_symer:.2f} at epoch {best_epoch + 1}")

    # Test the best validation model
    print("Evaluating best validation model over test data")
    prediction_model = keras.models.load_model(pred_model_filepath)
    test_symer, test_seqer = evaluate_model(prediction_model, test_images, test_labels, i2w)

    # Save fold logs
    # The last line on the CSV file is the one corresponding to the best validation model
    loss_acc.extend(["-", loss_acc[best_epoch]])
    val_symer_acc.extend(["-", val_symer_acc[best_epoch]])
    val_seqer_acc.extend(["-", val_seqer_acc[best_epoch]])
    logs = {
        "loss" : loss_acc, 
        "val_symer": val_symer_acc, "val_seqer": val_seqer_acc, 
        "test_symer": ["-"] * (len(val_symer_acc) - 1) + [test_symer], "test_seqer": ["-"] * (len(val_seqer_acc) - 1) + [test_seqer]
        }
    logs = pd.DataFrame.from_dict(logs)
    logs.to_csv(log_path, index=False)

    return

# -- EXPERIMENT TYPES -- #

# Utility function for performing a k-fold cross-validation experiment on a single dataset
def k_fold_experiment(epochs, folder_name=None, num_train_samples=13371):
    keras.backend.clear_session()
    gc.collect()

    # ---------- PRINT EXPERIMENT DETAILS

    print("k-fold cross-validation experiment")
    print(f"Data used {config.base_dir.stem}")

    # ---------- DATA COLLECTION

    train_folds_files = get_folds_filenames("train")
    val_folds_files = get_folds_filenames("val")
    test_folds_files = get_folds_filenames("test")

    assert len(train_folds_files) == len(val_folds_files) == len(test_folds_files)

    train_images_fnames, train_labels_fnames = get_datafolds_filenames(train_folds_files)
    val_images_fnames, val_labels_fnames = get_datafolds_filenames(val_folds_files)
    test_images_fnames, test_labels_fnames = get_datafolds_filenames(test_folds_files) 

    # ---------- K-FOLD EVALUATION

    # Start the k-fold evaluation scheme
    # k = len(train_images_fnames)
    # for i in range(k):

    # For the SMC 2022, we are going to do 0-Fold due to time restrictions
    # We use the first fold of the 5-crossval folder
    for i in range(1):
        # With 'clear_session()' called at the beginning,
        # Keras starts with a blank state at each iteration
        # and memory consumption is constant over time.
        keras.backend.clear_session()
        gc.collect()

        print(f"Fold {i}")

        # Set filepaths outputs
        output_dir = config.base_dir / "SMC-2022" / "Experiments" / "TaskSpecific" / config.task
        output_dir = output_dir / folder_name / f"Fold{i}" if folder_name is not None else output_dir / f"Fold{i}"
        os.makedirs(output_dir, exist_ok=True)
        pred_model_filepath = output_dir / "best_model.keras"
        w2i_filepath = output_dir / "w2i.json"
        log_path = output_dir / "logs.csv"

        # Get the current fold data
        train_images, train_labels = train_images_fnames[i], train_labels_fnames[i]
        val_images, val_labels = val_images_fnames[i], val_labels_fnames[i]
        test_images, test_labels = test_images_fnames[i], test_labels_fnames[i]

        # Get and save vocabularies: VERY IMPORTANT TO DO THIS STEP HERE!
        # So as to have the same architecture on both incremental experiments
        w2i, i2w = get_fold_vocabularies(train_labels)
        save_w2i_dictionary(w2i, w2i_filepath)

        # Select the desired number of training samples for the incremental scheme
        train_images, train_labels = train_images[:num_train_samples], train_labels[:num_train_samples]

        assert len(train_images) == len(train_labels)
        assert len(val_images) == len(val_labels)
        assert len(test_images) == len(test_labels)

        print(f"Train: {len(train_images)}")
        print(f"Validation: {len(val_images)}")
        print(f"Test: {len(test_images)}")

        # Build the models
        model, prediction_model = build_models(num_labels=len(w2i))

        # Train, validate, and test models
        # Save logs in CSV file
        train_and_test_model(
            data=(train_images, train_labels, val_images, val_labels, test_images, test_labels),
            vocabularies=(w2i, i2w),
            epochs=epochs,
            model=model, prediction_model=prediction_model,
            pred_model_filepath=pred_model_filepath, 
            log_path=log_path
        )

        # Clear memory
        del train_images, train_labels, val_images, val_labels, test_images, test_labels
        del model, prediction_model
        
    return

# --------------------

# Utility function for performing a k-fold cross-validation transfer learning experiment on a single dataset
def k_fold_transfer_learning_experiment(epochs, folder_name, frozen_layers, num_train_samples=13371, pretrained="omr"):
    keras.backend.clear_session()
    gc.collect()

    # ---------- PRINT EXPERIMENT DETAILS

    print("k-fold cross-validation transfer learning experiment")
    if pretrained == "omr":
        print("Using an OMR pretrained model to transfer its learning into an AMT model")
    else:
        print("Using an AMT pretrained model to transfer its learning into an OMR model")
    print(f"Layers to freeze: {frozen_layers}")
    print(f"Data used {config.base_dir.stem}")

    # ---------- DATA COLLECTION

    train_folds_files = get_folds_filenames("train")
    val_folds_files = get_folds_filenames("val")
    test_folds_files = get_folds_filenames("test")

    assert len(train_folds_files) == len(val_folds_files) == len(test_folds_files)

    train_images_fnames, train_labels_fnames = get_datafolds_filenames(train_folds_files)
    val_images_fnames, val_labels_fnames = get_datafolds_filenames(val_folds_files)
    test_images_fnames, test_labels_fnames = get_datafolds_filenames(test_folds_files) 

    # ---------- K-FOLD EVALUATION

    # Start the k-fold evaluation scheme
    # k = len(train_images_fnames)
    # for i in range(k):

    # For the SMC 2022, we are going to do 0-Fold due to time restrictions
    # We use the first fold of the 5-crossval folder
    for i in range(1):
        # With 'clear_session()' called at the beginning,
        # Keras starts with a blank state at each iteration
        # and memory consumption is constant over time.
        keras.backend.clear_session()
        gc.collect()

        print(f"Fold {i}")

        # Set filepaths outputs
        output_dir = config.base_dir / "SMC-2022" / "Experiments" / "TransferLearning" / config.task / folder_name /f"Fold{i}"
        os.makedirs(output_dir, exist_ok=True)
        pred_model_filepath = output_dir / "best_model.keras"
        w2i_filepath = output_dir / "w2i.json"
        log_path = output_dir / "logs.csv"

        # Get the current fold data
        train_images, train_labels = train_images_fnames[i], train_labels_fnames[i]
        val_images, val_labels = val_images_fnames[i], val_labels_fnames[i]
        test_images, test_labels = test_images_fnames[i], test_labels_fnames[i]

        # Get and save vocabularies: VERY IMPORTANT TO DO THIS STEP HERE!
        # The pretrained model was trained using the vocabulary of the whole dataset
        w2i, i2w = get_fold_vocabularies(train_labels)
        save_w2i_dictionary(w2i, w2i_filepath)

        # Select the desired number of training samples for the incremental scheme
        train_images, train_labels = train_images[:num_train_samples], train_labels[:num_train_samples]

        assert len(train_images) == len(train_labels)
        assert len(val_images) == len(val_labels)
        assert len(test_images) == len(test_labels)

        print(f"Train: {len(train_images)}")
        print(f"Validation: {len(val_images)}")
        print(f"Test: {len(test_images)}")

        # Build the model used only for training
        model = build_models(num_labels=len(w2i))[0]
        # Load the weights of a pretrained model into the previously created model
        pretrained_prediction_model_filepath = config.base_dir / "SMC-2022" / "Experiments" / "TaskSpecific" / f"{pretrained}" / f"Fold{i}" / "best_model.keras"
        model.load_weights(filepath=pretrained_prediction_model_filepath, by_name=True) 
        # Freeze some layers 
        for layer in model.layers:
            if layer.name in frozen_layers:
                layer.trainable = False
                print(f"Layer {layer.name} trainable? {layer.trainable}")
        # Interaction between trainable and compile() 
        # Source: https://keras.io/getting_started/faq/#how-can-i-freeze-layers-and-do-finetuning
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss={"ctc_loss" : lambda y_true, y_pred: y_pred}
        )
        # Create the prediction model
        prediction_model = keras.Model(model.get_layer("image").input, model.get_layer("Dense").output)

        # Train, validate, and test models
        # Save logs in CSV file
        train_and_test_model(
            data=(train_images, train_labels, val_images, val_labels, test_images, test_labels),
            vocabularies=(w2i, i2w),
            epochs=epochs,
            model=model, prediction_model=prediction_model,
            pred_model_filepath=pred_model_filepath, 
            log_path=log_path
        )

        # Clear memory
        del train_images, train_labels, val_images, val_labels, test_images, test_labels
        del model, prediction_model
        
    return

# --------------------

# Utility function for performing a k-fold test experiment
def k_fold_test_experiment(test_type: str, model_folder_name: str):
    keras.backend.clear_session()
    gc.collect()

    # ---------- PRINT EXPERIMENT DETAILS

    print("k-fold test experiment")
    print(f"Data used {config.base_dir.stem}")
    print(f"Evaluating model {model_folder_name} on {test_type.upper()} data")

    # ---------- DATA COLLECTION

    config.set_task(value=test_type)
    config.set_data_globals()
    config.set_arch_globals()
    test_folds_files = get_folds_filenames("test")
    test_images_fnames, test_labels_fnames = get_datafolds_filenames(test_folds_files) 

    # ---------- K-FOLD EVALUATION

    test_symer_acc = []
    test_seqer_acc = []

    # Start the k-fold evaluation scheme
    # k = len(test_images_fnames)
    # for i in range(k):

    # For the SMC 2022, we are going to do 0-Fold due to time restrictions
    # We use the first fold of the 5-crossval folder
    for i in range(1):
        # With 'clear_session()' called at the beginning,
        # Keras starts with a blank state at each iteration
        # and memory consumption is constant over time.
        keras.backend.clear_session()
        gc.collect()

        print(f"Fold {i}")

        # Set filepaths
        output_dir = model_folder_name / f"Fold{i}"
        pred_model_filepath = output_dir / "best_model.keras"
        w2i_filepath = output_dir / "w2i.json"
        log_path = output_dir / f"test_on_{test_type}_logs.csv"

        # Get the current fold data
        test_images, test_labels = test_images_fnames[i], test_labels_fnames[i]
        assert len(test_images) == len(test_labels)
        print(f"Test: {len(test_images)}")

        # Get vocabularies
        i2w = load_dictionaries(w2i_filepath)[1]

        # Load and test model
        prediction_model = keras.models.load_model(pred_model_filepath)
        test_symer, test_seqer = evaluate_model(prediction_model, test_images, test_labels, i2w)
        test_symer_acc.append(test_symer)
        test_seqer_acc.append(test_seqer)

        # Clear memory
        del test_images, test_labels
        del prediction_model

    # Save fold logs
    logs = {"test_symer": test_symer_acc, "test_seqer": test_seqer_acc}
    logs = pd.DataFrame.from_dict(logs)
    logs.to_csv(log_path, index=False)

    return