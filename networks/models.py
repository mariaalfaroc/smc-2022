from typing import List, Tuple

from tensorflow import keras
from tensorflow.keras import layers

# NOTE:
# OMR and AMT architecture based on:
# Miguel A. Román, Antonio Pertusa, Jorge Calvo-Zaragoza
# Data representations for audio-to-score monophonic music transcription

INPUT_HEIGHT = 256
POOLING_FACTORS = {"height_reduction": 4, "width_reduction": 2}


def build_models(num_labels: int) -> Tuple[keras.Model, keras.Model]:
    def ctc_loss_lambda(args):
        y_true, y_pred, input_length, label_length = args
        return keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

    # Input block
    image = keras.Input(
        shape=(INPUT_HEIGHT, None, 1),
        dtype="float32",
        name="image",
    )
    image_len = keras.Input(shape=(1,), dtype="int32", name="image_len")
    label = keras.Input(shape=(None,), dtype="int32", name="label")
    label_len = keras.Input(shape=(1,), dtype="int32", name="label_len")

    # Convolutional block
    x = layers.Conv2D(8, (10, 2), padding="same", use_bias=False, name="Conv2D_1")(
        image
    )
    x = layers.BatchNormalization(name="BN_1")(x)
    x = layers.LeakyReLU(0.2, name="LeakyReLU_1")(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="MaxPool2D_1")(x)

    x = layers.Conv2D(8, (8, 5), padding="same", use_bias=False, name="Conv2D_2")(x)
    x = layers.BatchNormalization(name="BN_2")(x)
    x = layers.LeakyReLU(0.2, name="LeakyReLU_2")(x)
    x = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), name="MaxPool2D_2")(x)

    # Intermediate block (preparation to enter the recurrent one)
    # [batch, height, width, channels] -> [batch, width, height, channels]
    x = layers.Permute((2, 1, 3), name="Permute")(x)
    # [batch, width, height, channels] -> [batch, width, height * channels]
    x = layers.Reshape((-1, x.shape[2] * x.shape[3]), name="Reshape")(x)

    # Recurrent block
    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, dropout=0.5, name="LSTM_1"),
        name="Bidirectional_1",
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, dropout=0.5, name="LSTM_2"),
        name="Bidirectional_2",
    )(x)

    # Dense layer
    # num_classes -> represents "num_labels + 1" classes,
    # where num_labels is the number of true labels,
    # and the largest value  "(num_classes - 1) = (num_labels + 1 - 1) = num_labels"
    # is reserved for the blank label
    # Range of true labels -> [0, len(voc_size))
    # Therefore, len(voc_size) is the default value for the CTC-blank index
    output = layers.Dense(num_labels + 1, activation="softmax", name="Dense")(x)

    # CTC-loss computation
    # Keras does not currently support loss functions with extra parameters,
    # so CTC loss is implemented in a Lambda layer
    ctc_loss = layers.Lambda(
        function=ctc_loss_lambda, output_shape=(1,), name="ctc_loss"
    )([label, output, image_len, label_len])

    # Create training model and predicition model
    model = keras.Model([image, image_len, label, label_len], ctc_loss)
    # The loss calculation is already done, so use a dummy lambda function for the loss
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss={"ctc_loss": lambda y_true, y_pred: y_pred},
    )
    # At inference time, we only have the image as input and the softmax prediction as output
    prediction_model = keras.Model(model.get_layer("image").input, output)

    return model, prediction_model


def build_models_from_pretrained(
    num_labels: int,
    pretrained_model_filepath: str,
    frozen_layers_names: List[str],
) -> Tuple[keras.Model, keras.Model]:
    # Build the model used only for training
    model, _ = build_models(num_labels)
    # Get layer names
    layer_names = [layer.name for layer in model.layers]
    # Check that all frozen layers are in the model
    for layer_name in frozen_layers_names:
        if layer_name not in layer_names:
            raise ValueError(
                f"Layer {layer_name} not found in the model."
                f"Available layers: {layer_names}"
            )
    # Load the weights of a pretrained model into the previously created model
    model.load_weights(filepath=pretrained_model_filepath, by_name=True)
    # Freeze some layers
    for layer in model.layers:
        if layer.name in frozen_layers_names:
            layer.trainable = False
            print(f"Layer {layer.name} trainable? {layer.trainable}")
    # Interaction between trainable and compile()
    # Source: https://keras.io/getting_started/faq/#how-can-i-freeze-layers-and-do-finetuning
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss={"ctc_loss": lambda y_true, y_pred: y_pred},
    )
    # Create the prediction model
    prediction_model = keras.Model(
        model.get_layer("image").input, model.get_layer("Dense").output
    )
    return model, prediction_model
