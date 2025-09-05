import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import mlflow
import numpy as np
import tensorflow as tf
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from tensorflow.keras import (Input, Model, callbacks, layers, models,
                              optimizers)

from configs.constants import (BATCH_SIZE, BUFFER_SIZE, EPOCHS, LEARNING_RATE,
                               ML_WINDOW, ML_WINDOW_OVERLAP)
from src.pipeline.deleteTFR import delete_old_files

from ..utils.tfrecord_utils import PerWindowNormalization, get_num_labels
from ..visualizer.source import Source
from .combineTFR import parse_tfrecord_fn

tfrecord_path = os.environ.get('TFRECORD_PATH')
log_path = os.environ.get('LOG_PATH')
ml_model_path = os.environ.get("ML_MODEL_PATH")
mlflow_path = os.environ.get('MLRUNS_PATH')
combined_emg_path = os.environ.get('COMBINED_EMG_FILE')
depedency_path = os.environ.get('DEPENDENCY_FILE')

mlflow.set_tracking_uri(uri=mlflow_path)
mlflow.set_experiment(experiment_name='gesture-classification')

num_emg_channels = Source.get_num_emg_channels()

def clear_tensorboard_logs() -> None:
    delete_old_files(selected_type=["train", "validation"], parent_path=log_path, file_type="*.v2")

def load_tfrecord(filename):
    raw_dataset = tf.data.TFRecordDataset([filename])
    return raw_dataset.map(parse_tfrecord_fn)

def load_dataset(selected_type):
    return load_tfrecord(filename=f"{tfrecord_path}/{selected_type}/{combined_emg_path}")


def run_model() -> None:
    with mlflow.start_run() as run:
        clear_tensorboard_logs()
        train_ds_size = sum(1 for _ in load_dataset(selected_type="train"))
        validation_ds_size = sum(1 for _ in load_dataset(selected_type="validate"))

        train_steps_per_epoch = int(train_ds_size // BATCH_SIZE)
        validation_steps_per_epoch = int(validation_ds_size // BATCH_SIZE)

        train_ds = load_dataset(selected_type="train").shuffle(buffer_size=BUFFER_SIZE).repeat()
        train_ds = (train_ds
            .batch(batch_size=BATCH_SIZE, drop_remainder=True)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            )

        validation_ds = load_dataset(selected_type="validate").shuffle(buffer_size=BUFFER_SIZE).repeat()
        validation_ds = (validation_ds
            .batch(batch_size=BATCH_SIZE, drop_remainder=True)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            )


        callback_list = [
            callbacks.ModelCheckpoint(filepath=ml_model_path, monitor="val_loss", save_best_only=True),
            callbacks.TensorBoard(log_dir=log_path),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=10, min_lr=0.00001, verbose=1)
        ]

        inputs = Input(shape=(ML_WINDOW, num_emg_channels))
        normalize = PerWindowNormalization()(inputs)
        embedding = layers.TimeDistributed(layers.Dense(units=170, activation='relu'))(normalize)
        lstm_one = layers.LSTM(400, return_sequences=True)(embedding)
        lstm_one = layers.Dropout(0.3)(lstm_one)
        lstm_two = layers.LSTM(400)(lstm_one)
        lstm_two = layers.Dropout(0.3)(lstm_two)
        outputs = layers.Dense(units=get_num_labels(), activation='softmax')(lstm_two)

        model = Model(inputs, outputs)
        model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

        with open('model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        mlflow.log_artifact('model_summary.txt')

        history = model.fit(train_ds, epochs=EPOCHS, steps_per_epoch=train_steps_per_epoch, validation_data=validation_ds, validation_steps=validation_steps_per_epoch, callbacks=callback_list)

        mlflow.log_artifacts(local_dir=log_path, artifact_path='tensorboard_logs')


        model_input = Schema(inputs=[TensorSpec(type=np.dtype(np.float32), shape= (-1, ML_WINDOW, num_emg_channels), name="EMG_Channels_1_to_8 (OpenBCI)")])
        model_output = Schema(inputs=[TensorSpec(type=np.dtype(np.float32), shape= (-1, get_num_labels()), name="Hand_Gestures (Classification)")])

        model_signature = ModelSignature(inputs=model_input, outputs=model_output)
        best_model = models.load_model(ml_model_path)
        mlflow.keras.log_model(best_model, 'best_model', pip_requirements = depedency_path, signature=model_signature)

        params = {
            'Batch Size' : BATCH_SIZE,
            'Sequence length' : ML_WINDOW,
            'Input Dimension' : num_emg_channels,
            'Overlap Data-Points' : ML_WINDOW_OVERLAP,
            'EPOCHS' : EPOCHS,
            'Optimizer Name' : 'adam',
            'Learning Rate' : LEARNING_RATE,
            'Buffer Size' : BUFFER_SIZE,
            'ReduceOnPlateau' : 'monitor:val_loss, factor:0.5, patience:10, min_lr:0.00001, verbose:1',
            'NumberOfGestures' : get_num_labels()
        }
        
        mlflow.log_params(params=params)

        metrics = {
            'Best Validation Accuracy' : max(history.history['val_accuracy']),
            'Best Validation Loss' : min(history.history['val_loss']),
            'Best Training Accuracy' : max(history.history['accuracy']),
            'Best Training Loss' : min(history.history['loss'])

        }

        mlflow.log_metrics(metrics)

##----Make model multi-input to accomodate baseline mertics as input into the model (so that there is no need to normalize data)
