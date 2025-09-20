import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
import yaml
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from tensorboard.plugins import projector
from tensorflow.keras import (Input, Model, callbacks, layers, models,
                              optimizers)

from configs.constants import (BATCH_SIZE, BUFFER_SIZE, EPOCHS, LATENT_DIM,
                               LEARNING_RATE, ML_WINDOW_OVERLAP,
                               TEST_BATCH_SIZE, TIMESTEPS)

from ..utils.tfrecord_utils import get_num_labels
from ..visualizer.source import Source
from .deleteTFR import delete_old_files

tfrecord_path = os.environ.get('TFRECORD_PATH')
scaled_path = os.environ.get('SCALED_EMG_FILE')
log_path = os.environ.get('LOG_PATH')
embedding_model_path = os.environ.get("EMBEDDING_MODEL_PATH")
artifact_path = os.environ.get('ARTIFACTS_PATH')
mlflow_path = os.environ.get('MLRUNS_PATH')
dvc_path = os.environ.get('DVC_PATH')
depedency_path = os.environ.get('DEPENDENCY_FILE')
embedding_visualization_path = os.environ.get('EMBEDDING_TENSORBOARD')
label_idx_mapping_path = os.environ.get('LABEL_IDX_MAPPING')

num_emg_channels = Source.get_num_emg_channels()

def parse_tfrecord_fn(example_proto):
    feature_description = {
        'sequence': tf.io.FixedLenFeature([], tf.string), ## add default_value field : [TECH DEBT]
        'label': tf.io.FixedLenFeature([get_num_labels()], tf.int64) ## add default value field : [TECH DEBT] 
    }
    parsed_example = tf.io.parse_single_example(serialized=example_proto, features=feature_description)
    window = tf.io.parse_tensor(serialized=parsed_example['sequence'], out_type=tf.float32)
    label = parsed_example['label']
    window.set_shape([TIMESTEPS, num_emg_channels])
    label.set_shape([get_num_labels()])
    return window, window

def clear_tensorboard_logs() -> None:
    delete_old_files(selected_type=["train", "validation"], parent_path=log_path, file_type="*.v2")

def clear_artifacts_logs() -> None:
    folder_path = Path(artifact_path)
    if folder_path.is_dir():
        try:
            for file in folder_path.glob(pattern="*.[png txt]"):
                file.unlink()
                print(f"Deleted: {file}")
        except OSError as e:
            print(f"Error deleting folder {folder_path}: {e}")
    else:
        print(f"Folder {folder_path} does not exists")
    delete_old_files(selected_type=["train", "validation"], parent_path=log_path, file_type="*.v2")

def load_tfrecord(filename):
    raw_dataset = tf.data.TFRecordDataset([filename])
    return raw_dataset.map(parse_tfrecord_fn)

def load_dataset(selected_type):
    return load_tfrecord(filename=f"{tfrecord_path}/{selected_type}/{scaled_path}")

def parse_tfrecord_fn_test(example_proto):
    feature_description = {
        'sequence': tf.io.FixedLenFeature([], tf.string), ## add default_value field : [TECH DEBT]
        'label': tf.io.FixedLenFeature([get_num_labels()], tf.int64) ## add default value field : [TECH DEBT] 
    }
    parsed_example = tf.io.parse_single_example(serialized=example_proto, features=feature_description)
    window = tf.io.parse_tensor(serialized=parsed_example['sequence'], out_type=tf.float32)
    label = parsed_example['label']
    window.set_shape([TIMESTEPS, num_emg_channels])
    label.set_shape([get_num_labels()])
    return window, label

def load_tfrecord_test(filename):
    raw_dataset = tf.data.TFRecordDataset([filename])
    return raw_dataset.map(parse_tfrecord_fn_test)

def load_dataset_test(selected_type):
    return load_tfrecord_test(filename=f"{tfrecord_path}/{selected_type}/{scaled_path}")

def read_dvc_version():
    try:
        with open(dvc_path, 'r') as f:
            dvc_meta = yaml.safe_load(f)
        dataset_version = dvc_meta['outs'][0]['md5']
    except Exception as e:
        print('Failed to read DVC File:', e)
        dataset_version = 'XXX'
    return dataset_version

def run_embedding() -> None:
    mlflow.set_tracking_uri(uri=mlflow_path)
    mlflow.set_experiment(experiment_name='emg-auto-encoder-embedding')
    with mlflow.start_run() as run:
        clear_tensorboard_logs()
        # clear_artifacts_logs()

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

        test_ds = load_dataset_test(selected_type='test')
        test_ds = (test_ds
            .batch(batch_size=TEST_BATCH_SIZE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            )

        callback_list = [
            callbacks.ModelCheckpoint(filepath=embedding_model_path, monitor="val_loss", save_best_only=True),
            callbacks.TensorBoard(log_dir=log_path),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=10, min_lr=0.00001, verbose=1)
        ]

        inputs = Input(shape=(TIMESTEPS, num_emg_channels))

        #Encoder
        encoded = layers.LSTM(100, return_sequences=True)(inputs)
        encoded = layers.LSTM(LATENT_DIM, return_sequences=False)(encoded)

        #Decoder
        decoded = layers.RepeatVector(TIMESTEPS)(encoded)
        decoded = layers.LSTM(30, return_sequences=True)(decoded)   # symmetric
        decoded = layers.LSTM(100, return_sequences=True)(decoded)
        outputs = layers.TimeDistributed(layers.Dense(units=num_emg_channels, activation=None))(decoded)

        encoder_model = Model(inputs, encoded)

        recurrent_autoencoder = Model(inputs, outputs)
        recurrent_autoencoder.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')


        with open(f'{artifact_path}/model_summary.txt', 'w') as f:
            recurrent_autoencoder.summary(print_fn=lambda x: f.write(x + '\n'))

        mlflow.log_artifact(f'{artifact_path}/model_summary.txt')
        mlflow_dataset_train = mlflow.data.tensorflow_dataset.from_tensorflow(train_ds.take(1), digest=read_dvc_version())
        mlflow.log_input(dataset=mlflow_dataset_train, context='training')

        history = recurrent_autoencoder.fit(train_ds, epochs=EPOCHS, steps_per_epoch=train_steps_per_epoch, validation_data=validation_ds, validation_steps=validation_steps_per_epoch, callbacks=callback_list)

        model_input = Schema(inputs=[TensorSpec(type=np.dtype(np.float32), shape= (-1, TIMESTEPS, num_emg_channels), name="EMG_Channels_1_to_8 (OpenBCI)")])
        model_output = Schema(inputs=[TensorSpec(type=np.dtype(np.float32), shape= (-1, TIMESTEPS, num_emg_channels), name="Auto-encoder")])

        model_signature = ModelSignature(inputs=model_input, outputs=model_output)
        best_model = models.load_model(embedding_model_path)

        mlflow.keras.log_model(best_model, 'best_model', pip_requirements = depedency_path, signature=model_signature)

        params = {
            'batch_size' : BATCH_SIZE,
            'sequence_length/timesteps' : TIMESTEPS,
            'input_dimension' : num_emg_channels,
            'window_overlap' : ML_WINDOW_OVERLAP,
            'epochs' : EPOCHS,
            'optimizer_name' : 'adam',
            'learning_rate' : LEARNING_RATE,
            'buffer_size' : BUFFER_SIZE,
            'ReduceOnPlateau' : 'monitor:val_loss, factor:0.5, patience:10, min_lr:0.00001, verbose:1',
            'number_of_classes' : get_num_labels(),
            'train_size' : train_ds_size,
            'validate_size' : validation_ds_size,
        }

        mlflow.log_params(params=params)

        metrics = {
            'best_val_loss' : min(history.history['val_loss']),
            'best_train_loss' : min(history.history['loss'])
        }

        mlflow.log_metrics(metrics)

        with open(label_idx_mapping_path) as file:
            label_idx_map = json.load(file)

        keys = tf.constant([int(k) for k in label_idx_map.keys()], dtype=tf.int64)
        values = tf.constant(list(label_idx_map.values()), dtype=tf.string)

        table = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(keys, values),
                                          default_value="UNKNOWN")

        def map_labels(window, label):
            idx = tf.argmax(label, axis=-1)
            gesture = table.lookup(idx)
            latent_vector = encoder_model(window, training=False)
            return latent_vector, gesture

        mapped_ds = test_ds.map(map_labels)

        test_labels = []
        test_embeddings = []

        for embedding, label in mapped_ds:
            test_embeddings.append(embedding.numpy())
            test_labels.extend([tf.compat.as_str_any(l.numpy()) for l in label])

        test_embeddings = np.concatenate(test_embeddings, axis=0)

        metadata_path = os.path.join(embedding_visualization_path, "metadata.tsv")

        with open(metadata_path, "w") as f:
            for label in test_labels:
                f.write(f"{label}\n")

        embedding_var = tf.Variable(test_embeddings, name="latent_embeddings")

        checkpoint = tf.train.Checkpoint(embedding=embedding_var)
        checkpoint.save(os.path.join(embedding_visualization_path, "embedding.ckpt"))

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.metadata_path = "metadata.tsv"  # relative path to log_dir
        projector.visualize_embeddings(embedding_visualization_path, config)

        mlflow.log_artifacts(local_dir=log_path, artifact_path='tensorboard_logs')
