import os

import mlflow
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from xgboost import train

from configs.constants import (BATCH_SIZE, LATENT_DIM, ML_WINDOW_OVERLAP,
                               NEG_DIST, POS_NEIGHBOR, TIMESTEPS)

from ..utils.tfrecord_utils import (PerWindowNormalization, get_all_files,
                                    get_all_labels, print_dataset_size)
from ..visualizer.source import Source
from .deleteTFR import delete_old_files

csv_path = os.environ.get('CSV_PATH')
log_path = os.environ.get('LOG_PATH')
mlflow_path = os.environ.get('MLRUNS_PATH')
depedency_path = os.environ.get('DEPENDENCY_FILE')
dvc_path = os.environ.get('DVC_PATH')


selected_type = 'train'
num_emg_channels = Source.get_num_emg_channels()

def clear_tensorboard_logs() -> None:
    delete_old_files(selected_type=["train", "validation"], parent_path=log_path, file_type="*.v2")

def load_and_parse_window_csv(filepath):
    text = tf.io.read_file(filepath)
    lines = tf.strings.split(text, sep='\n')
    lines = tf.boolean_mask(lines, tf.strings.length(lines) > 0)
    records = tf.io.decode_csv(
        lines,
        record_defaults=[0.0]*8,
        field_delim='\t'
        )
    all_rows = tf.stack(records, axis=1)  # shape [T, C]
    windows = tf.signal.frame(all_rows, TIMESTEPS, ML_WINDOW_OVERLAP, axis=0)
    num_windows = tf.shape(windows)[0]
    indices = tf.range(num_windows)

    pos_idx = tf.clip_by_value(
        indices + tf.random.uniform([num_windows], -POS_NEIGHBOR, POS_NEIGHBOR+1, dtype=tf.int32),
        0, num_windows-1
        )

    def sample_neg(idx):
        valid = tf.concat([
            tf.range(0, tf.maximum(0, idx-NEG_DIST)),
            tf.range(tf.minimum(num_windows, idx+NEG_DIST+1), num_windows)
        ], axis=0)
        neg_idx = valid[tf.random.uniform([], 0, tf.shape(valid)[0], dtype=tf.int32)]
        return windows[neg_idx]

    anchors = windows
    positives = tf.gather(windows, pos_idx)
    negatives = tf.map_fn(sample_neg, indices, fn_output_signature=tf.float32)

    return tf.data.Dataset.from_tensor_slices((anchors, positives, negatives))


def run_clr() -> None:
    mlflow.set_tracking_uri(uri=mlflow_path)
    mlflow.set_experiment(experiment_name='emg-clr-embedding')
    with mlflow.start_run() as run:
        clear_tensorboard_logs()

        # ---------------- DATASET PIPELINE ----------------
        def load_fileset(selected_type):
            label_classes = get_all_labels(selected_type)
            file_patterns = [f'{csv_path}/{selected_type}/{folder}/*.csv' for folder in label_classes]
            file_ds = get_all_files(pattern=file_patterns, shuffle_flag=False)
            return file_ds
        # ---------------- ---------------- ----------------

        train_file_ds = load_fileset('train')
        train_samples_ds = (
            train_file_ds
            .flat_map(load_and_parse_window_csv)  # one file at a time
        )

        print_dataset_size(train_samples_ds, 'train')

        train_ds = (
            train_samples_ds
            .shuffle(buffer_size=4096)
            # .repeat()
            .batch(batch_size=BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

        validation_file_ds = load_fileset('validate')
        validation_samples__ds = (
            validation_file_ds
            .flat_map(load_and_parse_window_csv)  # one file at a time
        )

        print_dataset_size(validation_file_ds, 'validate')

        validation_ds = (
            validation_samples__ds
            .batch(batch_size=BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

        ## check if the its working well end-to-end

        train_steps_per_epoch = sum(1 for _ in train_ds)
        validation_steps_per_epoch = sum(1 for _ in validation_ds)

    # ---------------- ENCODER ----------------
        def build_encoder(input_shape=(TIMESTEPS, num_emg_channels), latent_dim=LATENT_DIM):
            inputs = layers.Input(shape=input_shape)
            normalize = PerWindowNormalization()(inputs)
            x = layers.Conv1D(64, 5, strides=2, activation='relu', padding='same')(normalize)
            x = layers.Conv1D(128, 3, strides=2, activation='relu', padding='same')(x)
            x = layers.GlobalAveragePooling1D()(x)
            # x = layers.LSTM(64, return_sequences=True)(x)
            # x = layers.LSTM(32)(x)
            x = layers.Dense(latent_dim)(x)
            out = layers.Lambda(lambda z: tf.math.l2_normalize(z, axis=-1))(x)
            return Model(inputs, out)

        embedding_model = build_encoder()

        def triplet_loss(anchor, positive, negative, margin=0.5):
            pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
            neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
            loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
            return tf.reduce_mean(loss)

        class TripletModel(tf.keras.Model):
            def __init__(self, embedding_model, margin=0.5):
                super().__init__()
                self.embedding_model = embedding_model
                self.margin = margin
                self.loss_tracker = tf.keras.metrics.Mean(name="loss")
                self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")

            def call(self, inputs):
                a, p, n = inputs
                a_embed = self.embedding_model(a)
                p_embed = self.embedding_model(p)
                n_embed = self.embedding_model(n)
                return (a_embed, p_embed, n_embed)

            def train_step(self, data):
                with tf.GradientTape() as tape:
                    a_embed, p_embed, n_embed = self(data, training=True)
                    loss = triplet_loss(a_embed, p_embed, n_embed, self.margin)
                grads = tape.gradient(loss, self.embedding_model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.embedding_model.trainable_variables))
                self.loss_tracker.update_state(loss)
                return {"loss": self.loss_tracker.result()}

            def test_step(self, data):
                a_embed, p_embed, n_embed = self(data, training=False)
                loss = triplet_loss(a_embed, p_embed, n_embed, self.margin)
                self.val_loss_tracker.update_state(loss)
                return {"loss": self.val_loss_tracker.result()}

            @property
            def metrics(self):
                return [self.loss_tracker, self.val_loss_tracker]

        model = TripletModel(embedding_model)
        model.compile(optimizer = tf.keras.optimizers.Adam(0.01))

        history = model.fit(
            train_ds,
            epochs=5,
            steps_per_epoch=train_steps_per_epoch,
            validation_data = validation_ds,
            validation_steps=validation_steps_per_epoch
            )