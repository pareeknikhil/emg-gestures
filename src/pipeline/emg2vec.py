import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

from configs.constants import (BATCH_SIZE, LATENT_DIM, ML_WINDOW_OVERLAP,
                               NEG_DIST, POS_NEIGHBOR, TIMESTEPS)

from ..utils.tfrecord_utils import get_all_files, get_all_labels
from ..visualizer.source import Source

csv_path = os.environ.get('CSV_PATH')

selected_type = 'train'
num_emg_channels = Source.get_num_emg_channels()


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
    positives = tf.gather(windows, pos_idx)
    anchors = windows

    def sample_neg(idx):
        valid = tf.concat([
            tf.range(0, tf.maximum(0, idx-NEG_DIST)),
            tf.range(tf.minimum(num_windows, idx+NEG_DIST+1), num_windows)
        ], axis=0)
        neg_idx = valid[tf.random.uniform([], 0, tf.shape(valid)[0], dtype=tf.int32)]
        return windows[neg_idx]

    negatives = tf.map_fn(sample_neg, indices, fn_output_signature=tf.float32)

    ds_a = tf.data.Dataset.from_tensor_slices(anchors)
    ds_p = tf.data.Dataset.from_tensor_slices(positives)
    ds_n = tf.data.Dataset.from_tensor_slices(negatives)
    return tf.data.Dataset.zip((ds_a, ds_p, ds_n))
    # return tf.data.Dataset.from_tensor_slices((anchors, positives, negatives))


def run_clr() -> None:
    # ---------------- DATASET PIPELINE ----------------
    label_classes = get_all_labels(selected_type)
    file_patterns = [f'{csv_path}/{selected_type}/{folder}/*.csv' for folder in label_classes]
    file_ds = get_all_files(pattern=file_patterns, shuffle_flag=False)

    train_ds = (
        file_ds
        .flat_map(load_and_parse_window_csv)  # one file at a time
        .shuffle(buffer_size=4096)
        # .batch(batch_size=BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    anchor = []
    positive = []
    negative = []

    for a, p, n in train_ds:
        anchor.append(a)
        positive.append(p)
        negative.append(n)

    anchor_numpy = np.array(anchor)
    positive_numpy = np.array(positive)
    negative_numpy = np.array(negative)
    triplet_ds = tf.data.Dataset.from_tensor_slices((anchor_numpy, positive_numpy, negative_numpy))
    triplet_ds = triplet_ds.shuffle(4000).batch(BATCH_SIZE)


    # ---------------- ENCODER ----------------
    def build_encoder(input_shape=(TIMESTEPS, num_emg_channels), latent_dim=LATENT_DIM):
        inp = layers.Input(shape=input_shape)
        x = layers.Conv1D(64, 5, strides=2, activation='relu', padding='same')(inp)
        x = layers.Conv1D(128, 3, strides=2, activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling1D()(x)
        # x = layers.LSTM(64, return_sequences=True)(x)
        # x = layers.LSTM(32)(x)
        x = layers.Dense(latent_dim)(x)
        out = layers.Lambda(lambda z: tf.math.l2_normalize(z, axis=-1))(x)
        return Model(inp, out)

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
        
        def call(self, inputs):
            a, p, n = inputs
            a_embed = self.embedding_model(a)
            p_embed = self.embedding_model(p)
            n_embed = self.embedding_model(n)
            return (a_embed, p_embed, n_embed)

        def train_step(self, data):
            a, p, n = data
            with tf.GradientTape() as tape:
                a_embed, p_embed, n_embed = self((a, p, n), training=True)
                loss = triplet_loss(a_embed, p_embed, n_embed, self.margin)
            grads = tape.gradient(loss, self.embedding_model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.embedding_model.trainable_variables))
            return {"loss": loss}

    model = TripletModel(embedding_model)
    model.compile(optimizer = tf.keras.optimizers.Adam(0.01))

    history = model.fit(
        triplet_ds,
        epochs=2)