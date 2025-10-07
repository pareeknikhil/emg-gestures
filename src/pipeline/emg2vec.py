import os

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
    tf.print("****")
    tf.print(filepath)
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
    return tf.data.Dataset.from_tensor_slices((anchors, positives, negatives))

def run_clr() -> None:
    # ---------------- DATASET PIPELINE ----------------
    label_classes = get_all_labels(selected_type)
    file_patterns = [f'{csv_path}/{selected_type}/{folder}/*.csv' for folder in label_classes]
    file_ds = get_all_files(pattern=file_patterns, shuffle_flag=False)

    train_ds = (
        file_ds
        .flat_map(load_and_parse_window_csv)  # one file at a time
        .shuffle(4096)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ---------------- ENCODER ----------------
    def build_encoder(input_shape=(TIMESTEPS, num_emg_channels), latent_dim=LATENT_DIM):
        inp = layers.Input(shape=input_shape)
        x = layers.Conv1D(64, 5, strides=2, activation='relu', padding='same')(inp)
        x = layers.Conv1D(128, 3, strides=2, activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(latent_dim)(x)
        out = layers.Lambda(lambda z: tf.math.l2_normalize(z, axis=-1))(x)
        return Model(inp, out)

    encoder = build_encoder()

    # ---------------- CUSTOM TRIPLET MODEL ----------------
    class TripletModel(Model):
        def __init__(self, encoder, margin=1.0):
            super().__init__()
            self.encoder = encoder
            self.margin = margin

        def call(self, inputs, training=False):
            # inputs is a list of 3 tensors from dataset: [anchor, positive, negative]
            a, p, n = inputs
            z_a = self.encoder(a, training=training)
            z_p = self.encoder(p, training=training)
            z_n = self.encoder(n, training=training)

            # triplet loss
            pos_dist = tf.reduce_sum(tf.square(z_a - z_p), axis=-1)
            neg_dist = tf.reduce_sum(tf.square(z_a - z_n), axis=-1)
            loss = tf.reduce_mean(tf.maximum(pos_dist - neg_dist + self.margin, 0.0))
            self.add_loss(loss)
            return {"anchor": z_a, "positive": z_p, "negative": z_n}

    triplet_model = TripletModel(encoder)
    triplet_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

    # ---------------- TRAIN ----------------
    triplet_model.fit(train_ds, epochs=10)