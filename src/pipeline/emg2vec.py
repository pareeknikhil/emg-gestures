import os

import tensorflow as tf

from configs.constants import (BATCH_SIZE, ML_WINDOW_OVERLAP, NEG_DIST,
                               POS_NEIGHBOR, TIMESTEPS)

from ..utils.tfrecord_utils import get_all_files, get_all_labels

csv_path = os.environ.get('CSV_PATH')

selected_type = 'train'

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
    label_classes = get_all_labels(selected_type)
    file_patterns = [f'{csv_path}/{selected_type}/{folder}/*.csv' for folder in label_classes]
    file_ds = get_all_files(pattern=file_patterns, shuffle_flag=False)
    train_ds = file_ds.take(2).flat_map(load_and_parse_window_csv)
    for x in train_ds:
        print(tf.shape(x))