import os
from cProfile import label

import tensorflow as tf

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
    windows = tf.signal.frame(all_rows, 1, 1, axis=0)
    tf.print(tf.shape(windows))
    return tf.data.Dataset.from_tensors(windows)

def run_clr() -> None:
    label_classes = get_all_labels(selected_type)
    file_patterns = [f'{csv_path}/{selected_type}/{folder}/*.csv' for folder in label_classes]
    file_ds = get_all_files(pattern=file_patterns, shuffle_flag=False)
    train_ds = file_ds.take(2).flat_map(load_and_parse_window_csv)
    for x in train_ds:
        print(x)
        pass
