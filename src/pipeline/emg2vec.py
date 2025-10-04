import os
from cProfile import label
from venv import create

import tensorflow as tf
from sqlalchemy import false
from xgboost import train

from ..utils.tfrecord_utils import get_all_files, get_all_labels

csv_path = os.environ.get('CSV_PATH')

selected_type = 'train'

def parse(line):
    features = tf.io.decode_csv(records=line, record_defaults=[0.0]*8, field_delim='\t')
    tf.print(tf.stack(values=features))
    return tf.stack(values=features)

def load_and_parse_window_csv(filepath):
    csv_ds = tf.data.TextLineDataset(filenames=filepath)
    parse_ds = csv_ds.map(parse, num_parallel_calls=1, deterministic=True)
    return parse_ds

def run_clr() -> None:
    label_classes = get_all_labels(selected_type)
    for folder in label_classes:
        file_pattern = f'{csv_path}/{selected_type}/{folder}/*.csv'
        file_ds = get_all_files(pattern=file_pattern, shuffle_flag=True)
        train_ds = file_ds.map(load_and_parse_window_csv, num_parallel_calls=1, deterministic=True)
        for elm in train_ds.take(1):
            pass