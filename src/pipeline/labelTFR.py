import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tabulate import tabulate
from tensorflow.keras import layers

from configs.constants import ML_WINDOW_OVERLAP, TIMESTEPS

from ..utils.tfrecord_utils import (get_all_files, get_all_labels,
                                    write_tfrecord)
from ..visualizer.source import Source

csv_path = os.environ.get('CSV_PATH')
tfrecord_path = os.environ.get('TFRECORD_PATH')
mapping_path = os.environ.get('LABEL_IDX_MAPPING')

string_lookup = None

num_emg_channels = Source.get_num_emg_channels()

def parse(line):
    features = tf.io.decode_csv(records=line, record_defaults=[0.0]*8, field_delim='\t')
    return tf.stack(values=features)

def add_label(features, filepath):
    path_parts = tf.strings.split(input=filepath, sep=os.sep)
    label = path_parts[-2]
    label_one_hot =  tf.squeeze(input=string_lookup(label), axis=0)
    return features, label_one_hot

def load_and_parse_and_window_csv(filepath):
    csv_ds = tf.data.TextLineDataset(filenames=filepath)
    parse_ds = csv_ds.map(map_func=parse, num_parallel_calls=1, deterministic=True) ##explicit parameter 1 to avoid parallel process to maintain sequence order
    window_ds = parse_ds.window(size=TIMESTEPS, shift=ML_WINDOW_OVERLAP, drop_remainder=True)
    input_ds = window_ds.flat_map(map_func=lambda window: window.batch(TIMESTEPS, drop_remainder=True))
    return input_ds.map(map_func=lambda features: add_label(features=features, filepath=filepath))

def create_window(selected_type) -> None:
    global string_lookup
    label_classes = get_all_labels(selected_type)
    string_lookup = layers.StringLookup(vocabulary= label_classes, output_mode='one_hot', num_oov_indices=0)
    index_to_label = {i: str(object=label) for i, label in enumerate(iterable=string_lookup.get_vocabulary())}

    with open(file=mapping_path, mode='w') as f:
        json.dump(obj=index_to_label, fp=f, indent=2)

    for folder in label_classes:
        file_pattern = f'{csv_path}/{selected_type}/{folder}/*.csv'
        file_ds = get_all_files(pattern=file_pattern, shuffle_flag=True)
        windowed_ds = file_ds.interleave(load_and_parse_and_window_csv, cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE)
        write_tfrecord(windowed_ds, f'{tfrecord_path}/{selected_type}/{folder}.tfrecord')

def print_sample_collected(selected_type) -> None:
    print_list = []
    headers = ["Class", "Num of samples"]
    label_classes = get_all_labels(selected_type)
    for folder in label_classes:
        file_name = f'{tfrecord_path}/{selected_type}/{folder}.tfrecord'
        dataset = tf.data.TFRecordDataset(filenames=file_name)
        print_list.append([folder, sum(1 for _ in dataset)])
    print(tabulate(tabular_data=print_list, headers=headers, tablefmt="grid"))