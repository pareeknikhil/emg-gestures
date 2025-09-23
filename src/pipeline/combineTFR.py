import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tabulate import tabulate

from configs.constants import TIMESTEPS

from ..utils.tfrecord_utils import (get_all_files, get_num_labels,
                                    write_tfrecord)
from ..visualizer.source import Source

tfrecord_path = os.environ.get('TFRECORD_PATH')
combined_emg_file = os.environ.get("COMBINED_EMG_FILE")

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
    return window, label

def combine_labels(selected_type) -> None:
    file_pattern = f'{tfrecord_path}/{selected_type}/*.tfrecord'
    file_ds = get_all_files(pattern=file_pattern, shuffle_flag=False)

    dataset = tf.data.TFRecordDataset(filenames=file_ds, num_parallel_reads=get_num_labels())
    combined_dataset = dataset.map(map_func=parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE) ## also return dataset size here when returning transformed dataset (num_samples) : [TECH DEBT]

    write_tfrecord(dataset=combined_dataset, filename=f"{tfrecord_path}/{selected_type}/{combined_emg_file}")

def print_combine_sample_collected(selected_type) -> None:
    print_list = []
    headers = ["Class", "Num of samples"]
    file_name = f"{tfrecord_path}/{selected_type}/{combined_emg_file}"
    dataset = tf.data.TFRecordDataset(file_name)
    print_list.append(["All", sum(1 for _ in dataset)])
    print(tabulate(tabular_data=print_list, headers=headers, tablefmt="grid"))
