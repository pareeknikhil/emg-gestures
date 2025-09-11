import os

import tensorflow as tf
from tabulate import tabulate

from ..utils.tfrecord_utils import write_tfrecord
from .model import load_dataset

tfrecord_path = os.environ.get('TFRECORD_PATH')
scaled_path = os.environ.get('SCALED_EMG_FILE')

def standardize_window(window, label):
    mean = tf.reduce_mean(window, axis=0, keepdims=True)
    std = tf.math.reduce_std(window, axis=0, keepdims=True)
    scaled = (window - mean) / std + 1e-6
    return scaled, label


def scale(selected_type) -> None:
    unscaled_dataset = load_dataset(selected_type=selected_type)
    scaled_dataset = unscaled_dataset.map(standardize_window, num_parallel_calls=tf.data.AUTOTUNE)
    write_tfrecord(dataset=scaled_dataset, filename=f"{tfrecord_path}/{selected_type}/{scaled_path}")


def print_scaled_sample_collected(selected_type) -> None:
    print_list = []
    headers = ["Class", "Num of samples"]
    file_name = f"{tfrecord_path}/{selected_type}/{scaled_path}"
    dataset = tf.data.TFRecordDataset(file_name)
    print_list.append(["All", sum(1 for _ in dataset)])
    print(tabulate(tabular_data=print_list, headers=headers, tablefmt="grid"))
