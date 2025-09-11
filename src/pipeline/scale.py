import os

import tensorflow as tf

from ..utils.tfrecord_utils import write_tfrecord
from .model import load_dataset

tfrecord_path = os.environ.get('TFRECORD_PATH')
scaled_emg_file = os.environ.get('SCALED_EMG_FILE')

def standardize_window(window, label):
    mean = tf.reduce_mean(window, axis=0, keepdims=True)
    std = tf.math.reduce_std(window, axis=0, keepdims=True)
    scaled = (window - mean) / std + 1e-6
    return scaled, label


def scale(selected_type) -> None:
    unscaled_dataset = load_dataset(selected_type=selected_type)
    scaled_dataset = unscaled_dataset.map(standardize_window, num_parallel_calls=tf.data.AUTOTUNE)
    write_tfrecord(dataset=scaled_dataset, filename=f"{tfrecord_path}/{selected_type}/{scaled_emg_file}")
    scaled_ds_size = sum(1 for _ in scaled_dataset)
    print(scaled_ds_size)


### TO DO: add print in pipeline and verify