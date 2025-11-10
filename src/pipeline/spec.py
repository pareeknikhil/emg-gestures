import os

import librosa
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from tabulate import tabulate

from configs.constants import TIMESTEPS

from ..utils.tfrecord_utils import write_tfrecord
from .gesture_embedding import load_dataset

tfrecord_path = os.environ.get('TFRECORD_PATH')
spec_path = os.environ.get('SPEC_EMG_FILE')

def get_hann_window(window_size, skew=True) -> np.ndarray:
    hann = np.hanning(window_size)
    if skew:
        skew_factor = np.linspace(0, 10, window_size)
        skewed_window = hann * np.exp(skew_factor - 2)
        hann /= np.max(skewed_window)
    return hann

def stft_color(slices, min_db=-5, max_db=10):
    slices = librosa.amplitude_to_db(slices)
    slices = slices.clip(min_db, max_db)
    slices = (slices-min_db) / (max_db-min_db)
    # slices = COLOR_MAP(slices)
    # slices = (slices * 255).astype("u1")
    # return slices[:, :, :3]
    return slices


HANN_WINDOW = get_hann_window(window_size=TIMESTEPS, skew=True)
COLOR_MAP = cm.get_cmap(name='inferno')

def sample_fn(window, label):
    tf.print("hello")
    return label

def spectogram(selected_type) -> None:
    raw_dataset = load_dataset(selected_type)
    new_dataset = raw_dataset.map(sample_fn)
    for elm in new_dataset:
        print(elm.numpy())
        # window = window.numpy()
        # slices = np.fft.rfft(window*HANN_WINDOW, axis=1)
        # new_slice = stft_color(slices)
        # print(new_slice.shape)
        # new_window = tf.convert_to_tensor(new_slice, dtype=np.float32)

        # write_tfrecord(dataset=(new_window, label), filename=f"{tfrecord_path}/{selected_type}/{spec_path}")


def print_spectogram_sample_collected(selected_type) -> None:
    print_list = []
    headers = ["Class", "Num of samples"]
    file_name = f"{tfrecord_path}/{selected_type}/{spec_path}"
    dataset = tf.data.TFRecordDataset(file_name)
    print_list.append(["All", sum(1 for _ in dataset)])
    print(tabulate(tabular_data=print_list, headers=headers, tablefmt="grid"))
