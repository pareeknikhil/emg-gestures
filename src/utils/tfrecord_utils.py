import os
from typing import Any

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable

csv_path = os.environ.get('CSV_PATH')

def get_all_labels(selected_type="train") -> list[str]: ## default from train
    train_path = os.path.join(csv_path, selected_type)
    return tf.io.gfile.listdir(path=train_path)

def get_num_labels(selected_type="train") -> int: ## default from train
    train_path = os.path.join(csv_path, selected_type)
    return len(tf.io.gfile.listdir(path=train_path))

def get_all_files(pattern: Any, shuffle_flag: bool) -> tf.data.Dataset:
    return tf.data.Dataset.list_files(file_pattern=pattern, shuffle=shuffle_flag)

def write_tfrecord(dataset: tf.data.Dataset, filename: str) -> None:
    with tf.io.TFRecordWriter(path=filename) as writer:
        for window, label_one_hot in dataset:
            serial_sample = serialize(window=window, label_one_hot=label_one_hot)
            writer.write(serial_sample)

def serialize(window: tf.Tensor, label_one_hot: tf.Tensor) -> bytes:
    feature = {
        'sequence': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tensor=window).numpy()])
        ),
        'label': tf.train.Feature(
            int64_list=tf.train.Int64List(value=label_one_hot)
        )
    }
    proto_message = tf.train.Example(features=tf.train.Features(feature=feature))
    return proto_message.SerializeToString()

@register_keras_serializable()
class PerWindowNormalization(Layer):
    def call(self, data: tf.Tensor) -> tf.Tensor:
        mean = tf.reduce_mean(input_tensor=data, axis=1, keepdims=True)
        std = tf.math.reduce_std(input_tensor=data, axis=1, keepdims=True) + 1e-6
        return (data - mean)/std