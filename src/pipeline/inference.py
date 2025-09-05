import os

import pandas as pd
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.config import enable_unsafe_deserialization

from configs.constants import BATCH_SIZE

from ..utils.tfrecord_utils import get_all_labels
from ..visualizer.source import Source
from .combineTFR import parse_tfrecord_fn

enable_unsafe_deserialization()

num_emg_channels = Source.get_num_emg_channels()

ml_model_path = os.environ.get("ML_MODEL_PATH")
tfrecord_path = os.environ.get('TFRECORD_PATH')

def run_test() -> None:
    model = models.load_model(ml_model_path)

    BATCH_SIZE = 1

    ##----Accuracy-----
    print("***************Test**********")
    test_ds_size = sum(1 for _ in load_dataset(selected_type="test"))
    test_steps_per_epoch = int(test_ds_size // BATCH_SIZE)

    test_ds = load_dataset(selected_type="test")

    test_ds = (test_ds
        .shuffle(buffer_size=1000, seed=42, reshuffle_each_iteration=False)
        .batch(batch_size=BATCH_SIZE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

    model.evaluate(test_ds, steps=test_steps_per_epoch)

    #----Confusion matrix-----
    test_ds = load_dataset(selected_type="test").shuffle(buffer_size=1000)

    test_pred, test_label = [], []

    for feature, label in test_ds:
        logits = model(feature[None, ...])
        test_pred.append(tf.argmax(input=logits, axis=-1))
        test_label.append(tf.argmax(input=label, axis=-1))


    label_classes = get_all_labels(selected_type="train")

    cm = tf.math.confusion_matrix(labels=test_label, predictions=test_pred).numpy()

    df_cm = pd.DataFrame(data=cm,
                        index=[f'Actual: {label}' for label in label_classes],
                        columns=[f'Pred: {label}' for label in label_classes])

    print(df_cm.to_string())

def load_tfrecord(filename):
    raw_dataset = tf.data.TFRecordDataset([filename])
    return raw_dataset.map(parse_tfrecord_fn)

def load_dataset(selected_type):
    return load_tfrecord(filename=f"{tfrecord_path}/{selected_type}/EMGdata.tfrecord")