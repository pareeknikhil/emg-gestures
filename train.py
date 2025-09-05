import os
from gc import disable
from pathlib import Path

import mlflow
import tensorflow as tf
from tensorflow.keras import Input, Model, callbacks, layers, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_root():
    return Path(__file__).parents[0]

def parse_tfrecord_fn(example_proto):
    feature_description = {
        'sequence': tf.io.FixedLenFeature([], tf.string), ## add default_value field : [TECH DEBT]
        'label': tf.io.FixedLenFeature([12], tf.int64) ## add default value field : [TECH DEBT] 
    }
    parsed_example = tf.io.parse_single_example(serialized=example_proto, features=feature_description)
    window = tf.io.parse_tensor(serialized=parsed_example['sequence'], out_type=tf.float32)
    label = parsed_example['label']
    window.set_shape([100, 8])
    label.set_shape([12])
    return window, label

def load_tfrecord(path):
    raw_dataset = tf.data.TFRecordDataset([path])
    return raw_dataset.map(parse_tfrecord_fn)

def load_dataset(path):
    return load_tfrecord(path)

# set mlflow tracking uri
mlflow.set_tracking_uri(uri=(get_root() / 'mlruns').as_uri())
mlflow.set_experiment(experiment_name='emg-lstm')

# mlflow.tensorflow.autolog()

train_path = 'data/tfrecords/test/EMGdata.tfrecord'
validate_path = 'data/tfrecords/validate/EMGdata.tfrecord'

train_ds_size = sum(1 for _ in load_dataset(train_path))
validate_ds_size = sum(1 for _ in load_dataset(validate_path))

batch_size = 128
epoch = 5
train_steps_per_epoch = int(train_ds_size // batch_size)
validation_steps_per_epoch = int(validate_ds_size // batch_size)

train_ds = load_dataset(path=train_path).shuffle(1000).repeat(epoch).batch(batch_size=batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
validate_ds = load_dataset(path=validate_path).shuffle(1000).repeat(epoch).batch(batch_size=batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE) 


with mlflow.start_run(run_name='train_v1') as run:
    inputs = Input(shape=(100, 8))
    dense = layers.TimeDistributed(layers.Dense(10, activation='relu'))(inputs)
    lstm = layers.LSTM(10)(dense)
    outputs = layers.Dense(12, activation='softmax')(lstm)
    model = Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])    
    model.fit(train_ds, epochs=epoch, steps_per_epoch=train_steps_per_epoch, validation_data=validate_ds, validation_steps= validation_steps_per_epoch)
    with open("model_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    mlflow.log_artifact("model_summary.txt")
    
    mlflow.keras.log_model(model, artifact_path="model")
    