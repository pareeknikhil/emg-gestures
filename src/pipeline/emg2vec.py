import csv
import json
import os

import mlflow
import numpy as np
import tensorflow as tf
import yaml
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from tensorboard.plugins import projector
from tensorflow.keras import Model, callbacks, layers

from configs.constants import (BATCH_SIZE, BUFFER_SIZE, EPOCHS, LATENT_DIM,
                               LEARNING_RATE, ML_WINDOW_OVERLAP, NEG_DIST,
                               POS_NEIGHBOR, TEST_BATCH_SIZE, TIMESTEPS)

from ..utils.tfrecord_utils import (PerWindowNormalization, get_all_files,
                                    get_all_labels, get_num_labels,
                                    print_dataset_size)
from ..visualizer.source import Source
from .combineTFR import parse_tfrecord_fn
from .deleteTFR import delete_old_files

csv_path = os.environ.get('CSV_PATH')
log_path = os.environ.get('LOG_PATH')
mlflow_path = os.environ.get('MLRUNS_PATH')
depedency_path = os.environ.get('DEPENDENCY_FILE')
dvc_path = os.environ.get('DVC_PATH')
embedding_model_path = os.environ.get("EMBEDDING_MODEL_PATH")
artifact_path = os.environ.get('ARTIFACTS_PATH')
label_idx_mapping_path = os.environ.get('LABEL_IDX_MAPPING')
tfrecord_path = os.environ.get('TFRECORD_PATH')
combined_emg_path = os.environ.get('COMBINED_EMG_FILE')
embedding_visualization_path = os.environ.get('EMBEDDING_TENSORBOARD')


selected_type = 'train'
num_emg_channels = Source.get_num_emg_channels()

def clear_tensorboard_logs() -> None:
    delete_old_files(selected_type=["train", "validation"], parent_path=log_path, file_type="*.v2")

def load_and_parse_window_csv(filepath):
    text = tf.io.read_file(filepath)
    lines = tf.strings.split(text, sep='\n')
    lines = tf.boolean_mask(lines, tf.strings.length(lines) > 0)
    records = tf.io.decode_csv(
        lines,
        record_defaults=[0.0]*8,
        field_delim='\t'
        )
    all_rows = tf.stack(records, axis=1)  # shape [T, C]
    windows = tf.signal.frame(all_rows, TIMESTEPS, ML_WINDOW_OVERLAP, axis=0)
    num_windows = tf.shape(windows)[0]
    indices = tf.range(num_windows)

    pos_idx = tf.clip_by_value(
        indices + tf.random.uniform([num_windows], -POS_NEIGHBOR, POS_NEIGHBOR+1, dtype=tf.int32),
        0, num_windows-1
        )

    def sample_neg(idx):
        valid = tf.concat([
            tf.range(0, tf.maximum(0, idx-NEG_DIST)),
            tf.range(tf.minimum(num_windows, idx+NEG_DIST+1), num_windows)
        ], axis=0)
        neg_idx = valid[tf.random.uniform([], 0, tf.shape(valid)[0], dtype=tf.int32)]
        return windows[neg_idx]

    anchors = windows
    positives = tf.gather(windows, pos_idx)
    negatives = tf.map_fn(sample_neg, indices, fn_output_signature=tf.float32)

    return tf.data.Dataset.from_tensor_slices((anchors, positives, negatives))

def load_tfrecord(filename):
    raw_dataset = tf.data.TFRecordDataset([filename])
    return raw_dataset.map(parse_tfrecord_fn)

def load_dataset(selected_type):
    return load_tfrecord(filename=f"{tfrecord_path}/{selected_type}/{combined_emg_path}")

def read_dvc_version():
    try:
        with open(dvc_path, 'r') as f:
            dvc_meta = yaml.safe_load(f)
        dataset_version = dvc_meta['outs'][0]['md5']
    except Exception as e:
        print('Failed to read DVC File:', e)
        dataset_version = 'XXX'
    return dataset_version

def run_clr() -> None:
    mlflow.set_tracking_uri(uri=mlflow_path)
    mlflow.set_experiment(experiment_name='emg-clr-embedding')
    with mlflow.start_run() as run:
        clear_tensorboard_logs()

        # ---------------- DATASET PIPELINE ----------------
        def load_fileset(selected_type):
            label_classes = get_all_labels(selected_type)
            file_patterns = [f'{csv_path}/{selected_type}/{folder}/*.csv' for folder in label_classes]
            file_ds = get_all_files(pattern=file_patterns, shuffle_flag=False)
            return file_ds
        # ---------------- ---------------- ----------------

        train_file_ds = load_fileset('train')
        train_samples_ds = (
            train_file_ds
            .flat_map(load_and_parse_window_csv)  # one file at a time
        )

        train_ds_size = print_dataset_size(train_samples_ds, 'train')

        train_ds = (
            train_samples_ds
            .shuffle(buffer_size=4096)
            # .repeat()
            .batch(batch_size=BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

        validation_file_ds = load_fileset('validate')
        validation_samples_ds = (
            validation_file_ds
            .flat_map(load_and_parse_window_csv)  # one file at a time
        )

        validation_ds_size = print_dataset_size(validation_samples_ds, 'validate')

        validation_ds = (
            validation_samples_ds
            .batch(batch_size=BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

        test_ds = load_dataset(selected_type='test')
        test_ds = (test_ds
            .batch(batch_size=TEST_BATCH_SIZE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            )

        test_ds_size = print_dataset_size(load_dataset(selected_type='test'), 'test')

        ## check if the its working well end-to-end

        train_steps_per_epoch = sum(1 for _ in train_ds)
        validation_steps_per_epoch = sum(1 for _ in validation_ds)

        callback_list = [
                callbacks.ModelCheckpoint(filepath=embedding_model_path, monitor="val_loss", save_best_only=True),
                callbacks.TensorBoard(log_dir=log_path),
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience=10, min_lr=0.00001, verbose=1)
        ]

    # ---------------- ENCODER ----------------
        def build_encoder(input_shape=(TIMESTEPS, num_emg_channels), latent_dim=LATENT_DIM):
            inputs = layers.Input(shape=input_shape)
            normalize = PerWindowNormalization()(inputs)
            x = layers.Conv1D(64, 5, strides=2, activation='relu', padding='same')(normalize)
            x = layers.Conv1D(128, 3, strides=2, activation='relu', padding='same')(x)
            x = layers.GlobalAveragePooling1D()(x)
            # x = layers.LSTM(64, return_sequences=True)(x)
            # x = layers.LSTM(32)(x)
            x = layers.Dense(latent_dim)(x)
            out = layers.Lambda(lambda z: tf.math.l2_normalize(z, axis=-1))(x)
            return Model(inputs, out)

        embedding_model = build_encoder()

        def triplet_loss(anchor, positive, negative, margin=0.5):
            pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
            neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
            loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
            return tf.reduce_mean(loss)

        class TripletModel(tf.keras.Model):
            def __init__(self, embedding_model, margin=0.5):
                super().__init__()
                self.embedding_model = embedding_model
                self.margin = margin
                self.loss_tracker = tf.keras.metrics.Mean(name="loss")
                self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")

            def call(self, inputs):
                a, p, n = inputs
                a_embed = self.embedding_model(a)
                p_embed = self.embedding_model(p)
                n_embed = self.embedding_model(n)
                return (a_embed, p_embed, n_embed)

            def train_step(self, data):
                with tf.GradientTape() as tape:
                    a_embed, p_embed, n_embed = self(data, training=True)
                    loss = triplet_loss(a_embed, p_embed, n_embed, self.margin)
                grads = tape.gradient(loss, self.embedding_model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.embedding_model.trainable_variables))
                self.loss_tracker.update_state(loss)
                return {"loss": self.loss_tracker.result()}

            def test_step(self, data):
                a_embed, p_embed, n_embed = self(data, training=False)
                loss = triplet_loss(a_embed, p_embed, n_embed, self.margin)
                self.val_loss_tracker.update_state(loss)
                return {"loss": self.val_loss_tracker.result()}

            @property
            def metrics(self):
                return [self.loss_tracker, self.val_loss_tracker]

        model = TripletModel(embedding_model)
        model.compile(optimizer = tf.keras.optimizers.Adam(0.01))

        with open(f'{artifact_path}/model_summary.txt', 'w') as f:
            embedding_model.summary(print_fn=lambda x: f.write(x + '\n'))

        mlflow.log_artifact(f'{artifact_path}/model_summary.txt')
        mlflow_dataset_train = mlflow.data.tensorflow_dataset.from_tensorflow(train_ds.take(1), digest=read_dvc_version())
        mlflow.log_input(dataset=mlflow_dataset_train, context='training')

        params = {
            'batch_size' : BATCH_SIZE,
            'sequence_length/timesteps' : TIMESTEPS,
            'input_dimension' : num_emg_channels,
            'window_overlap' : ML_WINDOW_OVERLAP,
            'epochs' : EPOCHS,
            'optimizer_name' : 'adam',
            'learning_rate' : LEARNING_RATE,
            'buffer_size' : BUFFER_SIZE,
            'ReduceOnPlateau' : 'monitor:val_loss, factor:0.5, patience:10, min_lr:0.00001, verbose:1',
            'number_of_classes' : get_num_labels(),
            'train_size' : train_ds_size,
            'validate_size' : validation_ds_size,
            'test_size' : test_ds_size
        }

        mlflow.log_params(params=params)

        history = model.fit(
            train_ds,
            epochs=5,
            steps_per_epoch=train_steps_per_epoch,
            validation_data = validation_ds,
            validation_steps=validation_steps_per_epoch,
            callbacks=callback_list
            )
        

        model_input = Schema(inputs=[TensorSpec(type=np.dtype(np.float32), shape= (-1, TIMESTEPS, num_emg_channels), name="EMG_Channels_1_to_8 (OpenBCI)")])
        model_output = Schema(inputs=[TensorSpec(type=np.dtype(np.float32), shape= (-1, LATENT_DIM), name="Embedding")])

        model_signature = ModelSignature(inputs=model_input, outputs=model_output)

        mlflow.keras.log_model(embedding_model, 'EMG2Vec', pip_requirements = depedency_path, signature=model_signature)

        metrics = {
            'best_val_loss' : min(history.history['val_loss']),
            'best_train_loss' : min(history.history['loss'])
        }

        mlflow.log_metrics(metrics)

        with open(label_idx_mapping_path) as file:
            label_idx_map = json.load(file)

        keys = tf.constant([int(k) for k in label_idx_map.keys()], dtype=tf.int64)
        values = tf.constant(list(label_idx_map.values()), dtype=tf.string)

        table = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(keys, values),
                                          default_value="UNKNOWN")

        def map_labels(window, label):
            idx = tf.argmax(label, axis=-1)
            gesture = table.lookup(idx)
            latent_vector = embedding_model(window, training=False)
            return latent_vector, gesture

        mapped_ds = test_ds.map(map_labels)

        test_labels = []
        test_embeddings = []

        for embedding, label in mapped_ds:
            test_embeddings.append(embedding.numpy())
            test_labels.extend([tf.compat.as_str_any(l.numpy()) for l in label])

        test_embeddings = np.concatenate(test_embeddings, axis=0)

        metadata_path = os.path.join(embedding_visualization_path, "metadata.tsv")

        with open(metadata_path, "w") as f:
            for label in test_labels:
                f.write(f"{label}\n")

        embedding_var = tf.Variable(test_embeddings, name="latent_embeddings")

        checkpoint = tf.train.Checkpoint(embedding=embedding_var)
        checkpoint.save(os.path.join(embedding_visualization_path, "embedding.ckpt"))

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.metadata_path = "metadata.tsv"  # relative path to log_dir
        projector.visualize_embeddings(embedding_visualization_path, config)

        mlflow.log_artifacts(local_dir=log_path, artifact_path='tensorboard_logs')

        orange_csv_path = f"{artifact_path}/embeddings_with_labels.csv"

        with open(orange_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            # header: first column is label, then embedding_0 ... embedding_{D-1}
            header = ["label"] + [f"dim_{i}" for i in range(test_embeddings.shape[1])]
            writer.writerow(header)

            for label, emb in zip(test_labels, test_embeddings):
                writer.writerow([label] + emb.tolist())

        print(f"Saved {len(test_labels)} rows to {orange_csv_path}")

        mlflow.log_artifact(orange_csv_path)