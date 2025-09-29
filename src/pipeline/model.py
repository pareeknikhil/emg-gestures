import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import csv
import json

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import yaml
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from tensorboard.plugins import projector
from tensorflow.keras import (Input, Model, callbacks, layers, models,
                              optimizers)

from configs.constants import (BATCH_SIZE, BUFFER_SIZE, EPOCHS, LATENT_DIM,
                               LEARNING_RATE, ML_WINDOW_OVERLAP,
                               TEST_BATCH_SIZE, TIMESTEPS)

from ..utils.tfrecord_utils import (PerWindowNormalization, get_all_labels,
                                    get_num_labels)
from ..visualizer.source import Source
from .combineTFR import parse_tfrecord_fn
from .deleteTFR import delete_old_files

tfrecord_path = os.environ.get('TFRECORD_PATH')
log_path = os.environ.get('LOG_PATH')
ml_model_path = os.environ.get("ML_MODEL_PATH")
mlflow_path = os.environ.get('MLRUNS_PATH')
combined_emg_path = os.environ.get('COMBINED_EMG_FILE')
depedency_path = os.environ.get('DEPENDENCY_FILE')
dvc_path = os.environ.get('DVC_PATH')
artifact_path = os.environ.get('ARTIFACTS_PATH')
label_idx_mapping_path = os.environ.get('LABEL_IDX_MAPPING')
embedding_visualization_path = os.environ.get('EMBEDDING_TENSORBOARD')


num_emg_channels = Source.get_num_emg_channels()

def clear_tensorboard_logs() -> None:
    delete_old_files(selected_type=["train", "validation"], parent_path=log_path, file_type="*.v2")

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

def run_model() -> None:
    mlflow.set_tracking_uri(uri=mlflow_path)
    mlflow.set_experiment(experiment_name='emg-gesture')
    with mlflow.start_run() as run:
        clear_tensorboard_logs()

        train_ds_size = sum(1 for _ in load_dataset(selected_type="train"))
        validation_ds_size = sum(1 for _ in load_dataset(selected_type="validate"))
        test_ds_size = sum(1 for _ in load_dataset(selected_type="test"))


        train_steps_per_epoch = int(train_ds_size // BATCH_SIZE)
        validation_steps_per_epoch = int(validation_ds_size // BATCH_SIZE)
        test_steps_per_epoch = int(test_ds_size // TEST_BATCH_SIZE)


        train_ds = load_dataset(selected_type="train").shuffle(buffer_size=BUFFER_SIZE).repeat()
        train_ds = (train_ds
            .batch(batch_size=BATCH_SIZE, drop_remainder=True)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            )

        validation_ds = load_dataset(selected_type="validate").shuffle(buffer_size=BUFFER_SIZE).repeat()
        validation_ds = (validation_ds
            .batch(batch_size=BATCH_SIZE, drop_remainder=True)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            )

        test_ds = load_dataset(selected_type="test")
        test_ds = (test_ds
            .batch(batch_size=TEST_BATCH_SIZE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            )

        callback_list = [
            callbacks.ModelCheckpoint(filepath=ml_model_path, monitor="val_loss", save_best_only=True),
            callbacks.TensorBoard(log_dir=log_path),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=10, min_lr=0.00001, verbose=1)
        ]

        inputs = Input(shape=(TIMESTEPS, num_emg_channels))
        normalize = PerWindowNormalization()(inputs)
        embedding = layers.TimeDistributed(layers.Dense(units=170, activation='relu'))(normalize)
        lstm_one = layers.LSTM(400, return_sequences=True)(embedding)
        lstm_one = layers.Dropout(0.3)(lstm_one)
        lstm_two = layers.LSTM(LATENT_DIM, name="lstm_final")(lstm_one)
        lstm_two = layers.Dropout(0.3)(lstm_two)
        outputs = layers.Dense(units=get_num_labels(), activation='softmax')(lstm_two)

        model = Model(inputs, outputs)
        model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

        encoder_model = Model(inputs=model.input, outputs=model.get_layer('lstm_final').output)

        with open(f'{artifact_path}/model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

        mlflow.log_artifact(f'{artifact_path}/model_summary.txt')
        mlflow_dataset_train = mlflow.data.tensorflow_dataset.from_tensorflow(train_ds.take(1), digest=read_dvc_version())
        mlflow.log_input(dataset=mlflow_dataset_train, context='training')

        history = model.fit(train_ds, epochs=EPOCHS, steps_per_epoch=train_steps_per_epoch, validation_data=validation_ds, validation_steps=validation_steps_per_epoch, callbacks=callback_list)

        # mlflow.log_artifacts(local_dir=log_path, artifact_path='tensorboard_logs')

        model_input = Schema(inputs=[TensorSpec(type=np.dtype(np.float32), shape= (-1, TIMESTEPS, num_emg_channels), name="EMG_Channels_1_to_8 (OpenBCI)")])
        model_output = Schema(inputs=[TensorSpec(type=np.dtype(np.float32), shape= (-1, get_num_labels()), name="Hand_Gestures (Classification)")])

        model_signature = ModelSignature(inputs=model_input, outputs=model_output)
        best_model = models.load_model(ml_model_path)

        mlflow.keras.log_model(best_model, 'best_model', pip_requirements = depedency_path, signature=model_signature)

        params = {
            'batch_size' : BATCH_SIZE,
            'sequence_length' : TIMESTEPS,
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

        test_loss, test_acc = best_model.evaluate(test_ds, steps=test_steps_per_epoch)

        metrics = {
            'best_val_acc' : max(history.history['val_accuracy']),
            'best_val_loss' : min(history.history['val_loss']),
            'best_train_acc' : max(history.history['accuracy']),
            'best_train_loss' : min(history.history['loss']),
            'test_acc' : test_acc,
            'test_loss' : test_loss
        }

        mlflow.log_metrics(metrics)

        # test_pred, test_label = [], []

        # for feature, label in test_ds:
        #     logits = best_model(feature[None, ...]) ## [TECH DEBT: could be optimized, currently one sample batch prediction]
        #     test_pred.append(tf.argmax(input=logits, axis=-1))
        #     test_label.append(tf.argmax(input=label, axis=-1))

        # label_classes = get_all_labels(selected_type='train')

        # cm = tf.math.confusion_matrix(labels=test_label, predictions=test_pred).numpy()

        # df_cm = pd.DataFrame(data=cm,
        #                     index=[f'Actual: {label}' for label in label_classes],
        #                     columns=[f'Pred: {label}' for label in label_classes])

        # plt.figure(figsize=(10, 8))  # adjust size as needed
        # sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
        # plt.title('Confusion Matrix')
        # plt.ylabel('Actual')
        # plt.xlabel('Predicted')
        # plt.tight_layout()
        # plt.savefig(f'{artifact_path}/confusion_matrix.png')  # saves as PNG
        # mlflow.log_artifact(f'{artifact_path}/confusion_matrix.png')


        with open(label_idx_mapping_path) as file:
            label_idx_map = json.load(file)

        keys = tf.constant([int(k) for k in label_idx_map.keys()], dtype=tf.int64)
        values = tf.constant(list(label_idx_map.values()), dtype=tf.string)

        table = tf.lookup.StaticHashTable(initializer=tf.lookup.KeyValueTensorInitializer(keys, values),
                                          default_value="UNKNOWN")

        def map_labels(window, label):
            idx = tf.argmax(label, axis=-1)
            gesture = table.lookup(idx)
            latent_vector = encoder_model(window, training=False)
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


        # test_embeddings: shape (N, D)
        # test_labels:     length N

        csv_path = f"{artifact_path}/embeddings_with_labels.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            # header: first column is label, then embedding_0 ... embedding_{D-1}
            header = ["label"] + [f"dim_{i}" for i in range(test_embeddings.shape[1])]
            writer.writerow(header)

            for label, emb in zip(test_labels, test_embeddings):
                writer.writerow([label] + emb.tolist())

        print(f"Saved {len(test_labels)} rows to {csv_path}")

        mlflow.log_artifact(csv_path)
