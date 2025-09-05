import json
import os
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.config import enable_unsafe_deserialization

from configs.constants import HOP_SIZE, ML_WINDOW

from ..utils.tfrecord_utils import per_window_normalization
from .source import Source

enable_unsafe_deserialization()

ml_model_path = os.environ.get('ML_MODEL_PATH')
label_to_idx = os.environ.get('LABEL_IDX_MAPPING')


class Model:

    __instance = None

    @classmethod
    def get_instance(cls, logger) -> "Model":
        if cls.__instance is None:
            cls.__instance = cls(logger)
        return cls.__instance

    def __init__(self, logger) -> None:
        self.logger = logger
        self.model = models.load_model(ml_model_path,
                        custom_objects={"per_window_normalization": per_window_normalization})

        self.emg_feature = np.zeros((Source.get_num_emg_channels(), ML_WINDOW))
        self.label_lookup = json.loads(open(str(label_to_idx)).read())
        self.prediction = None

    def add(self, new_wave_data) -> None:
        self.logger.info(f"MODEL: Recvd data for buffer (no. of channels: {new_wave_data.shape[0]}), " 
            f"no. of data points in each channel: {new_wave_data.shape[1]}") 
        self.__add_new_wave(new_wave_data)

    def __add_new_wave(self, data) -> None:
        if data.shape[-1] == HOP_SIZE:
            self.emg_feature[:, :-HOP_SIZE] = self.emg_feature[:, HOP_SIZE:]
            self.emg_feature[:, -HOP_SIZE:] = data
            self.logger.info("MODEL: Updated data to buffer")

    def predict(self) -> Any:
        input_feature = self.emg_feature.transpose()
        input_feature = np.expand_dims(input_feature, axis=0)
        self.prediction = self.predict_fun(input_feature).numpy()
        return self.label_lookup[str(object=np.argmax(self.prediction))]

    @tf.function
    def predict_fun(self, data) -> Any:
        return(self.model(data, training=False))