import os

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim
from brainflow.data_filter import (AggOperations, DataFilter, FilterTypes,
                                   NoiseTypes)
from dotenv import load_dotenv

from configs.constants import IS_SYNTHETIC_BOARD

BOARDID = BoardIds.SYNTHETIC_BOARD if IS_SYNTHETIC_BOARD else BoardIds.CYTON_BOARD

def filter_data(data: np.ndarray) -> np.ndarray:
    data = np.array(data, dtype=np.float64)

    DataFilter.perform_rolling_filter(data, 3, AggOperations.MEAN.value)

    DataFilter.perform_bandstop(data=data, sampling_rate=BoardShim.get_sampling_rate(BOARDID), start_freq=58.0, stop_freq=62.0,
                                        order=4, filter_type=FilterTypes.BUTTERWORTH, ripple=1.0)

    DataFilter.perform_bandpass(data=data, sampling_rate=BoardShim.get_sampling_rate(BOARDID), start_freq=5.0, stop_freq=250.0,
                                        order=4, filter_type=FilterTypes.BUTTERWORTH, ripple=1.0)

    DataFilter.remove_environmental_noise(data=data, sampling_rate=BoardShim.get_sampling_rate(BOARDID), noise_type=NoiseTypes.FIFTY.value)

    DataFilter.remove_environmental_noise(data=data, sampling_rate=BoardShim.get_sampling_rate(BOARDID), noise_type=NoiseTypes.SIXTY.value)

    return data[500:]

def get_hann_window(window_size, skew=True) -> np.ndarray:
    hann = np.hanning(window_size)
    if skew:
        skew_factor = np.linspace(0, 10, window_size)
        skewed_window = hann * np.exp(skew_factor - 2)
        hann /= np.max(skewed_window)
    return hann

def load_env_variables() -> None: 
    load_dotenv(override=True)