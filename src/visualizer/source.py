import sys
import time

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter

from configs.constants import IS_SYNTHETIC_BOARD, SERIAL_PORT_LINUX

from ..utils.tfrecord_utils import get_all_files


class Source():
    BOARDID = BoardIds.SYNTHETIC_BOARD if IS_SYNTHETIC_BOARD else BoardIds.CYTON_BOARD

    __instance = None

    @classmethod
    def get_instance(cls, logger) -> "Source":
        if cls.__instance is None:
            cls.__instance = cls(logger)
        return cls.__instance

    def __init__(self, logger) -> None:
        BoardShim.enable_dev_board_logger()
        self.logger = logger

        _params = BrainFlowInputParams()
        _params.serial_port = SERIAL_PORT_LINUX

        self.emg_channels = Source.get_emg_channels()
        self.board = BoardShim(board_id=Source.BOARDID, input_params=_params)
        self.board.prepare_session()
        if not self.board.is_prepared():
            sys.exit(1)

        self.turn_off_srb(self.board)
        self.logger.info("SOURCE: Board initialized and SRB channels turned-off")

        self.is_recording = False

        self.emg_recording = []

    @staticmethod
    def get_emg_channels():
        return BoardShim.get_emg_channels(board_id=BoardIds.CYTON_BOARD) ## hard-coded: suppose to work for synthetic or cyton board (not any other config) 

    @staticmethod
    def get_num_emg_channels():
        return len(BoardShim.get_emg_channels(board_id=BoardIds.CYTON_BOARD))

    def turn_off_srb(self, board):
        for channel in self.emg_channels:
            board_response = board.config_board(f'x{channel}060100X')[:1]
            if board_response not in {'S', 'C'}:
                sys.exit(1)

    def start_stream(self) -> None:
        if self.board.is_prepared():
            self.board.start_stream()
            self.logger.info(f"SOURCE: Data Stream Started (synthetic data: {IS_SYNTHETIC_BOARD})")
        else:
            self.logger.error("SOURCE: Unable to connect with OpenBCI board")
            sys.exit(1)

    def get_data(self, num_of_samples_expctd) -> np.ndarray:
        board_data = self.board.get_board_data(num_samples=num_of_samples_expctd)
        emg_data = board_data[self.emg_channels, :]
        self.emg_recording.append(emg_data) if self.is_recording else None
        num_of_sampl_recvd = board_data.shape[-1]
        self.logger.info(f"SOURCE: Received data per channel from board: {num_of_sampl_recvd}, requested: {num_of_samples_expctd}")
        if num_of_sampl_recvd == num_of_samples_expctd:
            return emg_data

        padded_emg_data = np.zeros([len(self.emg_channels), num_of_samples_expctd])
        padded_emg_data[:, :emg_data.shape[1]] = emg_data
        self.logger.info(f"SOURCE: Padded data on per channel (new no. of samples {padded_emg_data.shape[1]})")
        return padded_emg_data

    def release_board(self) -> None:
        self.board.stop_stream()
        self.board.release_session()
        self.logger.info("SOURCE: OpenBCI Stream closed successfully")

    def flip_recording_flag(self) -> None:
        self.is_recording = not self.is_recording

    def write_to_disk(self, type, label) -> None:
        emg_numpy = np.concatenate(self.emg_recording, axis=1)
        file_path = f'data/csv/{type}/{label}/file_{str(int(time.time()))}.csv' ## hard-coded: [TECH DEBT]
        DataFilter.write_file(data=emg_numpy, file_name=file_path, file_mode='w')
        file_ds = get_all_files(pattern=f'data/csv/{type}/{label}/*.csv', shuffle_flag=False) ## hard-coded: [TECH DEBT]
        print(f"SOURCE: File written in folder: {type}/{label} (Total files: {sum(1 for _ in file_ds)})")
        self.emg_recording.clear()
        print("SOURCE: Source data storage cleaned(Reset)")