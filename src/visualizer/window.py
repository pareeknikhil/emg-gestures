from typing import Any

import moderngl
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QSurfaceFormat
from PyQt5.QtWidgets import (QAction, QApplication, QLabel, QMenu,
                             QOpenGLWidget, QPushButton, QShortcut,
                             QToolButton)

from configs.constants import (FRAME_RATE, GUI_HEIGHT, GUI_WIDTH, HOP_SIZE,
                               SPECTROGRAM_WINDOW)

from ..utils.tfrecord_utils import get_all_labels
from .model import Model
from .source import Source
from .spec import Spec
from .text import Text
from .time_series import Wave


def override(method) -> Any:
    """Custom decorator to indicate method overriding"""
    return method


class EMGSignalAnalyzer(QOpenGLWidget):

    def __init__(self, logger) -> None:
        super().__init__()

        self.logger = logger
        self.logger.info("WINDOW: Initializing EMG Signal Analyzer...")
        self.setWindowTitle("EMG Analyzer")
        self.setFixedSize(GUI_WIDTH, GUI_HEIGHT)

        _fmt = QSurfaceFormat()
        _fmt.setVersion(3,3)
        _fmt.setProfile(QSurfaceFormat.CoreProfile)
        _fmt.setDefaultFormat(_fmt)
        _fmt.setSamples(4)
        self.setFormat(_fmt)

        QShortcut(Qt.Key_Escape, self, self.close)

        self.__timer = QTimer()
        self.__timer.timeout.connect(self.update)
        self.__timer.start(int(1000/FRAME_RATE))

        self.add_buttons()
        self.cyton = Source.get_instance(self.logger)

        self.model = Model.get_instance(self.logger)


    def add_buttons(self) -> None:
        self.start_button = QPushButton("Start Recording", self)
        self.start_button.clicked.connect(self.on_click)
        self.start_button.setStyleSheet(self.get_stylesheet(color="green"))
        self.start_button.move(0, 30)

        self.label = QLabel("Ready", self)
        self.label.move(120, 30)
        self.label.setStyleSheet("font-size: 20px;")
        self.label.setStyleSheet("color: red;")

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_countdown)

        self.reset_button = QPushButton("Reset", self)
        self.reset_button.clicked.connect(self.on_reset)
        self.reset_button.move(0, 65)
        self.reset_button.setStyleSheet(self.get_stylesheet(color="purple"))

        self.type_dropdown = self.create_dropdown_button(
            label="Type",
            items=["train", "validate", "test"],
            color="blue",
            position=(0, 0),
            callback=self.on_type_selected
        )

        self.dropdown = self.create_dropdown_button(
            label="Activity",
            items=get_all_labels(),
            color="brown",
            position=(80, 0),
            callback=self.on_activity_selected
        )

        self.start_recording = True

    def create_dropdown_button(self, label, items, color, position, callback) -> QToolButton:
        button = QToolButton(self)
        button.setText(label)
        button.setPopupMode(QToolButton.MenuButtonPopup)
        button.setStyleSheet(self.get_stylesheet(color=color))
        button.move(*position)

        menu = QMenu(self)
        for item in items:
            action = QAction(item, self)
            action.triggered.connect(lambda checked=False, i=item: callback(i))
            menu.addAction(action)

        button.setMenu(menu)
        return button

    def on_activity_selected(self, name) -> None:
        self.dropdown.setText(name)
        self.location = name  # or use self.sender().text()

    def on_type_selected(self, type_name) -> None:
        self.type_dropdown.setText(type_name)
        self.selected_type = type_name

    def on_click(self) -> None:
        self.cyton.flip_recording_flag()
        self.remaining_time = 0

        if self.start_recording:
            self.timer.start()
            self.start_button.setText("Stop Recording")
            self.start_button.setStyleSheet(self.get_stylesheet(color="red"))

        else:
            self.timer.stop()
            self.label.setText(f"{self.remaining_time}")
            self.start_button.setText("Start Recording")
            self.start_button.setStyleSheet(self.get_stylesheet(color="green"))
            self.cyton.write_to_disk(self.selected_type, self.location, )

        self.start_recording = not self.start_recording

    def update_countdown(self) -> None:
        self.remaining_time += 1
        self.label.setText(f"{self.remaining_time}")

    def on_reset(self):
        self.time_series.reset()
        self.spec_series.reset()
        self.logger.info("WINDOW: Analyzer reset completed")

    @override
    def closeEvent(self, event) -> None:
        self.close_gui()
        super().closeEvent(event)

    @override
    def initializeGL(self) -> None:
        self.ctx = moderngl.create_context(require=330)
        self.ctx.enable(moderngl.BLEND) 
        self.ctx.multisample = True

        self.time_series = Wave.get_instance(self.ctx, self.logger)

        self.spec_series = Spec.get_instance(self.ctx, 40, 80, self.logger)

        self.cyton.start_stream()

        self.activity_prediction = Text.get_instance(self.ctx, 170, 80, align='center')
        self.activity_prediction.add(170, 80, 'center')

    @override
    def resizeGL(self, w, h) -> None:
        self.spec_series.size(w, h)
        self.activity_prediction.size(w, h)
        self.logger.info(f"WINDOW: Size - {w} , {h}")

    @override
    def paintGL(self):

        emg_data = self.cyton.get_data(num_of_samples_expctd=HOP_SIZE)

        self.time_series.add(new_wave_data=emg_data)
        self.time_series.draw()

        filtrd_emg = self.time_series.get_filtrd_emg(n_latest_samples=SPECTROGRAM_WINDOW)

        self.spec_series.add(filtrd_emg)
        self.spec_series.draw()

        self.model.add(new_wave_data=emg_data)
        predctd_label = self.model.predict()
        self.activity_prediction.draw(predctd_label)

    def close_gui(self) -> None:
        self.logger.info("WINDOW: Closing GUI resources...")
        self.cyton.release_board()
        self.time_series.release()
        self.spec_series.release()
        self.ctx.release()
        self.logger.info("WINDOW: Closed GUI successfully")
        self.logger.release()

    @staticmethod
    def get_stylesheet(color) -> str:
        return f"""
            QPushButton {{
                background-color: {color};
                font-size:15px;
                font-family: Arial;
            }}
        """

    @classmethod
    def run(cls, logger) -> None:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        window = QApplication([])
        main = cls(logger)
        main.show()
        window.exit(window.exec())
