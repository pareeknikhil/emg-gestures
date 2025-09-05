import moderngl
import numpy as np

from configs.constants import GUI_WIDTH, HOP_SIZE

from ..shaders.shader_loader import wave_fragment_shader, wave_vertex_shader
from ..utils.data_processing import filter_data
from .source import Source


class Wave:
    __instance = None

    @classmethod
    def get_instance(cls, ctx, logger) -> "Wave":
        if cls.__instance is None:
            cls.__instance = cls(ctx, logger)
        return cls.__instance

    def reset(self) -> None:
        self.time_series.fill(0)
        self.uv_plot.fill(0)


    def __init__(self, ctx, logger) -> None:
        self.logger = logger

        self.num_emg_channels = Source.get_num_emg_channels()

        self.time_series = np.zeros(shape=(self.num_emg_channels, GUI_WIDTH+500))
        self.uv_plot = np.zeros(shape=(self.num_emg_channels, GUI_WIDTH))
        self.x_points = np.linspace(start=-1, stop=1, num=GUI_WIDTH)

        self.prog = ctx.program(vertex_shader=wave_vertex_shader, fragment_shader=wave_fragment_shader)
        self.buffer = ctx.buffer(reserve=self.uv_plot.nbytes * 3, dynamic=True)
        self.vao = ctx.vertex_array(self.prog, self.buffer, "in_position")
        self.draw()

    def add(self, new_wave_data) -> None:
        self.logger.info(f"WAVE: Recvd data for buffer (no. of channels: {new_wave_data.shape[0]}), " 
                    f"no. of data points in each channel: {new_wave_data.shape[1]}")
        self.__add_new_wave(new_wave_data=new_wave_data)

    def draw(self) -> None:
        channels, width = self.uv_plot.shape

        y_max = self.uv_plot.max(axis=1, keepdims=True)
        y_min = self.uv_plot.min(axis=1, keepdims=True)
        y_norm = (self.uv_plot - y_max) * 0.05 / (y_max - y_min) + 1.0

        x_vals = np.tile(A=self.x_points, reps=(channels, 1))

        positions = np.stack(arrays=[x_vals, y_norm], axis=-1).reshape(-1, 2)
        self.buffer.write(positions.astype("f4"))
        for i in range(channels):
            self.prog['u_channel_index'].value = float(i)
            self.vao.render(moderngl.LINE_STRIP, vertices=width, first=i*width)

        self.logger.info("WAVE: Drawing")

    def get_filtrd_emg(self, n_latest_samples) -> np.ndarray:
        return self.uv_plot[:, -n_latest_samples:]

    def release(self) -> None:
        self.prog.release()
        self.buffer.release()
        self.vao.release()

    def __add_new_wave(self, new_wave_data) -> None:
        self.time_series[:, :-HOP_SIZE] = self.time_series[:, HOP_SIZE:]
        self.time_series[:, -HOP_SIZE:] = new_wave_data
        for count in range(self.num_emg_channels):
            self.uv_plot[count, :] = filter_data(self.time_series[count, :])
        self.logger.info("WAVE: Updated filtered data to buffer")