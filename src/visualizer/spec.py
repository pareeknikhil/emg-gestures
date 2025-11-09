import librosa
import matplotlib.cm as cm
import moderngl
import numpy as np
from pyrr import Matrix44

from configs.constants import GUI_WIDTH, SPECTROGRAM_WINDOW

from ..shaders.shader_loader import spec_fragment_shader, spec_vertex_shader
from ..utils.data_processing import get_hann_window
from .source import Source


## Captures 2(assuming 125 samples in one spectrogram-window) Hz to 125 Hz
class Spec:
    COLOR_MAP = cm.get_cmap(name='inferno')
    HANN_WINDOW = get_hann_window(window_size=SPECTROGRAM_WINDOW, skew=True)
    HANN_WINDOW.setflags(write=False)

    __instance = None

    @classmethod
    def get_instance(cls, ctx, y, h, logger) -> "Spec":
        if cls.__instance is None:
            cls.__instance = cls(ctx, y, h, logger)
        return cls.__instance


    def __init__(self, ctx, y, h, logger) -> None:
        self.logger = logger

        self.num_emg_channels = Source.get_num_emg_channels()

        self.frames = np.zeros((self.num_emg_channels, SPECTROGRAM_WINDOW//2 + 1, GUI_WIDTH, 3), dtype='u1')

        self.prog = ctx.program(vertex_shader=spec_vertex_shader, fragment_shader=spec_fragment_shader)

        vertices = []
        for i in range(self.num_emg_channels):
            y_offset = y + i * 125
            layer = float(i)
            vertices.extend([
            0, y_offset  , 0, 1, layer, # A
            0, y_offset+h, 0, 0, layer, # B
            GUI_WIDTH, y_offset+h, 1, 0, layer, # C
            0, y_offset  , 0, 1, layer, # A
            GUI_WIDTH, y_offset+h, 1, 0, layer, # C
            GUI_WIDTH, y_offset  , 1, 1, layer, # D,
        ])

        vertices = np.array(vertices, dtype='f4')
        self.buffer = ctx.buffer(vertices)
        self.vao = ctx.vertex_array(
                self.prog, [(self.buffer, '2f 2f 1f', 'in_position', 'in_uv', 'in_layer')])


        self.textures = ctx.texture_array(
        size=(GUI_WIDTH, SPECTROGRAM_WINDOW//2 + 1, self.num_emg_channels),
        components=3,
        data=self.frames)
        self.textures.repeat_x = False
        self.textures.repeat_y = True

    def reset(self) -> None:
        self.frames = np.zeros((self.num_emg_channels, SPECTROGRAM_WINDOW//2 + 1, GUI_WIDTH, 3), dtype='u1')

    def add(self, window) -> None:
        slices = Spec.stft_slice(window)
        new_slice = Spec.stft_color(slices)
        self.frames[:, :, :-1, :] = self.frames[:, :, 1:, :]
        self.frames[:, :, -1, :] = new_slice
        self.logger.info(f"SPEC: Adding window shape {window.shape}")

    def size(self, w, h) -> None:
        P = Spec.orthographic(w, h)
        self.prog['P'].write(P)

    def draw(self) -> None:
        self.textures.write(self.frames)
        self.textures.use(0)
        for i in range(self.num_emg_channels):
            self.vao.render(mode=moderngl.TRIANGLES, vertices=6, first=i*6)
        self.logger.info("SPEC: Spec rendered...")
    
    def release(self) -> None:
        self.prog.release()
        self.buffer.release()
        self.textures.release()
        self.vao.release()

    @staticmethod
    def stft_slice(window) -> np.ndarray:
        return np.fft.rfft(window*Spec.HANN_WINDOW, axis=1)

    @staticmethod
    def stft_color(slices, min_db=-5, max_db=10):
        slices = librosa.amplitude_to_db(slices)
        slices = slices.clip(min_db, max_db)
        slices = (slices-min_db) / (max_db-min_db)
        slices = Spec.COLOR_MAP(slices)
        slices = (slices * 255).astype("u1")
        return slices[:, :, :3]

    @staticmethod
    def orthographic(w, h) -> Matrix44:
        P = Matrix44.orthogonal_projection(
                0, w, h, 0, -1, 1, dtype='f4')
        return P
