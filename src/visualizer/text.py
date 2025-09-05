import os
from typing import Any

import freetype
import numpy as np
from pyrr import Matrix44

from configs.constants import PREDICTION_FONT_SIZE

from ..shaders.shader_loader import text_fragment_shader, text_vertex_shader

font_path = os.environ.get('FONT_PATH')

class CharacterSlot:

    def __init__(self, ctx, glyph) -> None:
        if not isinstance(glyph, freetype.GlyphSlot):
            raise RuntimeError('Unknown glyph type')

        self.width   = glyph.bitmap.width
        self.height  = glyph.bitmap.rows
        self.advance = glyph.advance.x

        size = (self.width, self.height)

        data = np.array(object=glyph.bitmap.buffer, dtype='u1')
        self.texture = ctx.texture(size, 1, data)
        self.texture.repeat_x = False
        self.texture.repeat_y = False

class Text:
    __instance = None

    @classmethod
    def get_instance(cls, ctx, x, y, align) -> "Text":
        if cls.__instance is None:
            cls.__instance = cls(ctx, x, y, align)
        return cls.__instance

    def __init__(self, ctx, x, y ,align) -> None:
        self.ctx = ctx
        self.prog = self.ctx.program(
                vertex_shader=text_vertex_shader,
                fragment_shader=text_fragment_shader)

        self.vbo = self.ctx.buffer(
                reserve=6*4*4, dynamic=True)

        self.vao = self.ctx.vertex_array(
                self.prog, self.vbo, 'vertex', 'uv')

        self.prog['color'] = (0.8, 0.6, 0.6, 1.0)

        self.init_font(font=font_path)

        self.texts = []

        self.add(x=x, y=y, align=align)


    def init_font(self, font) -> None:
        self.characters = dict()
        size = int(PREDICTION_FONT_SIZE)
        # Load the font face
        face = freetype.Face(path_or_stream=font)
        face.set_pixel_sizes(width=size, height=size)
        # Load ASCII characters from 30-128
        for i in range(30, 128):
            char = chr(i)
            face.load_char(char=char)
            character = CharacterSlot(self.ctx, face.glyph)
            self.characters[char] = character

    def set_geometry(self, x, y, w, h) -> None:
        vertices = np.array([
            x,   y,   0, 1,
            x+w, y,   1, 1,
            x+w, y-h, 1, 0,
            x,   y,   0, 1,
            x+w, y-h, 1, 0,
            x,   y-h, 0, 0,
        ])
        vertices = vertices.astype('f4')
        self.vbo.write(vertices)

    def text_width(self, text) -> Any:
        w = 0
        for c in text:
            character = self.characters[c]
            w += (character.advance >> 6)
        return w

    def add(self, x, y, align='left') -> None:
        self.texts.append((x, y, align))

    @staticmethod
    def orthographic(w, h) -> Matrix44:
        return Matrix44.orthogonal_projection(
                left=0, right=w, top=h, bottom=0, near=1, far=-1, dtype='f4')

    def size(self, w, h) -> None:
        P = Text.orthographic(w=w, h=h)
        self.prog['P'].write(P)

    def draw(self, text) -> None:
         for x, y, align in self.texts:
            if align == 'center':
                w = self.text_width(text=text)
                x -= w / 2
            if align == 'right':
                w = self.text_width(text=text)
                x -= w
            for i, c in enumerate(text):
                character = self.characters[c]
                character.texture.use(0)
                w = character.width
                h = character.height
                self.set_geometry(x=x, y=y, w=w, h=h)
                self.vao.render()
                x = x + (character.advance >> 6)