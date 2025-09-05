import os

wave_vertex__glsl_path = os.environ.get('WAVE_VERTEX_GLSL_PATH')
wave_fragment__glsl_path = os.environ.get('WAVE_FRAGMENT_GLSL_PATH')

spec_vertex__glsl_path = os.environ.get('SPEC_VERTEX_GLSL_PATH')
spec_fragment__glsl_path = os.environ.get('SPEC_FRAGMENT_GLSL_PATH')

text_vertex__glsl_path = os.environ.get('TEXT_VERTEX_GLSL_PATH')
text_fragment__glsl_path = os.environ.get('TEXT_FRAGMENT_GLSL_PATH')


def load_shadr_file(path):
    with open(path, 'r') as file:
        return file.read()

wave_vertex_shader = load_shadr_file(wave_vertex__glsl_path)

wave_fragment_shader = load_shadr_file(wave_fragment__glsl_path)

spec_vertex_shader = load_shadr_file(spec_vertex__glsl_path)

spec_fragment_shader = load_shadr_file(spec_fragment__glsl_path)

text_vertex_shader = load_shadr_file(text_vertex__glsl_path)

text_fragment_shader = load_shadr_file(text_fragment__glsl_path)