#version 330 core
uniform mat4 P;
in vec2 in_position;
in vec2 in_uv;
in float in_layer;
out vec2 v_uv;
out float v_layer;
void main() {
    vec2 pos = in_position;
    gl_Position = P * vec4(pos, 0, 1);
    v_uv = in_uv;
    v_layer = in_layer;
}