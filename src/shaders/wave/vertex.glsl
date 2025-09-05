#version 330
in vec2 in_position;
uniform float u_channel_index;
void main() {
    vec2 pos = in_position;
    pos.y -= u_channel_index * 0.25;  // vertical spacing
    gl_Position = vec4(pos, 0.0, 1.0);
}