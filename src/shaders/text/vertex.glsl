#version 330 core
uniform mat4 P;
in vec2 vertex;
in vec2 uv;
out vec2 v_uv;
void main() {
    gl_Position = P * vec4(vertex, 0, 1);
    v_uv = uv;
}