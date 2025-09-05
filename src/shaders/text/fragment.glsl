#version 330 core
in vec2 v_uv;
uniform sampler2D image;
uniform vec4 color;
out vec4 out_color;
void main() {
    float mask = texture(image, v_uv).r;
    out_color = vec4(color.rgb, color.a * mask);
}