#version 330 core
uniform sampler2DArray image;
in vec2 v_uv;
in float v_layer;
out vec4 out_color;
void main() {
    vec4 color = texture(image, vec3(v_uv, v_layer));
    out_color = vec4(color.rgb, 1);
}