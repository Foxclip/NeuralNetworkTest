#version 330 core

in vec3 pos;
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D texture1;

void main()
{
    FragColor = vec4(vec3(texture(texture1, TexCoord).r), 1.0);
}