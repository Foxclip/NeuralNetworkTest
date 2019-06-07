#version 330 core

out vec3 pos;
out vec2 TexCoord;

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

uniform mat4 projection;

void main()
{
    gl_Position = projection * vec4(aPos, 1.0);
    pos = aPos;
    TexCoord = aTexCoord;
}