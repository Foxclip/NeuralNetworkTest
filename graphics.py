import glfw
import sys
import numpy as np
import pyrr
import random
from OpenGL.GL import *

SCR_WIDTH = 512
SCR_HEIGHT = 512
ARR_SIZE_X = 4
ARR_SIZE_Y = 4
STEP_X = int(SCR_WIDTH / ARR_SIZE_X)
STEP_Y = int(SCR_HEIGHT / ARR_SIZE_Y)

def framebuffer_size_callback(window, width, height):
    glViewport(0, 0, width, height)

class Point:
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value

class Graphics:

    def __init__(self):
        self.shaderProgram = None
        self.VAO = None
        self.VBO = None
        self.EBO = None
        self.points = []

        for y in range(0, SCR_HEIGHT, STEP_Y):
            for x in range(0, SCR_WIDTH, STEP_X):
                result = random.random()
                new_point = Point(x, y, result)
                self.points.append(new_point)

    def setFloat(self, name, value):
        glUniform1f(glGetUniformLocation(self.shaderProgram, name), value);

    def setVec3(self, name, v1, v2, v3):
        glUniform3f(glGetUniformLocation(self.shaderProgram, name), v1, v2, v3)

    def setMat4(self, name, value):
        glUniformMatrix4fv(glGetUniformLocation(self.shaderProgram, name), 1, GL_FALSE, value)

    def setArray(self, name, size, value):
        glUniform1fv(glGetUniformLocation(self.shaderProgram, name), size, value);

    def compile_shader(self, shader_source, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, shader_source)
        glCompileShader(shader)
        success = glGetShaderiv(shader, GL_COMPILE_STATUS)
        if not success:
            infolog = glGetShaderInfoLog(shader)
            if shader_type == GL_VERTEX_SHADER:
                print("Vertex shader compilation failed")
            elif shader_type == GL_FRAGMENT_SHADER:
                print("Fragment shader compilation failed")
            print(infolog)
            sys.exit()
        return shader

    def create_shader_program(self, vertexShader, fragmentShader, name):
        shaderProgram = glCreateProgram()
        glAttachShader(shaderProgram, vertexShader)
        glAttachShader(shaderProgram, fragmentShader)
        glLinkProgram(shaderProgram)
        success = glGetProgramiv(shaderProgram, GL_LINK_STATUS)
        if not success:
            infolog = glGetShaderInfoLog(shaderProgram);
            print(name + " linking failed")
            print(infolog)
        return shaderProgram

    def compileShaders(self):
        vertexShaderString = open("vertex.vert", "r").read()
        fragmentShaderString = open("fragment.frag", "r").read()
        vertexShader = self.compile_shader(vertexShaderString, GL_VERTEX_SHADER)
        fragmentShader = self.compile_shader(fragmentShaderString, GL_FRAGMENT_SHADER)
        self.shaderProgram = self.create_shader_program(vertexShader, fragmentShader, "test")
        glUseProgram(self.shaderProgram)
        glDeleteShader(vertexShader)
        glDeleteShader(fragmentShader)

    def initGeometry(self):

        # vertices = np.array([
        #     -1.0, -1.0, 0.0,
        #      1.0, -1.0, 0.0,
        #      0.0,  1.0, 0.0
        # ], dtype=np.float32)

        vertices = np.array([
                  0.0,        0.0, 0.0,
            SCR_WIDTH,        0.0, 0.0,
            SCR_WIDTH, SCR_HEIGHT, 0.0,
                  0.0, SCR_HEIGHT, 0.0,
            #  0.5,  0.5, 0.0,    0.0, 0.0, 1.0,    1.0, 1.0,
            #  0.5, -0.5, 0.0,    1.0, 0.0, 0.0,    1.0, 0.0,
            # -0.5, -0.5, 0.0,    0.0, 1.0, 0.0,    0.0, 0.0,
            # -0.5,  0.5, 0.0,    1.0, 1.0, 0.0,    0.0, 1.0
        ], dtype=np.float32)

        # indices = np.array([0, 1, 3, 1, 2, 3], dtype=np.int32)

        self.VAO = glGenVertexArrays(1);
        self.VBO = glGenBuffers(1);
        # self.EBO = glGenBuffers(1)

        glBindVertexArray(self.VAO);
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO);
        # glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)

        glBufferData(GL_ARRAY_BUFFER, len(vertices)*4, vertices, GL_STATIC_DRAW)
        # glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices)*4, indices, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None) #in case of problems, try changing stride
        glEnableVertexAttribArray(0)

        orthoMatrix = pyrr.matrix44.create_orthogonal_projection(0, SCR_WIDTH, 0, SCR_HEIGHT, -1, 1, dtype=np.float32)
        self.setMat4("projection", orthoMatrix)
        self.setVec3("color", 1.0, 0.0, 0.0)
        for i in range(len(self.points)):
            point = self.points[i]
            self.setFloat("points[" + str(i) + "].x", point.x)
            self.setFloat("points[" + str(i) + "].y", point.y)
            self.setFloat("points[" + str(i) + "].value", point.value)

    def init(self):

        self.compileShaders()
        self.initGeometry()

    def render(self):

        # glEnableClientState(GL_VERTEX_ARRAY)
        # print(glGetError())

        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.shaderProgram)
        glBindVertexArray(self.VAO)
        # glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 6)

        # glDisableClientState(GL_VERTEX_ARRAY)
        # print(glGetError())

    def processInput(self, window):
    	if(glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS):
    	    glfw.set_window_should_close(window, True)

    def main(self):
        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3);
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3);
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE);

        window = glfw.create_window(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", None, None);
        if not window:
            print("Failed to create GLFW window")
            glfw.terminate();
            sys.exit()
        glfw.make_context_current(window);
        glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)

        self.init()

        while not glfw.window_should_close(window):

            self.processInput(window)

            self.render()

            glfw.swap_buffers(window)
            glfw.poll_events()

        glfw.terminate()

graphics = Graphics()
graphics.main()