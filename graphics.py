import glfw
import sys
import numpy as np
import pyrr
# import random
import multiprocessing
from OpenGL.GL import *

SCR_WIDTH = 512
SCR_HEIGHT = 512
ARR_SIZE_X = 16
ARR_SIZE_Y = 16
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
        self.window = None
        self.shaderProgram = None
        self.VAO = None
        self.VBO = None
        self.EBO = None
        self.points_queue = None
        self.texture = None

    def setInt(self, name, value):
        glUniform1i(glGetUniformLocation(self.shaderProgram, name), value)

    def setFloat(self, name, value):
        glUniform1f(glGetUniformLocation(self.shaderProgram, name), value)

    def setVec3(self, name, v1, v2, v3):
        glUniform3f(glGetUniformLocation(self.shaderProgram, name), v1, v2, v3)

    def setMat4(self, name, value):
        glUniformMatrix4fv(glGetUniformLocation(self.shaderProgram, name), 1, GL_FALSE, value)

    def setArray(self, name, size, value):
        glUniform1fv(glGetUniformLocation(self.shaderProgram, name), size, value)

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
            infolog = glGetShaderInfoLog(shaderProgram)
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

        vertices = np.array([

            0.0, 0.0, 0.0,
            0.0, 0.0,

            SCR_WIDTH, 0.0, 0.0,
            1.0, 0.0,

            SCR_WIDTH, SCR_HEIGHT, 0.0,
            1.0, 1.0,

            0.0, SCR_HEIGHT, 0.0,
            0.0, 1.0

        ], dtype=np.float32)

        # indices = np.array([0, 1, 3, 1, 2, 3], dtype=np.int32)

        # texture
        data = [0.5] * STEP_X * STEP_Y
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(
            GL_TEXTURE_2D,      # target
            0,                  # mipmap level
            GL_RED,             # internal format
            STEP_X,             # width
            STEP_Y,             # height
            0,                  # must be zero
            GL_RED,             # data format
            GL_FLOAT,           # data type, try GL_FLOAT in case of problems
            data                # pixel data
        )
        glGenerateMipmap(GL_TEXTURE_2D)

        self.VAO = glGenVertexArrays(1)
        self.VBO = glGenBuffers(1)
        # self.EBO = glGenBuffers(1)

        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        # glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)

        glBufferData(GL_ARRAY_BUFFER, len(vertices) * 4, vertices, GL_STATIC_DRAW)
        # glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices)*4, indices, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, None)  # in case of problems, try changing stride
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, 3 * 4)
        glEnableVertexAttribArray(1)

        orthoMatrix = pyrr.matrix44.create_orthogonal_projection(0, SCR_WIDTH, 0, SCR_HEIGHT, -1, 1, dtype=np.float32)
        self.setMat4("projection", orthoMatrix)
        self.setVec3("color", 1.0, 0.0, 0.0)
        self.setInt("column_count", int(SCR_WIDTH / STEP_X))
        self.setInt("stepX", STEP_X)
        self.setInt("stepY", STEP_Y)

    def initOpenGL(self):
        self.compileShaders()
        self.initGeometry()

    def render(self):

        # glEnableClientState(GL_VERTEX_ARRAY)
        # print(glGetError())

        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.shaderProgram)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glBindVertexArray(self.VAO)

        try:
            points = self.points_queue.get(block=False)
            for i in range(len(points)):
                point = points[i]
                self.setFloat("points[" + str(i) + "].x", point.x)
                self.setFloat("points[" + str(i) + "].y", point.y)
                self.setFloat("points[" + str(i) + "].value", point.value)
        except Exception as e:
            if type(e).__name__ != "Empty":
                print(e)

        # glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 6)

        # glDisableClientState(GL_VERTEX_ARRAY)
        # print(glGetError())

    def processInput(self, window):
        if(glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS):
            glfw.set_window_should_close(window, True)

    def mainCycle(self):

        while not glfw.window_should_close(self.window):

            self.processInput(self.window)
            self.render()

            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glfw.terminate()

    def initGLFW(self, name, queue):

        self.points_queue = queue

        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.window = glfw.create_window(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", None, None)
        if not self.window:
            print("Failed to create GLFW window")
            glfw.terminate()
            sys.exit()
        glfw.set_window_pos(self.window, 100, 100)
        glfw.make_context_current(self.window)
        glfw.set_framebuffer_size_callback(self.window, framebuffer_size_callback)

        self.initOpenGL()

        self.mainCycle()

    def start(self, queue):

        process = multiprocessing.Process(target=self.initGLFW, args=(1, queue))
        process.start()
