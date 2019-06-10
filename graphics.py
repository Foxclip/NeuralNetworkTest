import glfw
import sys
import numpy as np
import pyrr
# import random
import multiprocessing
import ctypes
import math
from OpenGL.GL import *

SCR_WIDTH = 512
SCR_HEIGHT = 512
ARR_SIZE_X = 32
ARR_SIZE_Y = 32
STEP_X = int(SCR_WIDTH / ARR_SIZE_X)
STEP_Y = int(SCR_HEIGHT / ARR_SIZE_Y)
CIRCLE_VERTEX_COUNT = 16
CIRCLE_RADIUS = 5.0

DATA_MAX_X = 200.0
DATA_MAX_Y = 200.0

FLOAT_SIZE = 4


def framebuffer_size_callback(window, width, height):
    glViewport(0, 0, width, height)


class Point:

    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color


class Graphics:

    def __init__(self):
        self.window = None
        self.shaderProgram = None
        self.VAO = None
        self.VBO = None
        self.EBO = None
        self.circleVAO = None
        self.circleVBO = None
        self.points_queue = None
        self.data_points = None
        self.texture = None
        self.rectangleShader = None
        self.circleShader = None

    def setInt(self, shader, name, value):
        glUniform1i(glGetUniformLocation(shader, name), value)

    def setFloat(self, shader, name, value):
        glUniform1f(glGetUniformLocation(shader, name), value)

    def setVec3(self, shader, name, v1, v2, v3):
        glUniform3f(glGetUniformLocation(shader, name), v1, v2, v3)

    def setMat4(self, shader, name, value):
        glUniformMatrix4fv(glGetUniformLocation(shader, name), 1, GL_FALSE, value)

    def setArray(self, shader, name, size, value):
        glUniform1fv(glGetUniformLocation(shader, name), size, value)

    def compileShader(self, shader_source, shader_type):
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

    def linkShaderProgram(self, vertexShader, fragmentShader, name):
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

    def createShader(self, name, vertexSourceFile, fragmentSourceFile):
        vertexShaderString = open(vertexSourceFile, "r").read()
        fragmentShaderString = open(fragmentSourceFile, "r").read()
        vertexShader = self.compileShader(vertexShaderString, GL_VERTEX_SHADER)
        fragmentShader = self.compileShader(fragmentShaderString, GL_FRAGMENT_SHADER)
        shaderProgram = self.linkShaderProgram(vertexShader, fragmentShader, name)
        glUseProgram(shaderProgram)
        glDeleteShader(vertexShader)
        glDeleteShader(fragmentShader)
        return shaderProgram

    def compileShaders(self):
        self.rectangleShader = self.createShader("rectangle", "vertex.vert", "fragment.frag")
        self.circleShader = self.createShader("circle", "circle.vert", "circle.frag")

    def initRectangle(self):

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

        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        self.VAO = glGenVertexArrays(1)
        self.VBO = glGenBuffers(1)

        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)

        glBufferData(GL_ARRAY_BUFFER, len(vertices) * FLOAT_SIZE, vertices, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * FLOAT_SIZE, None)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * FLOAT_SIZE, ctypes.c_void_p(3 * FLOAT_SIZE))
        glEnableVertexAttribArray(1)

    def initCircle(self):

        vertices = []
        for i in range(CIRCLE_VERTEX_COUNT):
            angle = float(i) / CIRCLE_VERTEX_COUNT * 360
            x = math.cos(angle * math.pi / 180)
            y = math.sin(angle * math.pi / 180)
            vertices.append(x)
            vertices.append(y)
        vertices = np.array(vertices, dtype=np.float32)

        self.circleVAO = glGenVertexArrays(1)
        self.circleVBO = glGenBuffers(1)

        glBindVertexArray(self.circleVAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.circleVBO)

        glBufferData(GL_ARRAY_BUFFER, len(vertices) * FLOAT_SIZE, vertices, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * FLOAT_SIZE, None)
        glEnableVertexAttribArray(0)

    def initGeometry(self):
        self.initRectangle()
        self.initCircle()

    def setUniforms(self):

        orthoMatrix = pyrr.matrix44.create_orthogonal_projection(0, SCR_WIDTH, 0, SCR_HEIGHT, -1, 1, dtype=np.float32)

        glUseProgram(self.rectangleShader)
        self.setMat4(self.rectangleShader, "projection", orthoMatrix)

        glUseProgram(self.circleShader)
        self.setMat4(self.circleShader, "projection", orthoMatrix)

    def initOpenGL(self):
        self.compileShaders()
        self.initGeometry()
        self.setUniforms()

    def drawRectangle(self):

        glUseProgram(self.rectangleShader)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glBindVertexArray(self.VAO)

        try:

            data = self.points_queue.get(block=False)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glTexImage2D(
                GL_TEXTURE_2D,      # target
                0,                  # mipmap level
                GL_RED,             # internal format
                ARR_SIZE_X,         # width
                ARR_SIZE_Y,         # height
                0,                  # must be zero
                GL_RED,             # data format
                GL_FLOAT,           # data type, try GL_FLOAT in case of problems
                data                # pixel data
            )
            glGenerateMipmap(GL_TEXTURE_2D)

        except Exception as e:
            if type(e).__name__ != "Empty":
                print(e)

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 6)

    def drawCircle(self, point):

        glUseProgram(self.circleShader)
        glBindVertexArray(self.circleVAO)

        scaleFactorX = SCR_WIDTH / DATA_MAX_X
        scaleFactorY = SCR_HEIGHT / DATA_MAX_Y

        scaleMatrix = pyrr.matrix44.create_from_scale([CIRCLE_RADIUS, CIRCLE_RADIUS, 1.0], dtype=np.float32)
        translationMatrix = pyrr.matrix44.create_from_translation([point.x * scaleFactorX, point.y * scaleFactorY, 0.0], dtype=np.float32)
        modelMatrix = pyrr.matrix44.multiply(scaleMatrix, translationMatrix)
        self.setVec3(self.circleShader, "color", *point.color)
        self.setMat4(self.circleShader, "model", modelMatrix)
        glDrawArrays(GL_TRIANGLE_FAN, 0, CIRCLE_VERTEX_COUNT)

    def render(self):

        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        self.drawRectangle()

        for point in self.data_points:
            self.drawCircle(point)

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

    def start(self, queue, data_points):
        self.data_points = data_points
        process = multiprocessing.Process(target=self.initGLFW, args=(1, queue))
        process.start()
