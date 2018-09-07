from OpenGL.GLUT import *
from OpenGL.GLU import *
import OpenGL.GLUT.fonts
from OpenGL.GL import *
import time

from pyoptix.enums import Format


class ImageWindow:
    def __init__(self, context, width, height, name="window"):
        self.width = width
        self.height = height
        self.fps = -1.0
        self.last_frame_count = 0
        self.last_update_time = time.time()
        self.frame_count = 0
        self.context = context
        self.gl_tex_id = 0
        self.display_callbacks = []

        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE)
        glutInitWindowSize(width, height)
        glutInitWindowPosition(100, 100)
        glutCreateWindow(name)
        glutHideWindow()

    def run(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, 1, 0, 1, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glViewport(0, 0, self.width, self.height)

        glutShowWindow()
        glutReshapeWindow(self.width, self.height)

        # callbacks
        glutDisplayFunc(self.glutDisplay)
        glutIdleFunc(self.glutDisplay)
        #glutReshapeFunc(glutResize)
        #glutKeyboardFunc(glutKeyboardPress)
        #glutMouseFunc(glutMousePress)
        #glutMotionFunc(glutMouseMotion)

        #registerExitHandler()

        glutMainLoop()

    def glutDisplay(self, output_buffer_name="output_buffer"):
        # update camera
        # launch context
        for callback in self.display_callbacks:
            callback(self.context)
        self.context.launch(0, self.width, self.height)

        buffer = self.context[output_buffer_name]

        if self.gl_tex_id == 0:
            glGenTextures(1, self.gl_tex_id)
            glBindTexture(GL_TEXTURE_2D, self.gl_tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        glBindTexture(GL_TEXTURE_2D, self.gl_tex_id)

        pboId = buffer.get_GLBO_id()
        buffer_format = buffer.get_format()
        imageData = 0
        if pboId:
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboId)
        else:
            imageData = buffer.to_array()

        if buffer_format == Format.unsigned_byte4:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, imageData)
        elif buffer_format == Format.float4:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self.width, self.height, 0, GL_RGBA, GL_FLOAT, imageData)
        else:
            raise RuntimeError("Format not handled : {}".format(buffer_format))

        if pboId:
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        glEnable(GL_TEXTURE_2D)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex2f(0.0, 0.0)
        glTexCoord2f(1.0, 0.0)
        glVertex2f(1.0, 0.0)
        glTexCoord2f(1.0, 1.0)
        glVertex2f(1.0, 1.0)
        glTexCoord2f(0.0, 1.0)
        glVertex2f(0.0, 1.0)
        glEnd()
        glDisable(GL_TEXTURE_2D)

        #self.frame_count += 1
        #self.display_fps(self.frame_count)
        glutSwapBuffers()

    def display_fps(self, frame_count):
        current_time = time.time()
        if (current_time - self.last_update_time > 0.5):
            self.fps = (frame_count - self.last_frame_count) / (current_time - self.last_update_time )
            self.last_frame_count = frame_count
            self.last_update_time = current_time
        print(self.fps)


        text = "hello"
        blending = False
        if glIsEnabled(GL_BLEND):
            blending = True

        glLoadIdentity()
        gluOrtho2D(0.0, 1.0, 0.0, 1.0)
        glMatrixMode(GL_MODELVIEW)

        glColor3f(1, 1, 1)
        glRasterPos2f(0, 0)
        for ch in text:
            glutStrokeCharacter(GLUT_STROKE_MONO_ROMAN, ctypes.c_int(ord(ch)))

        if not blending:
            glDisable(GL_BLEND)


def calculate_camera_variables(eye, lookat, up, fov, aspect_ratio, fov_is_vertical=False):
    import numpy as np
    import math

    W = np.array(lookat) - np.array(eye)
    wlen = np.linalg.norm(W)
    U = np.cross(W, np.array(up))
    U /= np.linalg.norm(U)
    V = np.cross(U, W)
    V /= np.linalg.norm(V)

    if fov_is_vertical:
        vlen = wlen * math.tan(0.5 * fov * math.pi / 180.0)
        V *= vlen
        ulen = vlen * aspect_ratio
        U *= ulen
    else:
        ulen = wlen * math.tan(0.5 * fov * math.pi / 180.0)
        U *= ulen
        vlen = ulen * aspect_ratio
        V *= vlen

    return U, V, W

