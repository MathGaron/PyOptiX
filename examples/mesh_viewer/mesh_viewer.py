import sys
from os.path import dirname
from OpenGL.GLUT import *
from OpenGL.GL import *
import math

from examples.optix_mesh import OptixMesh
from pyoptix.matrix4x4 import Matrix4x4

sys.path.append(dirname(dirname(dirname(__file__))))

import numpy as np
from pyoptix import Context, Compiler, Buffer, Program, Geometry, Material, GeometryInstance, EntryPoint, \
    GeometryGroup, Acceleration
from examples.image_window_base import ImageWindowBase, calculate_camera_variables

ESCAPE_KEY = 27

width = 512
height = 512
Compiler.add_program_directory(dirname(__file__))
mesh_g = None


class ImageWindow(ImageWindowBase):
    def __init__(self, context, width, height):
        global mesh_g
        super().__init__(context, width, height)
        # will be called before display
        self.display_callbacks.append(self.set_camera)
        self.frame_number = 1
        self.mouse_button = None
        self.moues_prev_pose = None
        center = bb_center(mesh_g.bbox_min, mesh_g.bbox_max)
        size = bb_extent(mesh_g.bbox_min, mesh_g.bbox_max)
        max_dim = max(size[0], size[1])
        self.camera_lookat = center
        self.camera_eye = center + [0., 0., max_dim*1.5]
        self.camera_rotate = Matrix4x4()

    def glut_resize(self, w, h):
        if self.width == w and self.height == h: return

        if w <= 0: w = 1
        if h <= 0: h = 1

        self.width = w
        self.height = h

        self.frame_number = 1
        self.context["output_buffer"].set_size(self.width, self.height)
        glViewport(0, 0, self.width, self.height)
        glutPostRedisplay()

    def glut_keyboard_press(self, k, x, y):
        if ord(k) == ESCAPE_KEY:
            exit()

    def glut_mouse_press(self, button, state, x, y):
        if state == GLUT_DOWN:
            self.mouse_button = button
            self.mouse_prev_pose = (x, y)

    def glut_mouse_motion(self, x, y):
        if self.mouse_button == GLUT_RIGHT_BUTTON:
            dx = float(x - self.mouse_prev_pose[0]) / float(self.width)
            dy = float(y - self.mouse_prev_pose[1]) / float(self.height)
            dmax = dx if abs(dx) > abs(dy) else dy
            scale = min(dmax, 0.9)
            self.camera_eye = self.camera_eye + (self.camera_lookat - self.camera_eye) * scale
            self.frame_number = 1

        #todo implement arcball rotation...

        self.mouse_prev_pose = (x, y)

    def set_camera(self):
        camera_up = np.array([0, 1, 0], dtype=np.float32)
        fov = 35.0
        aspect_ratio = float(width) / float(height)

        # claculate camera variables
        W = self.camera_lookat - self.camera_eye
        wlen = np.sqrt(np.sum(W ** 2))
        U = normalize(np.cross(W, camera_up))
        V = normalize(np.cross(U, W))

        vlen = wlen * math.tan(0.5 * fov * math.pi / 180)
        V *= vlen
        ulen = vlen * aspect_ratio
        U *= ulen

        # compute transformations
        frame = Matrix4x4.from_basis(normalize(U),
                                     normalize(V),
                                     normalize(-W),
                                     self.camera_lookat)
        frame_inv = frame.inverse()

        #  apply transformation

        # print(frame.to_parameters(True))

        self.context["frame_number"] = np.array(self.frame_number, dtype=np.uint32)
        self.context["eye"] = np.array(self.camera_eye, dtype=np.float32)
        self.context["U"] = np.array(U, dtype=np.float32)
        self.context["V"] = np.array(V, dtype=np.float32)
        self.context["W"] = np.array(W, dtype=np.float32)

        self.frame_number += 1


def main():
    context, entry_point = create_context()
    context.set_print_enabled(True)
    context.set_print_buffer_size(20000)
    create_geometry(context)

    entry_point.launch((width, height))

    window = ImageWindow(context, width, height)
    window.run()


def create_context():
    context = Context()

    context.set_ray_type_count(2)
    context.set_entry_point_count(1)
    context.set_stack_size(1800)

    context['scene_epsilon'] = np.array(1e-4, dtype=np.float32)
    context['radiance_ray_type'] = np.array(0, dtype=np.uint32)
    context['shadow_ray_type'] = np.array(1, dtype=np.uint32)

    context['output_buffer'] = Buffer.empty((height, width, 4), dtype=np.uint8, buffer_type='o', drop_last_dim=True)
    entry_point = EntryPoint(Program('pinhole_camera.cu', 'pinhole_camera'),
                             Program('pinhole_camera.cu', 'exception'),
                             Program('constantbg.cu', 'miss'))
    context['bad_color'] = np.array([1., 0., 1.], dtype=np.float32)
    context['bg_color'] = np.array([0.34, 0.55, 0.85], dtype=np.float32)

    return context, entry_point


def create_geometry_instance(geometry, material, variable, color):
    geometry_instance = GeometryInstance(geometry, material)
    geometry_instance[variable] = color
    return geometry_instance


def create_geometry(context):
    global mesh_g
    mesh_g = OptixMesh()
    mesh_g.load_from_file("../data/dragon.obj")
    basic_lights = np.zeros(3,
                            dtype=[('pos', ('<f4', 3)), ('color', ('<f4', 3)), ('cast_shadow', '<u4'), ('padd', '<u4')])
    basic_lights["pos"] = [[-0.5, 0.25, -1.0],
                           [-0.5, 0.0, 1.0],
                           [0.5, 0.5, 0.5]]
    basic_lights["color"] = [[0.2, 0.2, 0.25],
                             [0.1, 0.1, 0.1],
                             [0.7, 0.7, 0.65]]
    basic_lights["cast_shadow"] = [0, 0, 1]

    light_buffer = Buffer.from_array([x.tobytes() for x in basic_lights], buffer_type='i')
    context["lights"] = light_buffer

    group = GeometryGroup(children=([mesh_g.geometry_instance]))
    group.set_acceleration(Acceleration("Trbvh"))
    context['top_object'] = group
    context['top_shadower'] = group

def bb_center(bb_min, bb_max):
    return (bb_max + bb_min) / 2


def bb_extent(bb_min, bb_max):
    return bb_max - bb_min


def normalize(mat):
    return mat / np.linalg.norm(mat)


if __name__ == '__main__':
    main()
