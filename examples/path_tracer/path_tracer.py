import sys
from os.path import dirname
from OpenGL.GLUT import *
from OpenGL.GL import *
import math

from pyoptix.matrix4x4 import Matrix4x4

sys.path.append(dirname(dirname(dirname(__file__))))

import numpy as np
from pyoptix import Context, Compiler, Buffer, Program, Geometry, Material, GeometryInstance, EntryPoint, \
    GeometryGroup, Acceleration
from examples.common import ImageWindowBase, calculate_camera_variables

ESCAPE_KEY = 27

width = 512
height = 512
Compiler.add_program_directory(dirname(__file__))



class ImageWindow(ImageWindowBase):
    def __init__(self, context, width, height):
        super().__init__(context, width, height)
        # will be called before display
        self.display_callbacks.append(self.set_camera)
        self.frame_number = 1
        self.mouse_button = None
        self.moues_prev_pose = None
        self.camera_eye = np.array([278, 273, -900], dtype=np.float32)
        self.camera_lookat = np.array([278, 273, 0], dtype=np.float32)
        self.camera_rotate = Matrix4x4()

    def glut_resize(self, w, h):
        global frame_number
        if self.width == w and self.height == h: return

        if w <= 0: w = 1
        if h <= 0: h = 1

        self.width = w
        self.height = h

        frame_number = 1
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


class ParallelogramLight:
    def __init__(self):
        # structure : [corner, v1, v2, normal, emission] (all float3)
        self.buffer_numpy = np.zeros(15, dtype=np.float32)

    @property
    def buffer(self):
        return self.buffer_numpy.tobytes()

    def set_corner(self, x, y, z):
        self.buffer_numpy[0] = x
        self.buffer_numpy[1] = y
        self.buffer_numpy[2] = z

    def set_v1_v2(self, v1, v2):
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)
        self.buffer_numpy[3:6] = v1
        self.buffer_numpy[6:9] = v2
        self.buffer_numpy[9:12] = normal

    def set_emission(self, r, g, b):
        self.buffer_numpy[12] = r
        self.buffer_numpy[13] = g
        self.buffer_numpy[14] = b


def main():
    context, entry_point = create_context()
    # context.set_print_enabled(True)
    # context.set_print_buffer_size(20000)
    create_geometry(context)

    entry_point.launch((width, height))

    window = ImageWindow(context, width, height)
    window.run()


def create_context():
    context = Context()

    context.set_ray_type_count(2)
    context.set_entry_point_count(1)
    context.set_stack_size(1800)

    context['scene_epsilon'] = np.array(1e-3, dtype=np.float32)
    context['pathtrace_ray_type'] = np.array(0, dtype=np.uint32)
    context['pathtrace_shadow_ray_type'] = np.array(1, dtype=np.uint32)
    context['rr_begin_depth'] = np.array(1, dtype=np.uint32)

    context['output_buffer'] = Buffer.empty((height, width, 4), dtype=np.float32, buffer_type='o', drop_last_dim=True)
    Program('optixPathTracer.cu', 'pathtrace_camera')
    entry_point = EntryPoint(Program('optixPathTracer.cu', 'pathtrace_camera'),
                             Program('optixPathTracer.cu', 'exception'),
                             Program('optixPathTracer.cu', 'miss'))
    context['sqrt_num_samples'] = np.array(2, dtype=np.uint32)
    context['bad_color'] = np.array([1000000., 0., 1000000.], dtype=np.float32)
    context['bg_color'] = np.zeros(3, dtype=np.float32)

    return context, entry_point


def create_parallelogram(anchor, offset1, offset2, intersect_program, bb_program):
    parallelogram = Geometry(bounding_box_program=bb_program, intersection_program=intersect_program)
    parallelogram.set_primitive_count(1)
    normal = np.cross(offset1, offset2)
    normal /= np.linalg.norm(normal)
    d = np.dot(normal, anchor)
    plane = np.zeros(4, dtype=np.float32)
    plane[:3] = normal
    plane[3] = d

    v1 = offset1 / np.dot(offset1, offset1)
    v2 = offset2 / np.dot(offset2, offset2)

    parallelogram["plane"] = plane
    parallelogram["anchor"] = anchor
    parallelogram["v1"] = v1
    parallelogram["v2"] = v2
    return parallelogram


def create_geometry_instance(geometry, material, variable, color):
    geometry_instance = GeometryInstance(geometry, material)
    geometry_instance[variable] = color
    return geometry_instance


def create_geometry(context):
    light = ParallelogramLight()
    light.set_corner(343., 548.6, 227.)
    light.set_v1_v2(np.array([-130., 0., 0.]), np.array([0., 0., 105.]))
    light.set_emission(15, 15, 5)
    light_buffer = Buffer.from_array([light.buffer_numpy.tobytes()], buffer_type='i')
    context["lights"] = light_buffer

    diffuse = Material(closest_hit={0: Program('optixPathTracer.cu', 'diffuse')},
                       any_hit={1: Program('optixPathTracer.cu', 'shadow')})

    diffuse_light = Material(closest_hit={0: Program('optixPathTracer.cu', 'diffuseEmitter')})

    pgram_bounding_box = Program('parallelogram.cu', 'bounds')
    pgram_intersection = Program('parallelogram.cu', 'intersect')

    # colors
    white = np.array([0.8, 0.8, 0.8], dtype=np.float32)
    green = np.array([0.05, 0.8, 0.05], dtype=np.float32)
    red = np.array([0.8, 0.05, 0.05], dtype=np.float32)

    geometry_instances = []

    # floor
    floor_geometry = create_parallelogram(np.array([0., 0.0, 0.], dtype=np.float32),
                                          np.array([0., 0.0, 559.2], dtype=np.float32),
                                          np.array([556., 0, 0.], dtype=np.float32),
                                          pgram_intersection, pgram_bounding_box)
    geometry_instances.append(create_geometry_instance(floor_geometry, diffuse, "diffuse_color", white))

    # ceiling
    ceiling_geometry = create_parallelogram(np.array([0., 548.8, 0.], dtype=np.float32),
                                            np.array([556., 0.0, 0.], dtype=np.float32),
                                            np.array([0., 0, 559.2], dtype=np.float32),
                                            pgram_intersection, pgram_bounding_box)
    geometry_instances.append(create_geometry_instance(ceiling_geometry, diffuse, "diffuse_color", white))

    # back wall
    bwall_geometry = create_parallelogram(np.array([0., 0., 559.2], dtype=np.float32),
                                          np.array([0., 548.8, 0.], dtype=np.float32),
                                          np.array([556., 0, 0.], dtype=np.float32),
                                          pgram_intersection, pgram_bounding_box)
    geometry_instances.append(create_geometry_instance(bwall_geometry, diffuse, "diffuse_color", white))

    # right wall
    rwall_geometry = create_parallelogram(np.array([0., 0., 0.], dtype=np.float32),
                                          np.array([0., 548.8, 0.], dtype=np.float32),
                                          np.array([0., 0, 559.2], dtype=np.float32),
                                          pgram_intersection, pgram_bounding_box)
    geometry_instances.append(create_geometry_instance(rwall_geometry, diffuse, "diffuse_color", green))

    # left wall
    lwall_geometry = create_parallelogram(np.array([556., 0., 0.], dtype=np.float32),
                                          np.array([0., 0., 559.2], dtype=np.float32),
                                          np.array([0., 548.8, 0.], dtype=np.float32),
                                          pgram_intersection, pgram_bounding_box)
    geometry_instances.append(create_geometry_instance(lwall_geometry, diffuse, "diffuse_color", red))

    # short block
    short_a = create_parallelogram(np.array([130.0, 165.0, 65.0], dtype=np.float32),
                                   np.array([-48., 0., 160], dtype=np.float32),
                                   np.array([160., 0.0, 49.], dtype=np.float32),
                                   pgram_intersection, pgram_bounding_box)
    short_b = create_parallelogram(np.array([290., 0., 114.], dtype=np.float32),
                                   np.array([0., 165., 0.0], dtype=np.float32),
                                   np.array([-50., 0.0, 158.], dtype=np.float32),
                                   pgram_intersection, pgram_bounding_box)
    short_c = create_parallelogram(np.array([130., 0., 65.], dtype=np.float32),
                                   np.array([0., 165., 0.], dtype=np.float32),
                                   np.array([160., 0., 49.], dtype=np.float32),
                                   pgram_intersection, pgram_bounding_box)
    short_d = create_parallelogram(np.array([82., 0., 225.], dtype=np.float32),
                                   np.array([0., 165., 0.], dtype=np.float32),
                                   np.array([48., 0., -160.], dtype=np.float32),
                                   pgram_intersection, pgram_bounding_box)
    short_e = create_parallelogram(np.array([240., 0., 272.], dtype=np.float32),
                                   np.array([0., 165., 0.], dtype=np.float32),
                                   np.array([-158., 0., -47.], dtype=np.float32),
                                   pgram_intersection, pgram_bounding_box)

    geometry_instances.append(create_geometry_instance(short_a, diffuse, "diffuse_color", white))
    geometry_instances.append(create_geometry_instance(short_b, diffuse, "diffuse_color", white))
    geometry_instances.append(create_geometry_instance(short_c, diffuse, "diffuse_color", white))
    geometry_instances.append(create_geometry_instance(short_d, diffuse, "diffuse_color", white))
    geometry_instances.append(create_geometry_instance(short_e, diffuse, "diffuse_color", white))

    # short block
    tall_a = create_parallelogram(np.array([426.0, 330.0, 247.0], dtype=np.float32),
                                  np.array([-158., 0., 49], dtype=np.float32),
                                  np.array([49., 0.0, 159.], dtype=np.float32),
                                  pgram_intersection, pgram_bounding_box)
    tall_b = create_parallelogram(np.array([423., 0., 247.], dtype=np.float32),
                                  np.array([0., 330., 0.0], dtype=np.float32),
                                  np.array([49., 0.0, 159.], dtype=np.float32),
                                  pgram_intersection, pgram_bounding_box)
    tall_c = create_parallelogram(np.array([472., 0., 406.], dtype=np.float32),
                                  np.array([0., 330., 0.], dtype=np.float32),
                                  np.array([-158., 0., 50.], dtype=np.float32),
                                  pgram_intersection, pgram_bounding_box)
    tall_d = create_parallelogram(np.array([314., 0., 456.], dtype=np.float32),
                                  np.array([0., 330., 0.], dtype=np.float32),
                                  np.array([-49., 0., -160.], dtype=np.float32),
                                  pgram_intersection, pgram_bounding_box)
    tall_e = create_parallelogram(np.array([265., 0., 296.], dtype=np.float32),
                                  np.array([0., 330., 0.], dtype=np.float32),
                                  np.array([158., 0., -49.], dtype=np.float32),
                                  pgram_intersection, pgram_bounding_box)

    geometry_instances.append(create_geometry_instance(tall_a, diffuse, "diffuse_color", white))
    geometry_instances.append(create_geometry_instance(tall_b, diffuse, "diffuse_color", white))
    geometry_instances.append(create_geometry_instance(tall_c, diffuse, "diffuse_color", white))
    geometry_instances.append(create_geometry_instance(tall_d, diffuse, "diffuse_color", white))
    geometry_instances.append(create_geometry_instance(tall_e, diffuse, "diffuse_color", white))

    shadow_group = GeometryGroup(children=geometry_instances)
    shadow_group.set_acceleration(Acceleration("Trbvh"))
    context['top_shadower'] = shadow_group

    # light
    light_geometry = create_parallelogram(np.array([343., 548.6, 227.], dtype=np.float32),
                                          np.array([-130., 0.0, 0.], dtype=np.float32),
                                          np.array([0., 0, 105.], dtype=np.float32),
                                          pgram_intersection, pgram_bounding_box)
    light_instance = create_geometry_instance(light_geometry,
                                              diffuse_light,
                                              "emission_color",
                                              np.array([15., 15., 5.], dtype=np.float32))

    group = GeometryGroup(children=(geometry_instances + [light_instance]))
    group.set_acceleration(Acceleration("Trbvh"))
    context['top_object'] = group


def normalize(mat):
    return mat / np.linalg.norm(mat)


if __name__ == '__main__':
    main()
