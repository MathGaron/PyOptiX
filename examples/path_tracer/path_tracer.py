import sys
from os.path import dirname

import math

from pyoptix._driver import RTbuffertype, RTformat

from pyoptix.matrix4x4 import Matrix4x4

sys.path.append(dirname(dirname(dirname(__file__))))

import numpy as np
from pyoptix import Context, Compiler, Buffer, Program, Geometry, Material, GeometryInstance, EntryPoint, \
    GeometryGroup, Acceleration
from examples.common import ImageWindow, calculate_camera_variables

width = 1024
height = 768
camera_rotate = Matrix4x4()
Compiler.add_program_directory(dirname(__file__))

frame_number = 1


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
    context.set_print_enabled(True)
    context.set_print_buffer_size(20000)
    create_geometry(context)
    set_camera(context)

    entry_point.launch((width, height))

    window = ImageWindow(context, width, height)
    window.display_callbacks.append(set_camera)
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

    # floor
    floor_geometry = create_parallelogram(np.array([0., 0.0, 0.], dtype=np.float32),
                                          np.array([0., 0.0, 559.2], dtype=np.float32),
                                          np.array([556., 0, 0.], dtype=np.float32),
                                          pgram_intersection, pgram_bounding_box)
    floor_geometry['diffuse_color'] = white
    floor_instance = GeometryInstance(floor_geometry, diffuse)

    # ceiling
    ceiling_geometry = create_parallelogram(np.array([0., 548.8, 0.], dtype=np.float32),
                                          np.array([556., 0.0, 0.], dtype=np.float32),
                                          np.array([0., 0, 559.2], dtype=np.float32),
                                          pgram_intersection, pgram_bounding_box)
    ceiling_geometry['diffuse_color'] = white
    ceiling_instance = GeometryInstance(ceiling_geometry, diffuse)

    # back wall
    bwall_geometry = create_parallelogram(np.array([0., 0., 559.2], dtype=np.float32),
                                            np.array([0., 548.8, 0.], dtype=np.float32),
                                            np.array([556., 0, 0.], dtype=np.float32),
                                            pgram_intersection, pgram_bounding_box)
    bwall_geometry['diffuse_color'] = white
    bwall_instance = GeometryInstance(bwall_geometry, diffuse)

    shadow_group = GeometryGroup(children=[floor_instance, ceiling_instance, bwall_instance])
    shadow_group.set_acceleration(Acceleration("Trbvh"))
    context['top_shadower'] = shadow_group

    # light
    light_geometry = create_parallelogram(np.array([343., 548.6, 227.], dtype=np.float32),
                                          np.array([-130., 0.0, 0.], dtype=np.float32),
                                          np.array([0., 0, 105.], dtype=np.float32),
                                          pgram_intersection, pgram_bounding_box)
    light_geometry['emission_color'] = np.array([15., 15., 5.], dtype=np.float32)
    light_instance = GeometryInstance(light_geometry, diffuse_light)

    group = GeometryGroup(children=[floor_instance, ceiling_instance, light_instance, bwall_instance])
    group.set_acceleration(Acceleration("Trbvh"))
    context['top_object'] = group


def normalize(mat):
    return mat / np.linalg.norm(mat)


def set_camera(context):
    global frame_number
    camera_eye = np.array([278, 273, -900], dtype=np.float32)
    camera_lookat = np.array([278, 273, 0], dtype=np.float32)
    camera_up = np.array([0, 1, 0], dtype=np.float32)
    fov = 35.0
    aspect_ratio = float(width) / float(height)

    # claculate camera variables
    W = camera_lookat - camera_eye
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
                                 camera_lookat)
    frame_inv = frame.inverse()

    #  apply transformation

    #print(frame.to_parameters(True))

    context["frame_number"] = np.array(frame_number, dtype=np.uint32)
    context["eye"] = np.array(camera_eye, dtype=np.float32)
    context["U"] = np.array(U, dtype=np.float32)
    context["V"] = np.array(V, dtype=np.float32)
    context["W"] = np.array(W, dtype=np.float32)

    frame_number += 1


if __name__ == '__main__':
    main()
