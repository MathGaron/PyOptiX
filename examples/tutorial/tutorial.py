import sys
from os.path import dirname
from OpenGL.GLUT import *
from OpenGL.GL import *
import math

from examples.hdr import load_hdr
from pyoptix.matrix4x4 import Matrix4x4

sys.path.append(dirname(dirname(dirname(__file__))))

import numpy as np
from pyoptix import Context, Compiler, Buffer, Program, Geometry, Material, GeometryInstance, EntryPoint, \
    GeometryGroup, Acceleration, TextureSampler
from examples.image_window_base import ImageWindowBase, calculate_camera_variables

ESCAPE_KEY = 27

width = 512
height = 512
Compiler.add_program_directory(dirname(__file__))

tutorial_number = 10
tutorial_file = "tutorial{}.cu".format(tutorial_number)


class ImageWindow(ImageWindowBase):
    def __init__(self, context, width, height):
        super().__init__(context, width, height)
        # will be called before display
        self.display_callbacks.append(self.set_camera)
        self.frame_number = 1
        self.mouse_button = None
        self.moues_prev_pose = None
        self.camera_eye = np.array([7.0, 9.2, -6.0], dtype=np.float32)
        self.camera_lookat = np.array([0., 4., 0.], dtype=np.float32)
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
        fov = 60.0
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
        self.buffer_numpy = np.zeros(15, dtype='<f4')

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
    create_light(context)

    entry_point.launch((width, height))

    window = ImageWindow(context, width, height)
    window.run()


def create_context():
    context = Context()

    context.set_ray_type_count(2)
    context.set_entry_point_count(1)
    context.set_stack_size(4640)

    context['max_depth'] = np.array(100, dtype=np.int32)
    context['radiance_ray_type'] = np.array(0, dtype=np.uint32)
    context['shadow_ray_type'] = np.array(1, dtype=np.uint32)
    context['scene_epsilon'] = np.array(1e-4, dtype=np.float32)
    context['importance_cutoff'] = np.array(0.01, dtype=np.float32)
    context['ambient_light_color'] = np.array([0.31, 0.33, 0.28], dtype=np.float32)

    context['output_buffer'] = Buffer.empty((height, width, 4), dtype=np.uint8, buffer_type='o', drop_last_dim=True)

    # Ray generation program
    camera_name = "env_camera" if tutorial_number >= 11 else "pinhole_camera"
    ray_generation_program = Program(tutorial_file, camera_name)

    # Exception program
    exception_program = Program(tutorial_file, "exception")

    # Miss program
    miss_name = "envmap_miss" if tutorial_number >= 5 else "miss"
    miss_program = Program(tutorial_file, miss_name)

    entry_point = EntryPoint(ray_generation_program, exception_program, miss_program)
    context['sqrt_num_samples'] = np.array(2, dtype=np.uint32)
    context['bad_color'] = np.array([1., 1., 1.], dtype=np.float32)
    context['bg_color'] = np.array([0.34, 0.55, 0.85], dtype=np.float32)

    hdr_image = load_hdr("../data/CedarCity.hdr")
    hdr_image = np.flip(hdr_image, axis=0)
    texture = np.zeros((hdr_image.shape[0], hdr_image.shape[1], 4), np.float32)
    texture[:, :, :3] = hdr_image
    tex_buffer = Buffer.from_array(texture, buffer_type='i', drop_last_dim=True)
    tex_sampler = TextureSampler(tex_buffer, wrap_mode='repeat', indexing_mode='normalized_coordinates',
                                 read_mode='normalized_float', filter_mode='linear')
    context['envmap'] = tex_sampler

    noise = np.random.uniform(0, 1, 64*64*64).astype(np.float32)
    tex_buffer = Buffer.from_array(noise.reshape(64, 64, 64), buffer_type='i', drop_last_dim=False)
    noise_sampler = TextureSampler(tex_buffer, wrap_mode='repeat', indexing_mode='normalized_coordinates',
                                   read_mode='normalized_float', filter_mode='linear')
    context["noise_texture"] = noise_sampler

    return context, entry_point


def create_parallelogram(anchor, offset1, offset2, intersect_program, bb_program):
    parallelogram = Geometry(bounding_box_program=bb_program, intersection_program=intersect_program)
    parallelogram.set_primitive_count(1)
    normal = np.cross(offset2, offset1)
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


def make_plane(n, p):
    n = normalize(n)
    d = -np.dot(n, p)
    ret = np.zeros(4, np.float32)
    ret[:3] = n
    ret[3] = d
    return ret


def create_geometry(context):
    pgram_bounding_box = Program('box.cu', 'box_bounds')
    pgram_intersection = Program('box.cu', 'box_intersect')

    # Create box
    box = Geometry(bounding_box_program=pgram_bounding_box, intersection_program=pgram_intersection)
    box.set_primitive_count(1)
    box["boxmin"] = np.array([-2., 0., -2.], np.float32)
    box["boxmax"] = np.array([2., 7., 2.], np.float32)

    # Create chull
    if tutorial_number >= 9:
        chull = Geometry(bounding_box_program=Program(tutorial_file, 'chull_bounds'),
                         intersection_program=Program(tutorial_file, 'chull_intersect'))
        chull.set_primitive_count(1)
        nsides = 6
        chplane = []
        radius = 1
        xlate = np.array([-1.4, 0, -3.7], dtype=np.float32)
        for i in range(nsides):
            angle = float(i)/float(nsides) * math.pi * 2.
            x = math.cos(angle)
            y = math.sin(angle)
            chplane.append(make_plane(np.array([x, 0, y], dtype=np.float32),
                                      np.array([x*radius, 0, y*radius], dtype=np.float32) + xlate))
        min = 0.02
        max = 3.5
        chplane.append(make_plane(np.array([0, -1, 0], dtype=np.float32),
                                  np.array([0, min, 0], dtype=np.float32) + xlate))
        angle = 5/nsides * math.pi * 2
        chplane.append(make_plane(np.array([math.cos(angle), 0.7, math.sin(angle)], dtype=np.float32),
                                  np.array([0, max, 0], dtype=np.float32) + xlate))
        plane_buffer = Buffer.from_array([x.tobytes() for x in chplane], buffer_type='i')
        chull["planes"] = plane_buffer
        chull["chull_bbmin"] = np.array([-radius + xlate[0], min + xlate[1], -radius + xlate[2]], dtype=np.float32)
        chull["chull_bbmax"] = np.array([radius + xlate[0], max + xlate[1], radius + xlate[2]], dtype=np.float32)

    # Floor geometry
    floor_geometry = create_parallelogram(np.array([-64., 0.01, -64.], dtype=np.float32),
                                          np.array([128., 0., 0.], dtype=np.float32),
                                          np.array([0., 0, 128.], dtype=np.float32),
                                          Program('parallelogram.cu', 'intersect'),
                                          Program('parallelogram.cu', 'bounds'))

    # Materials
    box_chname = "closest_hit_radiance0"
    if tutorial_number >= 8:
        box_chname = "box_closest_hit_radiance"
    elif tutorial_number >= 3:
        box_chname = "closest_hit_radiance3"
    elif tutorial_number >= 2:
        box_chname = "closest_hit_radiance2"
    elif tutorial_number >= 1:
        box_chname = "closest_hit_radiance1"

    closest_hit = {0: Program(tutorial_file, box_chname)}
    any_hit = None
    if tutorial_number >= 3:
        any_hit = {1: Program(tutorial_file, "any_hit_shadow")}
    box_matl = Material(closest_hit=closest_hit, any_hit=any_hit)
    box_matl["Ka"] = np.array([0.3, 0.3, 0.3], dtype=np.float32)
    box_matl["Kd"] = np.array([0.6, 0.7, 0.8], dtype=np.float32)
    box_matl["Ks"] = np.array([0.8, 0.9, 0.8], dtype=np.float32)
    box_matl["phong_exp"] = np.array(88, dtype=np.float32)
    box_matl["reflectivity_n"] = np.array([0.2, 0.2, 0.2], dtype=np.float32)

    floor_name = "closest_hit_radiance0"
    if tutorial_number >= 7:
        floor_name = "floor_closest_hit_radiance"
    elif tutorial_number >= 6:
        floor_name = "floor_closest_hit_radiance5"
    elif tutorial_number >= 4:
        floor_name = "floor_closest_hit_radiance4"
    elif tutorial_number >= 3:
        floor_name = "closest_hit_radiance3"
    elif tutorial_number >= 2:
        floor_name = "closest_hit_radiance2"
    elif tutorial_number >= 1:
        floor_name = "closest_hit_radiance1"

    closest_hit = {0: Program(tutorial_file, floor_name)}
    any_hit = None
    if tutorial_number >= 3:
        any_hit = {1: Program(tutorial_file, "any_hit_shadow")}
    floor_matl = Material(closest_hit=closest_hit, any_hit=any_hit)
    floor_matl["Ka"] = np.array([0.3, 0.3, 0.1], dtype=np.float32)
    floor_matl["Kd"] = np.array([194./255.*0.6, 186./255.*0.6, 151./255.*0.6], dtype=np.float32)
    floor_matl["Ks"] = np.array([0.4, 0.4, 0.4], dtype=np.float32)
    floor_matl["reflectivity"] = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    floor_matl["reflectivity_n"] = np.array([0.05, 0.05, 0.05], dtype=np.float32)
    floor_matl["phong_exp"] = np.array(88, dtype=np.float32)
    floor_matl["tile_v0"] = np.array([0.25, 0., 0.15], dtype=np.float32)
    floor_matl["tile_v1"] = np.array([-0.15, 0., 0.25], dtype=np.float32)
    floor_matl["crack_color"] = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    floor_matl["crack_width"] = np.array(0.02, dtype=np.float32)

    # Glass material
    if tutorial_number >= 9:
        closest_hit = {0: Program(tutorial_file, "glass_closest_hit_radiance")}
        any_hit = {1: Program(tutorial_file, "glass_any_hit_shadow")} if tutorial_number >= 10 else {1: Program(tutorial_file, "any_hit_shadow")}
        glass_matl = Material(closest_hit=closest_hit, any_hit=any_hit)

        glass_matl["importance_cutoff"] = np.array(1e-2, dtype=np.float32)
        glass_matl["cutoff_color"] = np.array([0.34, 0.55, 0.85], dtype=np.float32)
        glass_matl["fresnel_exponent"] = np.array(3., dtype=np.float32)
        glass_matl["fresnel_minimum"] = np.array(0.1, dtype=np.float32)
        glass_matl["fresnel_maximum"] = np.array(1.0, dtype=np.float32)
        glass_matl["refraction_index"] = np.array(1.4, dtype=np.float32)
        glass_matl["refraction_color"] = np.array([1., 1., 1.], dtype=np.float32)
        glass_matl["reflection_color"] = np.array([1., 1., 1.], dtype=np.float32)
        glass_matl["refraction_maxdepth"] = np.array(100, dtype=np.int32)
        glass_matl["reflection_maxdepth"] = np.array(100, dtype=np.int32)
        extinction = np.array([0.8, 0.89, 0.75], dtype=np.float32)
        glass_matl["extinction_constant"] = np.log(extinction)
        glass_matl["shadow_attenuation"] = np.array([0.4, 0.7, 0.4], dtype=np.float32)

    # GI
    geometry_instances = [GeometryInstance(box, box_matl),
                          GeometryInstance(floor_geometry, floor_matl)]

    if tutorial_number >= 9:
        geometry_instances.append(GeometryInstance(chull, glass_matl))

    geometry_group = GeometryGroup(children=geometry_instances)
    geometry_group.set_acceleration(Acceleration("NoAccel"))
    context['top_object'] = geometry_group
    context['top_shadower'] = geometry_group


def create_light(context):
    basic_lights = np.zeros(1,
                            dtype=[('pos', ('<f4', 3)), ('color', ('<f4', 3)), ('cast_shadow', '<u4'), ('padd', '<u4')])
    basic_lights["pos"] = [[-5., 60., -16.0]]
    basic_lights["color"] = [[1., 1., 1.]]
    basic_lights["cast_shadow"] = [1]

    light_buffer = Buffer.from_array([x.tobytes() for x in basic_lights], buffer_type='i')
    context["lights"] = light_buffer

def normalize(mat):
    return mat / np.linalg.norm(mat)


if __name__ == '__main__':
    main()
