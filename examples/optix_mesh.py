import numpy as np
import pymesh

from examples.mesh import Mesh
from pyoptix import Buffer, GeometryInstance, Geometry, Program, Material


class OptixMesh:
    def __init__(self):

        # setup programs
        use_texture = False
        self.closest_hit_program = Program("phong.cu",
                                           "closest_hit_radiance_textured" if use_texture else "closest_hit_radiance")
        self.any_hit_program = Program("phong.cu", "any_hit_shadow")
        pgram_bounding_box = Program('triangle_mesh.cu', 'mesh_bounds')
        pgram_intersection = Program('triangle_mesh.cu', 'mesh_intersect')

        # setup material and geometry
        self.material = Material(closest_hit={0: self.closest_hit_program},
                                 any_hit={1: self.any_hit_program})
        self.geometry = Geometry(bounding_box_program=pgram_bounding_box, intersection_program=pgram_intersection)


        # init buffers
        self.n_vertices = 0
        self.n_triangles = 0
        self.positions_buffer = Buffer.from_array([], dtype=np.dtype('f4, f4, f4'), buffer_type='i')
        self.tri_indices = Buffer.from_array([], dtype=np.dtype('i4, i4, i4'), buffer_type='i')
        self.normals_buffer = Buffer.from_array([], dtype=np.dtype('f4, f4, f4'), buffer_type='i')
        self.texcoord_buffer = Buffer.from_array([], dtype=np.dtype('i4, i4'), buffer_type='i')
        self.material_buffer = Buffer.from_array([], dtype=np.dtype('i4'), buffer_type='i')

    def load_from_file(self, filename):
        mesh = pymesh.load_mesh(filename)
        #print(mesh.attribute_names)
        #print(np.sum(np.isinf(mesh.vertices)))
        #print(mesh.vertices)
        #print(mesh.bbox)
        #print(mesh.num_faces)
        #exit()
        self.bbox_min = mesh.bbox[0]
        self.bbox_max = mesh.bbox[1]
        self.n_triangles = mesh.num_faces
        self.n_vertices = mesh.num_vertices
        self.positions_buffer = Buffer.from_array([x.tobytes() for x in mesh.vertices.astype(np.float32)], buffer_type='i')
        self.tri_indices = Buffer.from_array([x.tobytes() for x in mesh.faces], buffer_type='i')

        # setup material
        self.Kd = np.array([0.7, 0.7, 0.7], np.float32)
        self.Ks = np.array([0, 0, 0], np.float32)
        self.Kr = np.array([0, 0, 0], np.float32)
        self.Ka = np.array([0, 0, 0], np.float32)
        self.exp = np.zeros(1, np.float32)

        self.material["Kd_mapped"] = np.zeros(1, np.int32)
        self.material["Kd"] = self.Kd
        self.material["Ks"] = self.Ks
        self.material["Kr"] = self.Kr
        self.material["Ka"] = self.Ka
        self.material["exp"] = self.exp


        self.material_indices = np.zeros(self.n_triangles, np.int32)
        self.material_buffer = Buffer.from_array(self.material_indices, dtype=np.dtype('i4'), buffer_type='i')

        self.geometry.set_primitive_count(self.n_triangles)
        self.geometry["vertex_buffer"] = self.positions_buffer
        self.geometry["index_buffer"] = self.tri_indices
        self.geometry["normal_buffer"] = self.normals_buffer
        self.geometry["texcoord_buffer"] = self.texcoord_buffer
        self.geometry["material_buffer"] = self.material_buffer
        self.geometry_instance = GeometryInstance(self.geometry, self.material)
