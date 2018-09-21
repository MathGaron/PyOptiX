import numpy as np
import trimesh
from pyoptix import Buffer, GeometryInstance, Geometry, Program, Material


class OptixMesh:
    """
    Does not support textures and material
    """
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
        mesh = trimesh.load(filename)
        self.bbox_min = mesh.bounding_box.bounds[0]
        self.bbox_max = mesh.bounding_box.bounds[1]
        self.n_triangles = mesh.faces.shape[0]
        self.n_vertices = mesh.vertices.shape[0]
        self.positions_buffer = Buffer.from_array([x.tobytes() for x in mesh.vertices.astype(np.float32)], buffer_type='i')
        self.tri_indices = Buffer.from_array([x.tobytes() for x in mesh.faces.astype(np.int32)], buffer_type='i')
        self.normals_buffer = Buffer.from_array([x.tobytes() for x in mesh.vertex_normals.astype(np.float32)], buffer_type='i')
        if "vertex_texture" in mesh.metadata:
            self.texcoord_buffer = Buffer.from_array([x.tobytes() for x in mesh.metadata["vertex_texture"].astype(np.int32)],
                                                    buffer_type='i')

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
