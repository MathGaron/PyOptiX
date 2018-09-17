import numpy as np
import pymesh


class Mesh:
    def __init__(self):
        self.n_vertices = 0
        self.positions = None

        self.normals = None
        self.texcoords = None

        self.n_triangles = 0
        self.tri_indices = None

        self.bbox_min = None
        self.bbox_max = None

        self.n_materials = 0
        self.mat_params = None

    def initialize_dummy_mesh(self):
        self.n_vertices = 7
        self.positions = np.array([[-0.045526, 0.033273, -0.015044],
                                   [-0.043676, 0.033473, -0.014789],
                                   [-0.039585, 0.034060, -0.014388],
                                   [-0.038543, 0.033936, -0.014123],
                                   [-0.034647, 0.033750, -0.014075],
                                   [-0.064301, 0.065583, -0.013945],
                                   [0.026319, 0.068888, -0.011068]], dtype=np.float32)

        self.n_triangles = 5
        self.tri_indices = np.array([[0, 1, 5],
                                     [0, 5, 2],
                                     [2, 5, 6],
                                     [2, 6, 3],
                                     [3, 6, 4]], dtype=np.int32)

        self.bbox_max = np.max(self.positions, axis=0)
        self.bbox_min = np.min(self.positions, axis=0)

        return self


if __name__ == '__main__':
    import os
    mesh = pymesh.load_mesh("./data/cow.obj")

    print(mesh)
    print(mesh.bbox)

