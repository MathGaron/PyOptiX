import math
import numpy as np


class Matrix4x4:
    def __init__(self):
        self.matrix = np.eye(4, dtype=np.float32)

    @staticmethod
    def from_basis(u, v, w, c):
        matrix = Matrix4x4()
        for i in range(3):
            matrix.matrix[i, 0] = u[i]
            matrix.matrix[i, 1] = v[i]
            matrix.matrix[i, 2] = w[i]
            matrix.matrix[i, 3] = c[i]
        return matrix

    def inverse(self):
        ret = Matrix4x4()
        ret.matrix[0:3, 0:3] = self.matrix[0:3, 0:3].transpose()
        ret.matrix[0:3, 3] = -ret.matrix[0:3, 0:3].dot(self.matrix[0:3, 3])
        return ret

    def to_parameters(self, as_degree=False):
        x, y, z = self.matrix[0:3, 3]
        a, b, c = self.mat2euler(self.matrix[0:3, 0:3])
        if as_degree:
            a = math.degrees(a)
            b = math.degrees(b)
            c = math.degrees(c)
        ret = [x, y, z, a, b, c]
        return np.array(ret)

    @staticmethod
    def mat2euler(M, cy_thresh=None):
        M = np.asarray(M)
        if cy_thresh is None:
            try:
                cy_thresh = np.finfo(M.dtype).eps * 4
            except ValueError:
                cy_thresh = np.finfo(float).eps * 4.0
        r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
        cy = math.sqrt(r33 * r33 + r23 * r23)
        if cy > cy_thresh:
            z = math.atan2(-r12, r11)
            y = math.atan2(r13, cy)
            x = math.atan2(-r23, r33)
        else:
            z = math.atan2(r21, r22)
            y = math.atan2(r13, cy)
            x = 0.0

        return x, y, z
