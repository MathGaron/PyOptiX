import sys
from os.path import dirname
sys.path.append(dirname(dirname(dirname(__file__))))

import numpy as np
from PIL import Image
from pyoptix import Context, Buffer, Program, EntryPoint, Compiler
from examples.common import ImageWindowBase


Compiler.add_program_directory(dirname(__file__))


def main():
    width = 512
    height = 384

    context = Context()

    context.set_ray_type_count(1)

    context['output_buffer'] = Buffer.empty((height, width, 4), buffer_type='o', dtype=np.float32, drop_last_dim=True)

    ray_gen_program = Program('draw_color.cu', 'draw_solid_color')

    ray_gen_program['draw_color'] = np.array([0.462, 0.725, 0.0], dtype=np.float32)

    entry_point = EntryPoint(ray_gen_program)
    entry_point.launch(size=(width, height))

    window = ImageWindowBase(context, width, height)
    window.run()

if __name__ == '__main__':
    main()
