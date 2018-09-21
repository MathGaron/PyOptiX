import sys
from os.path import dirname
sys.path.append(dirname(dirname(dirname(__file__))))

import numpy as np
from PIL import Image
from pyoptix import Context, Compiler, Buffer, Program, EntryPoint, TextureSampler
from examples.image_window_base import ImageWindowBase

Compiler.add_program_directory(dirname(__file__))


def main():
    tex_width = 64
    tex_height = 64

    trace_width = 512
    trace_height = 384

    context = Context()

    tex_data = []
    for j in range(tex_height):
        tex_data.append([])
        for i in range(tex_width):
            tex_data[j].append([
                (i + j) / (tex_width + tex_height) * 255,
                i / tex_width * 255,
                j / tex_height * 255,
                255
            ])

    tex_buffer = Buffer.from_array(np.array(tex_data, dtype=np.uint8), buffer_type='i', drop_last_dim=True)
    tex_sampler = TextureSampler(tex_buffer, wrap_mode='clamp_to_edge', indexing_mode='normalized_coordinates',
                                 read_mode='normalized_float', filter_mode='linear')

    context['input_texture'] = tex_sampler

    context['output_buffer'] = Buffer.empty((trace_height, trace_width, 4), dtype=np.float32,
                                            buffer_type='o', drop_last_dim=True)

    entry_point = EntryPoint(Program('draw_texture.cu', 'draw_texture'),
                             Program('draw_texture.cu', 'exception'))

    entry_point.launch((trace_width, trace_height))

    window = ImageWindowBase(context, trace_width, trace_height)
    window.run()

if __name__ == '__main__':
    main()
