import Imath
import OpenEXR

import numpy as np


def load_hdr(path):

    ext = path.split(".")[-1]

    if ext == "exr":
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        rgb_img_openexr = OpenEXR.InputFile(path)
        rgb_img = rgb_img_openexr.header()['dataWindow']
        size_img = (rgb_img.max.x - rgb_img.min.x + 1, rgb_img.max.y - rgb_img.min.y + 1)

        redstr = rgb_img_openexr.channel('R', pt)
        red = np.fromstring(redstr, dtype=np.float32)
        red.shape = (size_img[1], size_img[0])

        greenstr = rgb_img_openexr.channel('G', pt)
        green = np.fromstring(greenstr, dtype=np.float32)
        green.shape = (size_img[1], size_img[0])

        bluestr = rgb_img_openexr.channel('B', pt)
        blue = np.fromstring(bluestr, dtype=np.float32)
        blue.shape = (size_img[1], size_img[0])

        hdr_img = np.dstack((red, green, blue))
    elif ext == "rgbe" or ext == "hdr":
        import imageio
        hdr_img = imageio.imread(path, format="HDR-FI")
    else:
        raise RuntimeError("extension {} is not supported...".format(ext))

    return hdr_img