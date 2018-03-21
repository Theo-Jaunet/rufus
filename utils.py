import os
import numpy as np
import skimage.color, skimage.transform


def preprocess(img, resolution):
    """# Format input with given resolution"""

    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img


def getfile(path, extensions):
    """# Recursive search of files with given extensions (must be an array)"""

    files = []
    file = extensions
    for f in os.listdir(path):
        if not os.path.isdir(path + f):
            ext = os.path.splitext(f)[1]
            if ext.lower() in file:
                files.append(path + f)
        else:
            files += getfile(path + f + "/", extensions)

    return files


if __name__ == '__main__':
    print(os.path.join(os.getcwd() + "/logs"))
    test = getfile(os.path.join(os.getcwd() + "/logs/"), [".pth"])
    print(test)
