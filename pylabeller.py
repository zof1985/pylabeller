#! IMPORTS


import os
import sys
import h5py
from numpy.typing import NDArray
from PySide2.QtWidgets import QApplication
from pylabeller import Labeller, EllipseSegmenter, TriangleSegmenter


#! FUNCTIONS


def read_h5(file: str) -> NDArray:
    """
    read h5 files containing thermal images and return the readed data.

    Parameters
    ----------
    file: str
        the input file.

    Returns
    -------
    images: NDArray
        a (frames, height, width) NDArray
    """
    # check the file
    ext = file.rsplit(".", 1)[-1]
    if ext != "h5" or not os.path.isfile(file):
        raise TypeError("file must be a '.h5' file.")

    # get the thermal images
    with h5py.File(file, "r") as obj:
        images = obj["samples"][:]
    return images


#! MAIN


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # setup the labeller and the input reading formats
    face = EllipseSegmenter("FACE", (255, 0, 0, 255), 1, 1)
    resp = TriangleSegmenter("RESPIRATION", (0, 255, 0, 255), 1, 1)
    formats = {"h5": read_h5}
    labeller = Labeller([face, resp], **formats)

    # show the data
    labeller.show()
    sys.exit(app.exec_())
