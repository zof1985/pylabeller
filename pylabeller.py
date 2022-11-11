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


#! RUN
def run(path:str | None, segmenters:list | None, **formats) -> None:
    """
    Open the GUI allowing to interact with the provided labeller and images.

    Parameters
    ----------
    path: str | None, optional
        the path to the file to be automatically loaded once the labeller
        is opened.

    segmenters: Iterable[Segmenter] | Segmenter | None, optional
        pregenerated segmenters to be included in the Labeller.

    **formats: Keyworded arguments
        any number of keyworded arguments. Each should be the name of an
        extension file format accepted by the labeller. Each key must have
        an associated Method or Function that allows to extract the frames
        to be labelled from the selected file.
    """

    # setup the labeller GUI
    app = QApplication(sys.argv)
    labeller = Labeller(segmenters, **formats)

    # show the data
    labeller.show()

    # add the path
    if path is not None:
        labeller.set_input(path)

    # run until closed
    app.exec_()
    app.quit()
    del app


#! MAIN


if __name__ == "__main__":
    path = "sample.h5"
    face = EllipseSegmenter("FACE", (255, 0, 0, 255), 1, 1)
    resp = TriangleSegmenter("RESPIRATION", (0, 255, 0, 255), 1, 1)
    segmenters = [face, resp]
    formats = {"h5": read_h5}
    run(path, segmenters, **formats)
