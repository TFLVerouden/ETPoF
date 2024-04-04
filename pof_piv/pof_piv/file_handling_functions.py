import os
from natsort import natsorted
import numpy as np
import cv2 as cv
def read_image_directory(directory, prefix=None, image_type='png'):
    """
    Read all images in a directory and store them in a 3D array.

    PARAMETERS:
        directory (str): Path to the directory containing the images.
        prefix (str): Prefix of the image files to read.
        image_type (str): Type of the image files to read.

    RETURNS:
        images (np.array): array (i, y, x) containing the images.
    """

    # Get a list of files in the directory
    files = os.listdir(directory)

    # If a prefix is specified, filter the list of files
    if prefix is not None:
        files = [f for f in files if f.startswith(prefix)]

    # If a type is specified, filter the list of files
    if image_type is not None:
        files = [f for f in files if f.endswith(image_type)]

    # Sort the files
    files = natsorted(files)

    # Read the images and store them in a 3D array
    images = np.array([cv.imread(os.path.join(directory, f),
                                 cv.IMREAD_GRAYSCALE) for f in files],
                      dtype=np.uint64)

    return images