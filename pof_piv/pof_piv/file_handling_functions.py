import os
from natsort import natsorted
from natsort import index_natsorted
import cv2 as cv
import numpy as np


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


def read_image_sets(directory, prefix=None, image_type='png', grouping_ab=False):
    """
    Read sets of images in subdirectories and store them in a list of 3D arrays.

    PARAMETERS:
        directory (str): Path to the directory containing the image sets.
        prefix (str): Prefix of the image files to read.
        image_type (str): Type of the image files to read.

    RETURNS:
        image_sets (list): List of arrays (i, y, x) containing the image sets.
        set_names (list): List of strings containing subdirectory names.
    """

    # Initialize lists to store the image sets and their names
    images_sets = []
    set_names = []

    # Walk through the directory and its subdirectories
    for subdirectory, _, files in os.walk(directory):

        # Skip empty directories
        if len(files) < 2:
            continue

        # If a prefix is specified, filter the list of files
        if prefix is not None:
            files = [f for f in files if f.startswith(prefix)]

        # If specified, group files with common base names
        if grouping_ab & (len(files) > 2):

            # files = os.listdir(subdirectory)

            # Go through all files
            for file in files:
                # Get any files that have the same base name apart from the
                # character a or b at the end
                base_name = file.rstrip('ab.' + image_type)

                # If the base name is not in the list of set names...
                if base_name not in set_names:

                    # ...import the images
                    images = read_image_directory(directory, base_name,
                                                  image_type)
                    images_sets.append(images)

                    # Save the set name
                    set_names.append(base_name)

        else:
            # Read the images in the subdirectory
            images = read_image_directory(subdirectory, prefix, image_type=image_type)
            images_sets.append(images)

            # Save the subdirectory name
            set_names.append(subdirectory.split('/')[-1])

    # Look for any strings that exist within all set names
    common_str = os.path.commonprefix(set_names)

    # If a common string was found, remove it from the set names
    if common_str:
        set_names = [name.replace(common_str, '') for name in set_names]

    # Remove trailing underscores from the set names
    set_names = [name.rstrip('_') for name in set_names]

    # Naturally sort the set names and images by the set names
    sort_order = index_natsorted(set_names)
    image_sets = [images_sets[i] for i in sort_order]
    set_names = natsorted(set_names)

    return image_sets, set_names
