from typing import Any, List

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
from natsort import natsorted
import scipy.signal as sig


# indexing: c (count), j (window vertical), i (window horizontal)
# y (vertical top-bottom), x (horizontal left-right)
def read_image_directory(directory, prefix=None, image_type='png'):
    """
    TODO: Add documentation
    (i, y, x)
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
                                 cv.IMREAD_GRAYSCALE) for f in files])

    return images


def correlate_image_pair(image0, image1, method='correlate', plot=False):
    """
    TODO: Add documentation
    """

    # Compute the correlation between the two images using two methods
    if method == 'correlate':
        correlation = sig.correlate(image1 - np.mean(image1),
                                    image0 - np.mean(image0))
    elif method == 'convolve':
        correlation = sig.fftconvolve(image1, image0[::-1, ::-1])
    else:
        raise ValueError('Invalid method')

    # If the plot option was set...
    if plot:
        # Plot the frames
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].imshow(image0, cmap='gray')
        ax[0].set_title('Frame 0')
        ax[0].set_xlabel('y [px]')
        ax[0].set_ylabel('x [px]')

        ax[1].imshow(image1, cmap='gray')
        ax[1].set_title('Frame 1')
        ax[1].set_xlabel('y [px]')
        ax[1].set_yticklabels([])
        plt.show()

        # Plot the correlation
        # TODO: Set size of extent correctly for inequal frame sizes
        ax_extent = [-image0.shape[1] + 0.5, image0.shape[1] - 0.5,
                     -image0.shape[0] + 0.5, image0.shape[0] - 0.5]

        # TODO: Set ticks based on image size
        ax_ticks = np.round(np.array(ax_extent) / 10) * 10

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(correlation, cmap='gray', interpolation='none',
                  extent=ax_extent)
        ax.set_xticks(np.arange(ax_ticks[0], ax_ticks[1] + 1, 10))
        ax.set_yticks(np.arange(ax_ticks[2], ax_ticks[3] + 1, 10))
        ax.set_title('Correlation')
        ax.set_xlabel('dy [px]')
        ax.set_ylabel('dx [px]')
        plt.show()

    return correlation


def find_displacement(correlation, subpixel_method='gauss_neighbor',
                      plot=False):
    """
    TODO: Add documentation
    """

    # Calculate the peak value of the cross-correlation
    peak = np.argwhere(np.amax(correlation) == correlation)

    # TODO: can be made faster with https://stackoverflow.com/a/58652335

    # If multiple maxima were found...
    if len(peak) > 1:
        # Error
        raise ValueError('Multiple equal maxima found in cross-correlation')
        # TODO: Handle multiple (neighbouring) maxima, if necessary
    else:
        # Take the first value if only one peak was found
        peak = peak[0]

    # If the subpixel option was set...
    if subpixel_method is not None:
        # Refine the peak location
        correction = subpixel_refinement(correlation, peak, subpixel_method)
        peak = peak + np.array(correction)

    # Subtract the image center to get relative coordinates
    image_centre = (np.array(correlation.shape - np.ones_like((1, 2))) / 2)
    displacement = peak - image_centre

    return displacement


def subpixel_refinement(correlation, peak, method='gauss_neighbor'):
    """
    TODO: Add documentation
    """

    # With given method...
    if method == 'gauss_neighbor':

        # Get the neighbouring pixels in both dimensions
        neighbors = [correlation[(peak[0] - 1):(peak[0] + 2), peak[1]],
                     correlation[peak[0], (peak[1] - 1):(peak[1] + 2)]]

        # Three-point Gaussian fit in both dimensions
        correction = [(0.5 * (np.log(neighbor[0]) - np.log(neighbor[2]))
                       / ((np.log(neighbor[0])) + np.log(neighbor[2]) -
                          2 * np.log(neighbor[1]))) for neighbor in neighbors]

    else:
        raise ValueError('Invalid method')

    return correction


def divide_in_windows(image, window_size, overlap=0):
    """
    TODO: Add documentation
    """

    # Check whether the image can be evenly divided into windows of this size
    if not np.all([np.mod(image.shape[j], window_size[j]) == 0 for j in
                   range(len(image.shape))]):
        # Error
        raise ValueError(f'A {window_size} window does not fit into the image.')

    # Get the coordinates of the top left pixel of each window
    coordinates = np.array([[[y, x] for y in range(0, image.shape[0],
                                                   window_size[0])]
                            for x in range(0, image.shape[1], window_size[1])])

    # Divide the image into windows of size window_size
    windows = np.array([image[y:(y + window_size[0]), x:(x + window_size[1])]
                        for y in range(0, image.shape[0], window_size[0]) for x
                        in range(0, image.shape[1], window_size[1])])

    # Reshape the windows array to be 4D
    windows = windows.reshape((image.shape[0] // window_size[0],
                               image.shape[1] // window_size[1], window_size[0],
                               window_size[1]))

    return windows, coordinates
