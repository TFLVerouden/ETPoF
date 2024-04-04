from pof_piv import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig


def correlate_image_pair(image0, image1, method='correlate', plot=False):
    """
    Calculate the cross-correlation between two images.

    PARAMETERS:
        image0 (np.array): First image [y, x].
        image1 (np.array): Second image [y, x].
        method (str): Method to use for the cross-correlation.
            Options are 'correlate' and 'convolve'.
        plot (bool): Whether to plot the images and the correlation.

    RETURNS:
        correlation (np.array): Cross-correlation [y, x] between the two images.
    """

    # Compute the correlation between the two images using two methods
    if method == 'correlate':
        correlation = sig.correlate(image1, image0)
    elif method == 'convolve':
        correlation = sig.fftconvolve(image1, image0[::-1, ::-1])
    else:
        raise ValueError('Invalid method')

    # If the plot option was set...
    if plot:
        _, _ = plot_correlation(image0, image1, correlation)

    return correlation


def find_displacement(correlation, subpixel_method='gauss_neighbor'):
    """
    Find the displacement peak in a cross-correlation array.

    PARAMETERS:
        correlation (np.array): Cross-correlation array [y, x].
        subpixel_method (str): Method to use for subpixel refinement.
            Options are 'gauss_neighbor'.

    RETURNS:
        displacement (np.array): Displacement vector [y, x].
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

    # If the peak is at the edge of the correlation array...
    if np.any(peak == np.array(correlation.shape - np.ones_like((1, 2)))):
        # Throw a warning
        print('Peak is at the edge of the correlation array')

    # If the subpixel option was set...
    elif subpixel_method is not None:
        # Refine the peak location
        correction = subpixel_refinement(correlation, peak, subpixel_method)
        peak = peak + np.array(correction)

    # Subtract the image center to get relative coordinates
    image_centre = (np.array(correlation.shape - np.ones_like((1, 2))) / 2)
    displacement = peak - image_centre

    return displacement


def subpixel_refinement(correlation, peak, method='gauss_neighbor'):
    """
    Refine the peak location in a cross-correlation array to subpixel precision.

    PARAMETERS:
        correlation (np.array): Cross-correlation array [y, x].
        peak (np.array): Peak location [y, x].
        method (str): Method to use for subpixel refinement.
    """

    # Three-point offset calculation from the lecture
    if method == 'gauss_neighbor':

        # Get the neighbouring pixels in both dimensions
        neighbors = [correlation[(peak[0] - 1):(peak[0] + 2), peak[1]],
                     correlation[peak[0], (peak[1] - 1):(peak[1] + 2)]]

        # If the neighbors shape is not 3x3...
        if not all([neighbor.shape == (3,) for neighbor in neighbors]):
            fig, ax = plt.subplots()
            ax.imshow(correlation, cmap='gray')
            ax.plot(peak[1], peak[0], 'ro')
            plt.show()

        # Change all zeros to a small value to avoid division by zero
        # neighbors = [np.where(neighbor == 0, 1, neighbor)
        # for neighbor in neighbors]

        # Three-point Gaussian fit in both dimensions
        correction = [(0.5 * (np.log(neighbor[0]) - np.log(neighbor[2]))
                       / ((np.log(neighbor[0])) + np.log(neighbor[2]) -
                          2 * np.log(neighbor[1]))) for neighbor in neighbors]

    else:
        raise ValueError('Invalid method')

    return correction


def assume_squareness(size):
    """
    Assume that a window size is square, even if only one dimension is given.

    PARAMETERS:
        size (int or tuple): Size of a window in pixels. If an integer is
            supplied, the function turns this into a square window.

    RETURNS:
        size (np.array): Size of a window in pixels, as a 1D, 2-element array.
    """

    # Turn int or tuple into 1D array
    size = np.array([size]).flatten()

    # If the array has more than two elements, raise an error
    if size.size > 2:
        raise ValueError('The size should be a 1D array with 1 or 2 elements.')

    # If the size is a single element, assume a square window
    if len(size) == 1:
        size = np.array([size[0], size[0]])

    return size


def divide_in_windows(images, window_size):
    """
    Divide a set of images into windows of a given (equal!) size.

    PARAMETERS:
        images (np.array): Images [c, y, x].
        window_size (int or tuple): Size of a window in pixels.

    RETURNS:
        windows (np.array): Windows [c, j, i, j_y, i_x].
        coordinates (np.array): Coordinates of the centre of each window [y, x].
    """

    # Process window_size, which may be an integer or tuple
    window_size = assume_squareness(window_size)

    # Check whether one or multiple images was supplied
    if len(images.shape) == 2:
        # If only one image was supplied, add a dimension
        images = images[np.newaxis, :, :]

    # Check whether the images can be evenly divided into windows of this size
    if not np.all([np.mod(images.shape[j + 1], window_size[j]) == 0 for j in
                   range(len(images.shape) - 1)]):
        # Error
        raise ValueError(
                f'A {window_size} window does not fit into the images.')

    # Get the coordinates of the top left pixel of each window
    coordinates = np.array(
            [[[y, x] for x in range(0, images.shape[2], window_size[1])] for y
             in range(0, images.shape[1], window_size[0])])

    # Get the coordinates of the centre of each window
    coordinates = coordinates + np.array(window_size) / 2

    # Divide the images into windows of size window_size
    windows = np.array(
            [[[images[z, y:(y + window_size[0]), x:(x + window_size[1])]
               for x in range(0, images.shape[2], window_size[1])] for y in
              range(0, images.shape[1], window_size[0])] for z in
             range(images.shape[0])])

    return windows, coordinates


def filter_displacements(displacements, radius_range=None,
                         angle_range=None):
    """
    Filter out displacement vectors based on their magnitude and angle.

    PARAMETERS:
        displacements (np.array): Displacement vectors [j, i, y/x].
        radius_range (list): Range of magnitudes to keep.
        angle_range (list): Range of angles to keep.

    RETURNS:
        mask (np.array): Boolean mask [j, i] of the filtered vectors.
    """

    # Set default values
    if angle_range is None:
        angle_range = [-np.pi, np.pi]
    if radius_range is None:
        radius_range = [0, np.inf]

    # Calculate the magnitude and angle of the displacement vectors
    magnitudes = np.linalg.norm(displacements, axis=2)
    angles = np.arctan2(displacements[:, :, 1], displacements[:, :, 0])

    # If only nans are given, skip the filtering
    if np.all(np.isnan(radius_range + angle_range)):
        mask = np.zeros(displacements.shape[:2], dtype=bool)

    # Filter the displacements based on the given radius and angle ranges
    else:
        # Create a mask the same size as displacements
        mask = np.ones(displacements.shape[:2], dtype=bool)

        if not np.isnan(radius_range[0]):
            mask = mask & (magnitudes > radius_range[0])
        if not np.isnan(radius_range[1]):
            mask = mask & (magnitudes < radius_range[1])
        if not np.isnan(angle_range[0]):
            mask = mask & (angles > angle_range[0])
        if not np.isnan(angle_range[1]):
            mask = mask & (angles < angle_range[1])

    # Return the mask
    return mask
