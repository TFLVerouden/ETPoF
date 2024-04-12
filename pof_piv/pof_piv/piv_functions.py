from pof_piv.helper_functions import *
from pof_piv.plot_functions import *
import numpy as np


def simple_piv(images, window_size, calib_dist=None, calib_time=None,
               subpixel_method='gauss_neighbor', skip_errors=False,
               plot=False, plt_flow_params={}, plt_disp_params={}):
    """
    Perform a basic PIV analysis on two images.

    This function takes in two images and calculates the displacement of
    windows in the first image with respect to the second image. The
    displacement is calculated using the correlation between the two images
    and the subpixel method specified.

    PARAMETERS:
        images (np.array): Two images [c, y, x].
        window_size (int | tuple): Size of the windows [y, x].
        calib_dist (float): Calibration distance.
        calib_time (float): Time between the two images.
        subpixel_method (str): Subpixel method to use.
            Options are 'gauss_neighbor'.
        plot (bool): Whether to plot the flow field and displacements.
        plt_flow_params (dict): Parameters for the flow field plot.
        plt_disp_params (dict): Parameters for the displacements plot.

    RETURNS:
        velocities or displacements (np.array): Velocity or displacement
            vectors [j, i, y/x], depending on whether calibration data was
            supplied.
        coordinates (np.array): Coordinates of the windows [j, i, y/x].
    """

    # Check whether there are two or more images
    if images.shape[0] < 2:
        # Error
        raise ValueError('At least two images are required for PIV.')
    elif images.shape[0] > 2:
        # Warning
        raise NotImplementedError('Only two images are currently supported.')

    # Divide the images into windows
    windows, coordinates = divide_in_windows(images, window_size)

    # Calculate the correlation of each window [j, i] in frame 0 with the
    # corresponding window in frame 1
    correlations = np.array([[correlate_image_pair(windows[0, j, i],
                                                   windows[1, j, i], plot=False)
                              for i in range(windows.shape[2])]
                             for j in range(windows.shape[1])])

    # Calculate the displacement of each window [j, i] in frame 0 with the
    # corresponding window in frame 1
    displacements = np.array(
            [[find_displacement(correlation,
                                subpixel_method=subpixel_method,
                                skip_errors=skip_errors) for correlation in row]
             for row in correlations])

    if plot:
        # Plot the flow field
        plot_flow_field(displacements, coordinates, window_size,
                        **plt_flow_params)

        # Plot the displacements
        plot_displacements(displacements, **plt_disp_params)
        pass

    # Calculate the velocity field
    if calib_dist is not None and calib_time is not None:
        # Calibrate the displacement vectors
        displacements_calibrated = displacements * calib_dist

        # Calculate the velocity vectors
        velocities = displacements_calibrated / calib_time

        return velocities, coordinates
    else:
        return displacements, coordinates


def horizontal_flow_piv(images, window_counts, margins=[0, 0, 0, 0], debug=False):

    # Check whether there are two or more images
    if images.shape[0] < 2:
        # Error
        raise ValueError('At least two images are required for PIV.')
    elif images.shape[0] > 2:
        # Warning
        raise NotImplementedError('Only two images are currently supported.')

    # Divide the images into windows
    windows, centers, locations, sizes = divide_in_n_windows(images,
                                                             window_counts,
                                                             margins=margins)

    if debug:
        # Plot the images
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(images[0], cmap='gray')
        ax[1].imshow(images[1], cmap='gray')

        plt.show()

    # Check whether the windows in the vertical direction are of equal size
    try:
        assert np.all([np.all([window.shape[2] == windows[0][0].shape[2] for
                               window in row]) for row in windows])
    except AssertionError:
        raise ValueError(
            f'An image of width {images[0].shape[1] - margins[2] - margins[3]} could not be divided into {window_counts[1]} windows of equal width.')

    # plt.imshow(correlate_image_pair(images[0], images[1]))

    # Calculate the correlation of each window [j, i] in frame 0 with the
    # corresponding window in frame 1
    correlations = [[correlate_image_pair(windows[j][i][0],
                                              windows[j][i][1], plot=False)
                     for i in range(window_counts[1])]
                    for j in range(window_counts[0])]

    # Add up all correlation images in the horizontal direction
    correlation_rows = [np.sum(row, axis=0) for row in correlations]

    # # Average the x coordinates of the windows
    # centers = np.mean(centers, axis=1)

    # Calculate the displacement of each window [j, i] in frame 0 with the
    # corresponding window in frame 1
    displacements = np.array(
            [find_displacement(correlation, skip_errors=True) for correlation in
             correlation_rows])

    return displacements