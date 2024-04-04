from pof_piv import *
import numpy as np


def simple_piv(images, window_size, calib_dist=None, calib_time=None,
               subpixel_method='gauss_neighbor', plot=False,
               plt_flow_params={}, plt_disp_params={}):
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

    # Check whether there is two or more images
    if images.shape[0] < 2:
        # Error
        raise ValueError('At least two images are required for PIV.')
    elif images.shape[0] > 2:
        # Warning
        raise NotImplementedError('Only two images are currently supported.')

    # Divide the images into windows
    windows, coordinates = divide_in_windows(images, window_size)
    # print(f'wind: {windows.shape}')
    # print(f'coord: {coordinates.shape}')

    # Calculate the correlation of each window [j, i] in frame 0 with the
    # corresponding window in frame 1
    correlations = np.array([[correlate_image_pair(windows[0, j, i],
                                                   windows[1, j, i])
                              for i in range(windows.shape[2])]
                             for j in range(windows.shape[1])])
    # print(f'corr: {correlations.shape}')

    # Calculate the displacement of each window [j, i] in frame 0 with the
    # corresponding window in frame 1
    displacements = np.array(
            [[find_displacement(correlation, subpixel_method=subpixel_method)
              for correlation in row] for row in correlations])
    # print(f'disp: {displacements.shape}')

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
