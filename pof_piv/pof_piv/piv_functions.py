from pof_piv.file_handling_functions import *
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


def horizontal_flow_piv(images, window_counts, margins=[0, 0, 0, 0],
                        debug=False):
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

    # Average the x coordinates of the windows
    centers = np.mean(centers, axis=1)

    # Calculate the displacement of each window [j, i] in frame 0 with the
    # corresponding window in frame 1
    displacements = np.array(
            [find_displacement(correlation, skip_errors=True) for correlation in
             correlation_rows])

    return displacements, centers


def video_piv(position_nr, series_nr, window_counts, frame_rate=40000,
              cutoff=None, margins=(0, 0, 0, 0), save=False, plot=True):
    # Read all images in the folder
    series_name = 'pos' + str(position_nr) + '-' + str(series_nr)
    directory = 'data/' + series_name

    images = read_image_directory(directory, image_type='tif', timing=True)
    background = cv.imread('data/backgrounds/' + series_name + '.tif',
                           cv.IMREAD_GRAYSCALE)

    # Subtract the background from all images
    images = subtract_background(images, background)

    # If it does not exist, create a subfolder with the series name in the
    # 'processed' directory
    if save:
        processed_directory = 'processed/' + series_name
        if not os.path.exists(processed_directory):
            os.makedirs(processed_directory)

    # Plot average pixel intensity of the cropped images over time
    intensity = np.mean(images[:, margins[0]:(images.shape[1] - margins[1]),
                        margins[2]:(images.shape[2] - margins[3])], axis=(1, 2))

    if plot:
        fig, ax = plt.subplots()
        ax.semilogy(np.arange(len(images)) / frame_rate, intensity, color='y')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Average intensity')
        # ax.hlines(0.2, 0, 1)
        # ax.vlines(0.15, 0, 255, linestyles='--')
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0.05, 255)
        plt.show()

    # Pre-allocate displacements array
    displacements = np.zeros((len(images) - 1, window_counts[0], 2))
    centers = np.zeros((len(images) - 1, window_counts[0], 2))

    # Loop over all images except the very final one
    for i in trange(len(images) - 1):
        # Calculate the PIV between the current and the next image
        displacements[i], centers[i] = horizontal_flow_piv(images[i:i + 2],
                                                           window_counts,
                                                           margins=margins)

    # Save the displacements to a file
    if save:
        np.save(processed_directory + '/displacements1.npy', displacements)
        np.save(processed_directory + '/centers1.npy', centers)

    # Plot all displacement vectors as scatter plot
    if plot:
        fig, ax = plt.subplots()
        ax.scatter(displacements[0:6000, :, 1], displacements[0:6000, :, 0],
                   marker='.', s=0.005)
        ax.set_aspect('equal')
        ax.set_xlabel('Δx (px)')
        ax.set_ylabel('Δy (px)')
        ax.set_title('Displacement vectors')
        plt.show()

    # Load the relevant calibration file
    calibration = np.load(f'data/calibration/pos{position_nr}.npy')

    # Load displacement data
    displacements = np.load(
            f'processed/pos{position_nr}-{series_nr}/displacements1.npy')

    # Plot the mean displacement magnitude over time
    if plot:
        fig, ax = plt.subplots()
        ax.hlines(0, -1, 1, color='k', lw=0.5)

        mean_displ = np.empty(len(images) - 1)
        for i in np.arange(len(images) - 1):
            mean_displ[i] = np.nanmean(
                    np.linalg.norm(
                            (displacements[i, (displacements[i, :, 0] >= 0), :]),
                            axis=1))

        ax.plot(np.arange(len(images) - 1) / frame_rate, mean_displ / calibration[
            0] * frame_rate / 1000 * 0.01 * 0.02 * 1000)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Flow rate (L/s)')
        ax.set_title('Flow over time (pos. components only)')
        ax.set_xlim(0, 0.5)
        plt.show()

    # Clear images from memory
    del images

    return displacements, centers
