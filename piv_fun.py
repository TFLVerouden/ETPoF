import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted
import cv2 as cv
from tqdm import tqdm
from tqdm import trange
import os
from scipy.optimize import curve_fit
from scipy import signal as sig


def read_image_directory(directory, prefix=None, image_type='png',
                         timing=False):
    """
    Read all images in a directory and store them in a 3D array.

    PARAMETERS:
        directory (str): Path to the directory containing the images.
        prefix (str): Prefix of the image files to read.
        image_type (str): Type of the image files to read.
        timing (bool): Whether to show a progress bar.

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
                                 cv.IMREAD_GRAYSCALE) for f in
                       tqdm(files, desc='Reading images', disable=not timing)],
                      dtype=np.uint64)

    return images


def subtract_background(images, background):
    """
    Subtract a background image from a set of images.

    PARAMETERS:
        images (np.array): Images [c, y, x].
        background (np.array): Background image [y, x].

    RETURNS:
        images (np.array): Images with background subtracted [c, y, x].
    """

    # Check whether the images and background have the same shape
    if images.shape[1:] != background.shape:
        # Error
        raise ValueError(
                'The images and background do not have the same shape.')

    # Subtract the background from the images
    images = images - background

    # Set any integer overflowed values to zero
    images[images > 255] = 0

    return images


def displacement_1d(correlation, axis=1, max_disp=None, ignore_disp=0,
                    ignore_only_if_mult=True, subpixel_method=None, plot=False):
    # # Cross-correlate the images
    # correlation = sig.correlate(image1, image0)
    # TODO: Possibly better to calculate manually the correlation along 1 axis

    if axis == 1:
        # Get a slice of the correlation image
        correlation_slice = correlation[correlation.shape[0] // 2, :]

        # If a maximum displacement is specified, cut the slice from the center
        if max_disp is not None:
            correlation_slice = correlation_slice[
                                correlation.shape[1] // 2 - max_disp:
                                correlation.shape[1] // 2 + max_disp]

        # Define the image center
        center = correlation_slice.shape[0] // 2
    else:
        raise NotImplementedError('Only axis=1 is implemented')

    # Get local maxima in the correlation slice
    prominence = 0.5 * np.std(correlation_slice)
    peaks = sig.find_peaks(correlation_slice, prominence=prominence, width=1)[0]

    # Filter out peaks that are too far from the image center
    if max_disp is not None:
        peaks = peaks[np.abs(peaks - center) < max_disp]

    # If a displacement to be ignored is specified, and ignore is set to always,
    # or there are multiple peaks...
    if ignore_disp is not None and (len(peaks) > 1 or not ignore_only_if_mult):
        # Remove the specified peak
        peaks = peaks[peaks != ignore_disp + center]

        # Sort the peaks by correlation value in decreasing order
        peaks = peaks[np.argsort(correlation_slice[peaks])[::-1]]

    # If there are no peaks, no PIV can be done
    if len(peaks) == 0:
        return np.nan

    # Use the brightest peak as the displacement
    peak = peaks[0]

    # If subpixel resolution is requested, calculate it
    if subpixel_method is not None:
        # Calculate the subpixel displacement
        correction = subpixel(correlation_slice, peak,
                              method=subpixel_method)

        # Add the subpixel displacement to the integer displacement
        peak = peak + correction

    # Subtract the image center to get the displacement
    displacement = peak - center

    # Plot the correlation slice and the peak
    if plot:
        if max_disp is None:
            max_disp = correlation.shape[1] // 2

        x = np.arange(correlation.shape[1]) - correlation.shape[1] // 2

        fig, ax = plt.subplots()
        ax.plot(x, correlation[correlation.shape[0] // 2, :], 'o-')
        ax.axvline(displacement, color='r')
        ax.set_xlim(-max_disp, max_disp)
        ax.set_xlabel('Displacement [px]')
        ax.set_ylabel('Correlation')
        plt.show()

    return displacement


def displacement_2d(correlation, max_disp=None, subpixel_method='gauss_neighbor', plot=False):
    # Plot the correlation map
    if plot:
        extent = [-correlation.shape[1] // 2, correlation.shape[1] // 2,
                    -correlation.shape[0] // 2, correlation.shape[0] // 2]

        fig, ax = plt.subplots()
        ax.imshow(correlation, extent=extent)
        ax.set_xlabel('dx [px]')
        ax.set_ylabel('dy [px]')
        plt.show()

    # If all values in the correlation array are zero...
    if np.all(correlation == 0):
        # Return a nan displacement
        return np.array([np.nan, np.nan])

    # Set all values outside of a circle with radius max_displ to zero
    if max_disp is not None:

        # Set max_disp to the maximum possible displacement if it is too large
        max_disp = min(max_disp, correlation.shape[0] // 2 - 1, correlation.shape[1] // 2 - 1)

        # Create a grid of distances from the center
        x, y = np.meshgrid(np.arange(correlation.shape[1]) - correlation.shape[1] // 2,
                            np.arange(correlation.shape[0]) - correlation.shape[0] // 2)
        r = np.sqrt(x ** 2 + y ** 2)

        # Set all values outside of the circle to zero
        correlation[r > max_disp] = 0

    # Get the pixel with maximum brightness
    peaks = np.argwhere(np.amax(correlation) == correlation)

    # If multiple equal maxima were found...
    if len(peaks) > 1:

        # Use the peak with the largest sum of its neighbours
        peak = peaks[np.argmax([np.sum(correlation[p[0] - 1:p[0] + 2,
                                       p[1] - 1:p[1] + 2]) for p in peaks])]

    else:
        # Take the first value if only one peak was found
        peak = peaks[0]

    # If the subpixel option was set...
    if subpixel_method is not None:
        # Try calculating the subpixel correction
        try:
            # Get slices along both axes
            x_slice = correlation[peak[0], :]
            y_slice = correlation[:, peak[1]]

            # Calculate the subpixel displacement
            correction = np.array([
                subpixel(x_slice, peak[1], method=subpixel_method),
                subpixel(y_slice, peak[0], method=subpixel_method)])

            # Add the subpixel displacement to the integer displacement
            peak = peak + correction

        except (ValueError, FloatingPointError):
            # If the subpixel calculation failed, return a nan displacement
            return np.array([np.nan, np.nan])

        # Reject values within the floating point error

        # Add the subpixel displacement to the integer displacement
        peak = peak + correction

    # Subtract the image center to get the displacement
    center = (np.array(correlation.shape - np.ones_like((1, 2))) / 2)
    displacement = peak - center

    return displacement


def shift_displaced_image(images, displacement, axis=1):
    # If the displacement is non-zero...
    if displacement != 0:

        # Shift the image
        if axis == 1:
            images[1, :, :] = np.roll(images[1, :, :], -displacement)

            # Zero the pixels that were shifted out of the image
            if displacement > 0:
                images[1, :, -displacement:] = 0
            else:
                images[1, :, :-displacement] = 0
        else:
            raise NotImplementedError('Only axis=1 is implemented')

    return images


def subpixel(array, peak_index, method='gauss_neighbor'):
    """
    """

    # Three-point offset calculation from the lecture
    if method == 'gauss_neighbor':

        # Raise error if numpy encounters an exception
        with np.errstate(divide='raise', invalid='raise'):

            # Get the neighbouring pixels
            neighbors = array[(peak_index - 1):(peak_index + 2)]

            # Calculate the three-point Gaussian correction
            correction = (0.5 * (np.log(neighbors[0]) - np.log(neighbors[2]))
                           / ((np.log(neighbors[0])) + np.log(neighbors[2]) -
                              2 * np.log(neighbors[1])))

            # # If any of the neighbours is zero, throw an error
            # if np.any(neighbors == 0):
            #     raise ValueError('Cannot calculate subpixel correction: one of the'\
            #                      ' neighbouring pixels is zero.')
            #
            # # Try if the denominator can be calculated
            # try:
            #     denom = ()
            #
            #
            # except FloatingPointError:
            #     print(neighbors)
            #     raise ValueError('Cannot calculate subpixel correction: one of the'\
            #                      ' neighbouring pixels is zero.')
            # try:
            #
            # except FloatingPointError:
            #     print(((np.log(neighbors[0])) + np.log(neighbors[2])
            #                - 2 * np.log(neighbors[1])))

    else:
        raise ValueError('Invalid method')

    return correction


def plot_flow_field(displacements, window_centers, background=None,
                    arrow_color='k', arrow_scale=1, zero_displ_thr=0,
                    highlight_radius_range=[np.nan, np.nan],
                    highlight_angle_range=[np.nan, np.nan],
                    highlight_color='b', calib_dist=None, units=None,
                    title='Flow field', timing=False):
    """
    Plot the flow field with displacements as arrows.

    This function takes in the same keyword arguments as plot_displacements,
    but ignores some of them.

    PARAMETERS:
        displacements (np.array): Displacement vectors [j, i, y/x].
        window_centers (np.array): Coordinates of the windows [j, i, y/x].
        background (np.array): Background image to plot the flow field on.
        arrow_color (str): Color of the arrows.
        arrow_scale (float): Scale of the arrows.
        zero_displ_thr (float): Threshold for displacements to be plotted
            as dots.
        highlight_radius_range (list): Range of magnitudes to highlight.
        highlight_angle_range (list): Range of angles to highlight.
        highlight_color (str): Color of the highlighted arrows and dots.
        calib_dist (float): Calibration distance.
        units (str): Units of the calibration distance.
        title (str): Title of the plot.
        timing (bool): Whether to show a progress bar.

    RETURNS:
        fig (plt.figure): Figure object.
        ax (plt.axis): Axis object.
    """

    # Plot all displacement vectors at the center of each window
    fig, ax = plt.subplots()

    # If a background image is supplied, add it to the plot
    if background is not None:
        ax.imshow(background, cmap='gray')

    # If a calibration distance was specified, calibrate all values
    if calib_dist is not None:
        displacements = displacements * calib_dist
        window_centers = window_centers * calib_dist

    # Show a grid with the outline of each window and an arrow in the centre
    # indicating the displacement
    arrow_param = calib_dist if calib_dist is not None else 1

    # Get a list of indices that should be coloured
    highlight = filter_displacements(displacements,
                                     radius_range=highlight_radius_range,
                                     angle_range=highlight_angle_range)

    # Get a list of indices that are below the zero-threshold
    zero_displ = np.nan_to_num(np.linalg.norm(displacements, axis=2)) <= zero_displ_thr

    # Plot the indices below the threshold as dots
    ax.scatter(window_centers[zero_displ & highlight, 1],
               window_centers[zero_displ & highlight, 0],
               c=highlight_color, marker='.', s=arrow_scale)
    ax.scatter(window_centers[zero_displ & ~highlight, 1],
               window_centers[zero_displ & ~highlight, 0],
               c=arrow_color, marker='.', s=arrow_scale)

    # Plot the flow field window by window
    for j in trange(window_centers.shape[0], desc='Plotting arrows',
                    disable=not timing):
        for i in range(window_centers.shape[1]):

            # If the displacement is above the zero-threshold, plot an arrow
            if np.linalg.norm(displacements[j, i]) > zero_displ_thr:
                # If the displacement should be highlighted, set the color
                color = highlight_color if highlight[j, i] else arrow_color

                # Calculate the start and end of the arrow
                arrow_start = np.array(
                        [window_centers[j, i][0] -
                         arrow_scale * 0.5 * displacements[j, i][0],
                         window_centers[j, i][1] -
                         arrow_scale * 0.5 * displacements[j, i][1]])

                # Plot the arrow
                ax.arrow(arrow_start[1], arrow_start[0],
                         arrow_scale * displacements[j, i][1],
                         arrow_scale * displacements[j, i][0],
                         width=1.5 * arrow_param,
                         head_width=10 * arrow_param,
                         head_length=7 * arrow_param,
                         fc=color, ec=color, lw=1)

    # Aspect ratio should be 1
    ax.set_aspect('equal')

    # Set limits
    # ax.set_xlim([0, np.max(window_centers[:, :, 1]) + window_size[1] / 2])
    # ax.set_ylim([np.max(window_centers[:, :, 0] + window_size[0] / 2), 0])

    # If a calibration distance was specified, add units to the labels
    if calib_dist is not None:
        ax.set_xlabel(f'x [{units}]')
        ax.set_ylabel(f'y [{units}]')
    else:
        ax.set_xlabel('x [px]')
        ax.set_ylabel('y [px]')

    # If an arrow scale was specified, add it to the title
    if arrow_scale != 1:
        title = title + f' (arrows scaled ×{arrow_scale})'
    ax.set_title(title)
    plt.show()

    return fig, ax


def plot_displacements(displacements,
                       highlight_radius_range=None,
                       highlight_angle_range=None,
                       highlight_color='b', calib_dist=None, units=None,
                       legend=None):
    """
    Plot the displacement vectors.

    This function takes in the same keyword arguments as plot_flow_field,
    but ignores some of them.

    PARAMETERS:
        displacements (np.array): Displacement vectors [j, i, y/x].
        highlight_radius_range (list): Range of magnitudes to highlight.
        highlight_angle_range (list): Range of angles to highlight.
        highlight_color (str): Color of the highlighted arrows and dots.
        calib_dist (float): Calibration distance.
        units (str): Units of the calibration distance.
        legend (list): Legend of the plot.

    RETURNS:
        fig (plt.figure): Figure object.
        ax (plt.axis): Axis object.
    """

    # Set all default values
    if legend is None:
        legend = ['Highlighted', 'Out of range']
    if highlight_angle_range is None:
        highlight_angle_range = [np.nan, np.nan]
    if highlight_radius_range is None:
        highlight_radius_range = [np.nan, np.nan]

    # Plot all displacement vectors
    fig, ax = plt.subplots()

    # If a calibration distance was specified, calibrate all values
    if calib_dist is not None:
        displacements = displacements * calib_dist

    # Get a list of indices that should be coloured
    highlight = filter_displacements(displacements,
                                     radius_range=highlight_radius_range,
                                     angle_range=highlight_angle_range)

    # Plot the indices below the threshold as dots
    ax.scatter(displacements[highlight, 1], displacements[highlight, 0],
               marker='^', s=10, color=highlight_color)
    ax.scatter(displacements[~highlight, 1], displacements[~highlight, 0],
               marker='o', s=10, color='k')

    # Draw zero lines
    ax.axhline(0, color='darkgrey', lw=0.5)
    ax.axvline(0, color='darkgrey', lw=0.5)

    # Pad the limits, but use only finite values
    displacements_finite = displacements[np.any(np.isfinite(displacements),
                                                axis=2)]
    ax.set_xlim([np.nanmin(displacements_finite[:, 1]) - 1,
                 np.nanmax(displacements_finite[:, 1]) + 1])
    ax.set_ylim([np.nanmin(displacements_finite[:, 0]) - 1,
                 np.nanmax(displacements_finite[:, 0]) + 1])

    # Pad the x limits to make the plot square
    ax.set_xlim([np.amin([ax.get_xlim()[0], ax.get_ylim()[0]]),
                 np.amax([ax.get_xlim()[1], ax.get_ylim()[1]])])

    ax.set_aspect('equal')

    # If a calibration distance was specified, add units to the labels
    if calib_dist is not None:
        ax.set_xlabel(f'Δx [{units}]')
        ax.set_ylabel(f'Δy [{units}]')
    else:
        ax.set_xlabel('Δx [px]')
        ax.set_ylabel('Δy [px]')

    # If points were highlighted, add a legend
    if np.any(highlight):
        ax.legend(legend)

    ax.set_title('All displacement vectors')
    plt.show()

    return fig, ax


def filter_displacements(displacements, radius_range=None,
                         angle_range=None, template=None):
    """
    Filter out displacement vectors based on their magnitude and angle.

    PARAMETERS:
        displacements (np.array): Displacement vectors [j, i, y/x].
        radius_range (list): Range of magnitudes to keep.
        angle_range (list): Range of angles to keep.
        template (np.array): Template to use for filtering.

    RETURNS:
        mask (np.array): Boolean mask [j, i] of the filtered vectors.
    """

    # MODE 1: Polar coordinates
    if template is None:
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

    # MODE 2: Template-based filtering
    else:
        raise NotImplementedError('Template-based filtering is not implemented')
