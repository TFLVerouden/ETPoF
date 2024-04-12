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
        ax_extent = [-image0.shape[1] + 0.5, image0.shape[1] - 0.5,
                     -image0.shape[0] + 0.5, image0.shape[0] - 0.5]
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
                      skip_errors=False):
    """
    Find the displacement peak in a cross-correlation array.

    PARAMETERS:
        correlation (np.array): Cross-correlation array [y, x].
        subpixel_method (str): Method to use for subpixel refinement.
            Options are 'gauss_neighbor'.
        skip_errors (bool): Whether to skip errors in subpixel refinement.

    RETURNS:
        displacement (np.array): Displacement vector [y, x].
    """

    # If all values in the correlation array are zero...
    if np.all(correlation == 0):
        # Return a nan displacement
        return np.array([np.nan, np.nan])

    # Calculate the peak value indices of the cross-correlation
    peaks = np.argwhere(np.amax(correlation) == correlation)

    # If multiple maxima were found...
    if len(peaks) > 1:

        # Use the peak with the largest sum of its neighbours
        peak = peaks[np.argmax([np.sum(correlation[p[0] - 1:p[0] + 2,
                                        p[1] - 1:p[1] + 2]) for p in peaks])]

    else:
        # Take the first value if only one peak was found
        peak = peaks[0]

    # If the subpixel option was set...
    if subpixel_method is not None:
        # Try refining the peak location
        try:
            correction = subpixel_refinement(correlation, peak, subpixel_method)

        # If this gives an error, continue without refinement or throw an error
        except ValueError as e:
            if skip_errors:
                return np.array([np.nan, np.nan])
                # correction = np.zeros((2,))
            else:
                raise e

        peak = peak + np.array(correction)

    # Subtract the image center to get relative coordinates
    image_centre = (np.array(correlation.shape - np.ones_like((1, 2))) / 2)
    displacement = peak - image_centre

    return displacement


def intensity_at_peak(correlation, displacement):
    """
    Get the intensity at the peak of a cross-correlation array.

    PARAMETERS:
        correlation (np.array): Cross-correlation array [y, x].
        displacement (np.array): Displacement vector [y, x].

    RETURNS:
        intensity (float): Intensity at the peak of the cross-correlation.
    """

    # Convert the peak values back to indices
    image_centre = (np.array(correlation.shape - np.ones_like((1, 2))) / 2)
    peak = displacement + image_centre

    # Get the intensity at the peak indices
    intensity = correlation[int(peak[0]), int(peak[1])]

    return intensity


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

        # If any of the neighbors are zero
        if np.any([np.any(neighbor == 0) for neighbor in neighbors]):
            raise ValueError(
                    'Zero intensity pixels do not allow for Gaussian neighbor interpolation.')

        # If the neighbors shape is not 3x3...
        if not all([neighbor.shape == (3,) for neighbor in neighbors]):
            raise ValueError(
                    'Tried to calculate a maximum at the edge of the window.')

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


def divide_in_n_windows(images, window_counts, margins=[0, 0, 0, 0]):
    """
    Divide a set of images into windows of a given (equal!) size.

    PARAMETERS:
        images (np.array): Images [c, y, x].
        window_counts (tuple): Number of windows in each dimension [j, i].
        margins (list of ints): Number of pixels to cut off from the edges of the
            images [y0, y1, x0, x1].

    RETURNS:
        windows (list of lists): Windows [c, j, i, j_y, i_x].
        centers (np.array): Subpixel coordinates of the centre of each window [j, i, y/x].
        locations (np.array): Coordinates of the top-left corner of each window [y, x].
    """

    # Check whether one or multiple images was supplied
    if len(images.shape) == 2:
        # If only one image was supplied, add a dimension
        images = images[np.newaxis, :, :]

    # Cut off a number of pixels in each direction given by margins
    images = images[:, margins[0]:(images.shape[1] - margins[1]),
             margins[2]:(images.shape[2] - margins[3])]

    # Get the cropped image size
    crop_size = images.shape[1:]

    # Calculate the subpixel window size
    window_size = np.array(crop_size) / window_counts

    # Get the subpixel window centers
    centers_y = np.linspace(0.5 * window_size[0], crop_size[0]
                            - 0.5 * window_size[0], window_counts[0])
    centers_x = np.linspace(0.5 * window_size[1], crop_size[1]
                            - 0.5 * window_size[1], window_counts[1])

    # From these, calculate the top-left pixel indices of the windows within
    # the uncropped image
    locations = np.array([[[margins[0] + int(y - 0.5 * window_size[0]),
                            margins[2] + int(x - 0.5 * window_size[1])]
                           for x in centers_x] for y in centers_y])

    # Divide the images into windows of approximately equal size
    windows = [[images[:, int(y - 0.5 * window_size[0]):
                          int(y + 0.5 * window_size[0]),
                int(x - 0.5 * window_size[1]):
                int(x + 0.5 * window_size[1])]
                for x in centers_x] for y in centers_y]

    # Center coordinates
    centers = np.array([[[margins[0] + y, margins[2] + x] for x in centers_x]
                        for y in centers_y])

    # Finally, calculate the size of each window
    sizes = np.array([[window.shape[1:] for window in col] for col in windows])

    return windows, centers, locations, sizes


def divide_in_n_windows(images, window_counts, margins=[0, 0, 0, 0]):
    """
    Divide a set of images into windows of a given (equal!) size.

    PARAMETERS:
        images (np.array): Images [c, y, x].
        window_counts (tuple): Number of windows in each dimension [j, i].
        margins (list of ints): Number of pixels to cut off from the edges of the
            images [y0, y1, x0, x1].

    RETURNS:
        windows (list of lists): Windows [c, j, i, j_y, i_x].
        centers (np.array): Subpixel coordinates of the centre of each window [j, i, y/x].
        locations (np.array): Coordinates of the top-left corner of each window [y, x].
        sizes (np.array): Size of each window [j, i, y/x].
    """

    # Check whether one or multiple images was supplied
    if len(images.shape) == 2:
        # If only one image was supplied, add a dimension
        images = images[np.newaxis, :, :]

    # Cut off a number of pixels in each direction given by margins
    images = images[:, margins[0]:(images.shape[1] - margins[1]),
             margins[2]:(images.shape[2] - margins[3])]

    # Get the cropped image size
    crop_size = images.shape[1:]

    # Calculate the subpixel window size
    window_size = np.array(crop_size) / window_counts

    # Get the subpixel window centers
    centers_y = np.linspace(0.5 * window_size[0], crop_size[0]
                            - 0.5 * window_size[0], window_counts[0])
    centers_x = np.linspace(0.5 * window_size[1], crop_size[1]
                            - 0.5 * window_size[1], window_counts[1])

    # From these, calculate the top-left pixel indices of the windows within
    # the uncropped image
    locations = np.array([[[margins[0] + int(y - 0.5 * window_size[0]),
                            margins[2] + int(x - 0.5 * window_size[1])]
                           for x in centers_x] for y in centers_y])

    # Divide the images into windows of approximately equal size
    windows = [[images[:, int(y - 0.5 * window_size[0]):
                          int(y + 0.5 * window_size[0]),
                int(x - 0.5 * window_size[1]):
                int(x + 0.5 * window_size[1])]
                for x in centers_x] for y in centers_y]

    # Center coordinates
    centers = np.array([[[margins[0] + y, margins[2] + x] for x in centers_x]
                        for y in centers_y])

    # Finally, calculate the size of each window
    sizes = np.array([[window.shape[1:] for window in col] for col in windows])

    return windows, centers, locations, sizes


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
        # TODO: Implement template-based filtering


def remove_outliers(displacements, mask, windows, method='nan'):
    if method == 'nan':
        # Set the outliers to NaN
        displacements[~mask] = np.nan
    elif method == 'mean':
        # Set the outliers to the mean of the surrounding values
        raise NotImplementedError(
                'Mean-based outlier removal is not implemented')
        # TODO: Implement mean-based outlier removal

    return displacements


def shift_windows():
    pass


def merge_windows():
    pass


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
