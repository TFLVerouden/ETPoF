from scipy import signal as sig
import matplotlib.pyplot as plt
import numpy as np
def correlate_image_pair(image0, image1, method='correlate', plot=False):
    """
    TODO: Add documentation
    """

    # Compute the correlation between the two images using two methods
    if method == 'correlate':
        # correlation = sig.correlate(image1 - np.mean(image1),
        #                             image0 - np.mean(image0))
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


def find_displacement(correlation, subpixel_method='gauss_neighbor'):
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
    TODO: Add documentation
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
        # neighbors = [np.where(neighbor == 0, 1, neighbor) for neighbor in neighbors]

        # Three-point Gaussian fit in both dimensions
        correction = [(0.5 * (np.log(neighbor[0]) - np.log(neighbor[2]))
                       / ((np.log(neighbor[0])) + np.log(neighbor[2]) -
                          2 * np.log(neighbor[1]))) for neighbor in neighbors]

    else:
        raise ValueError('Invalid method')

    return correction


def assume_squareness(size):
    """

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
    TODO: Add documentation
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

def filter_displacements(displacements, radius_range=[0, np.inf],
                         angle_range=[-np.pi, np.pi]):
    """
    TODO: Add documentation
    """

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