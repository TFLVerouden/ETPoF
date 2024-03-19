import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import spatial
from tqdm import tqdm
from natsort import natsorted


def read_image(file_path, roi=None, grayscale=True):
    """
    Read an image from a file and crop it to a region of interest (ROI).

    PARAMETERS:
        file_path (str): The path to the image file.
        roi (list): A list of four integers [x0, x1, z0, z1], specifying the
            region of interest (default: None).
        grayscale (bool): Whether to read the image in grayscale
            (default: True).

    RETURNS:
        array: The cropped image.
    """

    # If grayscale is specified, read the image in grayscale
    if grayscale:
        image = cv.imread(file_path, 0)

    # If not, read the image in color
    else:
        image = cv.imread(file_path)

    # Crop the image to the region of interest
    if roi is not None:
        image = image[roi[2]:roi[3], roi[0]:roi[1]]

    return image


def read_image_series(directory, prefix=None, roi=None, grayscale=True,
                      timing=True):
    """
    Read a series of images from a directory and crop them to a region of
    interest (ROI).

    PARAMETERS:
        directory (str): The path to the directory containing the images.
        prefix (str): A common prefix for the image files (default: None).
        roi (list): A list of four integers [x0, x1, z0, z1], specifying the
            region of interest (default: None).
        grayscale (bool): Whether to read the images in grayscale
            (default: True).
        timing (bool): Whether to print the time taken to read the images
            (default: False).

    RETURNS:
        tuple: A tuple containing a list of images and a list of file names.
    """

    # Get a list of all files in the directory
    files = os.listdir(directory)

    # If a prefix is specified, filter the list of files
    if prefix is not None:
        files = [f for f in files if f.startswith(prefix)]

    # Sort the list of files
    files = natsorted(files)

    # Read the images and store them in a list
    images = [read_image(os.path.join(directory, f), roi, grayscale=grayscale)
              for f in tqdm(files, disable=not timing, desc="Reading images")]

    return images, files


def all_distances(x, y=None):
    """Calculate the distance between all points in a 2D (x,y) space

    PARAMETERS:
        x (np.array): The x-coordinates of the points
        y (np.array): The y-coordinates of the points

    RETURNS:
        distances_px (np.array): The distances between all points
    """

    # If only one array is given, assume the other one contains only zeros
    if y is None:
        y = np.zeros_like(x)

    # Combine the x and y coordinates into a single array
    xy = np.column_stack([x, y])

    # Calculate the distance between all points
    distances = spatial.distance.cdist(xy, xy, 'euclidean')

    return distances


def weighted_avg_and_std(values, weights, mask_diagonal=True,
                         verbose=False, precision=6, units="mm"):
    """
    Calculate the weighted average and standard deviation of a set of values.

    PARAMETERS:
        values (array-like): The values for which to calculate the weighted
            average and standard deviation.
        weights (array-like): The weights corresponding to each value.
        mask_diagonal (bool): Whether to mask the diagonal of the
            distance matrix (default: True).
        verbose (bool): Whether to print the results (default: False).
        precision (int): In the verbose output, the number of decimal places to
            round the results to (default: 5).
        units (str): In the verbose output, the length units of the values
            (default: "mm").

    RETURNS:
        tuple: A tuple containing the weighted average and standard deviation.
    """

    # Mask the diagonal to avoid division by zero
    if mask_diagonal:
        mask = np.eye(values.shape[0], dtype=bool).__invert__()
    else:
        mask = np.ones_like(values, dtype=bool)

    # Calculate the weighted average
    average = np.average(values[mask], weights=weights[mask])

    # Calculate the weighted variance and standard deviation
    variance = np.average((values[mask] - average) ** 2, weights=weights[mask])
    deviation = np.sqrt(variance)

    if verbose:
        print(f"The average resolution is {average:.{precision}f} {units}/px,")
        print(f"with a standard deviation of {np.sqrt(variance):.{precision}f}"
              f" {units}/px ({100 * deviation / average:.2} %).")

    # Output the average and standard deviation
    return average, deviation


def intensity_peaks(profile, method='weighted_avg', threshold=85):
    """
    Find the peaks in an intensity profile.

    PARAMETERS:
        profile (array-like): The intensity profile.
        method (str): The method to use for peak detection
            (default: 'weighted_avg').
        threshold (float): The threshold for peak detection (default: 85).

    RETURNS:
        array: The indices of the peaks in the profile.
    """

    # METHOD #1: WEIGHTED INTENSITY AVERAGE
    if method == 'weighted_avg':

        # For each z coordinate...
        peak_indices = []
        peak_intensities = []
        current_group = []
        for idx in range(len(profile)):

            # Check whether the current row is above the threshold
            if profile[idx] > threshold:

                # If so, append the row number to the list of peak indices
                current_group.append(idx)
            else:

                # If not, check whether this is the end of a peak
                if len(current_group) > 0:
                    # Calculate the average intensity
                    peak_intensities.append(np.mean(profile[current_group]))

                    # Calculate an average peak coordinate, weighted by the
                    # intensity
                    peak_indices.append(
                            np.dot(profile[current_group], current_group) /
                            np.sum(profile[current_group]))

                    # Reset the current peak group
                    current_group = []

    # METHOD 2: GAUSSIAN FITTING
    elif method == 'gaussian_fit':
        raise NotImplementedError("Gaussian fitting has not been implemented.")

    # Throw error if wrong method is specified
    else:
        raise ValueError(f"Unknown method '{method}'")

    # Return the (subpixel precision) indices of the peaks and their intensities
    return peak_indices, peak_intensities


def calibrate_camera(image, calib_dist, threshold, peak_method='weighted_avg',
                     plot=False, file_name=None, verbose=False, units="mm",
                     precision=3):
    """
    Calibrate the resolution of a camera using a calibration image.

    PARAMETERS:
        image (array-like): The calibration image.
        calib_dist (float): The known distance between the peaks in the image.
        threshold (float): The threshold for peak detection.
        peak_method (str): The method to use for peak detection
        plot (bool): Whether to plot the results (default: False).
        file_name (str): When plotting, the name of the file to use as
            the figure title
        verbose (bool): Whether to print the results (default: False).
        precision (int): In the verbose output, the number of decimal places to
            round the results to (default: 5).
        units (str): In the verbose output, the length units of the values
            (default: "mm").

    RETURNS:
        tuple: A tuple containing the average resolution, standard deviation,
            and the line heights.
    """

    # Average all pixels in the x-direction to get the intensity profile
    int_profile = np.mean(image, axis=1)

    # Calculate the peaks in the intensity profile
    line_height, peak_intensities\
        = intensity_peaks(int_profile, method=peak_method, threshold=threshold)
    # Idea: the peak intensities could be used as a measure of error

    # Calculate the distances between all peaks
    dist_px = all_distances(line_height)

    # Define an array of distances in real units and calculate all distances
    peak_re = calib_dist * np.arange(0, len(line_height))
    dist_re = all_distances(peak_re)

    # Ignoring division by zero...
    with np.errstate(divide='ignore', invalid='ignore'):
        # ...calculate the resolutions of the image in units length per pixel
        res_matrix = dist_re / dist_px

    # Calculate the weighted average and standard deviation of the resolutions
    res_avg, res_std = weighted_avg_and_std(res_matrix, dist_px,
                                            verbose=verbose, units=units,
                                            precision=precision)

    # Plot the cropped image and intensity profile side-by-side
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(6, 9))

        # Use file name to generate figure title
        if file_name is not None:
            fig.suptitle(file_name)

        # Plot the cropped image
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Calibration (cropped)')
        ax[0].set_xlabel('x [px]')
        ax[0].set_ylabel('z [px]')

        # Plot the intensity profile
        ax[1].plot(int_profile, np.arange(0, len(int_profile)), '.')
        ax[1].set_title('Intensity profile')
        ax[1].set_xlabel('|I| (a.u.)')

        # Plot the intensity peaks
        ax[1].plot(peak_intensities, line_height, 'rx')

        # Add a vertical line for the threshold
        ax[1].axvline(threshold, color='r', linestyle='--')

        # Flip and scale the profile axis to match the image
        ax[1].invert_yaxis()
        ax[1].set_ylim([len(int_profile), 0])

        # Add an extra vertical axis on the right containing the values in mm
        ax2 = ax[1].twinx()
        ax2.set_ylabel('z [m]')
        ax2.invert_yaxis()
        ax2.set_ylim([len(int_profile) * res_avg / 1000, 0])

        # Add legend
        ax[1].legend(['Mean intensity', 'Avg. peaks', 'Threshold'],
                     loc='upper left')
        plt.show()

    return res_avg, res_std, line_height


def calibrate_cameras(directory, roi, calib_dist, threshold,
                      peak_method='weighted_avg',
                      file_prefix='Calibration',
                      plot=False, verbose=True, units="m", precision=7):
    """
    Calibrate the resolution of a series of cameras using calibration images.

    PARAMETERS:
        directory (str): The path to the directory containing the calibration
            images.
        roi (list): A list of four integers [x0, x1, z0, z1], specifying the
            region of interest.
        calib_dist (float): The known distance between the peaks in the image.
        threshold (float): The threshold for peak detection.
        peak_method (str): The method to use for peak detection
            (default: 'weighted_avg').
        file_prefix (str): A common prefix for the image files
            (default: 'Calibration').
        plot (list(bool)): Which camera's results to plot (default: False).
        verbose (bool): Whether to print the results (default: True).
        units (str): In the verbose output, the length units of the values
            (default: "m").
        precision (int): In the verbose output, the number of decimal places to
            round the results to (default: 7).
    """

    # Read the calibration images in grayscale [z, x]
    calib_images, files = read_image_series(directory, prefix=file_prefix,
                                            roi=roi, timing=False)
    no_images = len(calib_images)

    # Generate an array of plot flags
    if plot is True:
        plot = np.ones(no_images, dtype=bool)
    elif plot is False:
        plot = np.zeros(no_images, dtype=bool)
    elif isinstance(plot, (list, tuple, np.ndarray)):
        plot = np.array(plot, dtype=bool)
    else:
        raise ValueError("Invalid value for 'plot'")

    # Pre-allocate
    res_avg, res_std, offset = (np.zeros(no_images), np.zeros(no_images),
                                np.zeros(no_images))

    # For each calibration image...
    for idx, calib_image in enumerate(calib_images):
        # Print the camera name
        if verbose:
            print(f"==> {files[idx]}:")

        # Get the average resolution and standard deviation
        res_avg[idx], res_std[idx], line_height_idx \
            = calibrate_camera(calib_image, calib_dist, threshold,
                               peak_method=peak_method,
                               plot=bool(plot[idx]), file_name=files[idx],
                               verbose=verbose, units=units,
                               precision=precision)

        # Store the line heights of the first camera
        if idx == 0:
            line_height_0 = line_height_idx
            offset[0] = 0

        # If this is not the first camera, compare the line locations
        else:
            line_diff = np.array(line_height_idx) - np.array(line_height_0)

            # The offset is given by the mean of the differences
            offset[idx] = res_avg[idx] * np.mean(line_diff)

            # Print the offset
            if verbose:
                print(f"The offfset is {offset[idx]:.{precision}f} {units}.")

    return res_avg, res_std, offset
