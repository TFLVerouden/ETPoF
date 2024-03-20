import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit

from . import all_distances, read_image_series


def remove_hits(img, mode):
    """ Remove specific patterns from an image using a hit-or-miss transform.

    ARGUMENTS:
        img (np.array): The image to be processed.
        mode (str): The mode of the hit-or-miss transform to be applied.

    RETURNS:
        img (np.array): The processed image.
    """

    # Define different kernels
    if mode == 'single':
        kernel = np.array([[0, -1, 0],
                           [-1, +1, -1],
                           [0, -1, 0]])
    elif mode == 'horizontal':
        kernel = np.array([[-1, -1, -1],
                           [0, +1, 0],
                           [-1, -1, -1]])
    elif mode == 'vertical':
        kernel = np.array([[-1, 0, -1],
                           [-1, +1, -1],
                           [-1, 0, -1]])
    else:
        raise ValueError('Mode not recognized')

    # Binarize the image
    img_binary = cv.threshold(img, 0, 255, cv.THRESH_BINARY)[1]

    # Perform a hit-or-miss transform on a binary version
    #   to remove matching pixels
    img_singles = cv.morphologyEx(img_binary, cv.MORPH_HITMISS, kernel)

    # Edit the original image
    img[img_singles == 255] = 0

    return img


def remove_background(img, contours=None, sigma_color1=50, sigma_space1=9,
                      sigma_color2=20, sigma_space2=3, noise_threshold=50):
    """
    Remove background from an image using a bilateral filter and
    a hit-or-miss transform.

    ARGUMENTS:
        img (np.array): The image to be processed.
        contours (list): The contours of the particles in the image
            (default: None).
        sigma_color1 (int): The color sigma of the first bilateral filter
            (default: 50).
        sigma_space1 (int): The space sigma of the first bilateral filter
            (default: 9).
        sigma_color2 (int): The color sigma of the second bilateral filter
            (default: 20).
        sigma_space2 (int): The space sigma of the second bilateral filter
            (default: 3).
        noise_threshold (int): The threshold for the to-zero threshold
            (default: 50).

    RETURNS:
        img_proc (np.array): The processed image.
    """

    # MODE 1: If contours are not given, clean up the image
    if contours is None:

        # Apply a bilateral filter to remove noise while keeping edges sharp
        img_proc = cv.bilateralFilter(img, d=-1, sigmaColor=sigma_color1,
                                      sigmaSpace=sigma_space1)

        # Apply another bilateral filter for finer detail removal
        img_proc = cv.bilateralFilter(img_proc, d=-1, sigmaColor=sigma_color2,
                                      sigmaSpace=sigma_space2)

        # Set all pixels below a certain threshold to zero with a
        #   to-zero threshold
        _, img_proc = cv.threshold(img_proc, noise_threshold, 255,
                                   cv.THRESH_TOZERO)

        # Remove a bunch of specific patterns
        img_proc = remove_hits(img_proc, 'horizontal')
        img_proc = remove_hits(img_proc, 'vertical')
        # img_proc = remove_hits(img_proc, 'single') # This one is not necessary

    # MODE 2: If contours are provided, remove all pixels but the contours
    else:
        # Create a mask of the contours and their interiors
        mask = np.zeros_like(img)
        mask = cv.drawContours(mask, contours, -1, 255, -1)

        # Set the pixels outside the contours to zero
        img_proc = np.where(mask == 255, img, 0)

    return img_proc


def remove_contours(img, contours, keep_index=None):
    """
    Remove specific contours from an image.

    ARGUMENTS:
        img (np.array): The image to be processed.
        contours (list): The contours to be removed.
        keep_index (int): The index of the contour to be kept (default: None).

    RETURNS:
        img_proc (np.array): The processed image.
    """

    # If a specific contour is to be kept, remove it from the list
    if keep_index is not None:
        contours = [contour for i, contour in enumerate(contours)
                    if i != keep_index]

    # Set the pixels on the contours to zero
    img_proc = img.copy()
    img_proc = cv.drawContours(img_proc, contours, -1, 0, -1)

    return img_proc


def remove_sublists(list_of_lists):
    """
    Remove sublists and duplicates from a list of lists.
    Source: https://geeksforgeeks.org/python-remove-sublists-that-are-present-in-another-sublist/

    ARGUMENTS:
        list_of_lists: list of lists (might be inhomogeneous)

    RETURNS:
        results: list of lists with sublists removed
    """

    # Pre-allocation
    current_sublist = []
    results = []

    # Convert each element to a set and sort by descending length
    for element in sorted(map(set, list_of_lists), key=len, reverse=True):

        # Add only sets that are not already part of current_sublist
        if not any(element <= number for number in current_sublist):
            current_sublist.append(element)
            results.append(list(element))

    return results


def split_clusters(peak_indices, neighbor_dist=1.5):
    """
    Split a set of maxima into clusters based on their Euclidean distance.

    ARGUMENTS:
        peak_indices (np.array): The indices of the maxima
            [coordinate, peak index].
        neighbor_dist (float): The maximum distance between maxima to be
            considered part of the same cluster (default: 1.5).

    RETURNS:
        peak_clusters (list): A list of lists, where each sublist contains
            the indices of the peaks that belong together in a cluster.
    """

    # Check the Euclidian distances between the maxima
    dists = all_distances(peak_indices[0, :], peak_indices[1, :])

    # Get the indices of all the groups that are within a certain distance
    peak_clusters = [[i for i, val in enumerate(dists[:, col])
                      if val < neighbor_dist] for col in range(dists.shape[1])]

    # Remove subsets and duplicates
    peak_clusters = remove_sublists(peak_clusters)

    return peak_clusters


def get_neighbors(peak_indices, peak_indices_new, peak_intensities_new,
                  neighbor_dist=1.5):
    """
    Get the points in a new set of maxima that are neighbors of the points in
    the original set.

    ARGUMENTS:
        peak_indices (np.array): The indices of the original maxima.
        peak_indices_new (np.array): The indices of the new maxima.
        peak_intensities_new (np.array): The intensities of the new maxima.
        neighbor_dist (float): The maximum distance between maxima to be
            considered neighbors (default: 1.5).

    RETURNS:
        peak_indices_new (np.array): The indices of the new maxima that are
            neighbors of the original maxima.
        peak_intensities_new (np.array): The intensities of the new maxima that
            are neighbors of the original maxima.
    """

    # Initialize an empty list to store the indices of peak_indices_new
    #   that are within 1 pixel distance
    indices_within_1_pixel = []

    # Iterate over each point in peak_indices_new
    for i, point_new in enumerate(peak_indices_new.T):
        # Calculate the Euclidean distance to all points in peak_indices
        distances = np.sqrt(
                np.sum((peak_indices - point_new.reshape(-1, 1)) ** 2, axis=0))

        # If the minimum distance is less than or equal to a given distance,
        # append the index to indices_within_1_pixel
        if np.min(distances) <= neighbor_dist:
            indices_within_1_pixel.append(i)

    # Select the points from peak_indices_new that are within 1 pixel distance
    peak_indices_new = peak_indices_new[:, indices_within_1_pixel]

    # Also get the corresponding intensities
    peak_intensities_new = peak_intensities_new[indices_within_1_pixel]

    return peak_indices_new, peak_intensities_new


def destroy_neighbors(box_indices, box_intensities, peak_indices,
                      peak_intensities, max_iter=3, neighbor_dist=1.5):
    """
    Iteratively remove maxima in a cluster until only one maximum remains.

    ARGUMENTS:
        box_indices (np.array): The indices of the region.
        box_intensities (np.array): The intensities of the region.
        peak_indices (np.array): The indices of the maxima.
        peak_intensities (np.array): The intensities of the maxima.
        max_iter (int): The maximum number of iterations to remove neighbouring
            maxima (default: 3).
        neighbor_dist (float): The maximum distance between maxima to be
            considered part of the same cluster (default: 1.5).

    RETURNS:
        peak_indices (np.array): The indices of the final maxima.
        peak_intensities (np.array): The intensities of the final maxima.
    """

    # Keep track of the number of iterations
    iteration = 0

    # If there are (still) neighbors in the set of maxima...
    while peak_indices.shape[1] > 1:

        # Perform a small Gaussian blur on the region
        box_intensities = cv.GaussianBlur(box_intensities, (3, 3), 0,
                                          borderType=cv.BORDER_CONSTANT)

        # Get the new local maxima
        peak_indices_new, peak_intensities_new \
            = find_local_max(box_indices, box_intensities, min_intensity=0)

        # Delete any maxima that are not direct (diagonal)
        #   neighbours from the original set
        peak_indices_new, peak_intensities_new \
            = get_neighbors(peak_indices, peak_indices_new,
                            peak_intensities_new,
                            neighbor_dist=neighbor_dist)

        # If all neighbours were removed, signifying a very small island...
        if peak_indices_new.shape[1] == 0:

            # Pick the first maximum and break the loop
            peak_indices = peak_indices[:, [0]]
            peak_intensities = peak_intensities[[0]]
            break

        # Else, update the maxima
        else:
            peak_indices = peak_indices_new
            peak_intensities = peak_intensities_new

        # Go to the next iteration
        iteration += 1

        # If the max number of iterations is reached...
        if iteration >= max_iter:
            # Pick the first maximum and break the loop
            peak_indices = peak_indices[:, [0]]
            peak_intensities = peak_intensities[[0]]
            break

    return peak_indices, peak_intensities


def find_local_max(all_indices, all_intensities, min_intensity=100):
    """
    Find the local maxima in a region of interest.

    ARGUMENTS:
        all_indices (np.array): The indices of the region.
        all_intensities (np.array): The intensities of the region.
        min_intensity (int): The minimum intensity of a local maximum
            (default: 100).

    RETURNS:
        peak_indices (np.array): The indices of the maxima.
        peak_intensities (np.array): The intensities of the maxima.
    """

    # Apply a maximum filter on this (processed) region
    peak_intensities = ndimage.maximum_filter(all_intensities, size=3)

    # Filter out maxima of 0 at the edges (these are not real maxima)
    peak_intensities = np.where(peak_intensities == 0, -1, peak_intensities)

    # Filter out maxima with a low intensity
    peak_intensities = np.where(peak_intensities < min_intensity, -1,
                                peak_intensities)

    # Get the indices corresponding to the peaks
    peak_indices = all_indices[:, all_intensities == peak_intensities]

    # Get the intensities of the peaks
    peak_intensities = all_intensities[all_intensities == peak_intensities]

    # If no maxima were detected
    if peak_indices.shape[1] == 0:
        # Return nothing
        return None, None

    return peak_indices, peak_intensities


# noinspection PyTupleAssignmentBalance
def fit_gaussian_2d(box_indices, box_intensities, peak_index,
                    peak_sigma=None, plot=False):
    """
    Fit a Gaussian in two directions to the intensities around a local maximum.

    ARGUMENTS:
        box_indices (np.array): The indices of the region.
        box_intensities (np.array): The intensities of the region.
        peak_index (np.array): The index of the peak.
        peak_sigma (list): The spread of the peak in both directions
            (default: None).
        plot (bool): Whether to plot the box intensities with the fitted
            Gaussian.

    RETURNS:
        peak_fitted_index (np.array): The subpixel coordinates of the maxima.
        peak_fitted_intensity (np.array): The fitted intensities of the maxima.
        peak_fitted_index_std (np.array): The errors in the subpixel coordinates
            of the maxima.
        peak_fitted_intensity_std (np.array): The errors in the fitted
            intensities of the maxima.
    """

    # If no spread is specified, set it to 1 in both directions
    if peak_sigma is None:
        peak_sigma = [1, 1]

    # Define a Gaussian function to be used for fitting
    def gaussian(x, a, x0, sigma):
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

    # Define a set of slices around the peak in 2 directions
    x_indices = box_indices[1, peak_index[0], :].flatten()
    z_indices = box_indices[0, :, peak_index[1]].flatten()

    # Get the corresponding intensities
    x_intensities = box_intensities[peak_index[0], :].flatten()
    z_intensities = box_intensities[:, peak_index[1]].flatten()

    # Get the local maximum intensity
    peak_height = box_intensities[peak_index[0], peak_index[1]]

    # Use a try-except structure to catch failed fits
    try:
        # Fit the Gaussian in both directions
        x_parameters, x_covariance \
            = curve_fit(gaussian, x_indices, x_intensities,
                        p0=[peak_height, peak_index[1], peak_sigma[0]])
        z_parameters, z_covariance \
            = curve_fit(gaussian, z_indices, z_intensities,
                        p0=[peak_height, peak_index[0], peak_sigma[1]])

    # If it cannot be fitted, return nothing
    except RuntimeError:
        return None, None, None, None

    # Get the standard errors from the (diagonals of the) covariance matrices
    x_errors = np.sqrt(np.diag(x_covariance))
    z_errors = np.sqrt(np.diag(z_covariance))

    # Get the optimized peak indices (to be converted back to coordinates)
    peak_fitted_index = np.array([z_parameters[1], x_parameters[1]])
    peak_fitted_intensity = np.array([z_parameters[0], x_parameters[0]])

    # Get the errors
    peak_fitted_index_std = np.array([z_errors[1], x_errors[1]])
    peak_fitted_intensity_std = np.array([z_errors[0], x_errors[0]])

    # # Plot the box intensities with the fitted Gaussian
    if plot:
        # Define more detailed indices for the plot
        x_indices_fit = np.linspace(np.min(x_indices), np.max(x_indices), 100)
        z_indices_fit = np.linspace(np.min(z_indices), np.max(z_indices), 100)

        # Plot the box intensities with the fitted Gaussian
        fig, ax = plt.subplots()
        ax.plot(x_indices, x_intensities, 'b+', label='x data')
        ax.plot(z_indices, z_intensities, 'r+', label='z data')
        ax.plot(x_indices_fit, gaussian(x_indices_fit, *x_parameters),
                'b-', label='x fit')
        ax.plot(z_indices_fit, gaussian(z_indices_fit, *z_parameters),
                'r-', label='z fit')
        ax.legend()

    return (peak_fitted_index, peak_fitted_intensity,
            peak_fitted_index_std, peak_fitted_intensity_std)


def find_particle_center(img, contours, contour_no, min_intensity=100,
                         neighbor_dist=2, box_margin=2, max_iter=3,
                         plot_mult=False, plot_all=False):
    """
    Find the center of a particle at a contour in an image using
    subpixel fitting.

    ARGUMENTS:
        img (np.array): The image to be processed.
        contours (list): The contours in the image.
        contour_no (int): The index of the contour to be processed.
        min_intensity (int): The minimum intensity of a local maximum
            (default: 100).
        neighbor_dist (float): The maximum distance between maxima to be
            considered part of the same cluster (default: 2).
        box_margin (int): The margin to be added to the bounding box of the
            contour (default: 2).
        max_iter (int): The maximum number of iterations to remove neighbouring
            maxima (default: 3).
        plot_mult (bool): Whether to plot the isolated contour box if multiple
            clusters are found (default: False).
        plot_all (bool): Whether to plot all maxima found originally
            (default: False).

    RETURNS:
        peak_coord_fit (np.array): The subpixel coordinates of the maxima.
        peak_intensity_fit (np.array): The fitted intensities of the maxima.
        peak_coord_err (np.array): The errors in the subpixel coordinates of the
            maxima.
        peak_intensity_err (np.array): The errors in the fitted intensities of
            the maxima.
    """

    # === PART 1: GETTING THE BOUNDING BOX ===
    # Get the contour and calculate its bounding box
    cnt = contours[contour_no]
    box_x, box_z, box_w, box_h = cv.boundingRect(cnt)

    # Create a background-less version by setting all pixels that are not on
    #   this particular contour to black
    img_nobg = remove_background(img, [cnt])

    # Also set all pixels on other contours to black on the original image,
    #   while keeping the background
    img_isol = remove_contours(img, contours, keep_index=contour_no)

    # Check if bounding box + margin is within the image boundaries
    img_height, img_width = img_isol.shape
    box_x0 = max(0, box_x - box_margin)
    box_z0 = max(0, box_z - box_margin)
    box_x1 = min(img_width, box_x + box_w + box_margin)
    box_z1 = min(img_height, box_z + box_h + box_margin)

    # Calculate new height and width
    box_w = box_x1 - box_x0
    box_h = box_z1 - box_z0

    # Get the indices of the box pixels, and the corresponding
    #   intensities as 2D arrays
    box_indices = np.indices((box_h, box_w))
    box_intensities_nobg \
        = img_nobg[box_z0:box_z1, box_x0:box_x1].reshape(box_h, box_w)
    box_intensities_isol \
        = img_isol[box_z0:box_z1, box_x0:box_x1].reshape(box_h, box_w)

    # === PART 2: FINDING LOCAL MAXIMA ===
    # Get the local maxima in the region
    peak_indices, peak_intensities \
        = find_local_max(box_indices, box_intensities_nobg,
                         min_intensity=min_intensity)

    # If no maxima were detected, return nothing, this contour is skipped
    if peak_indices is None:
        return None, None, None, None

    # Split into clusters
    clusters = split_clusters(peak_indices, neighbor_dist=neighbor_dist)

    # Set up plot conditions
    if plot_all | ((len(clusters) > 1) & plot_mult):
        plot = True
    else:
        plot = False

    # Set up a plot of the isolated contour box
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(img_isol, cmap='gray')
        ax.invert_yaxis()
        ax.set_xlim(box_x - box_margin - 0.5, box_x + box_w + box_margin + 0.5)
        ax.set_ylim(box_z + box_h + box_margin + 0.5, box_z - box_margin - 0.5)
        ax.set_xlabel('x [px]')
        ax.set_ylabel('z [px]')

        # Plot all maxima found originally
        peak_coords = peak_indices + np.array(
                [box_z - box_margin, box_x - box_margin])[:, None]
        plt1 = ax.scatter(*np.flipud(peak_coords),
                          c='b', s=75, marker='s', alpha=0.2)

    # Pre-allocate empty arrays
    peak_index, peak_intensity, peak_coord = [], [], []
    peak_spread_x, peak_spread_z = [], []

    # For each cluster (which might only be one in many cases)...
    for cluster_n, cluster in enumerate(clusters):
        # Save the range in x and z of the original maxima
        peak_spread_x.append((np.max(peak_indices[1, cluster]) - np.min(
                peak_indices[1, cluster]) + 1) / 2)
        peak_spread_z.append((np.max(peak_indices[0, cluster]) - np.min(
                peak_indices[0, cluster]) + 1) / 2)

        # Deal with neighbouring maxima to get only one single maximum pixel
        peak_index_current, peak_intensity_current = destroy_neighbors(
                box_indices, box_intensities_nobg,
                peak_indices[:, cluster], peak_intensities[cluster],
                neighbor_dist=neighbor_dist, max_iter=max_iter)
        peak_index.append(peak_index_current.flatten())
        peak_intensity.append(peak_intensity_current.flatten())

        # Get the real x and z coordinates
        peak_coord.append(peak_index[-1] +
                          np.array([box_z - box_margin, box_x - box_margin]))

        # Add to plot
        if plot:
            plt2 = ax.scatter(*np.flipud(peak_coord[-1]), c='r', s=15,
                              marker='s')

    # === PART 3: SUBPIXEL FITTING ===
    # Pre-allocate empty arrays
    peak_coord_fit = np.empty((0, 2))
    peak_intensity_fit = np.empty((0, 2))
    peak_coord_err = np.empty((0, 2))
    peak_intensity_err = np.empty((0, 2))

    # For each cluster...
    for current in range(len(peak_intensity)):

        # Edit the box_intensities_isol to adjust for other clusters present
        box_intensities_mult = box_intensities_isol.copy()

        # By looping over each other cluster...
        for other in range(len(peak_intensity)):
            if other != current:

                # Calculate the midpoint between the two clusters
                midpoint = ((peak_index[other] +
                             peak_index[current]) / 2).flatten()

                # Check whether the bisector between these two points is
                # vertical or not
                if peak_index[other][0] - peak_index[current][0] != 0:

                    # If not, calculate the slope and intercept of the bisector
                    slope = -(peak_index[current][1] - peak_index[other][1]) \
                            / (peak_index[current][0] - peak_index[other][0])
                    intercept = midpoint[0] - slope * midpoint[1]

                    # Get the z coordinates of each pixel relative to
                    # the bisector
                    bisector = (slope * box_indices[1, :]
                                - box_indices[0, :] + intercept)

                    # Check whether the current cluster is above or below the
                    # bisector, and generate a mask
                    if (peak_index[current][0] >
                            slope * peak_index[current][1] + intercept):
                        mask = bisector > 0  # Peak above bisector
                    else:
                        mask = bisector < 0  # Peak below bisector

                # If the bisector is vertical, generate a mask based on
                # the x coordinate
                else:
                    if peak_index[current][1] < midpoint[1]:
                        # Peak left of bisector
                        mask = box_indices[1, :] > midpoint[1]
                    else:
                        # Peak right of bisector
                        mask = box_indices[1, :] < midpoint[1]

                # Color all masked pixels black
                box_intensities_mult[mask] = 0

        # Fit a 2D gaussian over the masked intensities
        fit_params = fit_gaussian_2d(box_indices, box_intensities_mult,
                                     peak_index[current],
                                     peak_sigma=[peak_spread_x[current],
                                                 peak_spread_z[current]])

        # If the fit failed, break the loop, this cluster is rejected
        if fit_params[0] is None:
            break

        # Save the values
        peak_coord_fit = np.vstack((peak_coord_fit, fit_params[0] + np.array(
                [box_z - box_margin, box_x - box_margin])))
        peak_intensity_fit = np.vstack((peak_intensity_fit, fit_params[1]))
        peak_coord_err = np.vstack((peak_coord_err, fit_params[2]))
        peak_intensity_err = np.vstack((peak_intensity_err, fit_params[3]))

        # Add to plot
        if plot:
            plt3 = ax.scatter(*np.flipud(peak_coord_fit[-1]), c='g', s=75,
                              marker='+')

    # Finish plot
    if plot:
        plt.legend([plt1, plt2, plt3],
                   ['Local max. pixels', 'Post-blur maxima',
                    'Subpixel refinement'])
        plt.show()

    return (peak_coord_fit, peak_intensity_fit,
            peak_coord_err, peak_intensity_err)


def analyze_image(img, min_contour_area=1, min_intensity=100, neighbor_dist=2.5,
                  box_margin=2, max_iter=3, plot_all=False, plot_image=False,
                  bg_sigma_color1=50, bg_sigma_space1=9, bg_sigma_color2=20,
                  bg_sigma_space2=3, bg_noise_threshold=50):
    """
    Analyze an image to find the coordinates and intensities of the particles.

    ARGUMENTS:
        img (np.array): The image to be processed.
        min_contour_area (int): The minimum area of a contour to be considered
            a particle (default: 1).
        min_intensity (int): The minimum intensity of a local maximum
            (default: 100).
        neighbor_dist (float): The maximum distance between maxima to be
            considered part of the same cluster (default: 2.5).
        box_margin (int): The margin to be added to the bounding box of the
            contour (default: 2).
        max_iter (int): The maximum number of iterations to remove neighbouring
            maxima (default: 3).
        plot_all (bool): Whether to plot all maxima found originally
            (default: False).
        plot_image (bool): Whether to plot the image with the particle
            coordinates on top (default: False).
        bg_sigma_color1 (int): The color sigma of the first bilateral filter
            (default: 50).
        bg_sigma_space1 (int): The space sigma of the first bilateral filter
            (default: 9).
        bg_sigma_color2 (int): The color sigma of the second bilateral filter
            (default: 20).
        bg_sigma_space2 (int): The space sigma of the second bilateral filter
            (default: 3).
        bg_noise_threshold (int): The threshold for the to-zero threshold
            (default: 50).

    RETURNS:
        coords (np.array): The subpixel coordinates of the maxima.
        intensities (np.array): The fitted intensities of the maxima.
        coords_err (np.array): The errors in the subpixel coordinates of the
            maxima.
        intensities_err (np.array): The errors in the fitted intensities of the
            maxima.
    """

    # Remove background using the background parameters specified
    img_nobg = remove_background(img, sigma_color1=bg_sigma_color1,
                                 sigma_space1=bg_sigma_space1,
                                 sigma_color2=bg_sigma_color2,
                                 sigma_space2=bg_sigma_space2,
                                 noise_threshold=bg_noise_threshold)

    # Find the contours in the background-less image
    contours, _ = cv.findContours(img_nobg, cv.RETR_EXTERNAL,
                                  cv.CHAIN_APPROX_SIMPLE)

    # Filter out contours that are too small
    contours = [cnt for cnt in contours
                if cv.contourArea(cnt) > min_contour_area]

    # Pre-allocate empty arrays
    coords = np.empty((0, 2))
    intensities = np.empty((0, 2))
    coords_err = np.empty((0, 2))
    intensities_err = np.empty((0, 2))

    # Loop over contours found
    for cnt_idx in range(len(contours)):

        # Find the particle center(s) corresponding to this contour
        results = find_particle_center(img, contours, cnt_idx,
                                       min_intensity=min_intensity,
                                       neighbor_dist=neighbor_dist,
                                       box_margin=box_margin,
                                       max_iter=max_iter,
                                       plot_all=plot_all)

        # If no maxima were detected, skip this contour
        if results[0] is None:
            pass

        # Else, save the results
        else:
            coords = np.vstack((coords, results[0]))
            intensities = np.vstack((intensities, results[1]))
            coords_err = np.vstack((coords_err, results[2]))
            intensities_err = np.vstack((intensities_err, results[3]))

    # Plot the particles coordinates on top of the image
    if plot_image:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(img, cmap='gray')
        ax.scatter(coords[:, 1], coords[:, 0], c='y', s=15, marker='+')
        ax.invert_yaxis()
        ax.set_xlim(-0.5, img.shape[1] - 0.5)
        ax.set_ylim(img.shape[0] - 0.5, -0.5)
        ax.set_xlabel('x [px]')
        ax.set_ylabel('z [px]')
        plt.show()

    return coords, intensities, coords_err, intensities_err


def analyze_camera(directory, prefix, resolution, offset, min_contour_area=1,
                   min_intensity=100, neighbor_dist=2.5, box_margin=2,
                   max_iter=3, plot_all=False, plot_image=False,
                   plot_particles=True, verbose=False,
                   bg_sigma_color1=50, bg_sigma_space1=9,
                   bg_sigma_color2=20, bg_sigma_space2=3,
                   bg_noise_threshold=50):
    """
    Analyze a series of images from a given camera.

    ARGUMENTS:
        directory (str): The directory where the images are stored.
        prefix (str): The prefix of the image files.
        resolution (float): The resolution of the camera in meters per pixel.
        offset (np.array): The offset of the camera in meters.
        min_contour_area (int): The minimum area of a contour to be considered
            a particle (default: 1).
        min_intensity (int): The minimum intensity of a local maximum
            (default: 100).
        neighbor_dist (float): The maximum distance between maxima to be
            considered part of the same cluster (default: 2.5).
        box_margin (int): The margin to be added to the bounding box of the
            contour (default: 2).
        max_iter (int): The maximum number of iterations to remove neighbouring
            maxima (default: 3).
        plot_all (bool): Whether to plot all maxima found originally
            (default: False).
        plot_image (bool): Whether to plot the image with the particle
            coordinates on top (default: False).
        plot_particles (bool): Whether to plot the number of particles found in
            each image (default: True).
        verbose (bool): Whether to print out the number of particles found in
            each image (default: False).
        bg_sigma_color1 (int): The color sigma of the first bilateral filter
            (default: 50).
        bg_sigma_space1 (int): The space sigma of the first bilateral filter
            (default: 9).
        bg_sigma_color2 (int): The color sigma of the second bilateral filter
            (default: 20).
        bg_sigma_space2 (int): The space sigma of the second bilateral filter
            (default: 3).
        bg_noise_threshold (int): The threshold for the to-zero threshold
            (default: 50).
    """

    # Read the images
    images, files = read_image_series(directory, prefix)

    # Pre-allocate empty arrays
    coords = np.empty((0, 3))
    intensities = np.empty((0, 3))
    coords_err = np.empty((0, 3))
    intensities_err = np.empty((0, 3))
    particles_found = np.empty(len(images), dtype=int)

    # Loop over the images
    for img_no, img in enumerate(images):
        # Analyze the image
        coo_img, int_img, coo_err_img, int_err_img \
            = analyze_image(img, min_contour_area=min_contour_area,
                            min_intensity=min_intensity,
                            neighbor_dist=neighbor_dist,
                            box_margin=box_margin, max_iter=max_iter,
                            plot_all=plot_all, plot_image=plot_image,
                            bg_sigma_color1=bg_sigma_color1,
                            bg_sigma_space1=bg_sigma_space1,
                            bg_sigma_color2=bg_sigma_color2,
                            bg_sigma_space2=bg_sigma_space2,
                            bg_noise_threshold=bg_noise_threshold)

        # Convert coordinates to calibrated values, with offset
        coo_img = coo_img * resolution - offset
        coo_err_img = coo_err_img * resolution

        # Horizontally stack the image number with the coords
        img_no_arr = np.ones((len(coo_img), 1)) * (img_no + 1)
        coo_img = np.hstack((coo_img, img_no_arr))
        coo_err_img = np.hstack((coo_err_img, img_no_arr))
        int_img = np.hstack((int_img, img_no_arr))
        int_err_img = np.hstack((int_err_img, img_no_arr))

        # Save the data by vertically stacking with the previous data
        coords = np.vstack((coords, coo_img))
        intensities = np.vstack((intensities, int_img))
        coords_err = np.vstack((coords_err, coo_err_img))
        intensities_err = np.vstack((intensities_err, int_err_img))

        particles_found[img_no] = len(coo_img)

    # Plot the number of particles found in each image
    if plot_particles:
        plt.figure()
        plt.plot(particles_found)
        plt.xlabel('Image number')
        plt.ylabel('Number of particles found')
        plt.title('Number of particles found in each image')
        plt.show()

    # Print the number of particles found in each image
    if verbose:
        print(f'Found between {np.min(particles_found)} and '
              f'{np.max(particles_found)} particles in each image, '
              f'with a median of {np.median(particles_found)}.')

    return coords, intensities, coords_err, intensities_err, particles_found
