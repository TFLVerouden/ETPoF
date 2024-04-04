from pof_piv import *
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange


def plot_flow_field(displacements, coordinates, window_size,
                    background=None, plot_windows=True,
                    arrow_color='k', arrow_scale=1, zero_displ_thr=0,
                    highlight_radius_range=None,
                    highlight_angle_range=None,
                    highlight_color='b', calib_dist=None, units=None,
                    title='Flow field', legend=None, timing=False):
    """
    Plot the flow field with displacements as arrows.

    This function takes in the same keyword arguments as plot_displacements,
    but ignores some of them.

    PARAMETERS:
        displacements (np.array): Displacement vectors [j, i, y/x].
        coordinates (np.array): Coordinates of the windows [j, i, y/x].
        window_size (int | tuple): Size of the windows [y, x].
        background (np.array): Background image to plot the flow field on.
        plot_windows (bool): Whether to plot the windows.
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
        legend (list): Unused.
        timing (bool): Whether to show a progress bar.

    RETURNS:
        fig (plt.figure): Figure object.
        ax (plt.axis): Axis object.
    """

    # Assume the window size is square
    if highlight_angle_range is None:
        highlight_angle_range = [np.nan, np.nan]
    if highlight_radius_range is None:
        highlight_radius_range = [np.nan, np.nan]
    window_size = assume_squareness(window_size)

    # Plot all displacement vectors at the center of each window
    fig, ax = plt.subplots()

    # If a background image is supplied, add it to the plot
    if background is not None:
        ax.imshow(background, cmap='gray')

    # If a calibration distance was specified, calibrate all values
    if calib_dist is not None:
        displacements = displacements * calib_dist
        coordinates = coordinates * calib_dist
        window_size = window_size * calib_dist

    # Show a grid with the outline of each window and an arrow in the centre
    # indicating the displacement
    arrow_param = calib_dist if calib_dist is not None else 1

    # Get a list of indices that should be coloured
    highlight = filter_displacements(displacements,
                                     radius_range=highlight_radius_range,
                                     angle_range=highlight_angle_range)

    # Get a list of indices that are below the zero-threshold
    zero_displ = np.linalg.norm(displacements, axis=2) < zero_displ_thr

    # Plot the indices below the threshold as dots
    ax.scatter(coordinates[zero_displ & highlight, 1],
               coordinates[zero_displ & highlight, 0],
               c=highlight_color, marker='.', s=arrow_scale)
    ax.scatter(coordinates[zero_displ & ~highlight, 1],
               coordinates[zero_displ & ~highlight, 0],
               c=arrow_color, marker='.', s=arrow_scale)

    # Plot the flow field window by window
    for j in trange(coordinates.shape[0], desc='Plotting arrows',
                    disable=not timing):
        for i in range(coordinates.shape[1]):

            # Plot the window
            if plot_windows:
                ax.add_patch(
                        plt.Rectangle(
                                (coordinates[j, i][1] - window_size[1] / 2,
                                 coordinates[j, i][0] - window_size[0] / 2),
                                window_size[1],
                                window_size[0], fill=None,
                                edgecolor='darkgrey',
                                linewidth=1))

            # If the displacement is above the zero-threshold, plot an arrow
            if np.linalg.norm(displacements[j, i]) > zero_displ_thr:
                # If the displacement should be highlighted, set the color
                color = highlight_color if highlight[j, i] else arrow_color

                # Calculate the start and end of the arrow
                arrow_start = np.array(
                        [coordinates[j, i][0] -
                         arrow_scale * 0.5 * displacements[j, i][0],
                         coordinates[j, i][1] -
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
    ax.set_xlim([0, np.max(coordinates[:, :, 1]) + window_size[1] / 2])
    ax.set_ylim([np.max(coordinates[:, :, 0] + window_size[0] / 2), 0])

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
                       background=None, plot_windows=True,
                       arrow_color='k', arrow_scale=1, zero_displ_thr=0,
                       highlight_radius_range=None,
                       highlight_angle_range=None,
                       highlight_color='b', calib_dist=None, units=None,
                       title=None, legend=None,
                       timing=False):
    """
    Plot the displacement vectors.

    This function takes in the same keyword arguments as plot_flow_field,
    but ignores some of them.

    PARAMETERS:
        displacements (np.array): Displacement vectors [j, i, y/x].
        background (np.array): Unused.
        plot_windows (bool): Unused.
        arrow_color (str): Unused.
        arrow_scale (float): Unused.
        zero_displ_thr (float): Unused.
        highlight_radius_range (list): Range of magnitudes to highlight.
        highlight_angle_range (list): Range of angles to highlight.
        highlight_color (str): Color of the highlighted arrows and dots.
        calib_dist (float): Calibration distance.
        units (str): Units of the calibration distance.
        title (str): Unused.
        legend (list): Legend of the plot.
        timing (bool): Whether to show a progress bar.

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

    # Pad the limits
    ax.set_xlim([np.amin(displacements[:, :, 1]) - 1,
                 np.amax(displacements[:, :, 1]) + 1])
    ax.set_ylim([np.amin(displacements[:, :, 0]) - 1,
                 np.amax(displacements[:, :, 0]) + 1])

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
