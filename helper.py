import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from dataclasses import dataclass
from scipy.optimize import curve_fit

def hello():
    print("Hello,, from helper.py!")


import numpy as np
import matplotlib.pyplot as plt

def plot_pixel_matrix(matrix, title, show_zeros=False):
    """
    Displays a 16x16 pixel matrix with optional zero display.

    Parameters:
        matrix (np.ndarray): 16x16 pixel intensity matrix.
        title (str): Title of the plot.
        show_zeros (bool): Whether to show zero values on the plot. Default is False.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title, pad=12)

    # Set minor ticks for grid lines
    ax.set_xticks(np.arange(17) - 0.5, minor=True)
    ax.set_yticks(np.arange(17) - 0.5, minor=True)
    ax.grid(which="minor", color='black', linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Remove major ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Use gray background
    ax.imshow(np.ones_like(matrix), cmap='gray_r')

    # Annotate non-zero values (or all if show_zeros=True)
    for i in range(16):
        for j in range(16):
            val = int(matrix[i, j])
            if val != 0 or show_zeros:
                ax.text(j, i, str(val), ha='center', va='center', fontsize=8, color='black')

    plt.tight_layout()
    plt.show()




def plot_pixel_matrix_with_cg(matrix, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title, pad=12)

    ax.set_xticks(np.arange(17) - 0.5, minor=True)
    ax.set_yticks(np.arange(17) - 0.5, minor=True)
    ax.grid(which="minor", color='black', linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.imshow(np.ones_like(matrix), cmap='gray_r')

    # Text values
    for i in range(16):
        for j in range(16):
            if matrix[i, j] > 0:
                ax.text(j, i, str(int(matrix[i, j])), ha='center', va='center', fontsize=8, color='black')

    # Compute Center of Gravity (CG)
    Y, X = np.indices(matrix.shape)
    total_intensity = np.sum(matrix)
    x_cg = np.sum(X * matrix) / total_intensity
    y_cg = np.sum(Y * matrix) / total_intensity

    # Plot CG as a red dot
    ax.plot(x_cg, y_cg, 'ro', markersize=6, label='CG')

    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    print(f"Center of Gravity (x_cg, y_cg): ({x_cg:.2f}, {y_cg:.2f})")

def plot_single_event_histogram(pixels, amplitude, mean, std, bin_step=2, event_index=None):
    """
    Plots a histogram of pixel intensities for a single event along with a fitted Gaussian curve.

    Parameters:
    -----------
    pixels : np.ndarray
        A 2D array (e.g., 16x16) of pixel intensity values for a single event.
    
    amplitude : float
        The amplitude of the Gaussian fit (height of the peak).
    
    mean : float
        The mean (μ) of the fitted Gaussian distribution.
    
    std : float
        The standard deviation (σ) of the fitted Gaussian distribution.
    
    bin_step : int, optional (default=2)
        The width of each histogram bin.
    
    event_index : int or None, optional
        The event index to annotate the plot title. If None, a generic title is used.

    Returns:
    --------
    None
        Displays a plot with the histogram and fitted Gaussian curve, including threshold lines.
    """


    # Set custom bin range based on max value in pixel data
    max_value = int(np.max(pixels)) + bin_step
    bins = np.arange(0, max_value + bin_step, bin_step)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot the histogram
    count, bin_edges, _ = ax.hist(
        pixels.flatten(),
        bins=bins,
        alpha=0.7,
        color='skyblue',
        density=True,
        label='Pixel Histogram'
    )

    # Gaussian curve
    x_fit = np.linspace(0, max_value, 500)
    y_fit = amplitude * np.exp(-((x_fit - mean) ** 2) / (2 * std ** 2))
    ax.plot(x_fit, y_fit, 'r-', label='Fitted Gaussian')

    # Thresholds
    boundary_threshold = mean + 3 * std
    picture_threshold = mean + 5 * std
    ax.axvline(boundary_threshold, color='orange', linestyle='--', linewidth=2, label='Boundary Threshold (μ + 3σ)')
    ax.axvline(picture_threshold, color='green', linestyle='--', linewidth=2, label='Picture Threshold (μ + 5σ)')

    # Axis labels and title
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Density")
    title = f"Event {event_index} Histogram and Fitted Gaussian" if event_index is not None else "Histogram and Fitted Gaussian"
    ax.set_title(title)

    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.show()



def reshape_to_event_matrices(data: np.ndarray) -> np.ndarray:
    """
    Reshapes the input data into a 3D array of shape (total_events, 16, 16).

    Parameters:
        data (np.ndarray): A 2D NumPy array where the first column is event number,
                           and the remaining 256 columns are pixel data.

    Returns:
        np.ndarray: A 3D array of shape (total_events, 16, 16).
    """
    event_numbers = data[:, 0]
    pixel_data = data[:, 1:]

    total_events = len(event_numbers) // 16
    event_data = pixel_data.reshape(total_events, 16, 16)

    return event_data




def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu)**2) / (2 * sigma**2))

def fit_gaussian_to_events(
    event_data: np.ndarray,
    bin_step: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits a Gaussian distribution to the histogram of each event in the event_data.

    Parameters:
        event_data (np.ndarray): A 3D array of shape (num_events, height, width) representing event matrices.
        bin_step (int): Step size for histogram binning. Default is 2.

    Returns:
        Tuple containing arrays of amplitudes, means, and standard deviations for each event.
    """
    total_events = event_data.shape[0]
    amplitudes = np.zeros(total_events)
    means = np.zeros(total_events)
    std_devs = np.zeros(total_events)

    for i in range(total_events):
        data = event_data[i].flatten()
        max_val = int(np.max(data))
        bins = np.arange(0, max_val + bin_step + 1, bin_step)

        count, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        try:
            initial_guess = [np.max(count), np.mean(data), np.std(data)]
            popt, _ = curve_fit(gaussian, bin_centers, count, p0=initial_guess)
            amplitudes[i], means[i], std_devs[i] = popt
        except RuntimeError:
            # If fit fails, fall back to statistical estimates
            amplitudes[i] = np.max(count)
            means[i] = np.mean(data)
            std_devs[i] = np.std(data)

    return amplitudes, means, std_devs

def plot_single_event_gaussian(
    event_matrix: np.ndarray,
    amplitude: float,
    mean: float,
    std_dev: float,
    event_index: int = None,
    bin_step: int = 2
):
    """
    Plots histogram and fitted Gaussian for a single event.

    Parameters:
        event_matrix (np.ndarray): 2D pixel matrix for one event (e.g., 16x16).
        amplitude (float): Fitted Gaussian amplitude.
        mean (float): Fitted Gaussian mean.
        std_dev (float): Fitted Gaussian standard deviation.
        event_index (int, optional): Index of the event, for labeling. Default is None.
        bin_step (int): Histogram bin step size. Default is 2.
    """
    pixel_values = event_matrix.flatten()
    max_value = int(np.max(pixel_values)) + bin_step
    bins = np.arange(0, max_value + bin_step, bin_step)

    plt.figure(figsize=(10, 5))
    count, bin_edges, _ = plt.hist(
        pixel_values, bins=bins, alpha=0.7, color='skyblue', density=True, label='Pixel Histogram'
    )

    x_fit = np.linspace(0, max_value, 500)
    y_fit = gaussian(x_fit, amplitude, mean, std_dev)
    plt.plot(x_fit, y_fit, 'r-', label='Fitted Gaussian')

    plt.xlabel("Pixel Intensity")
    plt.ylabel("Density")
    title = f"Event {event_index} Histogram and Gaussian" if event_index is not None else "Histogram and Gaussian"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def apply_thresholds_to_dataset(event_data, means, stds):
    """
    Applies boundary and picture thresholds to each event's pixel matrix in the dataset.
    
    Parameters:
        event_data (np.ndarray): Shape (N, 16, 16) array of pixel matrices for N events.
        means (np.ndarray): Array of mean values for each event.
        stds (np.ndarray): Array of standard deviation values for each event.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays of shape (N, 16, 16) containing:
            - boundary_thresholded_data: pixels < boundary threshold (μ + 3σ) set to 0
            - picture_thresholded_data: pixels < picture threshold (μ + 5σ) set to 0
    """
    boundary_thresholded = []
    picture_thresholded = []

    for i in range(len(event_data)):
        pixels = event_data[i]
        mean = means[i]
        std = stds[i]

        boundary_threshold = mean + 3 * std
        picture_threshold = mean + 5 * std

        # Step 1: Zero out values below boundary threshold
        step1 = pixels - boundary_threshold
        step1[step1 < 0] = 0

        # Step 2: Zero out values below picture threshold (relative to step1)
        step2 = step1.copy()
        step2[step2 < (picture_threshold - boundary_threshold)] = 0

        boundary_thresholded.append(step1)
        picture_thresholded.append(step2)

    return np.array(boundary_thresholded), np.array(picture_thresholded)


def process_event_batch_with_suppression(event_data: np.ndarray,
                                         amplitudes: np.ndarray,
                                         means: np.ndarray,
                                         std_devs: np.ndarray) -> np.ndarray:
    """
    Apply bt/pt thresholding and isolated pixel suppression across a batch of events.

    Parameters:
        event_data (np.ndarray): Shape (N, 16, 16) – array of N event matrices.
        amplitudes (np.ndarray): Amplitudes from Gaussian fits, shape (N,).
        means (np.ndarray): Means from Gaussian fits, shape (N,).
        std_devs (np.ndarray): Std devs from Gaussian fits, shape (N,).

    Returns:
        np.ndarray: Suppressed pixel matrices, shape (N, 16, 16).
    """
    def apply_thresholds(matrix, mean, std):
        bt = mean + 3 * std
        pt = mean + 5 * std
        # Step 1: Apply boundary threshold
        mat_bt = matrix.copy()
        mat_bt -= bt
        mat_bt[mat_bt < 0] = 0

        # Step 2: Apply picture threshold (pt)
        mat_pt = mat_bt.copy()
        mat_pt[mat_pt < (pt - bt)] = 0
        return mat_pt

    def suppress_isolated_pixels(matrix):
        result = matrix.copy()

        # Horizontal check
        for i in range(16):
            for j in range(1, 15):
                if result[i, j - 1] == 0 and result[i, j + 1] == 0:
                    result[i, j] = 0

        # Vertical check
        for i in range(1, 15):
            for j in range(16):
                if result[i - 1, j] == 0 and result[i + 1, j] == 0:
                    result[i, j] = 0

        return result

    total_events = event_data.shape[0]
    output = np.zeros_like(event_data)

    for i in range(total_events):
        thresholded = apply_thresholds(event_data[i], means[i], std_devs[i])
        suppressed = suppress_isolated_pixels(thresholded)
        output[i] = suppressed

    return output



@dataclass
class ShowerAnalysisResult:
    """
    Data class representing the geometrical and statistical properties of a particle shower.
    """
    centroid_position: Tuple[float, float]
    longitudinal_extension: float
    transverse_extension: float
    orientation_angle_degrees_delta: float
    elliptical_eccentricity: float
    edge_leakage_ratio: float
    axis_miss_distance: float
    azimuthal_spread: float
    distribution_asymmetry: float
    radial_displacement: float
    radial_orientation_angle: float


def analyze_shower(
    photon_electron_matrix: np.ndarray,
    x_positions: np.ndarray = None,
    y_positions: np.ndarray = None,
    generate_visualization: bool = True
) -> ShowerAnalysisResult:
    """
    Analyze a 2D photon-electron matrix from a sensor to extract spatial and statistical 
    characteristics of a particle shower.

    Parameters
    ----------
    photon_electron_matrix : np.ndarray
        2D array (typically 16x16) representing the energy deposited (in PEs) in each sensor pixel.
    
    x_positions : np.ndarray, optional
        Array of x-axis positions corresponding to sensor columns. If None, they are centered around 0.
    
    y_positions : np.ndarray, optional
        Array of y-axis positions corresponding to sensor rows. If None, they are centered around 0.
    
    generate_visualization : bool, default=True
        If True, plots a heatmap with the fitted ellipse, centroid, and major axis.

    Returns
    -------
    ShowerAnalysisResult
        An object containing:
        - Centroid position (x, y)
        - Longitudinal and transverse extensions
        - Orientation angle in degrees
        - Elliptical eccentricity
        - Edge leakage ratio
        - Axis miss distance
        - Azimuthal spread
        - Distribution asymmetry
        - Radial displacement
        - Radial orientation angle
    """

    # Set default x/y positions (centered around origin)
    if x_positions is None:
        sensor_center_x = (photon_electron_matrix.shape[1] - 1) / 2
        x_positions = np.arange(photon_electron_matrix.shape[1]) - sensor_center_x
    if y_positions is None:
        sensor_center_y = (photon_electron_matrix.shape[0] - 1) / 2
        y_positions = sensor_center_y - np.arange(photon_electron_matrix.shape[0])

    total_photon_electrons = np.sum(photon_electron_matrix)
    if total_photon_electrons == 0:
        raise ValueError("Photon-electron matrix contains no detectable signal")

    # Edge leakage
    left = np.sum(photon_electron_matrix[:, 0])
    right = np.sum(photon_electron_matrix[:, -1])
    top = np.sum(photon_electron_matrix[0, :])
    bottom = np.sum(photon_electron_matrix[-1, :])
    edge_leakage_ratio = (left + right + top + bottom) / total_photon_electrons

    # Coordinate grid
    x_grid, y_grid = np.meshgrid(x_positions, y_positions, indexing='xy')

    # Centroid
    centroid_x = np.sum(photon_electron_matrix * x_grid) / total_photon_electrons
    centroid_y = np.sum(photon_electron_matrix * y_grid) / total_photon_electrons

    # Second moments
    var_x = np.sum(photon_electron_matrix * x_grid**2) / total_photon_electrons - centroid_x**2
    var_y = np.sum(photon_electron_matrix * y_grid**2) / total_photon_electrons - centroid_y**2
    cov_xy = np.sum(photon_electron_matrix * x_grid * y_grid) / total_photon_electrons - centroid_x * centroid_y

    # Orientation
    delta_var = var_y - var_x
    theta_rad = 0.5 * np.arctan2(2 * cov_xy, delta_var)
    orientation_angle_degrees_delta = np.degrees(theta_rad) % 180

    # Principal axes
    root_term = np.sqrt(delta_var**2 + 4 * cov_xy**2)
    longitudinal_extension = np.sqrt((var_x + var_y + root_term) / 2)
    transverse_extension = np.sqrt(abs((var_x + var_y - root_term) / 2))

    # Eccentricity
    elliptical_eccentricity = (
        np.sqrt(1 - (transverse_extension / longitudinal_extension)**2)
        if longitudinal_extension != 0 else 0
    )

    # Miss distance
    if cov_xy != 0:
        major_slope = (delta_var + root_term) / (2 * cov_xy)
        intercept = centroid_y - major_slope * centroid_x
        axis_miss_distance = abs(intercept) / np.sqrt(1 + major_slope**2)
    else:
        axis_miss_distance = abs(centroid_y)

    # Radial vector
    radial_displacement = np.sqrt(centroid_x**2 + centroid_y**2)
    radial_angle_degrees = np.degrees(np.arctan2(centroid_y, centroid_x)) % 180

    # Orientation mismatch between radial and major axis
    radial_slope = centroid_y / centroid_x if centroid_x != 0 else np.inf
    axis_slope = np.tan(theta_rad)
    denom = 1 + radial_slope * axis_slope
    radial_orientation_angle = 90.0 if denom == 0 else np.degrees(np.arctan(abs((axis_slope - radial_slope) / denom)))

    # Azimuthal spread
    azimuthal_spread = np.sqrt(
        (longitudinal_extension * np.sin(np.radians(radial_orientation_angle)))**2 +
        (transverse_extension * np.cos(np.radians(radial_orientation_angle)))**2
    )

    # Third moment for asymmetry
    third_x = np.sum(photon_electron_matrix * (x_grid - centroid_x)**3)
    third_y = np.sum(photon_electron_matrix * (y_grid - centroid_y)**3)
    raw_asym = (third_x + third_y) / total_photon_electrons
    distribution_asymmetry = abs(raw_asym)**(1/3) / longitudinal_extension if longitudinal_extension != 0 else 0

    # Visualization
    if generate_visualization:
        plt.figure(figsize=(10, 8))
        x_edges = np.linspace(x_positions[0] - 0.5, x_positions[-1] + 0.5, photon_electron_matrix.shape[1] + 1)
        y_edges = np.linspace(y_positions[0] - 0.5, y_positions[-1] + 0.5, photon_electron_matrix.shape[0] + 1)

        mesh_plot = plt.pcolormesh(x_edges, y_edges, photon_electron_matrix, cmap='viridis',
                                   edgecolors='black', linewidth=0.5)

        # Ellipse
        theta = np.linspace(0, 2*np.pi, 200)
        a = 1.4 * longitudinal_extension
        b = 1.4 * transverse_extension
        ellipse_x = centroid_x + a * np.cos(theta) * np.cos(theta_rad) - b * np.sin(theta) * np.sin(theta_rad)
        ellipse_y = centroid_y + a * np.cos(theta) * np.sin(theta_rad) + b * np.sin(theta) * np.cos(theta_rad)
        plt.plot(ellipse_x, ellipse_y, 'w-', lw=2, label='Fitted Ellipse')

        # Centroid and lines
        plt.scatter([centroid_x], [centroid_y], color='magenta', edgecolors='black', label='Energy Centroid', zorder=3)
        plt.plot([0, centroid_x], [0, centroid_y], 'cyan', linestyle='--', linewidth=1.5, label='Center-to-Centroid Vector')

        dx = a * np.cos(theta_rad)
        dy = a * np.sin(theta_rad)
        plt.plot([centroid_x - dx, centroid_x + dx], [centroid_y - dy, centroid_y + dy],
                 'red', linestyle='-', linewidth=2, label='Major Axis Vector')

        plt.axhline(0, color='gray', linestyle='--')
        plt.axvline(0, color='gray', linestyle='--')

        plt.xticks(x_positions, rotation=45)
        plt.yticks(y_positions)
        plt.grid(False)
        cbar = plt.colorbar(mesh_plot)
        cbar.set_label("Photon-Electron Count")
        plt.title("Particle Shower Profile with Elliptical Approximation", pad=20)
        plt.xlabel("X Position (Sensor Units)")
        plt.ylabel("Y Position (Sensor Units)")
        plt.legend(loc='upper right', framealpha=0.9)
        plt.tight_layout()
        plt.show()

    return ShowerAnalysisResult(
        centroid_position=(centroid_x, centroid_y),
        longitudinal_extension=longitudinal_extension,
        transverse_extension=transverse_extension,
        orientation_angle_degrees_delta=orientation_angle_degrees_delta,
        elliptical_eccentricity=elliptical_eccentricity,
        edge_leakage_ratio=edge_leakage_ratio,
        axis_miss_distance=axis_miss_distance,
        azimuthal_spread=azimuthal_spread,
        distribution_asymmetry=distribution_asymmetry,
        radial_displacement=radial_displacement,
        radial_orientation_angle=radial_orientation_angle
    )
