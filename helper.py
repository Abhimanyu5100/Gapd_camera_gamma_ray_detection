import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy.ndimage import label


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
    bin_step: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits a Gaussian distribution to the histogram of each event in the event_data.

    Parameters:
        event_data (np.ndarray): A 3D array of shape (num_events, height, width) representing event matrices.
        bin_step (int): Step size for histogram binning. Default is 1.

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
    bin_step: int = 1
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



def apply_thresholds_to_dataset(event_data: np.ndarray,
                                means: np.ndarray,
                                stds: np.ndarray):
    """
    Applies boundary (μ + 3σ) and picture (μ + 5σ) thresholds to every event.

    Parameters
    ----------
    event_data : (N, 16, 16) ndarray
        Pixel matrices for N events.
    means, stds : (N,) ndarrays
        Per-event mean and standard deviation.

    Returns
    -------
    boundary_thr_data, picture_thr_data : (N, 16, 16) ndarrays
        event_data with pixels below each threshold zeroed.
    """

    # reshape thresholds to broadcast over the 16×16 frames
    boundary_thr = (means + 3 * stds)[:, None, None]   # shape (N,1,1)
    picture_thr  = (means + 5 * stds)[:, None, None]

    # Step 1  – boundary threshold
    step1 = np.where(event_data >= boundary_thr,
                     event_data - boundary_thr, 0)

    # Step 2  – picture threshold applied to step1
    step2 = np.where(event_data >= picture_thr, step1, 0)

    return step1, step2


def suppress_isolated_pixels(matrix: np.ndarray, min_cluster_size: int = 3) -> np.ndarray:
    """
    Keeps only the connected cluster (8-connected) that contains the brightest pixel.
    All other clusters are removed, even if their size is >= min_cluster_size.

    Parameters:
        matrix (np.ndarray): 2D array of pixel values.
        min_cluster_size (int): Minimum number of pixels required to keep a cluster (applied only to the brightest cluster).

    Returns:
        np.ndarray: Matrix with only the brightest pixel's cluster retained (if large enough).
    """
    # Step 1: Binary mask for all non-zero pixels
    binary_mask = (matrix > 0).astype(int)

    # Step 2: Label connected components (8-connectivity)
    structure = np.ones((3, 3), dtype=int)
    labeled_array, num_features = label(binary_mask, structure=structure)

    # Step 3: Locate the brightest pixel
    max_pos = np.unravel_index(np.argmax(matrix), matrix.shape)
    target_label = labeled_array[max_pos]

    # Step 4: Create output matrix with only the cluster containing the brightest pixel
    cleaned = np.zeros_like(matrix)
    if target_label != 0:
        component_mask = (labeled_array == target_label)
        if np.sum(component_mask) >= min_cluster_size:
            cleaned[component_mask] = matrix[component_mask]

    return cleaned
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
        
        
            # End of apply_thresholds

    total_events = event_data.shape[0]
    output = np.zeros_like(event_data)

    for i in range(total_events):
        thresholded = apply_thresholds(event_data[i], means[i], std_devs[i])
        suppressed = suppress_isolated_pixels(thresholded)
        output[i] = suppressed

    return output



def plot_pixel_intensity_map(pixel_data):
    rows, cols = pixel_data.shape
    x = np.arange(cols + 1)
    y = np.arange(rows + 1)

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(x, y, pixel_data, cmap='jet', edgecolors='black', linewidth=0.5)

    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label("Pixel Intensity")

    # Set ticks in center of each pixel
    plt.xticks(np.arange(cols) + 0.5, np.arange(cols))
    plt.yticks(np.arange(rows) + 0.5, np.arange(rows))
    plt.gca().invert_yaxis()

    plt.title("Pixel Intensity Map (Pixel-Only Style)")
    plt.grid(False)
    plt.show()



def analyze_shower(
    photon_electron_matrix: np.ndarray,
    x_positions: np.ndarray | None = None,
    y_positions: np.ndarray | None = None,
    generate_visualization: bool = True
) -> tuple[float, ...]:
    """
    Weighted-PCA analysis of a single 16 × 16 photon–electron image.

    The routine extracts both geometric and statistical observables that
    characterise a particle shower recorded by the GAPD camera and, if
    requested, draws a heat-map with the fitted principal–axis ellipse.

    Computational pipeline
    ----------------------
    1. Build the physical x- and y-coordinate grids (centre of sensor = 0, 0).
       Custom grids can be passed in `x_positions` / `y_positions`.
    2. Flatten the image, remove zero-count pixels and treat the remaining
       counts as *weights*.
    3. Compute the weighted centroid (⟨x⟩, ⟨y⟩).
    4. Form the 2 × 2 weighted covariance matrix of the centred points and
       perform SVD → principal directions and variances.
    5. Semi-axes:
          a = 1.4 √λ₁ (major)  b = 1.4 √λ₂ (minor)     (λ₁ ≥ λ₂)  
       The 1.4 factor matches the legacy IDL implementation.
    6. Convert PCA output to high-level shower metrics
       (orientation, eccentricity, radial geometry, miss distance, etc.).
    7. Optional figure:
       • viridis pcolormesh of raw counts  
       • white ellipse (2 a × 2 b) at the PCA orientation  
       • centroid marker, centre-to-centroid vector, and major-axis vector.

    Parameters
    ----------
    photon_electron_matrix : (16, 16) ndarray
        2-D pixel counts for one event.
    x_positions, y_positions : 1-D ndarrays or None, optional
        Physical coordinates of pixel centres.  If *None* a symmetric
        (-7.5 … +7.5) grid is generated from the matrix size.
    generate_visualization : bool, default True
        If True, displays the matplotlib figure described above.

    Returns
    -------
    tuple (length 22)
        0  centroid_x                    ⟨x⟩ (sensor units)  
        1  centroid_y                    ⟨y⟩  
        2  2a                            full major-axis length  
        3  2b                            full minor-axis length  
        4  eccentricity                 √(1 – (b/a)²)  
        5  radial_displacement           √(⟨x⟩² + ⟨y⟩²)  
        6  radial_angle                  atan2(⟨y⟩,⟨x⟩)  in ° [0,180)  
        7  orientation_angle             PCA major-axis angle in ° [0,180)  
        8  alpha                         |radial-orientation| modulo 180°  
        9  orientation_slope             tan(orientation_angle)  
        10 radial_slope                  ⟨y⟩ / ⟨x⟩ (inf if ⟨x⟩ = 0)  
        11 total_active_pixels           count(px > 0)  
        12 total_photon_electrons        sum of all counts  
        13 azimuthal_spread              projected width at alpha  
        14 axis_miss_distance            centroid–to-axis perpendicular dist.  
        15 edge_leakage_ratio            PE on sensor border / total PE  
        16 brightest_pixel_distance      brightest-pixel → centroid distance  
        17 brightest_pixel_x             x of brightest pixel  
        18 brightest_pixel_y             y of brightest pixel  
        19 flag                          always 1 (reserved)  
        20 distribution_asymmetry        3ʳᵈ-moment asymmetry / a  
        21 frac2                         (two brightest counts) / total PE

    Notes
    -----
    • Uses 8-bit connectivity kernels and matplotlib for plotting
      (see memory entries on data-visualisation practice[1][2]).  
    • The output field order matches the legacy `helper.analyze_shower`
      so downstream code remains unchanged.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    if x_positions is None:
        cols = photon_electron_matrix.shape[1]
        x_positions = np.arange(cols) + 0.5 - cols / 2

    if y_positions is None:
        rows = photon_electron_matrix.shape[0]
        y_positions = rows / 2 - (np.arange(rows) + 0.5)

    total_photon_electrons = np.sum(photon_electron_matrix)
    total_active_pixels = np.count_nonzero(photon_electron_matrix)

    if total_active_pixels == 0:
        return tuple([0.0] * 21)

    x_grid, y_grid = np.meshgrid(x_positions, y_positions, indexing='xy')
    data = photon_electron_matrix.flatten()
    points = np.stack([x_grid.flatten(), y_grid.flatten()], axis=1)

    valid = data > 0
    weights = data[valid]
    points = points[valid]

    # Weighted centroid
    centroid = np.average(points, axis=0, weights=weights)
    centroid_x, centroid_y = centroid

    # Weighted covariance matrix
    centered = points - centroid
    cov_matrix = np.cov(centered.T, aweights=weights)

    # SVD for principal axes
    U, S, Vt = np.linalg.svd(cov_matrix)
    angle_rad = np.arctan2(Vt[0, 1], Vt[0, 0])  # orientation
    a = 1.4 * np.sqrt(S[0])  # semi-major axis
    b = 1.4 * np.sqrt(S[1])  # semi-minor axis

    orientation_angle = np.degrees(angle_rad) % 180
    eccentricity = np.sqrt(1 - (b / a)**2) if a != 0 else 0

    # Radial geometry
    radial_displacement = np.sqrt(centroid_x**2 + centroid_y**2)
    radial_angle = np.degrees(np.arctan2(centroid_y, centroid_x)) % 180
    radial_slope = centroid_y / centroid_x if centroid_x != 0 else np.inf
    orientation_slope = np.tan(angle_rad)

    denom = 1 + radial_slope * orientation_slope
    radial_orientation_angle = 90.0 if denom == 0 else np.degrees(np.arctan(abs((orientation_slope - radial_slope) / denom)))

    # Alpha: angle between radial direction and ellipse major axis, folded to [0°, 90°]
    alpha = abs((radial_angle - orientation_angle) % 180)
    if alpha > 90:
        alpha = 180 - alpha


    if orientation_slope != np.inf:
        intercept = centroid_y - orientation_slope * centroid_x
        axis_miss_distance = abs(intercept) / np.sqrt(1 + orientation_slope**2)
    else:
        axis_miss_distance = abs(centroid_x)

    azimuthal_spread = np.sqrt((a * np.sin(np.radians(radial_orientation_angle)))**2 +
                               (b * np.cos(np.radians(radial_orientation_angle)))**2)

    left = np.sum(photon_electron_matrix[:, 0])
    right = np.sum(photon_electron_matrix[:, -1])
    top = np.sum(photon_electron_matrix[0, :])
    bottom = np.sum(photon_electron_matrix[-1, :])
    corners = (
        photon_electron_matrix[0, 0] +
        photon_electron_matrix[0, -1] +
        photon_electron_matrix[-1, 0] +
        photon_electron_matrix[-1, -1]
    )

    edge_sum = left + right + top + bottom - corners
    edge_leakage_ratio = edge_sum / total_photon_electrons

    max_idx = np.unravel_index(np.argmax(photon_electron_matrix), photon_electron_matrix.shape)
    brightest_pixel_value = photon_electron_matrix[max_idx]
    brightest_pixel_distance = np.sqrt((x_grid[max_idx] - centroid_x)**2 + (y_grid[max_idx] - centroid_y)**2)

    cos_theta = np.cos(np.radians(orientation_angle))
    sin_theta = np.sin(np.radians(orientation_angle))

    # Compute 3rd order directional moments
    x_shift = x_grid - centroid_x
    y_shift = y_grid - centroid_y
    weight = photon_electron_matrix

    sig_x3 = np.sum(weight * x_shift**3) 
    sig_y3 = np.sum(weight * y_shift**3)  
    sig_x2y = np.sum(weight * x_shift**2 * y_shift) 
    sig_xy2 = np.sum(weight * x_shift * y_shift**2) 

    asym_projection = abs(
        sig_x3 * cos_theta**3 +
        sig_y3 * sin_theta**3 +
        3 * sig_x2y * cos_theta**2 * sin_theta +
        3 * sig_xy2 * sin_theta**2 * cos_theta
    )

    distribution_asymmetry = asym_projection**(1/3) if asym_projection > 0 else 0



    top_two_values = np.partition(photon_electron_matrix.flatten(), -2)[-2:]
    frac2 = np.sum(top_two_values) / total_photon_electrons

    if generate_visualization:
        plt.figure(figsize=(10, 8))
        x_edges = np.linspace(x_positions[0] - 0.5, x_positions[-1] + 0.5, photon_electron_matrix.shape[1] + 1)
        y_edges = np.linspace(y_positions[0] - 0.5, y_positions[-1] + 0.5, photon_electron_matrix.shape[0] + 1)

        mesh = plt.pcolormesh(x_edges, y_edges, photon_electron_matrix, cmap='viridis',
                              edgecolors='black', linewidth=0.5)

        ellipse = Ellipse(xy=(centroid_x, centroid_y), width=2*a, height=2*b,
                          angle=np.degrees(angle_rad), edgecolor='w', facecolor='none', lw=2, label='Fitted Ellipse')
        plt.gca().add_patch(ellipse)

        plt.scatter([centroid_x], [centroid_y], color='magenta', edgecolors='black', label='Energy Centroid', zorder=3)
        plt.plot([0, centroid_x], [0, centroid_y], 'cyan', linestyle='--', linewidth=1.5, label='Center-to-Centroid Vector')

        dx = a * np.cos(angle_rad)
        dy = a * np.sin(angle_rad)
        plt.plot([centroid_x - dx, centroid_x + dx], [centroid_y - dy, centroid_y + dy],
                 'red', linestyle='-', linewidth=2, label='Major Axis Vector')

        plt.axhline(0, color='gray', linestyle='--')
        plt.axvline(0, color='gray', linestyle='--')
        plt.colorbar(mesh).set_label("Photon-Electron Count")
        plt.xticks(x_positions, rotation=45)
        plt.yticks(y_positions)
        plt.title("Shower Profile with PCA Ellipse")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.grid(False)
        plt.show()

    return (
        centroid_x,
        centroid_y,
        2*a,  # 2a
        2*b,  # 2b
        eccentricity,
        radial_displacement,
        radial_angle,
        orientation_angle,
        alpha,
        orientation_slope,
        radial_slope,
        total_active_pixels,
        total_photon_electrons,
        azimuthal_spread,
        axis_miss_distance,
        edge_leakage_ratio,
        brightest_pixel_distance,
        brightest_pixel_value,
        1,
        distribution_asymmetry,
        frac2
    )


def save_shower_analysis_to_txt(events: np.ndarray, bt_values: np.ndarray, pt_values: np.ndarray, output_file: str):
    """
    Analyze a batch of photon-electron events and save results to a tab-separated .txt file,
    including a header and extra features bt and pt.

    Parameters:
        events (np.ndarray): Shape (N, H, W), a batch of 2D photon-electron matrices.
        bt_values (np.ndarray): Shape (N,), additional scalar per event.
        pt_values (np.ndarray): Shape (N,), additional scalar per event.
        output_file (str): Path to save the tab-separated result file.
    """

    headers = [
        "centroid_x", "centroid_y", "2a", "2b", "eccentricity",
        "radial_distance", "radial_angle", "orientation_angle", "alpha",
        "orientation_slope", "radial_slope", "total_active_pixels", "total_PE_count",
        "azimuthal_spread", "miss_distance", "edge_leakage_ratio", "brightest_pixel_distance",
        "brightest_pixel_value",  # <-- Updated field
        "flag", "distribution_asymmetry", "frac2",
        "bt", "pt"  # <-- Extra appended fields
    ]

    with open(output_file, 'w') as f:
        f.write('\t'.join(headers) + '\n')

        for i, event in enumerate(events):
            try:
                result = analyze_shower(event, generate_visualization=False)
            except Exception:
                result = tuple([0.0] * 21)
            
            # Append bt and pt to the result
            full_result = list(result) + [bt_values[i], pt_values[i]]

            line = '\t'.join(f'{val:.6f}' if isinstance(val, float) else str(val) for val in full_result)
            f.write(line + '\n')
