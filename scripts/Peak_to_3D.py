import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Constants
c = 3e8         # Speed of light in m/s
dt = 1          # Time resolution in ns
T_max = 500     # Maximum time in ns

# Time axis (in ns)
time = np.arange(0, T_max + 1, dt)

# -----------------------------
# 1. Generate a Synthetic Histogram with Multiple Peaks
# -----------------------------
min_peak_no = 1
max_peak_no = 4
# Randomly choose the number of peaks between min_peak_no and max_peak_no
num_peaks = np.random.randint(min_peak_no, max_peak_no)
peaks_parameters = []
# np.random.seed(42)  # Uncomment for reproducibility

# For each peak, define:
# - center: the center time (t0) in ns
# - amplitude: the photon count (A)
# - sigma_peak: the standard deviation (σ) in ns
for i in range(num_peaks):
    center = np.random.randint(50, T_max - 50)         # Avoid edges
    amplitude = np.random.randint(20, 81)                # Amplitude between 20 and 81 counts
    sigma_peak = np.random.uniform(2, 8)                 # Sigma between 2 and 8 ns
    peaks_parameters.append((center, amplitude, sigma_peak))

# Sum the contributions from each peak to form the true signal
true_signal = np.zeros_like(time, dtype=float)
for (center, amplitude, sigma_peak) in peaks_parameters:
    true_signal += amplitude * np.exp(-0.5 * ((time - center) / sigma_peak)**2)

# Simulate ambient noise with a mean around 25 counts (uniform between 20 and 30)
ambient_noise = np.random.randint(20, 31, size=time.shape)

# Combined raw histogram (true signal + ambient noise)
raw_histogram = true_signal + ambient_noise

# -----------------------------
# 2. Apply a Matched Filter (Gaussian Kernel)
# -----------------------------
sigma_filter = 5  # Filter sigma (in ns)
kernel_range = np.arange(-20, 21)  # Kernel spans from -20 ns to +20 ns
gaussian_kernel = np.exp(-0.5 * (kernel_range / sigma_filter)**2)
gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)  # Normalize the kernel

# Convolve the raw histogram with the Gaussian kernel to obtain the filtered histogram
filtered_histogram = np.convolve(raw_histogram, gaussian_kernel, mode='same')

# -----------------------------
# 3. Thresholding and Peak Detection
# -----------------------------
# Compute ambient noise statistics
sigma_noise = np.std(ambient_noise)
mean_noise = np.mean(ambient_noise)

# Adaptive threshold: T = (threshold factor) * sigma_noise + mean_noise
threshold_factor = 3
threshold = threshold_factor * sigma_noise + mean_noise

# Detect peaks in the filtered histogram above the threshold.
# The 'distance' parameter ensures peaks are at least 5 ns apart.
peaks, properties = find_peaks(filtered_histogram, height=threshold, distance=5)

# -----------------------------
# 4. Extract and Validate Peaks Based on SNR
# -----------------------------
SNR_threshold = 2  # Minimum acceptable SNR

valid_peaks = []
for peak in peaks:
    peak_value = filtered_histogram[peak]
    SNR = peak_value / mean_noise  # SNR = peak value / ambient noise mean
    if SNR >= SNR_threshold:
        valid_peaks.append((peak, peak_value, SNR))

# -----------------------------
# 5. Intensity Calculation for Each Valid Peak
# -----------------------------
# Define a window (in ns) around the peak to sum the photon counts
window = 5  # 5 ns on each side of the peak
intensity_results = []
for (peak, peak_value, SNR) in valid_peaks:
    start_idx = max(0, peak - window)
    end_idx = min(len(raw_histogram), peak + window + 1)
    # Cast the sum to an integer since intensity must be an integer
    intensity = int(np.sum(raw_histogram[start_idx:end_idx]))
    intensity_results.append((peak, intensity))

# -----------------------------
# 6. Convert Peak Times to Range for Each Valid Peak
# -----------------------------
results = []
for (peak, peak_value, SNR) in valid_peaks:
    t_peak_seconds = peak * dt * 1e-9  # Convert peak time from ns to seconds
    range_m = (c * t_peak_seconds) / 2   # Range conversion (round-trip)
    results.append((peak, peak_value, range_m, SNR))

# -----------------------------
# 7. Direction Analysis & Coordinate Conversion (for one example return)
# -----------------------------
# Simple camera sensor model for direction
sensor_width, sensor_height = 640, 480
horizontal_fov = np.deg2rad(60)  # Horizontal Field-of-View (60° in radians)
vertical_fov = np.deg2rad(45)    # Vertical Field-of-View (45° in radians)

# Example pixel coordinate for detected return
pixel_x, pixel_y = 350, 260
center_x, center_y = sensor_width / 2, sensor_height / 2

# Calculate the angle per pixel in each dimension
angle_per_pixel_x = horizontal_fov / sensor_width
angle_per_pixel_y = vertical_fov / sensor_height

# Convert pixel offsets to azimuth (φ) and elevation (θ) angles
azimuth = (pixel_x - center_x) * angle_per_pixel_x
elevation = (pixel_y - center_y) * angle_per_pixel_y

# Use the first valid peak for coordinate conversion (if available)
if results:
    peak_index, peak_value, range_m, SNR_value = results[0]
    X = range_m * np.cos(elevation) * np.sin(azimuth)
    Y = range_m * np.sin(elevation)
    Z = range_m * np.cos(elevation) * np.cos(azimuth)
else:
    X = Y = Z = np.nan

# -----------------------------
# 8. Plotting the Results
# -----------------------------
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Plot the raw and filtered histograms
ax[0].plot(time, raw_histogram, label='Raw Histogram', color='skyblue')
ax[0].plot(time, filtered_histogram, label='Filtered Histogram', color='orange', linewidth=2)
ax[0].axhline(threshold, color='red', linestyle='--', label='Threshold')
ax[0].scatter(time[peaks], filtered_histogram[peaks], marker='x', color='green', label='Detected Peaks')
ax[0].set_xlabel("Time (ns)")
ax[0].set_ylabel("Photon Counts")
ax[0].set_title("LiDAR Histogram and Matched Filter Output")
ax[0].legend()

# Prepare text with computed values, including intensity from intensity_results
valid_peaks_text = "\n".join(
    [f"Peak at {p} ns: Value={v:.2f}, Range={(c * p * dt * 1e-9) / 2:.2f} m, SNR={snr:.2f}, Intensity={intensity}"
     for (p, v, _, snr), (_, intensity) in zip(results, intensity_results)]
)
coord_text = f"Using first valid peak:\nAzimuth: {np.rad2deg(azimuth):.2f}°\nElevation: {np.rad2deg(elevation):.2f}°\nX: {X:.2f} m, Y: {Y:.2f} m, Z: {Z:.2f} m"

textstr = "\n".join((
    f"Ambient Noise Mean: {mean_noise:.2f} counts",
    f"Noise Sigma: {sigma_noise:.2f}",
    f"Threshold: {threshold:.2f} counts",
    "",
    "Valid Peaks (after SNR thresholding):",
    valid_peaks_text,
    "",
    coord_text
))
ax[1].axis('off')
ax[1].text(0.05, 0.5, textstr, fontsize=12, verticalalignment='center')

plt.tight_layout()
plt.show()

# -----------------------------
# Print Results to Console
# -----------------------------
print("Detected LiDAR Returns (Valid Peaks):")
for (peak, peak_value, range_m, SNR_value), (_, intensity) in zip(results, intensity_results):
    print(f" - Peak at {peak} ns, Value: {peak_value:.2f} counts, Range: {range_m:.2f} m, SNR: {SNR_value:.2f}, Intensity: {intensity}")

print("\nDirection and 3D Coordinates (using first valid peak):")
print(f" - Azimuth: {np.rad2deg(azimuth):.2f}°")
print(f" - Elevation: {np.rad2deg(elevation):.2f}°")
print(f" - X: {X:.2f} m, Y: {Y:.2f} m, Z: {Z:.2f} m")
