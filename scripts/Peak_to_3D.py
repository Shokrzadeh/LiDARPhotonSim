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
num_peaks = np.random.randint(min_peak_no, max_peak_no)
peaks_parameters = []
#np.random.seed(42)  # for reproducibility

# Increase amplitude range to help ensure multiple peaks exceed threshold
for i in range(num_peaks):
    center = np.random.randint(50, T_max - 50)         # Avoid edges
    amplitude = np.random.randint(20, 81)                # Random amplitude between 40 and 80 counts
    sigma_peak = np.random.uniform(2, 8)                 # Random sigma between 3 and 8 ns
    peaks_parameters.append((center, amplitude, sigma_peak))

# Sum the contributions from each peak
true_signal = np.zeros_like(time, dtype=float)
for (center, amplitude, sigma_peak) in peaks_parameters:
    true_signal += amplitude * np.exp(-0.5 * ((time - center) / sigma_peak)**2)

# Simulate ambient noise with a mean around 25 counts.
# We'll use a uniform distribution between 20 and 30.
ambient_noise = np.random.randint(20, 31, size=time.shape)

# Combined raw histogram (true signal + ambient noise)
raw_histogram = true_signal + ambient_noise

# -----------------------------
# 2. Apply a Matched Filter (Gaussian Kernel)
# -----------------------------
sigma_filter = 5  # ns
kernel_range = np.arange(-20, 21)  # from -20 ns to +20 ns
gaussian_kernel = np.exp(-0.5 * (kernel_range / sigma_filter)**2)
gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)  # Normalize

# Convolve the raw histogram with the Gaussian kernel (matched filtering)
filtered_histogram = np.convolve(raw_histogram, gaussian_kernel, mode='same')

# -----------------------------
# 3. Thresholding and Peak Detection
# -----------------------------
# Compute the standard deviation (sigma) and mean of the ambient noise
sigma_noise = np.std(raw_histogram)
mean_noise = np.mean(raw_histogram)

# Set the threshold as a factor of the ambient noise sigma plus the mean value
threshold_factor = 3
threshold = threshold_factor * sigma_noise + mean_noise

# Use scipy.signal.find_peaks to detect all peaks above the threshold
# The 'distance' parameter ensures that peaks are separated by at least 5 ns.
peaks, properties = find_peaks(filtered_histogram, height=threshold, distance=5)
peak_heights = properties["peak_heights"]

# -----------------------------
# 4. Extract and Validate Peaks Based on SNR
# -----------------------------
# Define SNR threshold: peaks with SNR below this value will be discarded.
SNR_threshold = 2

valid_peaks = []
for peak in peaks:
    peak_value = filtered_histogram[peak]
    SNR = peak_value / mean_noise  # Using ambient mean as noise level
    if SNR >= SNR_threshold:
        valid_peaks.append((peak, peak_value, SNR))

# -----------------------------
# 5. Convert Peak Times to Range for Each Valid Peak
# -----------------------------
results = []
for (peak, peak_value, SNR) in valid_peaks:
    t_peak_seconds = peak * dt * 1e-9  # convert ns to seconds
    range_m = (c * t_peak_seconds) / 2   # round-trip conversion
    results.append((peak, peak_value, range_m, SNR))

# -----------------------------
# 6. Direction Analysis & Coordinate Conversion (for one example return)
# -----------------------------
sensor_width, sensor_height = 640, 480
horizontal_fov = np.deg2rad(60)  # 60 degrees in radians
vertical_fov = np.deg2rad(45)    # 45 degrees in radians

# Example: simulate return detected at pixel (350, 260)
pixel_x, pixel_y = 350, 260
center_x, center_y = sensor_width / 2, sensor_height / 2
angle_per_pixel_x = horizontal_fov / sensor_width
angle_per_pixel_y = vertical_fov / sensor_height

azimuth = (pixel_x - center_x) * angle_per_pixel_x  # Horizontal angle
elevation = (pixel_y - center_y) * angle_per_pixel_y  # Vertical angle

# Use the first valid peak for coordinate conversion (if available)
if results:
    peak_index, peak_value, range_m, SNR_value = results[0]
    X = range_m * np.cos(elevation) * np.sin(azimuth)
    Y = range_m * np.sin(elevation)
    Z = range_m * np.cos(elevation) * np.cos(azimuth)
else:
    X = Y = Z = np.nan

# -----------------------------
# 7. Plotting the Results
# -----------------------------
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Plot the raw and filtered histograms
ax[0].plot(time, raw_histogram, label='Raw Histogram', color='skyblue')
ax[0].plot(time, filtered_histogram, label='Filtered Histogram', color='orange', linewidth=2)
ax[0].axhline(threshold, color='red', linestyle='--', label='Threshold')
ax[0].scatter(time[peaks], filtered_histogram[peaks], marker='x', color='green', label='All Detected Peaks')
ax[0].set_xlabel("Time (ns)")
ax[0].set_ylabel("Photon Counts")
ax[0].set_title("LiDAR Histogram and Matched Filter Output")
ax[0].legend()

# Display the computed values on the second subplot
valid_peaks_text = "\n".join(
    [f"Peak at {p} ns: Value={v:.2f}, Range={(c * p * dt * 1e-9) / 2:.2f} m, SNR={snr:.2f}" 
     for (p, v, _, snr) in results]
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
for (peak, peak_value, range_m, SNR_value) in results:
    print(f" - Peak at {peak} ns, Value: {peak_value:.2f} counts, Range: {range_m:.2f} m, SNR: {SNR_value:.2f}")

print("\nDirection and 3D Coordinates (using first valid peak):")
print(f" - Azimuth: {np.rad2deg(azimuth):.2f}°")
print(f" - Elevation: {np.rad2deg(elevation):.2f}°")
print(f" - X: {X:.2f} m, Y: {Y:.2f} m, Z: {Z:.2f} m")
