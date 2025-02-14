import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# ----------------------
# Simulation Parameters
# ----------------------
sim_time = 500      # Simulation time for forming the histogram (ns)
final_time = 750    # Total animation time (ns) including matched filter phase
dt = 1              # 1 ns per frame
num_frames = final_time + 1
time_array = np.arange(sim_time + 1)  # x-axis: 0 to 500 ns

# Histogram: photon counts per ns (all counts are integers)
hist_counts = np.zeros(sim_time + 1, dtype=int)

# ----------------------
# Matched Filter Parameters
# ----------------------
sigma_red = 5      # standard deviation (ns) for the Gaussian kernel
kernel_range = np.arange(-20, 21)  # Kernel window from -20 ns to +20 ns
kernel = np.exp(- (kernel_range**2) / (2 * (sigma_red ** 2)))
# Normalize kernel so that its sum equals 1 (smoothing filter)
kernel = kernel / np.sum(kernel)
# Define max_red (used in red pulse simulation)
max_red = 50

# ----------------------
# Scene Setup (Figure & Axes)
# ----------------------
fig, (ax_hist, ax_scene) = plt.subplots(2, 1, figsize=(10, 8))

# Top Panel: Display histogram.
# We'll use one line (purple) for the display.
(line_hist,) = ax_hist.plot([], [], color='purple', lw=2, label="Histogram")
# And one line for the moving kernel (orange dashed).
(line_kernel,) = ax_hist.plot([], [], color='orange', lw=2, linestyle='--', label="Kernel")
ax_hist.set_xlim(0, sim_time)
ax_hist.set_ylim(0, 60)
ax_hist.set_xlabel("Time (ns)")
ax_hist.set_ylabel("Photon Counts")
ax_hist.set_title("Photon Counts vs. Time")
ax_hist.legend(loc="upper right")

# Bottom Panel: Scene (schematic view)
ax_scene.set_xlim(0, 1)
ax_scene.set_ylim(0, 1)
ax_scene.axis('off')

# ----------------------
# Static Scene Elements
# ----------------------
# LiDAR/Receiver (blue circle)
emitter = np.array([0.2, 0.5])
emitter_circle = plt.Circle(emitter, 0.03, color='blue', zorder=2)
ax_scene.add_patch(emitter_circle)

# Object (gray rectangle)
object_center = np.array([0.8, 0.5])
object_rect = plt.Rectangle((object_center[0]-0.03, object_center[1]-0.1),
                            0.06, 0.2, color='gray', zorder=2)
ax_scene.add_patch(object_rect)

# Sun (yellow circle)
sun_pos = np.array([0.6, 0.9])
sun_circle = plt.Circle(sun_pos, 0.05, color='yellow', zorder=2)
ax_scene.add_patch(sun_circle)

# ----------------------
# Add Text Labels
# ----------------------
ax_scene.text(sun_pos[0], sun_pos[1]-0.08, "Sun", color='black', fontsize=12, ha='center')
ax_scene.text(object_center[0], object_center[1]-0.15, "Object", color='black', fontsize=12, ha='center')
ax_scene.text(emitter[0], emitter[1]-0.08, "LiDAR", color='black', fontsize=12, ha='center')

# ----------------------
# Dynamic Scatter Plots
# ----------------------
sun_scatter = ax_scene.scatter([], [], color='yellow', s=60, zorder=3)
red_scatter = ax_scene.scatter([], [], color='red', s=150, zorder=4)

# ----------------------
# Sun Photon Parameters and Storage
# ----------------------
# Each sun photon is stored as a dict.
sun_photons = []

# Photon travel times (in ns)
travel_time_direct = 10          # sun -> LiDAR
travel_time_to_object = 10       # sun -> object
travel_time_object_to_emitter = 10  # object -> LiDAR

# ----------------------
# Laser (Red Pulse) Parameters
# ----------------------
T_emit_red = 0    # ns: start of red pulse emission
T_bounce   = 140  # ns: red pulse reaches the object and bounces
T_arrival  = 280  # ns: red pulse returns to LiDAR

# ----------------------
# Update Function for Animation
# ----------------------
def update(frame):
    global hist_counts, sun_photons
    t = frame  # current time in ns

    # Phase 1: Simulation (0 <= t < sim_time)
    if t < sim_time:
        # Generate new sun photons:
        # Direct photons: from sun directly to LiDAR.
        n_direct = random.randint(15, 25)
        for _ in range(n_direct):
            vel = (emitter - sun_pos) / travel_time_direct
            sun_photons.append({'pos': sun_pos.copy(), 'vel': vel, 'route': 'direct'})
        # Reflected photons: from sun to object, then bounce toward LiDAR.
        n_reflected = random.randint(2, 5)
        for _ in range(n_reflected):
            vel = (object_center - sun_pos) / travel_time_to_object
            sun_photons.append({'pos': sun_pos.copy(), 'vel': vel, 'route': 'to_object'})
        
        # Update sun photons and check for arrival:
        updated_sun_photons = []
        sun_positions = []  # positions for scatter plot
        threshold = 0.03  # distance threshold for "arrival"
        for photon in sun_photons:
            photon['pos'] += photon['vel'] * dt
            if photon['route'] == 'to_object':
                if np.linalg.norm(photon['pos'] - object_center) < threshold:
                    photon['vel'] = (emitter - object_center) / travel_time_object_to_emitter
                    photon['route'] = 'to_emitter'
            if photon['route'] in ['direct', 'to_emitter']:
                if np.linalg.norm(photon['pos'] - emitter) < threshold:
                    hist_counts[t] += 1  # count photon
                    continue  # photon is consumed
            updated_sun_photons.append(photon)
            sun_positions.append(photon['pos'].copy())
        sun_photons = updated_sun_photons
        
        # Red pulse animation:
        red_positions = []
        if T_emit_red <= t <= T_arrival:
            if t < T_bounce:
                frac = (t - T_emit_red) / (T_bounce - T_emit_red)
                pos = emitter + frac * (object_center - emitter)
            else:
                frac = (t - T_bounce) / (T_arrival - T_bounce)
                pos = object_center - frac * (object_center - emitter)
                # Count red photons with a Gaussian envelope.
                if T_arrival - 20 <= t <= T_arrival:
                    count = int(round(max_red * np.exp(-((t - T_arrival)**2) / (2 * sigma_red**2))))
                    hist_counts[t] += count
            num_red_dots = 10
            for i in range(num_red_dots):
                offset = (i - num_red_dots / 2) * 0.005
                red_positions.append([pos[0], pos[1] + offset])
        else:
            red_positions = []
        
        # Update the raw histogram display (only up to current t).
        line_hist.set_data(time_array[:t+1], hist_counts[:t+1])
        # Hide the kernel line during simulation.
        line_kernel.set_data([], [])
    
    # Phase 2: Matched Filter Demo (sim_time <= t <= final_time)
    else:
        # Freeze the raw histogram.
        # Compute the smoothed (matched filter) output.
        smoothed = np.convolve(hist_counts, kernel, mode='same')
        # To preserve the overall amplitude, re-scale the smoothed output:
        if np.max(smoothed) > 0:
            scale_output = np.max(hist_counts) / np.max(smoothed)
        else:
            scale_output = 1
        smoothed_scaled = smoothed * scale_output
        
        # Determine the kernel's current center as it moves from 0 to sim_time.
        frac_time = (t - sim_time) / (final_time - sim_time)  # from 0 to 1
        kernel_center = frac_time * sim_time
        
        # Build the display:
        # For times ≤ kernel_center, show the smoothed value; for times > kernel_center, show raw histogram.
        display_y = np.where(time_array <= kernel_center, smoothed_scaled, hist_counts)
        line_hist.set_data(time_array, display_y)
        
        # Draw the moving kernel.
        # For visual purposes, we set a separate scale for the kernel display.
        kernel_vis_scale = np.max(hist_counts) * 0.5  if np.max(hist_counts) > 0 else 10
        x_kernel = kernel_range + kernel_center
        y_kernel = kernel * kernel_vis_scale
        line_kernel.set_data(x_kernel, y_kernel)
        
        # Clear photon scatter during Phase 2.
        sun_scatter.set_offsets(np.empty((0, 2)))
        red_scatter.set_offsets(np.empty((0, 2)))
    
    # Update scatter plots (only during Phase 1).
    if t < sim_time:
        if len(sun_positions) > 0:
            sun_scatter.set_offsets(np.array(sun_positions))
        else:
            sun_scatter.set_offsets(np.empty((0, 2)))
        if len(red_positions) > 0:
            red_scatter.set_offsets(np.array(red_positions))
        else:
            red_scatter.set_offsets(np.empty((0, 2)))
    
    # Adjust y-axis based on current display values.
    current_max = max(np.max(hist_counts), np.max(smoothed_scaled) if t>=sim_time else 60)
    ax_hist.set_ylim(0, current_max + 10)
    
    # Stop the animation after final_time.
    if t >= final_time:
        anim.event_source.stop()
    
    return line_hist, line_kernel, sun_scatter, red_scatter

# ----------------------
# Create and Run Animation
# ----------------------
anim = FuncAnimation(fig, update, frames=num_frames, interval=10, blit=True)
plt.tight_layout()
plt.show()
