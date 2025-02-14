# LiDARPhotonSim: LiDAR Simulation & Signal Processing

LiDARPhotonSim is a collection of Python scripts that simulate key concepts in LiDAR signal processing. The repository includes a simulation that visualizes photon counting from a LiDAR system, featuring:

- **Photon Simulation:** Visualizes the interaction of sun photons and a red laser pulse with a LiDAR receiver, an object, and a sun source.
- **Real-Time Histogram:** Builds an integer histogram of photon counts versus time (in nanoseconds) to illustrate signal accumulation.
- **Matched Filter Demonstration:** Shows how applying a Gaussian matched filter can smooth out noise and enhance the signal. A moving kernel is overlaid on the histogram to visually demonstrate the transition from raw to filtered output.

> **Important Note:**  
> This simulation is an educational and abstract demonstration of LiDAR signal processing concepts. It is a simplified model and does **not** capture the full complexity or realistic behavior of a SPAD LiDAR system.

Feel free to explore, experiment, and adapt the code for your own projects. Contributions and feedback are welcome!
