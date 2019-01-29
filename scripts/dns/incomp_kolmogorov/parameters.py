"""Parameters file."""

import numpy as np

# Domain
L = 2 * np.pi
N = 16
Lx = Ly = Lz = L
Nx = Ny = Nz = N

# Physical parameters
V = 1
k = 2 * np.pi / L
ν = 1

# Taylor-Green forcing
Fx = 0
Fy = 0
Fz = 0

# Initial conditions
ux = lambda x, y, z: V * np.sin(k*x) * np.cos(k*x) * np.cos(k*z)
uy = lambda x, y, z: -V * np.cos(k*x) * np.sin(k*y) * np.cos(k*z)
uz = lambda x, y, z: 0

# Temporal parameters
dt = 1e-2
stop_sim_time = 10
stop_wall_time = np.inf
stop_iteration = np.inf
snapshot_iter = 100


