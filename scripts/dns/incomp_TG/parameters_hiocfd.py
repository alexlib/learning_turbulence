"""Parameters file."""

import numpy as np

# Domain
L = 1
N = 256
Bx = By = Bz = (-np.pi*L, np.pi*L)
Nx = Ny = Nz = N

# Physical parameters
Re = 1600  # V * L / ν
V = L  # tc = L / V = 1
k = 1 / L
ν = V * L / Re

# Taylor-Green forcing
Fx = 0
Fy = 0
Fz = 0

# Initial conditions
ux = lambda x, y, z: V * np.sin(k*x) * np.cos(k*y) * np.cos(k*z)
uy = lambda x, y, z: -V * np.cos(k*x) * np.sin(k*y) * np.cos(k*z)
uz = lambda x, y, z: 0
p = lambda x, y, z: V**2/16 * (np.cos(2*k*x) + np.cos(2*k*y)) * (np.cos(2*k*z) + 2)

# Temporal parameters
dt = 1e-2
stop_sim_time = np.inf
stop_wall_time = np.inf
stop_iteration = (20 // dt) + 1
snapshots_iter = int(1 // dt)
slices_iter = int(0.1 // dt)
scalars_iter = int(0.01 // dt)


