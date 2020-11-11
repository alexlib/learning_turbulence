"""Parameters file."""

import numpy as np
from param_dns import L, ε, kf, kfw, seed, η, ν, α, Uα

# Domain
N = 512
mesh = None

# Physical parameters
# Prescribed
lh_kmax = 3  # Enstrophy hyperdissipation scale
# Derived
kmax = N * np.pi / L
lh = lh_kmax / kmax
h = lh**4 * η**(1/3)

# Temporal parameters
dx = L / N
safety = 0.5
dt = safety * dx / Uα
ts = "RK443"
stop_sim_time = np.inf
stop_wall_time = np.inf
scalars_iter = 10
snapshots_iter = 100
stop_iteration = 100000

