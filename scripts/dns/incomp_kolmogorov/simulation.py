
import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools
import parameters as param

import logging
logger = logging.getLogger(__name__)


# Bases and domain
x_basis = de.Fourier('x', param.Nx, interval=(0, param.Lx), dealias=3/2)
y_basis = de.Fourier('y', param.Ny, interval=(0, param.Ly), dealias=3/2)
z_basis = de.Fourier('z', param.Nz, interval=(0, param.Lz), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)

# Problem
problem = de.IVP(domain, variables=['p','ux','uy','uz'])
problem.parameters['ν'] = param.ν
problem.parameters['Fx'] = param.Fx
problem.parameters['Fy'] = param.Fy
problem.parameters['Fz'] = param.Fz
problem.substitutions['ke'] = "(ux*ux + uy*uy + uz*uz) / 2"
problem.substitutions['ωx'] = "dy(uz) - dz(uy)"
problem.substitutions['ωy'] = "dz(ux) - dx(uz)"
problem.substitutions['ωz'] = "dx(uy) - dy(ux)"
problem.substitutions['L(a)'] = "dx(dx(a)) + dy(dy(a)) + dz(dz(a))"
problem.substitutions['A(a)'] = "ux*dx(a) + uy*dy(a) + uz*dz(a)"
problem.add_equation("dt(ux) - ν*L(ux) + dx(p) = -A(ux) + Fx")
problem.add_equation("dt(uy) - ν*L(uy) + dy(p) = -A(uy) + Fy")
problem.add_equation("dt(uz) - ν*L(uz) + dz(p) = -A(uz) + Fz")
problem.add_equation("dx(ux) + dy(uy) + dz(uz) = 0", condition="(nx != 0) or (ny != 0) or (nz != 0)")
problem.add_equation("p = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK443)
logger.info('Solver built')

# Initial conditions
# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]

x, y, z = domain.grids(1)
ux = solver.state['ux']
uy = solver.state['uy']
uz = solver.state['uz']
ux['g'] = param.ux(x, y, z)
uy['g'] = param.uy(x, y, z)
uz['g'] = param.uz(x, y, z)

# Integration parameters
solver.stop_sim_time = param.stop_sim_time
solver.stop_wall_time = param.stop_wall_time
solver.stop_iteration = param.stop_iteration

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', iter=param.snapshot_iter, max_writes=1)
snapshots.add_system(solver.state)
snapshots.add_task("ke")
snapshots.add_task("ωx")
snapshots.add_task("ωy")
snapshots.add_task("ωz")

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("integ(ke)", name='KE')

# Main loop
dt = param.dt
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Total KE = %f' %flow.max('KE'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
