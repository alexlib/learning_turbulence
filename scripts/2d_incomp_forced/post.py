"""Post-processing tools."""

import numpy as np
import dedalus.public as de
import h5py


def build_domain(param, comm=None):
    """Build domain object."""
    x_basis = de.Fourier('x', param.N, interval=(0, param.L), dealias=3/2)
    y_basis = de.Fourier('y', param.N, interval=(0, param.L), dealias=3/2)
    domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64, comm=comm)
    return domain


def load_field(filename, domain, task, index=-1):
    """Load a field from HDF5 file."""
    with h5py.File(filename, 'r') as file:
        field = domain.new_field(name=task)
        field['g'] = file['tasks'][task][index]
    return field

