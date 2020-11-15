
"""
Plot planes from joint analysis files.

Usage:
    filter_snapshots.py <N_filt> <mlog10_ep> <files>... [--output=<dir>] [--parallel]

Options:
    --output=<dir>  Output directory [default: ./filtered]
    --parallel      Distribute analysis over COMM_WORLD

"""

import h5py
import post
import filter
import xarray
import numpy as np
import param_dns


def save_subgrid_fields(filename, N_filt, epsilon, comm, output_path):
    """Compute and save subgrid velocity, stress, and strain components."""
    print(filename)
    out = {}
    # Load velocities
    domain = post.build_domain(param_dns.N, param_dns.L, comm=comm)
    ux = post.load_field_hdf5(filename, domain, 'ux', 0)
    uy = post.load_field_hdf5(filename, domain, 'uy', 0)
    print('Done loading fields')
    # Filter velocities
    F = filter.build_gaussian_filter(domain, N_filt, epsilon)
    out['ux'] = F_ux = F(ux).evaluate()
    out['uy'] = F_uy = F(uy).evaluate()
    print('Done filtering fields')
    # Compute implicit subgrid stress components
    out['im_txx'] = (F(ux*ux) - F_ux*F_ux).evaluate()
    out['im_tyy'] = (F(uy*uy) - F_uy*F_uy).evaluate()
    out['im_txy'] = (F(ux*uy) - F_ux*F_uy).evaluate()
    # Compute explicit subgrid stress components
    out['ex_txx'] = (F(ux*ux) - F(F_ux*F_ux)).evaluate()
    out['ex_tyy'] = (F(uy*uy) - F(F_uy*F_uy)).evaluate()
    out['ex_txy'] = (F(ux*uy) - F(F_ux*F_uy)).evaluate()
    print('Done computing stresses')
    # Truncate and convert to xarray
    for key in out:
        field = out[key]
        field.require_coeff_space()
        field.set_scales(N_filt / param_dns.N)
        out[key] = post.field_to_xarray(field, layout='g')
    print('Done converting to xarray')
    # Save to netcdf
    ds = xarray.Dataset(out)
    input_path = pathlib.Path(filename)
    output_filename = output_path.joinpath(input_path.stem).with_suffix('.nc')
    ds.to_netcdf(output_filename)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools.parallel import Sync
    from mpi4py import MPI

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    # Distribute
    if args['--parallel']:
        comm = MPI.COMM_WORLD
        files = args['<files>']
        raise NotImplementedError()
    else:
        rank = MPI.COMM_WORLD.rank
        size = MPI.COMM_WORLD.size
        comm = MPI.COMM_SELF
        files = args['<files>'][rank::size]
    # Run
    for file in files:
        save_subgrid_fields(file, int(args['<N_filt>']), float(args['<mlog10_ep>']), comm, output_path)

