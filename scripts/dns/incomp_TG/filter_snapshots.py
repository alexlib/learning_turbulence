
"""
Plot planes from joint analysis files.

Usage:
    filter_snapshots.py <N> <files>... [--output=<dir>] [--parallel]

Options:
    --output=<dir>  Output directory [default: ./filtered]
    --parallel      Distribute analysis over COMM_WORLD

"""

import h5py
import post as post_tools
import filter
import xarray
import parameters as params


def field_to_xarray(field, layout='g'):
    """Convert Dedalus field to xarray dataset."""
    data = field[layout]
    domain = field.domain
    layout = domain.dist.get_layout_object(layout)
    coords = []
    for axis in range(domain.dim):
        basis = domain.bases[axis]
        meta = field.meta[basis.name]
        if layout.grid_space[axis]:
            label = basis.name
            scale = basis.grid(meta['scale'])
        else:
            label = basis.element_name
            scale = basis.elements
        coords.append((label, scale))
    xr_data = xarray.DataArray(data, coords=coords)
    return xr_data


def save_subgrid_fields(filename, N, comm, output_path):
    """Compute and save subgrid velocity, stress, and strain components."""
    print(filename)
    out = {}
    # Load velocities
    domain = post_tools.build_domain(params, comm=comm)
    ux = post_tools.load_field(filename, domain, 'ux', 0)
    uy = post_tools.load_field(filename, domain, 'uy', 0)
    uz = post_tools.load_field(filename, domain, 'uz', 0)
    # Filter velocities
    filt = filter.build_filter(domain, N)
    out['ux'] = filt_ux = filt(ux).evaluate()
    out['uy'] = filt_uy = filt(uy).evaluate()
    out['uz'] = filt_uz = filt(uz).evaluate()
    # Compute subgrid stress components
    out['txx'] = filt(filt_ux*filt_ux - ux*ux).evaluate()
    out['tyy'] = filt(filt_uy*filt_uy - uy*uy).evaluate()
    out['tzz'] = filt(filt_uz*filt_uz - uz*uz).evaluate()
    out['txy'] = filt(filt_ux*filt_uy - ux*uy).evaluate()
    out['tyz'] = filt(filt_uy*filt_uz - uy*uz).evaluate()
    out['tzx'] = filt(filt_uz*filt_ux - uz*ux).evaluate()
    # Compute resolved strain components
    dx = domain.bases[0].Differentiate
    dy = domain.bases[1].Differentiate
    dz = domain.bases[2].Differentiate
    out['Sxx'] = dx(filt_ux).evaluate()
    out['Syy'] = dy(filt_uy).evaluate()
    out['Szz'] = dz(filt_uz).evaluate()
    out['Sxy'] = (0.5*(dx(filt_uy) + dy(filt_ux))).evaluate()
    out['Syz'] = (0.5*(dy(filt_uz) + dz(filt_uy))).evaluate()
    out['Szx'] = (0.5*(dz(filt_ux) + dx(filt_uz))).evaluate()
    # Save all outputs
    for key in out:
        field = out[key]
        field.set_scales(N / params.N)
        out[key] = field_to_xarray(field, layout='g')
    ds = xarray.Dataset(out)
    input_path = pathlib.Path(filename)
    output_filename = output_path.joinpath(input_path.stem).with_suffix('.nc')
    ds.to_netcdf(output_filename)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
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
    else:
        rank = MPI.COMM_WORLD.rank
        size = MPI.COMM_WORLD.size
        comm = MPI.COMM_SELF
        files = args['<files>'][rank::size]
    # Run
    for file in files:
        save_subgrid_fields(file, int(args['<N>']), comm, output_path)
