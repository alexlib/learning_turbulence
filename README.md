# Learning Turbulence

## Data location

Data is stored on rusty at `/mnt/home/kburns/work_rusty/learning_turbulence/tg_dns/s5/filtered`.

Data can be loaded using the xarray package as

```
# Load entire xarray dataset for a given snapshot
import xarray
dataset = xarray.open_dataset('snapshots_s1.nc')
# Get numpy array for the field txx
dataset['txx'].data
```
