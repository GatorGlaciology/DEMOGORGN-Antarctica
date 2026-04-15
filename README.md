# DEMOGORGN Antarctica
Code for developing and accessing geostatistical Antarctic bed elevation models 

DEMOGORGN (Digital Elevation Models of Geostatistical ORiGiN) is an ensemble of geostatistically simulated bed topographies. This ensemble is hosted on Source Cooperative (https://source.coop/englacial/demogorgn). Here are instructions for uploading data to Source Coop, making updates, and downloading the data.

DEMOGORGN is a living data product, which means we are continuing to make updates and improvements. Please see our blog for information on DEMOGORGN updates: https://www.gatorglaciology.com/demogorgn. 

## Upload data to Source Coop


## Download data


## Merging mass conserving region with SGS region

The scripts in `Patch update scripts/` handle merging a mass-conserving (MCMC) bed inversion into the existing DEMOGORGN ensemble and running SGS to fill the transition zone.

**Files:**
- `MCMC_to_demogorgn.ipynb` — development notebook that works through the merge pipeline step by step
- `demogorgn_update.py` — the `DEMOGORGN_update` class that packages the pipeline into reusable methods
- `updata_demo.ipynb` — demo notebook showing how to use `DEMOGORGN_update` end to end

**What the pipeline does:**

1. **Load inputs.** Load 100 MCMC bed realizations from `MCMC_results/MCMC_*.npy` and the base DEMOGORGN realization from `tmp_realization.nc`.

2. **Merge the mass-conserving region.** Inside the high-velocity mask (`highvel_mask == 1`), the MCMC bed values replace the base bed. Outside the mask the base bed is kept.

3. **Build a buffer zone.** A 5 km buffer around the edge of the mass-conserving region is identified. Cells that are already part of the inversion domain, ocean (BedMachine source = 3), NaN in BedMap, or within 10 pixels of the grid edge are excluded from the buffer. The remaining buffer pixels are filled with radar-derived bed elevations (`bedmap_surf - radar_thickness`) as a temporary placeholder before SGS.

4. **Run SGS on the buffer zone.** Sequential Gaussian Simulation (SGS) is run to fill the buffer pixels in each realization. The surrounding non-buffer bed values are used as conditioning data after a normal-score transform. A Matern variogram (range ~140 km, sill 0.85, smoothness 0.66) is used with 50 nearest neighbours and a 30 km search radius.

**Usage with `DEMOGORGN_update`:**

```python
from demogorgn_update import DEMOGORGN_update

updata = DEMOGORGN_update()
updata.build_ice_stream_stack()   # loads MCMC realizations
updata.build_ice_chunk_stack()    # loads base bed from NetCDF
updata.build_ice_chunk_updated()  # merges and applies buffer mask
final_stack = updata.sgs_simulation()  # fills buffer with SGS
```

The final stack is stored in `updata.ice_chunk_stack_final` and returned by `sgs_simulation()`.

