import numpy as np
import pandas as pd
import verde as vd
class DEMOGORGN_update:
    """
    Handles updating and gap-filling glaciological bed elevation datasets by
    combining ice stream MCMC realizations with a base ice chunk (bed surface),
    applying spatial masks, and running Sequential Gaussian Simulation (SGS)
    to fill buffer zones around the ice stream area.
    """

    def __init__(
        self,
        ice_chunk=None,
        ice_stream_stack=None,
        nc_path="tmp_realization.nc",
        coord_csv="MertzGlacierDataGridded_2.csv",
        x_coord_name="x",
        y_coord_name="y",
    ):
        """
        Parameters
        ----------
        ice_chunk : array-like, optional
            2D or 3D base bed elevation array. If 2D, it will be broadcast
            across realizations when needed.
        ice_stream_stack : array-like, optional
            3D stack of ice stream bed realizations with shape
            (n_realizations, ny, nx).
        nc_path : str
            Path to the NetCDF file containing the base bed realization.
        coord_csv : str
            Path to the CSV file with gridded spatial coordinates and mask
            columns (e.g. highvel_mask, bedmap_mask, bedmap_surf,
            radar_thickness).
        x_coord_name : str
            Name of the x-coordinate dimension in the NetCDF file.
        y_coord_name : str
            Name of the y-coordinate dimension in the NetCDF file.
        """
        self.ice_chunk_stack = np.asarray(ice_chunk) if ice_chunk is not None else None
        self.ice_stream_stack = np.asarray(ice_stream_stack) if ice_stream_stack is not None else None
        self.ice_chunk_stack_updated = None
        self.ice_chunk_stack_final = None
        self.nc_path = nc_path
        self.coord_csv = coord_csv
        self.x_coord_name = x_coord_name
        self.y_coord_name = y_coord_name

        df = pd.read_csv(self.coord_csv)
        self.x_uniq = np.unique(df.x)
        self.y_uniq = np.unique(df.y)
        xx, yy = np.meshgrid(self.x_uniq, self.y_uniq)
        self.coord = (xx, yy)
        self.mask = None

    def build_ice_stream_stack(self):
        """
        Load and stack 100 MCMC bed realization arrays from the MCMC_results/
        directory into a single (100, ny, nx) array.

        Expects files named MCMC_0.npy through MCMC_99.npy. Raises ValueError
        if exactly 100 files are not found.

        Returns
        -------
        ice_stream_stack : np.ndarray
            Array of shape (100, ny, nx) containing all MCMC realizations.
        """
        from pathlib import Path

        mcmc_files = sorted(
            Path("MCMC_results").glob("MCMC_*.npy"),
            key=lambda path: int(path.stem.split("_")[1]),
        )
        if len(mcmc_files) != 100:
            raise ValueError(f"Expected 100 MCMC files, found {len(mcmc_files)}")
        ice_stream_stack = np.stack([np.load(path) for path in mcmc_files])
        self.ice_stream_stack = ice_stream_stack
        return ice_stream_stack

    def build_ice_chunk_stack(self):
        """
        Load the base bed realization from the NetCDF file and broadcast it
        into a stack of n_realizations identical copies.

        Reads the DataArray from nc_path, subsets it to the unique x/y
        coordinates from the coordinate CSV, ensures (y, x) dimension order,
        and repeats the 2D array n_realizations times along a new leading axis.
        n_realizations defaults to 100 or matches ice_stream_stack if available.

        Returns
        -------
        ice_chunk_stack : np.ndarray
            Array of shape (n_realizations, ny, nx).
        """
        import xarray as xr

        da = xr.open_dataarray(self.nc_path).squeeze(drop=True)
        trimmed = da.sel({self.x_coord_name: self.x_uniq, self.y_coord_name: self.y_uniq})
        if tuple(trimmed.dims) == (self.x_coord_name, self.y_coord_name):
            trimmed = trimmed.transpose(self.y_coord_name, self.x_coord_name)

        n_realizations = 100
        if self.ice_stream_stack is not None:
            n_realizations = self.ice_stream_stack.shape[0]

        ice_chunk_stack = np.repeat(np.asarray(trimmed.values)[np.newaxis, :, :], n_realizations, axis=0)
        self.ice_chunk_stack = ice_chunk_stack
        return ice_chunk_stack

    def build_ice_chunk_updated(self, ice_chunk=None, ice_stream_stack=None):
        """
        Merge the ice stream realizations into the base ice chunk and apply
        spatial masking to produce an updated bed elevation stack.

        For each realization, pixels inside the high-velocity mask are replaced
        by the corresponding ice stream value. A buffer zone (5 km distance from
        the ice stream edge, excluding the stream itself, BedMachine ocean
        cells, NaN bedmap regions, and a 10-pixel border) is identified and
        filled with radar-derived bed elevations. The buffer mask is stored in
        self.mask for use by sgs_simulation.

        Parameters
        ----------
        ice_chunk : array-like, optional
            2D or 3D base bed array. Falls back to self.ice_chunk_stack.
        ice_stream_stack : array-like, optional
            3D MCMC realization stack. Falls back to self.ice_stream_stack.

        Returns
        -------
        ice_chunk_stack_updated : np.ndarray
            Array of shape (n_realizations, ny, nx) with ice stream values
            merged in and buffer pixels filled with radar bed elevations.
        """
        ice_chunk = np.asarray(ice_chunk) if ice_chunk is not None else self.ice_chunk_stack
        ice_stream_stack = np.asarray(ice_stream_stack) if ice_stream_stack is not None else self.ice_stream_stack

        if ice_chunk.ndim == 2:
            ice_chunk = np.repeat(ice_chunk[np.newaxis, :, :], ice_stream_stack.shape[0], axis=0)
        if ice_stream_stack.ndim != 3:
            raise ValueError("ice_stream_stack must be a 3D array with shape (n_realizations, nx, ny)")
        if ice_chunk.ndim != 3:
            raise ValueError("ice_chunk must be a 2D or 3D array")
        if ice_chunk.shape != ice_stream_stack.shape:
            raise ValueError(
                f"ice_chunk and ice_stream_stack must have matching shapes, got {ice_chunk.shape} and {ice_stream_stack.shape}"
            )

        df = pd.read_csv(self.coord_csv)
        xx, yy = self.coord

        mask = df["highvel_mask"].values.reshape(xx.shape)
        bedmap_mask = df["bedmap_mask"].values.reshape(xx.shape)
        df['bed_cond'] = df.bedmap_surf - df.radar_thickness
        radar_bed = df['bed_cond'].values.reshape(xx.shape)

        reference_stream = ice_stream_stack[0]
        inv_masked = np.where(mask == 1, reference_stream, np.nan)
        area = np.isfinite(inv_masked)

        x_points = xx[area]
        y_points = yy[area]

        buffer_mask = vd.distance_mask(
            data_coordinates=(x_points, y_points),
            maxdist=5000,
            coordinates=(xx, yy),
        ).astype(int)

        edge_buffer = np.zeros_like(buffer_mask, dtype=bool)
        n = 10
        edge_buffer[:n, :] = True
        edge_buffer[-n:, :] = True
        edge_buffer[:, :n] = True
        edge_buffer[:, -n:] = True

        exclude = (
            (~np.isnan(inv_masked))
            | (mask == 1)
            | (bedmap_mask == 3)
            | np.isnan(bedmap_mask)
            | edge_buffer
        )
        buffer_mask[exclude] = 0

        self.mask = buffer_mask
        self.inv_masked_stack = np.where(mask[np.newaxis, :, :] == 1, ice_stream_stack, self.ice_chunk_stack)

        # Notebook-equivalent blank filling:
        # beds_updated = np.where(buffer_mask == 1, radar_bed, beds)
        ice_chunk_stack_updated = np.where(buffer_mask[np.newaxis, :, :] == 1, radar_bed[np.newaxis, :, :], self.inv_masked_stack)
        self.ice_chunk_stack_updated = ice_chunk_stack_updated
        return ice_chunk_stack_updated
    
    def sgs_simulation(self, ice_chunk_stack=None):
        """
        Fill the buffer zone pixels in each realization using Sequential
        Gaussian Simulation (SGS) conditioned on surrounding bed values.

        For each realization the surrounding finite, non-buffer pixels are used
        as conditioning data. Values are normal-score transformed via a
        QuantileTransformer before kriging-based SGS (okrige_sgs) and
        back-transformed afterwards. Realizations where the conditioning set is
        empty or has fewer than 2 unique values are skipped.

        Variogram parameters (Matern covariance, isotropic range ~140 km,
        sill 0.85, shape 0.66) and search parameters (k=50 neighbours,
        30 km search radius) are fixed.

        Must be called after build_ice_chunk_updated (requires self.mask).

        Parameters
        ----------
        ice_chunk_stack : array-like, optional
            3D updated bed stack to simulate into. Falls back to
            self.ice_chunk_stack_updated.

        Returns
        -------
        simulated_stack : np.ndarray
            Array of shape (n_realizations, ny, nx) with buffer pixels replaced
            by SGS-simulated values.
        """
        import gstatsim as gs
        from sklearn.preprocessing import QuantileTransformer

        if self.mask is None:
            raise ValueError("build_ice_chunk_updated must be called before sgs_simulation")

        ice_chunk_stack = np.asarray(ice_chunk_stack) if ice_chunk_stack is not None else self.ice_chunk_stack_updated
        if ice_chunk_stack is None:
            raise ValueError("ice_chunk_stack is not available")

        buffer_mask = self.mask.astype(bool)
        if not np.any(buffer_mask):
            self.ice_chunk_stack = ice_chunk_stack.copy()
            return self.ice_chunk_stack

        xx, yy = self.coord
        prediction_grid = np.column_stack((xx[buffer_mask], yy[buffer_mask]))
        simulated_stack = np.array(ice_chunk_stack, copy=True)

        azimuth = 0
        nugget = 0
        major_range = 140743
        minor_range = 140743
        sill = 0.85
        vtype = "Matern"
        s = 0.66
        vario = [azimuth, nugget, major_range, minor_range, sill, vtype, s]
        k = 50
        rad = 30000
        rng = np.random.default_rng(seed=2002)

        for idx in range (3):#(simulated_stack.shape[0]):
            surface = simulated_stack[idx]
            condition_mask = (~buffer_mask) & np.isfinite(surface)
            if not np.any(condition_mask):
                continue

            conditioning_values = surface[condition_mask].reshape(-1, 1)
            n_quantiles = min(500, conditioning_values.shape[0])
            if n_quantiles < 2:
                continue

            nst_trans = QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution="normal",
                random_state=0,
            ).fit(conditioning_values)

            df_grid = pd.DataFrame(
                {
                    "X": xx[condition_mask],
                    "Y": yy[condition_mask],
                    "Nbed": nst_trans.transform(conditioning_values).ravel(),
                }
            )

            sim_normal = gs.Interpolation.okrige_sgs(
                prediction_grid,
                df_grid,
                "X",
                "Y",
                "Nbed",
                k,
                vario,
                rad,
                seed=rng,
                quiet=True,
            ).reshape(-1, 1)

            surface_filled = surface.copy()
            surface_filled[buffer_mask] = nst_trans.inverse_transform(sim_normal).ravel()
            simulated_stack[idx] = surface_filled

        self.ice_chunk_stack_final = simulated_stack
        return simulated_stack
