from pathlib import Path
import pandas as pd
import numpy as np
import lumos.io as lumio
import lumos.photometry.detect as detect
from lumos.utils.helpers import progress_bar
from astropy.table import QTable
import astropy.units as u
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points

class PhotometrySession:
    def __init__(self,
                 metadata: pd.DataFrame = pd.DataFrame(),
                 lightcurves = None,
                 ref_stars: pd.DataFrame = pd.DataFrame(),
                 ref_image: str = "") -> None:
        self.metadata = metadata  #Same metadata as in CalibrationFrames class
        self.lightcurves = lightcurves if lightcurves is not None else {} #Dict of astropy QTables
        self.ref_stars = ref_stars  #Reference catalog (VizieR, Gaia, or custom CSV) (Index, RA, Dec, WavelengthMag)
        self.ref_image = ref_image  #Reference image filename
        self.wcs_files = lumio.find_WCS_files(self.metadata)["CLN_FILENAME"]

    def add_wcs(self) -> None:
        """
        Approximately add WCS to images based on reference image

        Parameters
        ----------
        filename : str
            The filename of the image to add WCS to (based on self.ref_image)
        
        Returns
        -------
        None
        """
        ref_hdul = fits.open(self.ref_image)
        ref_data = ref_hdul[0].data; ref_header = ref_hdul[0].header # type: ignore
        ref_source = detect.data_star_identification(ref_data)
        x_ref, y_ref = ref_source['xcentroid'], ref_source['ycentroid']
        ref_wcs = WCS(ref_header)
        ref_ra, ref_dec = ref_wcs.all_pix2world(x_ref, y_ref, 0)
        ref_coords = SkyCoord(ra=ref_ra*u.deg, dec=ref_dec*u.deg) # pyright: ignore[reportAttributeAccessIssue]
        for fname in self.metadata["CLN_FILENAME"]:
            if fname in set(self.wcs_files):
                continue
            with fits.open(fname,
                           mode='update',
                           output_verify = 'silentfix') as curr_hdul:
                data = curr_hdul[0].data; header = curr_hdul[0].header # type: ignore
                current_source = detect.data_star_identification(data)
                x_curr, y_curr = current_source['xcentroid'], current_source['ycentroid']
                approx_ra, approx_dec = ref_wcs.all_pix2world(x_curr, y_curr, 0)
                approx_coords = SkyCoord(ra=approx_ra*u.deg, dec=approx_dec*u.deg) # pyright: ignore[reportAttributeAccessIssue]
                idx, d2d, d3d = match_coordinates_sky(approx_coords, ref_coords)
                print(d2d)
                matched = d2d < 20 * u.arcsec # pyright: ignore[reportAttributeAccessIssue]
                x_curr_matched = x_curr[matched]; y_curr_matched = y_curr[matched]
                ref_matched = ref_coords[idx[matched]]
                xy = np.vstack([x_curr_matched, y_curr_matched]).T # pyright: ignore[reportCallIssue, reportArgumentType]
                wcs_curr = fit_wcs_from_points(xy, ref_matched)
                header.update(wcs_curr.to_header()) # pyright: ignore[reportAttributeAccessIssue]
                curr_hdul.flush()
                fits.writeto(fname, data, header, overwrite=True)
        ref_hdul.close()
        return None

    def find_source(self, fwhm: float = 15.0, threshold: float = 5.0,
                    source_dir: str = './source_CSV/',
                    subject_name: str = '',
                    metadata_dir: str = './') -> None:
        """
        Detect light sources in metadata clean images and store the results
        as csv files in source_dir.

        Parameters
        ----------
        fwhm : float
            The full width at half maximum for the Gaussian kernel.
        threshold : float
            The absolute image value above which to select sources.
        source_dir : str
            The directory to save the source CSV files.
        subject_name : str
            The subject name to use in the source CSV filenames.
        metadata_dir : str
            The directory where the metadata CSV file will be saved.
        
        Returns
        -------
        None
        """
        print(f"Source CSV files will be saved to: '{source_dir}'")
        Path(source_dir).mkdir(parents = True, exist_ok=True)
        print("Detecting sources in images...")
        for i, row in enumerate(self.metadata.query('CAL_STATUS == "SUCCESS"').itertuples()):
            sources = detect.data_star_identification(fits.getdata(row.CLN_FILENAME),
                                                      fwhm=fwhm,
                                                      threshold=threshold)
            basename = Path(row.FILENAME).name # pyright: ignore[reportArgumentType]
            source_filename = Path(source_dir).joinpath(basename)
            source_filename = source_filename.with_suffix('.csv')
            sources.write(source_filename, format='csv', overwrite=True)
            self.metadata.loc[row.Index, ['SOURCE_FILENAME']] = [str(source_filename)]
            progress_bar(i, len(self.metadata.query('CAL_STATUS == "SUCCESS"')))
        
        self.metadata.to_csv(f'{metadata_dir}{subject_name}_metadata.csv', index=False)
        print(f'Current metadata saved to {metadata_dir}{subject_name}_metadata.csv')
        return None
    

# class PhotometryTEST:
#     def __init__(self):
#         # Image metadata (per frame)
#         self.metadata = QTable(
#             names=("filename", "time", "filter", "exptime", "wcs"),
#             dtype=("U200", object, "U20", float, object)
#         )

#         # Light curves per object (dict of QTables)
#         self.lightcurves = {}

#         # Reference image for registration fallback
#         self.ref_index = 0
#         self.ref_stars = None  # detected stars in reference frame

#     # --------------------------
#     # Step 1: Load metadata
#     # --------------------------
#     def add_image(self, filename, header):
#         time = Time(header["DATE-OBS"])
#         filt = header.get("FILTER", "Unknown")
#         exptime = header.get("EXPTIME", 0.0)
#         wcs = WCS(header)

#         self.metadata.add_row((filename, time, filt, exptime, wcs))

#     # --------------------------
#     # Step 2: Register objects
#     # --------------------------
#     def add_object(self, name, skycoord):
#         # Make a table to store this object’s light curve
#         self.lightcurves[name] = QTable(
#             names=("time", "x", "y", "flux", "flux_err"),
#             dtype=(object, float, float, float, float)
#         )
#         self.lightcurves[name].meta["skycoord"] = skycoord

#     # --------------------------
#     # Step 3: Locate objects in frame
#     # --------------------------
#     def locate_objects(self, idx, fallback=True):
#         row = self.metadata[idx]
#         wcs = row["wcs"]

#         positions = {}
#         for name, lc in self.lightcurves.items():
#             coord = lc.meta["skycoord"]
#             try:
#                 x, y = wcs.world_to_pixel(coord)
#             except Exception:
#                 if not fallback:
#                     raise
#                 # Fallback: register to reference image
#                 x, y = self._register_to_reference(idx, coord)
#             positions[name] = (x, y)
#         return positions

#     def _register_to_reference(self, idx, coord):
#         # --- stub: implement registration here ---
#         # Use source detection + geometric transform to align
#         # For now, just return dummy coords
#         return (np.nan, np.nan)

#     # --------------------------
#     # Step 4: Measure fluxes
#     # --------------------------
#     def measure_fluxes(self, idx, data, positions, aperture_radius=5.0):
#         # stub: implement with photutils aperture_photometry
#         for name, (x, y) in positions.items():
#             flux, flux_err = np.nan, np.nan  # replace with real measurement
#             self.lightcurves[name].add_row((self.metadata[idx]["time"], x, y, flux, flux_err))

#     # --------------------------
#     # Step 5: Export
#     # --------------------------
#     def export_lightcurve(self, name, filename):
#         self.lightcurves[name].write(filename, overwrite=True)
