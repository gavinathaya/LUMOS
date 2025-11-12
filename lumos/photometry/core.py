from pathlib import Path
import pandas as pd
import numpy as np
import lumos.io as lumio
import lumos.photometry.detect as detect
import matplotlib.pyplot as plt
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

    def add_lightcurves(self, object_name: str,
                        object_coordinates: SkyCoord) -> None:
        """
        Add light curves for a specific astronomical object.

        Parameters
        ----------
        object_name : str
            The name of the astronomical object.
        object_coordinates : SkyCoord
            The coordinates of the astronomical object.
        """
        self.lightcurves[object_name] = {
            "coordinates": object_coordinates,
            "lightcurve": QTable()
        }

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

    def plot_sources(self, plot_dir:str = './cal_plots/', origin: str = "lower") -> None:
        """
        Plot detected sources for all exposures marked as successfully calibrated.

        This method iterates over rows in self.metadata where CAL_STATUS == "SUCCESS",
        opens the corresponding FITS image and source CSV table, overlays detected
        sources on the image using the image WCS, and saves a PNG per exposure to the
        specified output directory.

        Parameters
        ----------
        plot_dir : str, optional
            Directory where per-exposure source plots will be written. If the directory
            does not exist it will be created. Default is './cal_plots/'.
        origin : str, optional
            Origin parameter passed to the plotting routine (commonly "lower" or
            "upper") to control image origin when plotting. Default is "lower".

        Returns
        -------
        None

        Side effects
        ------------
        - Creates the output directory if it does not exist.
        - Opens FITS files and reads their headers/data.
        - Reads source tables (expected CSV format) into an Astropy QTable.
        - Calls an external plotting routine (detect.plot_source) to generate a
          Matplotlib Figure for each exposure.
        - Saves one PNG file per exposure named "<exposure_stem>_sources.png" into
          plot_dir.
        - Prints progress messages to stdout and updates a progress bar.

        Exceptions
        ----------
        Any IO-related or parsing errors raised by astropy.io.fits.open,
        astropy.table.QTable.read, or the plotting/saving calls will propagate. Common
        failure modes include missing or unreadable FITS/source files and errors when
        writing the output PNG files.

        Notes
        -----
        - The method filters metadata rows by the string CAL_STATUS == "SUCCESS".
        - The plotting routine is provided the WCS from the FITS header so RA/Dec
          coordinates are overplotted when available.
        - The caller should ensure that self.metadata and the referenced filenames
          (e.g. row.CLN_FILENAME, row.SOURCE_FILENAME, row.FILENAME) are valid.
        """
        print(f"Source plots will be saved to: '{plot_dir}'")
        Path(plot_dir).mkdir(parents = True, exist_ok=True)
        print("Plotting sources on images...")
        for i, row in enumerate(self.metadata.query('CAL_STATUS == "SUCCESS"').itertuples()):
            hdul = fits.open(row.CLN_FILENAME)  # pyright: ignore[reportCallIssue]
            data = hdul[0].data  # pyright: ignore[reportAttributeAccessIssue]
            header = hdul[0].header  # pyright: ignore[reportAttributeAccessIssue]
            wcs = WCS(header)
            source_data = QTable.read(row.SOURCE_FILENAME, format='csv')
            fig = detect.plot_source(data, f"Detected Sources in {Path(row.FILENAME).name}", # pyright: ignore[reportArgumentType]
                                     "Pixel Coordinates", "RA/Dec (J2000)", wcs=wcs, source=source_data,
                                     origin=origin)
            hdul.close()
            plot_filename = Path(plot_dir).joinpath(Path(row.FILENAME).stem + '_sources.png') # pyright: ignore[reportArgumentType]
            fig.savefig(plot_filename)
            plt.close(fig)
            progress_bar(i, len(self.metadata.query('CAL_STATUS == "SUCCESS"')))
        return None