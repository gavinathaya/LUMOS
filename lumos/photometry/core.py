import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import lumos.io as lumio
import lumos.photometry.detect as detect
import lumos.photometry.aperture as aperture
import matplotlib.pyplot as plt
from lumos.utils.helpers import progress_bar
from astropy.table import QTable, MaskedColumn
from astropy.time import Time
from astropy.timeseries import TimeSeries
import astropy.units as u
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import fit_wcs_from_points, pixel_to_skycoord

class WCSDegenerateWarning(RuntimeWarning):
    pass
warnings.simplefilter("once", WCSDegenerateWarning)
warnings.simplefilter("ignore", FITSFixedWarning)

class PhotometrySession:
    def __init__(self,
                 metadata: pd.DataFrame = pd.DataFrame(),
                 lightcurves = None,
                 ref_stars: pd.DataFrame = pd.DataFrame(),
                 ref_image: str = "") -> None:
        self.metadata = metadata  #Same metadata as in CalibrationFrames class
        self.lightcurves = lightcurves if lightcurves is not None else {} #Dict of astropy Time Series QTables
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
        ts = TimeSeries(time = Time([], format='isot'),
                        masked = True) #Empty TimeSeries to start
        
        for filter in self.metadata['FILTER'].unique():
            ts.add_column(MaskedColumn(name = f'mag_{filter}',
                                       dtype = float,
                                       mask = True))

        self.lightcurves[object_name] = {
            "coordinates": object_coordinates,
            "lightcurve": ts
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
                    detection_dir: str = './detection_dir/',
                    subject_name: str = '',
                    metadata_dir: str = './') -> None:
        """
        Detect light sources in metadata clean images and store the results
        as csv files in detection_dir.

        Parameters
        ----------
        fwhm : float
            The full width at half maximum for the Gaussian kernel.
        threshold : float
            The absolute image value above which to select sources.
        detection_dir : str
            The directory to save the source CSV files.
        subject_name : str
            The subject name to use in the source CSV filenames.
        metadata_dir : str
            The directory where the metadata CSV file will be saved.
        
        Returns
        -------
        None
        """
        #Only process successfully calibrated files
        success_meta = self.metadata.query('CAL_STATUS == "SUCCESS"')
        
        print(f"Source CSV files will be saved to: '{detection_dir}'")
        Path(detection_dir).mkdir(parents = True, exist_ok=True)
        print("Detecting sources in images...")
        for i, row in enumerate(success_meta.itertuples()):
            source = detect.data_star_identification(fits.getdata(row.CLN_FILENAME),
                                                      fwhm=fwhm,
                                                      threshold=threshold)
            basename = Path(row.FILENAME).name # pyright: ignore[reportArgumentType]
            source_filename = Path(detection_dir).joinpath(basename)
            source_filename = source_filename.with_suffix('.csv')
            source.write(source_filename, format='csv', overwrite=True)
            self.metadata.loc[row.Index, ['DETECTION_FILENAME']] = [str(source_filename)]
            progress_bar(i, len(success_meta))
        
        self.metadata.to_csv(f'{metadata_dir}{subject_name}_metadata.csv', index=False)
        print(f'Current metadata saved to {metadata_dir}{subject_name}_metadata.csv')
        return None

    def plot_sources(self, plot_dir:str = './source_plots/', origin: str = "lower") -> None:
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
        #Only process successfully calibrated files
        success_meta = self.metadata.query('CAL_STATUS == "SUCCESS"')
        
        print(f"Source plots will be saved to: '{plot_dir}'")
        Path(plot_dir).mkdir(parents = True, exist_ok=True)
        print("Plotting sources on images...")
        for i, row in enumerate(success_meta.itertuples()):
            hdul = fits.open(row.CLN_FILENAME)  # pyright: ignore[reportCallIssue]
            data = hdul[0].data  # pyright: ignore[reportAttributeAccessIssue]
            header = hdul[0].header  # pyright: ignore[reportAttributeAccessIssue]
            wcs = WCS(header)
            if lumio.is_valid_wcs(wcs) is False:
                warnings.warn(f"WCS not valid in file {row.CLN_FILENAME}, skipping plot.", WCSDegenerateWarning)
                hdul.close()
                continue

            source_data = QTable.read(row.DETECTION_FILENAME, format='csv')
            fig = detect.plot_source(data, f"Detected sources in {Path(row.FILENAME).name}", # pyright: ignore[reportArgumentType]
                                     "Pixel Coordinates", "RA/Dec (J2000)", wcs=wcs, source=source_data,
                                     origin=origin)
            hdul.close()
            plot_filename = Path(plot_dir).joinpath(Path(row.FILENAME).stem + '_sources.png') # pyright: ignore[reportArgumentType]
            fig.savefig(plot_filename)
            plt.close(fig)
            progress_bar(i, len(success_meta))
        return None

    def apply_aperture_photometry(self, n_fwhm: float = 3.0, 
                                  fit_shape: int = 15, phot_dir: str = './phot_CSV/',
                                  subject_name: str = '', metadata_dir: str = './') -> None:
            """
            Detect light sources in metadata clean images and store the results
            as csv files in phot_dir.

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
            #Only process successfully calibrated files
            success_meta = self.metadata.query('CAL_STATUS == "SUCCESS"')

            print(f"Photometry CSV files will be saved to: '{phot_dir}'")
            Path(phot_dir).mkdir(parents = True, exist_ok=True)
            print("Applying aperture photometry in images...")
            for i, row in enumerate(success_meta.itertuples()):
                hdul = fits.open(row.CLN_FILENAME)  # pyright: ignore[reportCallIssue]
                image = hdul[0].data  # pyright: ignore[reportAttributeAccessIssue]
                header = hdul[0].header  # pyright: ignore[reportAttributeAccessIssue]
                wcs = WCS(header)
                
                #Verify WCS
                if lumio.is_valid_wcs(wcs) is False:
                    warnings.warn(f"WCS not valid in file {row.CLN_FILENAME}, skipping plot.", WCSDegenerateWarning)
                    hdul.close()
                    continue

                #Read and generate source table
                source = QTable.read(row.DETECTION_FILENAME, format='csv')

                coords = pixel_to_skycoord(source['xcentroid'].data, source['ycentroid'].data, wcs)
                xypos = np.column_stack((source['xcentroid'].data, source['ycentroid'].data))
                phot_table = aperture.apply_phot_aperture(image, xypos, n_fwhm=n_fwhm, fit_shape=fit_shape)
                phot_table.add_columns([coords.ra, coords.dec], names = ['ra', 'dec'])

                catalogue = self.ref_stars.dropna()
                phot_coord = SkyCoord(phot_table['ra'], phot_table['dec'], frame = 'icrs', unit='deg')
                idx, d2d, d3d = match_coordinates_sky(phot_coord,
                                                      SkyCoord(catalogue['ra'], catalogue['dec'], frame = 'icrs', unit='deg'))
                tol = d2d < 10 * u.arcsec # pyright: ignore[reportAttributeAccessIssue]
                matched_catalogue = catalogue.iloc[idx[tol]]
                matched_table = phot_table[tol]
                mag, mag_inst = aperture.calibrate_mag(matched_table['aperture_sum'], matched_catalogue[(f'mag_{row.FILTER}').lower()], phot_table['aperture_sum'])
                phot_table['mag'] = mag
                phot_table['instrumental_mag'] = mag_inst

                #lightcurve logic
                for obj in self.lightcurves.keys():
                    obj_coord = self.lightcurves[obj]["coordinates"]
                    d2d_obj = obj_coord.separation(phot_coord)
                    matched_tol = d2d_obj < 10 * u.arcsec # pyright: ignore[reportAttributeAccessIssue]
                    if np.any(matched_tol):
                        matched_table = phot_table[matched_tol]
                        obs_time = row.DATE_OBS  # pyright: ignore[reportAttributeAccessIssue]
                        self.lightcurves[obj]["lightcurve"].add_row({'time': obs_time,
                                     (f'mag_{row.FILTER}'): matched_table['mag'][0]})
                    
                    #Sort lightcurve by time
                    self.lightcurves[obj]["lightcurve"].sort('time')

                basename = Path(row.FILENAME).name # pyright: ignore[reportArgumentType]
                phot_table_filename = Path(phot_dir).joinpath(basename)
                phot_table_filename = phot_table_filename.with_suffix('.csv')
                hdul.close()
                
                #Because we use plain CSV, convert RA/Dec to degrees and drop units
                phot_table['ra'] = phot_table['ra'].to_value(u.deg) # type: ignore
                phot_table['dec'] = phot_table['dec'].to_value(u.deg) # type: ignore

                phot_table.write(phot_table_filename, format='ascii.fast_csv', overwrite=True)
                self.metadata.loc[row.Index, ['PHOT_FILENAME']] = [str(phot_table_filename)]
                progress_bar(i, len(success_meta))
            
            self.metadata.to_csv(f'{metadata_dir}{subject_name}_metadata.csv', index=False)
            print(f'Current metadata saved to {metadata_dir}{subject_name}_metadata.csv')
            return None