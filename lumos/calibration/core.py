import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os; import sys
from astropy.io import fits
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
import lumos.utils as lumutils
import lumos.visualization as lumvis

class CalibrationFrames:
    """
    # CalibrationFrames

    High-level container and convenience routines for building, storing, applying,
    and inspecting CCD/CMOS calibration frames (bias, darks, flats) and for
    post-calibration processing (background removal, plotting). Designed to work
    with FITS files and a metadata table describing raw science frames.
    Primary responsibilities
    - Load sets of raw calibration frames from disk (bias, darks, flats), combine
        them into master calibration frames (median combination), and store them on
        the instance.
    - Maintain a metadata table for science frames and write/update per-file
        calibration results.
    - Apply calibration (bias subtraction, dark subtraction, flat division) to
        science images and write calibrated FITS files.
    - Estimate and subtract a spatially varying background from calibrated frames.
    - Produce simple comparison plots of raw vs calibrated, and calibrated vs
        cleaned frames.
    
    Attributes
    ----------
    bias : numpy.ndarray or None
            The master bias frame (2D array) or None if not set.
    darks : dict
            Mapping from exposure time (hashable key) to master dark frame (2D array).
            Example: {60.0: numpy.ndarray(...), 120.0: numpy.ndarray(...)}.
    flats : dict
            Mapping from filter identifier (string or hashable) to master flat (2D array).
    metadata : pandas.DataFrame
            Table describing science frames and their calibration state. Expected or
            used columns include:
                - 'FILENAME' : path to the raw FITS file
                - 'EXPTIME'  : exposure time (used to select dark)
                - 'FILTER'   : filter name (used to select flat)
            During processing additional columns are written/updated such as:
                - 'CAL_FILENAME', 'CAL_STATUS', 'CLN_FILENAME', etc.
    
    Dependencies
    -------------
    This class expects the following libraries for full functionality:
    - numpy
    - pandas
    - astropy.io.fits (getdata, getheader, PrimaryHDU, writeto)
    - astropy.stats.SigmaClip
    - photutils.background.Background2D and MedianBackground (or equivalent)
    - matplotlib (for saving figures created by lumvis)
    - lumutils (progress_bar)
    - lumvis (plot_comparison)
    Adjust imports or provide compatible replacement functions if your environment
    differs.

    Usage patterns
    --------------
    1. Build masters from file lists or DataFrame:
         - load_bias(df_or_list) -> median-combined bias frame (and optionally store).
         - load_darks(df_or_list) -> dictionary keyed by EXPTIME of median darks.
         - load_flats(df_or_list) -> dictionary keyed by FILTER of median-normalized flats.
         Each loader accepts:
             - a pandas.DataFrame with required columns, or
             - a list / numpy.ndarray of filenames (in which case the code reads FITS
                 headers to build the necessary columns).
    2. Add masters programmatically:
         - add_bias(files), add_dark(exptime, files), add_flat(filter_name, files)
         These create masters directly from provided file lists.
    3. Apply calibration to a single array:
         - apply_array(raw_image, exptime, filter_used) -> calibrated array (float64)
         The method performs: calibrated = (raw - bias) - (dark - bias); then divide by flat.
         Preconditions:
             - bias must be set (otherwise ValueError).
             - a matching dark for exptime must exist in self.darks (KeyError if missing).
             - a matching flat for filter_used must exist in self.flats (KeyError if missing).
         Notes:
             - All input frames (bias, dark, flat) are expected to have the same shape
                 as raw_image. No explicit per-pixel masking for zeros in flats; division by
                 zero will yield infinities/NaNs as in numpy.
    4. Batch processing using metadata:
         - apply_self(calibrated_dir=..., subject_name=..., metadata_dir=..., warn=True)
             Iterates self.metadata rows and attempts to calibrate each listed file,
             writing calibrated FITS and updating the metadata table with status and output
             filename. Records failures for missing calibration data (ValueError/KeyError)
             and continues. On completion writes an updated CSV of metadata.
    5. Background estimation and removal:
         - remove_background(data, sigma=3.0, box_size=50, filter_size=(3,3))
             Uses a median-based Background2D estimator and SigmaClip to compute a
             background map and subtract it from the input 2D array.
         - remove_background_self(clean_dir=..., subject_name=..., metadata_dir=..., warn=True, ...)
             Runs background removal on files previously calibrated (metadata entries
             with CAL_STATUS == "SUCCESS"), writes cleaned FITS files, and updates
             metadata with the clean filenames.
    6. Plotting helpers:
         - plot_calibration(plot_dir=..., origin='lower'): saves raw vs calibrated
             comparison images for successfully calibrated frames.
         - plot_background(plot_dir=..., origin='lower'): saves calibrated vs cleaned
             comparison images.
    
    Error handling and edge cases
    -----------------------------
    - File I/O errors (missing files, unreadable FITS) and astropy errors will
        propagate unless explicitly caught by callers. apply_self records only
        ValueError/KeyError calibration failures per file and continues; other
        exceptions will likely abort processing.
    - Loader methods assume frames grouped for combination share identical shape;
        otherwise numpy.median on a list of differing-shaped arrays will raise.
    - load_flats normalizes each input flat by its median; frames with median==0
        or containing NaN/Inf values will produce invalid master flats. Consider
        pre-filtering or masking such frames.
    - remove_background requires an input 2D array and a background estimator from
        photutils or compatible implementation; box_size must be appropriate for the
        image size.
    
    Performance and numeric considerations
    --------------------------------------
    - Median combination is used throughout for robustness; for large stacks and
        big images memory usage may be significant because frames are collected in
        Python lists before combination.
    - apply_array converts the input to float64 to preserve precision; if many
        images are processed in a tight loop, consider memory implications.
    - No explicit multi-threading or parallel I/O is provided; consider parallel
        processing externally if needed.
    
    Example (conceptual)
    ---------------------
    # 1) Build masters
    c = CalibrationFrames()
    c.load_bias(bias_file_list)
    c.load_darks(dark_file_list)
    c.load_flats(flat_file_list)
    # 2) Calibrate science frames listed in c.metadata
    c.apply_self(calibrated_dir='/data/cal/', subject_name='targetA', metadata_dir='/data/meta/')
    # 3) Remove backgrounds from successfully calibrated frames
    c.remove_background_self(clean_dir='/data/clean/', subject_name='targetA', metadata_dir='/data/meta/')
    This class is organized to be straightforward and pragmatic for small to
    moderate-sized datasets. For production-scale reductions you may want to:
    - add robust masking and flat-field zero handling,
    - use memory-mapped arrays, chunked I/O, or on-disk combination to reduce RAM
        footprint,
    - validate header keywords and types more strictly,
    - add logging instead of printing, and
    - add finer-grained exception handling when reading/writing many files.
    """
    def __init__(self,
                 bias = None, 
                 darks = None, 
                 flats = None, 
                 metadata = pd.DataFrame()):
        self.bias = bias #Bias frame
        self.darks = darks if darks is not None else {}   #keyed by exposure time
        self.flats = flats if flats is not None else {}   #keyed by filter name
        self.metadata = metadata  #DataFrame to hold metadata

    #--- Load Frames ---
    def load_bias(self, df, selfupdate = True):
        """
        Load and combine bias frames into a master bias frame.
        Parameters
        ----------
        self
            The instance on which this method is called. If `selfupdate` is True, the result
            will be stored on `self.bias`.
        df : pandas.DataFrame or numpy.ndarray or list
            Input describing the bias files to load.
            - If a pandas.DataFrame is provided, it must contain a column named 'FILENAME'
              with paths to the FITS files.
            - If a numpy.ndarray or list is provided, it is interpreted as an iterable of
              file paths and will be converted to a DataFrame with a single column
              'FILENAME'.
        selfupdate : bool, optional (default=True)
            If True, assign the computed master bias frame to `self.bias` before returning it.
        Returns
        -------
        numpy.ndarray
            The median-combined bias frame (2D array) computed by stacking all loaded frames
            and taking the median along the frame axis (axis=0).
        Notes
        -----
        - FITS files are read using astropy.io.fits.getdata(path).
        - A progress indicator is updated via lumutils.progress_bar(current_index, total_count)
          for each file processed.
        - All input FITS files must have compatible shapes; otherwise the median computation
          will raise an error.
        - If no frames are provided (empty input), computing the median will raise an error.
        Raises
        ------
        OSError, FileNotFoundError, astropy.io.fits-related errors
            Propagated if any input FITS file cannot be opened or read.
        ValueError
            May be raised if provided inputs are empty or frames have incompatible shapes.
        Examples
        --------
        # Provide a list/array of filenames
        master_bias = self.load_bias(['bias1.fits', 'bias2.fits'])
        # Provide a DataFrame with a 'FILENAME' column and do not update self
        master_bias = self.load_bias(pd.DataFrame({'FILENAME': ['b1.fits', 'b2.fits']}), selfupdate=False)
        """
        if type(df) is np.ndarray or type(df) is list:
            biasdata = pd.DataFrame(df, columns=['FILENAME'])
            df = biasdata
        
        frames = []
        print("Loading Bias Frames:")
        for row in df.itertuples():
            bias_now = fits.getdata(row.FILENAME)
            frames.append(bias_now)
            lumutils.progress_bar(row.Index, len(df))
        result = np.median(frames, axis=0)

        if selfupdate:
            self.bias = result
        return result

    def load_darks(self, df, inplace=True):
        """
        Load and combine dark frames grouped by exposure time.

        Parameters
        ----------
        df : pandas.DataFrame or list or numpy.ndarray
            If a pandas.DataFrame, it must contain the columns:
              - 'FILENAME': path to the dark FITS file
              - 'EXPTIME': exposure time value used for grouping
            If a list or numpy.ndarray, it is interpreted as an iterable of dark FITS
            file paths. In this case the function will read each file header to build
            a temporary DataFrame with 'FILENAME' and 'EXPTIME' (from the FITS header).
        inplace : bool, optional
            If True (default), store the resulting mapping of exposure time to median
            dark frame in self.darks. If False, do not modify the instance.

        Returns
        -------
        dict
            A dictionary mapping each exposure time (EXPTIME) to the median dark
            frame (numpy.ndarray) computed across all dark frames with that exposure
            time. The median is computed with numpy.median along the stack axis.

        Side effects
        ------------
        - Reads FITS headers and data using astropy.io.fits (fits.getheader / fits.getdata).
        - Calls lumutils.progress_bar to report per-group progress.
        - If inplace is True, assigns the resulting dict to self.darks.

        Notes
        -----
        - All frames grouped under a given EXPTIME must have the same shape; otherwise
          numpy.median will raise an error when stacking.
        - When df is provided as filenames, each file must contain the 'EXPTIME' keyword
          in the FITS header; missing keys will raise an exception.
        - File I/O errors (e.g., missing files, unreadable FITS) will propagate to the
          caller.
        - The function uses numpy.median to combine frames; the output dtype follows
          numpy's dtype promotion rules.

        Examples
        --------
        # From a DataFrame:
        # df = pd.DataFrame({'FILENAME': ['d1.fits', 'd2.fits'], 'EXPTIME': [1.0, 1.0]})
        # darks = self.load_darks(df, inplace=False)

        # From a list of filenames:
        # darks = self.load_darks(['d1.fits', 'd2.fits'])
        """
        if type(df) is np.ndarray or type(df) is list:
            darkdata = pd.DataFrame([
                    {
                        'FILENAME': darkfile,
                        'EXPTIME': fits.getheader(darkfile).get('EXPTIME'),
                    }
                    for darkfile in df
            ]).sort_values(by = 'EXPTIME'); df = darkdata

        result = {}
        for exptime, group in df.groupby("EXPTIME"):
            frames = []
            print(f"Dark Exptime: {exptime}")
            for i, row in enumerate(group.itertuples()):
                dark_now = fits.getdata(row.FILENAME)
                frames.append(dark_now)
                lumutils.progress_bar(i, len(group))
            result[exptime] = np.median(frames, axis=0)

        if inplace:
            self.darks = result
        return result

    def load_flats(self, df, inplace=True):
        """
        Load and combine flat-field frames into master flats per filter.
        Parameters
        ----------
        df : pandas.DataFrame or list or numpy.ndarray
            If a pandas DataFrame, it is expected to contain at least the columns:
            - 'FILENAME': path to a FITS flat frame
            - 'FILTER': filter identifier string
            If df is a list or ndarray, it is interpreted as an iterable of filenames;
            a temporary DataFrame will be constructed by reading the 'FILTER' header
            keyword from each FITS file.
        inplace : bool, optional
            If True (default), store the resulting dictionary of master flats on
            self.flats. Always returns the result dictionary regardless of this flag.
        Returns
        -------
        dict
            A mapping from filter identifier (as read from the FITS header or the
            DataFrame's 'FILTER' column) to a 2D numpy.ndarray containing the
            median-combined, normalized master flat for that filter.
        Behavior
        --------
        - If df is a list/ndarray of filenames, a DataFrame is created with columns
          'FILENAME' and 'FILTER' (FILTER read from the FITS header) and sorted by
          FILTER.
        - The function groups input rows by FILTER. For each group it:
          - Reads each flat frame data with astropy.io.fits.getdata.
          - Subtracts self.bias if self.bias is not None.
          - Normalizes the frame by its median.
          - Collects normalized frames and displays progress via lumutils.progress_bar.
          - Computes the pixel-wise median across frames to produce the master flat.
        - The resulting master flats are collected into a dict keyed by filter.
        - If inplace is True, the dict is assigned to self.flats before being returned.
        Notes and edge cases
        --------------------
        - Requires astropy.io.fits (fits.getdata, fits.getheader), numpy and pandas.
        - If a FITS file lacks a FILTER header keyword, the constructed DataFrame
          will contain None for that entry which may affect grouping.
        - If a group contains no valid frames, numpy.median on an empty list will
          raise an error.
        - The function normalizes each input frame by its median; if a frame has a
          median of zero or contains NaNs/Infs, the normalization or median
          combination may produce NaNs or raise exceptions. It is recommended to
          ensure input frames are well-formed and to handle masking if necessary.
        Example
        -------
        # assuming `self` has an optional numeric attribute `bias`
        # and `flat_files` is a list of FITS filenames:
        masters = self.load_flats(flat_files, inplace=True)
        # access master for filter 'V':
        v_master = masters.get('V')
        """
        if type(df) is np.ndarray or type(df) is list:
            flatdata = pd.DataFrame([
                    {
                        'FILENAME': flatfile,
                        'FILTER': fits.getheader(flatfile).get('FILTER'),
                    }
                    for flatfile in df
            ]).sort_values(by = 'FILTER'); df = flatdata
        
        result = {}
        for flt, group in df.groupby("FILTER"):
            frames = []
            print(f"Flat Filter: {flt}")
            for i, row in enumerate(group.itertuples()):
                flat_now = fits.getdata(row.FILENAME) - (self.bias if self.bias is not None else 0)
                flat_now /= np.median(flat_now)
                frames.append(flat_now)
                lumutils.progress_bar(i, len(group))
            result[flt] = np.median(frames, axis=0)
        
        if inplace:
            self.flats = result
        return result

    #--- Add frames ---
    def add_bias(self, files):
        frames = [fits.getdata(f) for f in files]
        self.bias = np.median(frames, axis=0)
        
    def add_dark(self, exptime, files):
        frames = [fits.getdata(f) for f in files]
        self.darks[exptime] = np.median(frames, axis=0)

    def add_flat(self, filter_name, files):
        frames = []
        for f in files:
            flat_now = fits.getdata(f)
            if self.bias is not None:
                flat_now = flat_now - self.bias
            flat_now /= np.median(flat_now)   # normalize
            frames.append(flat_now)
        self.flats[filter_name] = np.median(frames, axis=0)

    # --- Apply calibration ---
    def apply_array(self, raw_image, exptime, filter_used):
        """
        Apply bias, dark, and flat-field calibrations to a raw detector image.

        This method performs a standard CCD/CMOS image calibration sequence:
        1. Subtract the instrument bias frame.
        2. Subtract the dark contribution for the given exposure time (dark - bias).
        3. Divide by the flat field for the specified filter.

        The input image is copied and converted to numpy.float64 before any arithmetic,
        so the original array is not modified.

        Parameters
        ----------
        raw_image : array_like
            Raw detector image (typically a 2D numeric array). Shapes of bias, dark,
            and flat frames are expected to match this array's shape.
        exptime : hashable
            Exposure time identifier used to look up the matching dark frame in
            self.darks (e.g. a float or string key). Must be present in self.darks.
        filter_used : hashable
            Filter identifier used to look up the matching flat frame in self.flats
            (e.g. a string). Must be present in self.flats.

        Returns
        -------
        numpy.ndarray
            Calibrated image as a numpy.ndarray with dtype numpy.float64. The returned
            array has the same shape as raw_image.

        Raises
        ------
        ValueError
            If self.bias is None.
            If self.darks is empty/falsey.
            If self.flats is empty/falsey.
        KeyError
            If exptime is not found in self.darks.
            If filter_used is not found in self.flats.

        Notes
        -----
        - The dark correction applied is (dark_frame - bias), consistent with dark
          frames that include the bias level.
        - No explicit checks are performed for division by zero in the flat field;
          if any flat-field pixels are zero, the result will contain infinities/NaNs
          as produced by numpy division.
        - All inputs (bias, darks[exptime], flats[filter_used]) are expected to be
          aligned and the same shape as raw_image. Mismatched shapes will cause numpy
          broadcasting or runtime errors.
        - The method uses floating-point arithmetic (float64) to preserve precision.

        Examples
        --------
        >>> # assuming self.bias, self.darks, self.flats are properly populated
        >>> calibrated = self.apply_array(raw_image, exptime=60.0, filter_used='R')
        """
        calibrated = raw_image.copy().astype(np.float64)  #force for float64

        #Bias
        if self.bias is None:
            raise ValueError("No bias available!")
        calibrated -= self.bias

        #Dark
        if not self.darks:
            raise ValueError("No darks available!")
        if exptime not in self.darks:
            raise KeyError(f"No dark for exptime={exptime}")
        calibrated -= (self.darks[exptime] - self.bias)

        #Flat
        if not self.flats:
            raise ValueError("No flats available!")
        if filter_used not in self.flats:
            raise KeyError(f"No flat for filter={filter_used}")
        calibrated /= self.flats[filter_used]

        return calibrated

    def apply_self(self, calibrated_dir: str = './calibrated_FITS/',
                   subject_name: str = '',
                   metadata_dir: str = './', warn=True):
        """
        Apply calibration to all FITS files listed in self.metadata and save results + updated metadata.
        
        Parameters
        ----------
        calibrated_dir : str, optional
            Directory where calibrated FITS files will be written. Default: './calibrated_FITS/'.
            The directory is created if it does not exist.
        subject_name : str, optional
            Base name used when writing the updated metadata CSV file. Default: ''.
        metadata_dir : str, optional
            Directory where the metadata CSV will be saved. Default: './'.
        warn : bool, optional
            If True (default) print progress and status messages for each file. If False,
            suppress NumPy warnings and reduce printed output.
        
        Behavior / Side effects
        -----------------------
        - Expects self.metadata to be an iterable of rows (e.g., a pandas DataFrame) containing at least
          the columns: 'FILENAME', 'EXPTIME', and 'FILTER'.
        - For each entry in self.metadata:
            1. Reads the raw image data and header from the FITS file at row.FILENAME using astropy.io.fits.
            2. Calls self.apply_array(raw_data, exptime, filter_used) to produce the calibrated array.
            3. On success, writes a PrimaryHDU containing the calibrated data and the original header to
               a file in calibrated_dir. The filename is constructed by replacing the '.fit' suffix of the
               original file with '_calibrated.fits' (keeping the original basename).
            4. Updates self.metadata at the current row with:
               - CAL_FILENAME: path to the calibrated file (or NaN on failure)
               - CAL_STATUS: 'SUCCESS' on success, or "FAIL - {error}" if calibration failed for that row.
            5. Calls lumutils.progress_bar to report overall progress.
        - After processing all rows, saves self.metadata to a CSV at:
            f'{metadata_dir}{subject_name}_metadata.csv'
        
        Error handling
        --------------
        - If self.apply_array raises ValueError or KeyError for a particular file, that file is skipped,
          the failure is recorded in self.metadata, and processing continues with the next file.
        - Other exceptions (for example I/O errors, unexpected exceptions from astropy, or pandas errors)
          are not caught here and will propagate to the caller.
        
        Returns
        -------
        None
        
        Example
        -------
        self.apply_self(calibrated_dir='/data/calibrated/', subject_name='targetA', metadata_dir='/data/meta/', warn=True)
        """
        #Suppress warnings for cleaner output
        if not warn:
            np.seterr(all='ignore')
        
        #Output directory creation,
        if warn:
            print(f"Successfully calibrated files will be saved to '{calibrated_dir}'")
        os.makedirs(calibrated_dir, exist_ok=True)

        print('Applying Calibration...')
        
        for i, row in enumerate(self.metadata.itertuples()):
            raw_data = fits.getdata(row.FILENAME)
            header = fits.getheader(row.FILENAME)
            exptime = row.EXPTIME
            filter_used = row.FILTER

            try:
                calibrated_data = self.apply_array(raw_data, exptime, filter_used)
            except (ValueError, KeyError) as e:
                if warn:
                    print(f"Skipping calibration for {row.FILENAME}: {e}")
                self.metadata.loc[row.Index, ['CAL_FILENAME', 'CAL_STATUS']] = [np.nan, f"FAIL - {e}"]
                continue
            
            calibrated_filename = os.path.join(calibrated_dir,
                                           os.path.basename(row.FILENAME).replace('.fit', '_calibrated.fits'))
            hdu = fits.PrimaryHDU(data = calibrated_data,
                                  header = header)
            hdu.writeto(calibrated_filename, overwrite=True)
            self.metadata.loc[row.Index, ['CAL_FILENAME', 'CAL_STATUS']] = [calibrated_filename, "SUCCESS"]
            lumutils.progress_bar(i, len(self.metadata))

            if warn:
                print(f"Calibrated {row.FILENAME} -> {calibrated_filename}")
    
        self.metadata.to_csv(f'{metadata_dir}{subject_name}_metadata.csv', index=False)
        if warn:
            print(f"Current metadata saved to {metadata_dir}{subject_name}_metadata.csv")
        
        return None
    
    # --- Background Removal ---
    def remove_background(self, data: np.ndarray,
                          sigma: float = 3.0, box_size: int = 50,
                          filter_size = (3,3)):
        """
        Remove the estimated background from a 2D array using a median background estimator.

        Parameters
        ----------
        data : numpy.ndarray
            2D array (image or similar) from which to estimate and subtract the background.
        sigma : float, optional
            Sigma value used by astropy.stats.SigmaClip for clipping outliers when estimating
            the background. Default is 3.0.
        box_size : int, optional
            Size (in pixels) of the box used by Background2D to compute the local background.
            Default is 50.
        filter_size : tuple of int or None, optional
            Size of the smoothing filter applied to the background map (e.g. (3, 3)).
            Set to None to skip filtering. Default is (3, 3).

        Returns
        -------
        numpy.ndarray
            Background-subtracted array with the same shape as `data` (i.e., data - background).

        Raises
        ------
        TypeError
            If `data` is not a numpy.ndarray.
        ValueError
            If `data` is not 2D or if `box_size`/`filter_size` values are incompatible with the input shape.

        Notes
        -----
        - This function uses a median-based background estimator together with sigma-clipping
          to produce a robust background map and subtracts it from the input data.
        - Tuning `sigma`, `box_size`, and `filter_size` can improve results for different
          spatial scales and noise characteristics.

        Example
        -------
        >>> # remove background from a 2D numpy array `image`
        >>> clean = remove_background(image, sigma=3.0, box_size=64, filter_size=(3,3))
        """
        bkg_estimator = MedianBackground()
        sigma_clip = SigmaClip(sigma=sigma)
        bkg = Background2D(data, box_size, filter_size=filter_size,
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        cln_data = data - bkg.background
        return cln_data
    
    def remove_background_self(self, clean_dir:str = './clean_FITS/',
                              subject_name: str = '', metadata_dir: str = './',
                              warn = True, *, sigma: float = 3.0,
                              box_size: int = 50, filter_size = (3,3)):
        #Suppress warnings for cleaner output
        if not warn:
            np.seterr(all='ignore')
        
        #Output directory creation,
        if warn:
            print(f"Successfully calibrated files will be saved to '{clean_dir}'")
        os.makedirs(clean_dir, exist_ok=True)

        print('Removing Background...')
        for i, row in enumerate(self.metadata.query('CAL_STATUS == "SUCCESS"').itertuples()):
            cal_data = fits.getdata(row.CAL_FILENAME)
            clean_data = self.remove_background(cal_data, sigma=sigma,
                                             box_size=box_size,
                                             filter_size=filter_size)
            header = fits.getheader(row.CAL_FILENAME)

            clean_filename = os.path.join(clean_dir,
                                           os.path.basename(row.CAL_FILENAME).replace('.fits', '_clean.fits'))
            hdu = fits.PrimaryHDU(data = clean_data,
                                  header = header)
            hdu.writeto(clean_filename, overwrite=True)
            self.metadata.loc[row.Index, ['CLN_FILENAME']] = [clean_filename]
            lumutils.progress_bar(i, len(self.metadata.query('CAL_STATUS == "SUCCESS"')))

            if warn:
                print(f"Calibrated {row.FILENAME} -> {clean_filename}")
    
        self.metadata.to_csv(f'{metadata_dir}{subject_name}_metadata.csv', index=False)
        if warn:
            print(f"Current metadata saved to {metadata_dir}{subject_name}_metadata.csv")
        
        return None


    # --- Plotting ---
    def plot_calibration(self, plot_dir: str = './cal_plots/', origin: str = 'lower'):
        os.makedirs(plot_dir, exist_ok=True)
        print(f"Calibration plots will be saved to '{plot_dir}'")
        for i, row in enumerate(self.metadata.query('CAL_STATUS == "SUCCESS"').itertuples()):
            raw_data = fits.getdata(row.FILENAME)
            cal_data = fits.getdata(row.CAL_FILENAME)
            fig = lumvis.plot_comparison(raw_data, cal_data,
                                         f"Raw vs Calibrated: {os.path.basename(row.FILENAME)}",
                                         "Raw", "Calibrated", origin=origin)
            plot_filename = os.path.join(plot_dir,
                                         os.path.basename(row.FILENAME).replace('.fit', '_comparison.png'))
            fig.savefig(plot_filename)
            plt.close(fig)
            lumutils.progress_bar(i, len(self.metadata.query('CAL_STATUS == "SUCCESS"')))
        return None
    
    def plot_background(self, plot_dir: str = './cal_plots/', origin: str = 'lower'):
        os.makedirs(plot_dir, exist_ok=True)
        print(f"Background plots will be saved to '{plot_dir}'")
        for i, row in enumerate(self.metadata.query('CAL_STATUS == "SUCCESS"').itertuples()):
            cal_data = fits.getdata(row.CAL_FILENAME)
            cln_data = fits.getdata(row.CLN_FILENAME)
            fig = lumvis.plot_comparison(cal_data, cln_data,
                                         f"Calibrated vs Clean: {os.path.basename(row.FILENAME)}",
                                         "Calibrated", "Clean", origin=origin)
            plot_filename = os.path.join(plot_dir,
                                         os.path.basename(row.FILENAME).replace('.fit', '_clean.png'))
            fig.savefig(plot_filename)
            plt.close(fig)
            lumutils.progress_bar(i, len(self.metadata.query('CAL_STATUS == "SUCCESS"')))
        return None

    # --- Friendly representation ---
    def __repr__(self):
        lines = [
            f"<CalibrationFrames at {hex(id(self))}:",
            f"  Bias frame: {'present' if self.bias is not None else 'missing'}",
            f"  Darks: {sorted(self.darks.keys()) if self.darks else 'none'}",
            f"  Flats: {sorted(self.flats.keys()) if self.flats else 'none'}",
            f"  Metadata: {len(self.metadata)} entries" if not self.metadata.empty else "  Metadata: missing",
            ">"
        ]
        return "\n".join(lines)

