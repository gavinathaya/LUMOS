"""
I/O tools (:mod:`lumos.io`)
==========================================================
Tools for input output functions in LUMOS.
"""
import glob, fnmatch
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

def findfiles(dir: str = "./",
             raw_name: str = "*.fits",
             dark_name: str = "*dark.fits",
             bias_name: str = "*bias.fits",
             flat_name: str = "*flat.fits", *,
             recursive: bool = True):
    """
    Finds raw, dark, bias, and flat FITS files in a given directory.

    Uses glob and fnmatch to search for files matching the provided patterns.
    Raw files are filtered to exclude any files also matching dark, bias, or flat patterns.

    Parameters
    ----------
    dir : str, optional
        Directory to search for files. Default is "./".
    raw_name : str, optional
        Pattern for raw files. Default is "*.fits".
    dark_name : str, optional
        Pattern for dark files. Default is "*dark.fits".
    bias_name : str, optional
        Pattern for bias files. Default is "*bias.fits".
    flat_name : str, optional
        Pattern for flat files. Default is "*flat.fits".
    recursive : bool, optional
        Whether to search recursively in subdirectories. Default is True.

    Returns
    -------
    raw_files : np.ndarray
        Array of raw file paths, excluding any that match dark, bias, or flat patterns.
    dark_files : np.ndarray
        Array of dark file paths.
    bias_files : np.ndarray
        Array of bias file paths.
    flat_files : np.ndarray
        Array of flat file paths.
    """
    files = glob.glob(f"{dir}", recursive = recursive)
    raw_files = fnmatch.filter(files, raw_name)
    dark_files = fnmatch.filter(files, dark_name)
    bias_files = fnmatch.filter(files, bias_name)
    flat_files = fnmatch.filter(files, flat_name)
    raw_files = [f for f in raw_files 
                if f not in dark_files 
                and f not in bias_files 
                and f not in flat_files]
    return (np.asarray(raw_files), np.asarray(dark_files),
            np.asarray(bias_files), np.asarray(flat_files))

def metadata_gen(raw_files: np.ndarray | list, HDUnum: int = 0) -> pd.DataFrame:
    """
    Generates a metadata DataFrame for a list of raw FITS files.
    Required for lumos CalibrationFrame class.
    
    Extracts relevant header information from each FITS file and organizes it into a pandas DataFrame.
    Adds columns for calibration and cleaning status, initialized as empty or default values.

    Parameters
    ----------
    raw_files : array-like, with str dtype.
        Array of raw FITS file paths.
    HDUnum : int, optional
        n-th HDU to read from each FITS file. Default is 0 (primary HDU).
        1 is first extension HDU, etc.

    Returns
    -------
    metadata : pd.DataFrame
        DataFrame containing metadata for each file, including:
        - FILENAME: Name of the FITS file.
        - DATE_OBS: Observation date from FITS header.
        - EXPTIME: Exposure time from FITS header.
        - FILTER: Filter used from FITS header.
        - TELESCOPE: Telescope name from FITS header.
        - CAL_FILENAME: List for calibration file names (initially empty).
        - CAL_STATUS: Calibration status (initially 'UNCALIBRATED').
        - CLN_FILENAME: List for cleaned file names (initially empty).
    """

    metadata = pd.DataFrame([
        {
            'FILENAME': filename,
            'DATE_OBS': (header := fits.getheader(filename)).get('DATE-OBS'),
            'EXPTIME': header.get('EXPTIME'),
            'FILTER': header.get('FILTER'),
            'TELESCOPE': header.get('TELESCOP'),
            
            #---------WCS Stuffs-------------
            'CRVAL1': header.get('CRVAL1'),  #RA reference value (deg)
            'CRVAL2': header.get('CRVAL2'),  #DEC reference value (deg)
            'CRPIX1': header.get('CRPIX1'),  #Reference pixel X
            'CRPIX2': header.get('CRPIX2'),  #Reference pixel Y
            'CDELT1': header.get('CDELT1'),  #Pixel scale (deg/pix)
            'CDELT2': header.get('CDELT2'),  #Pixel scale (deg/pix)
            'CTYPE1': header.get('CTYPE1'),  #Projection type (e.g., RA---TAN)
            'CTYPE2': header.get('CTYPE2'),  #Projection type (e.g., DEC--TAN)

            #---------Calibration Stuffs-------------
            'CAL_FILENAME': [],             #Initial empty column for calibration filename
            'CAL_STATUS': 'UNCALIBRATED',   #Initial status
            'CLN_FILENAME': [],             #Initial empty column for cleaned filename
        }
        for filename in raw_files
    ]).sort_values(by = 'DATE_OBS').reset_index(drop=True)
    return metadata

def rebuild_wcs(row: pd.Series) -> WCS:
    """
    Rebuilds an astropy.wcs.WCS object from a metadata DataFrame row.

    Parameters
    ----------
    row : pandas.Series
        A single row from the metadata DataFrame. Must contain
        CRVAL1, CRVAL2, CRPIX1, CRPIX2, CDELT1, CDELT2, CTYPE1, and CTYPE2.

    Returns
    -------
    wcs : astropy.wcs.WCS
        Reconstructed WCS object.
    """
    hdr = fits.Header()

    # Basic WCS keywords (RA/DEC TAN projection default)
    hdr['CTYPE1'] = row.get('CTYPE1', 'RA---TAN')
    hdr['CTYPE2'] = row.get('CTYPE2', 'DEC--TAN')
    hdr['CRVAL1'] = float(row.get('CRVAL1', 0.0))
    hdr['CRVAL2'] = float(row.get('CRVAL2', 0.0))
    hdr['CRPIX1'] = float(row.get('CRPIX1', 0.0))
    hdr['CRPIX2'] = float(row.get('CRPIX2', 0.0))
    hdr['CDELT1'] = float(row.get('CDELT1', -1.0/3600))  # default: -1 arcsec/pix
    hdr['CDELT2'] = float(row.get('CDELT2',  1.0/3600))

    # Optional rotation matrix (CD or PC terms) if stored
    for key in ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']:
        if key in row and pd.notna(row[key]):
            hdr[key] = float(row[key])

    return WCS(hdr)

def find_WCS_files(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Returns rows from metadata where WCS headers are present and not default/placeholder values.

    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata DataFrame with WCS columns.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only files that have proper WCS headers.
    """
    #Define default values that indicates missing WCS
    def is_valid_wcs(row):
        #Check for nonzero, non-default, non-NaN values
        try:
            return (
                pd.notna(row['CRVAL1']) and pd.notna(row['CRVAL2']) and
                row['CRVAL1'] != 0.0 and row['CRVAL2'] != 0.0 and
                pd.notna(row['CRPIX1']) and pd.notna(row['CRPIX2']) and
                row['CRPIX1'] != 0.0 and row['CRPIX2'] != 0.0 and
                pd.notna(row['CDELT1']) and pd.notna(row['CDELT2']) and
                abs(row['CDELT1']) != 1.0/3600 and abs(row['CDELT2']) != 1.0/3600
            )
        except KeyError:
            return False
    wcs_files = metadata[metadata.apply(is_valid_wcs, axis=1)].reset_index(drop=True)
    return wcs_files