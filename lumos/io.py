"""
I/O tools (:mod:`lumos.io`)
==========================================================
Tools for input output functions in LUMOS.
"""
import numpy as np
import glob, fnmatch

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
