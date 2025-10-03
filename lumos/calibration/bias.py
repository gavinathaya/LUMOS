import numpy as np
import pandas as pd
from astropy.io import fits
from lumos.utils.helpers import progress_bar

def gen_master_bias(data):
    """
    Generates a master bias frame from a collection of bias FITS files.

    Loads each bias frame, displays a progress bar during loading, and computes the median across all frames to produce the master bias.

    Parameters
    ----------
    data : array-like with dtype str or pandas.DataFrame
        List, numpy array, or DataFrame containing bias FITS file paths.
        If array-like, should contain file names as strings.

    Returns
    -------
    master_bias : np.ndarray
        2D array representing the master bias frame, computed as the median of all input bias frames.
    """
    if type(data) is np.ndarray or type(data) is list:
        biasdata = pd.DataFrame(data, columns=['FILENAME'])
        data = biasdata
    
    frames = []
    print("Loading Bias Frames:")
    for row in data.itertuples():
        bias_now = fits.getdata(row.FILENAME)
        frames.append(bias_now)
        progress_bar(row.Index, len(data))
    master_bias = np.median(frames, axis=0)
    return master_bias