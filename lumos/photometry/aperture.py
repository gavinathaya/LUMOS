import os
import numpy as np
from lumos.visualization import plot_ccd
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
import astropy.units as u
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.detection import DAOStarFinder


def data_star_identification(image, fwhm=15.0, threshold=5.0):
    """
    Detect stars in the image using DAOStarFinder.

    Parameters
    ----------
    image : 2D array
        The input image image. Must be background-subtracted.
    fwhm : float
        The full width at half maximum for the Gaussian kernel.
    threshold : float
        The absolute image value above which to select sources.

    Returns
    -------
    sources : astropy.QTable
        An astropy QTable containing the detected sources.
    """
    std = np.nanstd(image)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*std)
    sources = daofind.find_stars(image)
    return sources

def star_identification(path, fwhm=15.0, threshold=5.0):
    """
    Load an image from the given path and detect stars in it.

    Parameters
    ----------
    path : str or list of paths str
        The file path to the image.
    fwhm : float
        The full width at half maximum for the Gaussian kernel.
    threshold : float
        The absolute image value above which to select sources.
    plot : bool
        Whether to plot and save the detected sources on the image.

    Returns
    -------
    sources : astropy.QTable
        An astropy QTable containing the detected sources.
    """
    image = None
    if isinstance(path, str):
        image = fits.getdata(path)
    elif isinstance(path, list):
        if not path:
            raise ValueError("path list is empty")
        for filename in path:
            image = fits.getdata(filename)
    else:
        raise ValueError("path must be a string or a list of strings")
    sources = data_star_identification(image, fwhm=fwhm, threshold=threshold)
    return sources

def plot_source_single(image, source):
    """
    Plot the detected stars on the image.

    Parameters
    ----------
    image : 2D array
        The input image array.
    sources : astropy.QTable
        The table of detected sources.
    plot_dir : str
        The file path to save the output plot.
    
    Returns
    -------
    fig : matplotlib.Figure.figure
        Figure of detected stars plot.
    """
    fig, ax = plt.subplots()
    plot_ccd(image, fig, ax, title = "Detected stars")

    #Masking to differentiate big vs small circles
    bright = source['flux'].value > 800
    dim = source['flux'].value <= 800

    ax.plot((source[bright])['xcentroid'], (source[bright])['xcentroid'],
            'or', ms = 5, mfc = 'none', lw = 0.25)
    ax.plot((source[dim])['xcentroid'], (source[dim])['xcentroid'],
            '.r', ms = 1)

    # fig = lumvis.plot_comparison(image, image, main_title="Detected Stars",
    #                              left_title="Original Image", right_title="Detected Stars")
    # ax = fig.axes[1]
    # positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    # apertures = CircularAperture(positions, r=10.)
    # apertures.plot(color='red', lw=1.5, alpha=0.5, ax=ax)
    # fig.savefig(output_path)
    # plt.close(fig)
    return fig