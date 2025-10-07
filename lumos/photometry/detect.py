import numpy as np
from photutils.detection import DAOStarFinder
from astropy.io import fits
from astropy.table import QTable
import matplotlib.pyplot as plt
from lumos.visualization import plot_ccd

def data_star_identification(image, fwhm=15.0, threshold=5.0) -> QTable:
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
    sources = daofind(image) #Equivalent to daofind.find_stars(image)
    sources = sources[np.isfinite(sources['mag'])]

    #Assign units, maybe later? Depends if it's needed
    # sources['xcentroid'].unit = u.pixel
    # sources['ycentroid'].unit = u.pixel
    # sources['flux'].unit = u.adu
    return sources

def star_identification(path, fwhm=15.0, threshold=5.0):
    """
    Load an image from the given path and detect stars in it.

    Parameters
    ----------
    path : str or list of str
        The file path(s) to the image(s).
    fwhm : float
        The full width at half maximum for the Gaussian kernel.
    threshold : float
        The absolute image value above which to select sources.

    Returns
    -------
    sources : astropy.QTable or list of astropy.QTable
        An astropy QTable or list of QTables containing the detected sources.
    """
    sources = []
    path_array = np.atleast_1d(path)
    for filename in path_array:
        image = fits.getdata(filename)
        sources.append(data_star_identification(image, fwhm=fwhm, threshold=threshold))
    if len(sources) == 1:
        return sources[0]
    return sources

def plot_source_single(image, source, projection=None):
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
    fig, ax = plt.subplots(subplot_kw={'projection': projection})
    plot_ccd(image, fig, ax, title = "Detected stars")
    # ax.plot((source['xcentroid']) , (source['ycentroid']),
    #         'or', ms = 5, mfc = 'none', lw = 0.25)
    #Masking to differentiate big vs small circles
    mid = np.nanmedian(source['flux'])
    bright = source['flux'].value > mid
    dim = source['flux'].value <= mid

    ax.plot((source[bright])['xcentroid'], (source[bright])['ycentroid'],
            'or', ms = 5, mfc = 'none', lw = 0.25, alpha = 0.5)
    ax.plot((source[dim])['xcentroid'], (source[dim])['ycentroid'],
            '.r', ms = 1, alpha = 0.5)

    # fig = lumvis.plot_comparison(image, image, main_title="Detected Stars",
    #                              left_title="Original Image", right_title="Detected Stars")
    # ax = fig.axes[1]
    # positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    # apertures = CircularAperture(positions, r=10.)
    # apertures.plot(color='red', lw=1.5, alpha=0.5, ax=ax)
    # fig.savefig(output_path)
    # plt.close(fig)
    return fig