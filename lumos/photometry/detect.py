import numpy as np
import pandas as pd
from photutils.detection import DAOStarFinder
from astropy.io import fits
from astropy.table import QTable
from astropy.visualization.wcsaxes import WCSAxes
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from lumos.visualization import plot_ccd
from lumos.utils.helpers import progress_bar

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
    for index, filename in enumerate(path_array):
        image = fits.getdata(filename)
        sources.append(data_star_identification(image, fwhm=fwhm, threshold=threshold))
        progress_bar(index, len(path_array))
    if len(sources) == 1:
        return sources[0]
    return sources

def plot_source_single(image, fig, ax, source, title="", clip=99.5, level = 250, origin = "lower", wcs = None):
    """
    Plot detected sources on a CCD image, highlighting relatively bright and dim detections.

    This function displays an image using the helper `plot_ccd` and overplots source centroids.
    Sources are split into two groups based on the median of the `source['flux']` values:
    "bright" sources (flux > median) are shown as open red circles, and "dim" sources
    (flux <= median) are shown as small filled red points.

    Parameters
    ----------
    image : array-like or astropy.nddata.NDData
        2D image array (e.g., numpy.ndarray or CCDData) to be displayed.
    fig : matplotlib.figure.Figure
        Matplotlib Figure instance to contain the axes.
    ax : matplotlib.axes.Axes
        Matplotlib Axes on which the image and source markers will be drawn.
    source : table-like
        Table or structured array of detected sources. Must provide the columns:
        - 'xcentroid' : x pixel coordinate of centroid
        - 'ycentroid' : y pixel coordinate of centroid
        - 'flux' : measured flux; may be an astropy Quantity or object with a `.value` attribute.
        The function uses `np.nanmedian(source['flux'])` to separate bright vs dim sources.
    title : str, optional
        Title to pass to the underlying image plotting routine (default: "").
    clip : float, optional
        Percentile used by `plot_ccd` for clipping/stretching the image display (default: 99.5).
    level : float or int, optional
        Contrast or display level forwarded to `plot_ccd` (default: 250).
    origin : {'lower', 'upper'}, optional
        Origin argument forwarded to `plot_ccd` / imshow to control axis origin (default: "lower").
    wcs : astropy.wcs.WCS or None, optional
        World Coordinate System to pass to `plot_ccd` for annotated axes; if None,
        pixel coordinates are used (default: None).

    Notes
    -----
    - The function calls `plot_ccd(image, fig, ax, title, clip, level, origin, wcs)` to draw
      the image; any behavior of image scaling/annotation is delegated to that function.
    - Marker styles:
      - Bright sources: open red circles ('o', red edge), marker size 5, facecolor none,
        line width 0.25, alpha 0.5.
      - Dim sources: small filled red points ('.'), marker size 1, alpha 0.5.
    - The flux comparison uses the numeric values (i.e., `source['flux'].value` when present).
      Ensure `source['flux']` is not all NaN.

    Returns
    -------
    None
        This function draws on the provided Axes and does not return a value.

    Examples
    --------
    Assuming `ax` and `fig` are matplotlib objects and `catalog` is an astropy Table:
    plot_source_single(image, fig, ax, catalog, title="Field", clip=99.0, level=200)
    """
    plot_ccd(image, fig, ax, title, clip, level, origin, wcs)

    #Masking to differentiate big vs small circles
    mid = np.nanmedian(source['flux'])
    bright = source['flux'].value > mid
    dim = source['flux'].value <= mid

    ax.plot((source[bright])['xcentroid'], (source[bright])['ycentroid'],
            'or', ms = 5, mfc = 'none', lw = 0.25, alpha = 0.5)
    ax.plot((source[dim])['xcentroid'], (source[dim])['ycentroid'],
            '.r', ms = 1, alpha = 0.5)
    return None

def plot_source(image, main_title = "Detected Sources",
                    left_title="In pixel coordinates", right_title="In world coordinates",
                    wcs = None, source = None,
                    clip=99.5, level=250, origin="lower"):
    """
    Plot detected sources on an image in both pixel and world coordinates.
    This convenience function creates a two-panel matplotlib figure showing the
    same image twice: the left panel displays the image in pixel coordinates and
    the right panel displays it using a provided WCS (world coordinate system).
    Detected sources from a catalog are overplotted on both panels. The actual
    rendering and annotation of sources is delegated to plot_source_single.
    
    Parameters
    ----------
    image : array-like
        2D image array (e.g. NumPy array) to be displayed.
    main_title : str, optional
        Overall figure title. Default is "Detected Sources".
    left_title : str, optional
        Title for the left (pixel) panel. Default is "In pixel coordinates".
    right_title : str, optional
        Title for the right (world) panel. Default is "In world coordinates".
    wcs : astropy.wcs.WCS or None, optional
        World Coordinate System object used to create the projection for the
        right-hand axis. If None, the right panel will still be created but
        without a valid WCS projection (behavior depends on matplotlib/astropy).
    source : sequence or astropy.table.Table
        Source catalog containing detected object positions. Must be provided;
        entries should include pixel coordinates (e.g. x, y) and, if plotting
        in world coordinates, columns or a representation compatible with the
        supplied `wcs`. If not provided, a ValueError is raised.
    clip : float, optional
        Percentile for image clipping used when rendering the image contrast.
        Higher values clip more extreme values. Default is 99.5.
    level : float or int, optional
        Contour or display level threshold passed to plot_source_single for
        highlighting sources or features. Default is 250.
    origin : {'lower', 'upper'} or str, optional
        Array origin passed to the image display routines. Default is "lower".

    Returns
    -------
    matplotlib.figure.Figure
        The created Figure object containing two Axes where the image and
        sources have been plotted. The caller can further modify or save this
        figure.

    Raises
    ------
    ValueError
        If `source` is None.

    Notes
    -----
    - This function relies on plot_source_single to do the per-panel plotting;
      it only constructs the figure and axes and calls that helper for each panel.
    - The right-hand panel uses the provided `wcs` to set the axes projection;
      ensure `wcs` is compatible with matplotlib/astropy WCSAxes if world
      coordinates are required.

    Examples
    --------
    fig = plot_source(image_data, wcs=my_wcs, source=detected_table)
    fig.savefig("sources.png")
    """
    if source is None:
        raise ValueError("Source catalog must be provided for plotting detected sources.")
    if wcs is None:
        raise ValueError("WCS must be provided for plotting in world coordinates.") 
    fig= plt.figure(figsize=(12, 6))
    ax: list = []
    ax.append(fig.add_subplot(1,2,1))
    ax.append(fig.add_subplot(1,2,2, projection = wcs))

    plot_source_single(image, fig, ax[0],
                       source=source,
                       title=left_title, clip=clip,
                       level=level, origin=origin)
    plot_source_single(image, fig, ax[1],
                       source=source, wcs = wcs,
                       title=right_title, clip=clip,
                       level=level, origin=origin)
    fig.suptitle(main_title, y = 0.87)

    return fig