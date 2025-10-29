"""
Visualization tools (:mod:`lumos.visualization`)
==========================================================
Tools for visualization & plotting functions in LUMOS.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from astropy.wcs import WCS
from astropy.visualization import (ImageNormalize, PercentileInterval,
                                   PowerDistStretch)

def plot_ccd(image, fig, ax, title="", clip=99.5, level = 250, origin = "lower", wcs = None):
    """
    Plot a CCD image with percentile-based clipping + PowerDist stretch.
    
    Parameters
    ----------
    image : ndarray
        The CCD image array
    fig : matplotlib.figure.Figure
        Figure to plot on
    ax : matplotlib.axes.Axes
        Axis to plot on
    title : str (optional)
        Title for the subplot
    clip : float (optional)
        Percentile for clipping (e.g., 99.5 keeps middle 99.5% of values)
    level : float (optional)
        PowerDist stretch level; follows y = (level^x -1 )/(level - 1).
        level -> 1: linear, level -> inf: exponential, level -> (0,1): logarithmic.
    origin : str (optional)
        Origin for imshow ("lower" or "upper").
    wcs : None or astropy.wcs.WCS (optional)
        World Coordinate System for axis (if provided, will use WCSAxes).
        Will override origin to "lower" if WCS is provided.
    
    Returns
    -------
    None
    """
    norm = ImageNormalize(image,
                          interval=PercentileInterval(clip),
                          stretch=PowerDistStretch(level))
    im = ax.imshow(image, cmap="gray", origin=origin, norm=norm)
    if wcs is not None:
        origin = "lower"
        xname=wcs.axis_type_names[0]
        yname=wcs.axis_type_names[1]
        ax.coords.grid(color='orange', alpha = 0.25, ls='dotted')
    else:
        xname="X Pixel"
        yname="Y Pixel"
    
    ax.set(title = title, xlabel = xname, ylabel=yname)
    fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)

def plot_comparison(left_image, right_image, main_title = "Comparison Plot",
                    left_title="Left", right_title="Right",
                    left_wcs = None, right_wcs = None,
                    clip=99.5, level=250, origin="lower"):
    """
    Plot a side-by-side comparison of two CCD images with optional WCS, clipping, and stretching.

    Parameters
    ----------
    left_image : ndarray
        The left CCD image array.
    right_image : ndarray
        The right CCD image array.
    main_title : str, optional
        Title for the entire figure.
    left_title : str, optional
        Title for the left subplot.
    right_title : str, optional
        Title for the right subplot.
    left_wcs : astropy.wcs.WCS or None, optional
        WCS for the left image (if any). Will override origin to "lower" if provided.
    right_wcs : astropy.wcs.WCS or None, optional
        WCS for the right image (if any). Will override origin to "lower" if provided.
    clip : float, optional
        Percentile for clipping (e.g., 99.5 keeps middle 99.5% of values).
    level : float, optional
        PowerDist stretch level; follows y = (level^x -1 )/(level - 1).
    origin : str, optional
        Origin for imshow ("lower" or "upper").

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the comparison plot.
    """
    fig= plt.figure(figsize=(12, 6))
    ax: list[Axes] = []
    for i, wcs_stat in enumerate([left_wcs, right_wcs]):
        ax[i] = fig.add_subplot(1,2,i+1, projection = wcs_stat)

    plot_ccd(left_image, fig, ax[0], wcs = left_wcs, title=left_title, clip=clip, level=level, origin=origin)
    plot_ccd(right_image, fig, ax[1], wcs = right_wcs, title=right_title, clip=clip, level=level, origin=origin)
    fig.suptitle(main_title, y = 0.87)

    return fig