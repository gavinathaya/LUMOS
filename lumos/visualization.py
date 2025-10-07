"""
Visualization tools (:mod:`lumos.visualization`)
==========================================================
Tools for visualization & plotting functions in LUMOS.
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import (ImageNormalize, PercentileInterval,
                                   PowerDistStretch)

def plot_ccd(image, fig, ax, title="", clip=99.5, level = 250, origin = "lower"):
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
    """
    norm = ImageNormalize(image,
                          interval=PercentileInterval(clip),
                          stretch=PowerDistStretch(level))
    
    im = ax.imshow(image, cmap="gray", origin=origin, norm=norm)
    ax.set_title(title)
    ax.set_xlabel("X Pixel")
    ax.set_ylabel("Y Pixel")
    fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)

def plot_comparison(left_image, right_image, main_title = "Comparison Plot",
                    left_title="Left", right_title="Right",
                    clip=99.5, level=250, origin="lower"):
    """
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plot_ccd(left_image, fig, ax[0], title=left_title, clip=clip, level=level, origin=origin)
    plot_ccd(right_image, fig, ax[1], title=right_title, clip=clip, level=level, origin=origin)
    fig.suptitle(main_title, y = 0.87)

    return fig