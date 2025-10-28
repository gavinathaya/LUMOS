import numpy as np
from lumos.visualization import plot_ccd
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from astropy.table import QTable
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

def calc_fwhm(image) -> float:
    return None

def apply_phot_aperture(image, positions, fwhm) -> QTable:
    
    aperture = CircularAperture(positions, r = fwhm)
    return result