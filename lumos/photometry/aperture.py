import numpy as np
from lumos.visualization import plot_ccd
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from astropy.table import QTable
from photutils.psf import fit_fwhm
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

def apply_phot_aperture(image, positions, n_fwhm) -> QTable:
    fwhm_vals = fit_fwhm(image, xypos=positions)
    aperture = CircularAperture(positions, r=n_fwhm * fwhm_vals)
    result = aperture_photometry(image, aperture)
    return result