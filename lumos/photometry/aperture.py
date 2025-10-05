import numpy as np
from lumos.visualization import plot_ccd
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry