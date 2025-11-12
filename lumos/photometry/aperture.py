import numpy as np
from lumos.visualization import plot_ccd
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from astropy.table import QTable
from photutils.psf import fit_fwhm
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

def apply_phot_aperture(image, positions, n_fwhm= 3, fit_shape = 15) -> QTable:
    fwhm_vals = fit_fwhm(image, xypos=positions, fit_shape=fit_shape)
    print(fwhm_vals.shape)
    aperture = CircularAperture(positions, r=n_fwhm * np.median(fwhm_vals))
    result = aperture_photometry(image, aperture)
    return result

def calibrate_mag(std_count, std_mag, list_count, ret_median = True) -> np.ndarray:
    #Ensure 1D inputs. Accept any shape that can be reshaped to 1D.
    std_count = np.asarray(std_count).reshape(-1)
    std_mag = np.asarray(std_mag).reshape(-1)
    list_count = np.asarray(list_count).reshape(-1)

    #Reshape for broadcasting: (N_std, 1) and (1, N_targets)
    std_count = std_count.reshape(-1, 1)   # shape (N_std, 1)
    std_mag = std_mag.reshape(-1, 1)       # shape (N_std, 1)
    list_count = list_count.reshape(1, -1) # shape (1, N_targets)

    zero_point = std_mag + 2.5 * np.log10(std_count)
    mag = -2.5 * np.log10(list_count) + zero_point
    if ret_median: mag = np.median(mag, axis=0)  #Median across standards
    return mag

if __name__ == "__main__":
    # Testing
    list_count = np.asarray([15000, 20000, 30000, 1000, 45000])
    std_count = (30000, 1000)
    std_mag = (6.0, 9.69)
    mag = calibrate_mag(std_count, std_mag, list_count)
    print("Calibrated Magnitudes:", mag)
    print("Shape of calibrated magnitudes:", mag.shape)