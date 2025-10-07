from astropy.table import QTable
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import pandas as pd
import numpy as np

class PhotometrySession:
    def __init__(self,
                 metadata: pd.DataFrame = pd.DataFrame(),
                 lightcurves = None,
                 ref_index: pd.DataFrame = pd.DataFrame(),
                 ref_stars: pd.DataFrame = pd.DataFrame()) -> None:
        self.metadata = metadata  #Same metadata as in CalibrationFrames class
        self.lightcurves = lightcurves if lightcurves is not None else {} #Dict of astropy QTables
        self.ref_index = ref_index  #Dataframe of detected stars id -> reference stars id per image
        self.ref_stars = ref_stars  #Reference catalog (VizieR, Gaia, or custom CSV) (Index, RA, Dec, WavelengthMag)
        
        return None

class PhotometryTEST:
    def __init__(self):
        # Image metadata (per frame)
        self.metadata = QTable(
            names=("filename", "time", "filter", "exptime", "wcs"),
            dtype=("U200", object, "U20", float, object)
        )

        # Light curves per object (dict of QTables)
        self.lightcurves = {}

        # Reference image for registration fallback
        self.ref_index = 0
        self.ref_stars = None  # detected stars in reference frame

    # --------------------------
    # Step 1: Load metadata
    # --------------------------
    def add_image(self, filename, header):
        time = Time(header["DATE-OBS"])
        filt = header.get("FILTER", "Unknown")
        exptime = header.get("EXPTIME", 0.0)
        wcs = WCS(header)

        self.metadata.add_row((filename, time, filt, exptime, wcs))

    # --------------------------
    # Step 2: Register objects
    # --------------------------
    def add_object(self, name, skycoord):
        # Make a table to store this object’s light curve
        self.lightcurves[name] = QTable(
            names=("time", "x", "y", "flux", "flux_err"),
            dtype=(object, float, float, float, float)
        )
        self.lightcurves[name].meta["skycoord"] = skycoord

    # --------------------------
    # Step 3: Locate objects in frame
    # --------------------------
    def locate_objects(self, idx, fallback=True):
        row = self.metadata[idx]
        wcs = row["wcs"]

        positions = {}
        for name, lc in self.lightcurves.items():
            coord = lc.meta["skycoord"]
            try:
                x, y = wcs.world_to_pixel(coord)
            except Exception:
                if not fallback:
                    raise
                # Fallback: register to reference image
                x, y = self._register_to_reference(idx, coord)
            positions[name] = (x, y)
        return positions

    def _register_to_reference(self, idx, coord):
        # --- stub: implement registration here ---
        # Use source detection + geometric transform to align
        # For now, just return dummy coords
        return (np.nan, np.nan)

    # --------------------------
    # Step 4: Measure fluxes
    # --------------------------
    def measure_fluxes(self, idx, data, positions, aperture_radius=5.0):
        # stub: implement with photutils aperture_photometry
        for name, (x, y) in positions.items():
            flux, flux_err = np.nan, np.nan  # replace with real measurement
            self.lightcurves[name].add_row((self.metadata[idx]["time"], x, y, flux, flux_err))

    # --------------------------
    # Step 5: Export
    # --------------------------
    def export_lightcurve(self, name, filename):
        self.lightcurves[name].write(filename, overwrite=True)
