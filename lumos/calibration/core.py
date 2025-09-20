import numpy as np
import pandas as pd
import os
from astropy.io import fits
import lumos.utils as lumutils

class CalibrationFrames:
    def __init__(self,
                 bias = None, 
                 darks = None, 
                 flats = None, 
                 metadata = pd.DataFrame()):
        self.bias = bias #Bias frame
        self.darks = darks if darks is not None else {}   #keyed by exposure time
        self.flats = flats if flats is not None else {}   #keyed by filter name
        self.metadata = metadata  #DataFrame to hold metadata

    #--- Load Frames ---
    def load_bias(self, df, selfupdate = True):
        """Make master bias from bias frames in df"""
        if type(df) is np.ndarray or type(df) is list:
            biasdata = pd.DataFrame(df, columns=['FILENAME'])
            df = biasdata
        
        frames = []
        print("\nLoading Bias Frames:")
        for row in df.itertuples():
            bias_now = fits.getdata(row.FILENAME)
            frames.append(bias_now)
            lumutils.progress_bar(row.Index, len(df))
        result = np.median(frames, axis=0)

        if selfupdate:
            self.bias = result
        return result

    def load_darks(self, df, inplace=True):
        """Group darks by EXPTIME and make masters automatically"""
        if type(df) is np.ndarray or type(df) is list:
            darkdata = pd.DataFrame([
                    {
                        'FILENAME': darkfile,
                        'EXPTIME': fits.getheader(darkfile).get('EXPTIME'),
                    }
                    for darkfile in df
            ]).sort_values(by = 'EXPTIME'); df = darkdata

        result = {}
        for exptime, group in df.groupby("EXPTIME"):
            frames = []
            print(f"\nDark Exptime: {exptime}")
            for i, row in enumerate(group.itertuples()):
                dark_now = fits.getdata(row.FILENAME)
                frames.append(dark_now)
                lumutils.progress_bar(i, len(group))
            result[exptime] = np.median(frames, axis=0)

        if inplace:
            self.darks = result
        return result

    def load_flats(self, df, inplace=True):
        """Group flats by FILTER and make masters automatically"""
        if type(df) is np.ndarray or type(df) is list:
            flatdata = pd.DataFrame([
                    {
                        'FILENAME': flatfile,
                        'FILTER': fits.getheader(flatfile).get('FILTER'),
                    }
                    for flatfile in df
            ]).sort_values(by = 'FILTER'); df = flatdata
        
        result = {}
        for flt, group in df.groupby("FILTER"):
            frames = []
            print(f"\nFlat Filter: {flt}")
            for i, row in enumerate(group.itertuples()):
                flat_now = fits.getdata(row.FILENAME) - (self.bias if self.bias is not None else 0)
                flat_now /= np.median(flat_now)
                frames.append(flat_now)
                lumutils.progress_bar(i, len(group))
            result[flt] = np.median(frames, axis=0)
        
        if inplace:
            self.flats = result
        return result

    #--- Add frames ---
    def add_bias(self, files):
        frames = [fits.getdata(f) for f in files]
        self.bias = np.median(frames, axis=0)
        
    def add_dark(self, exptime, files):
        frames = [fits.getdata(f) for f in files]
        self.darks[exptime] = np.median(frames, axis=0)

    def add_flat(self, filter_name, files):
        frames = []
        for f in files:
            flat_now = fits.getdata(f)
            if self.bias is not None:
                flat_now = flat_now - self.bias
            flat_now /= np.median(flat_now)   # normalize
            frames.append(flat_now)
        self.flats[filter_name] = np.median(frames, axis=0)

    # --- Apply calibration ---
    def apply_array(self, raw_image, exptime, filter_used):
        """Apply bias, dark, flat to a raw image array."""
        calibrated = raw_image.copy()

        #Bias
        if self.bias is None:
            raise ValueError("No bias available!")
        calibrated -= self.bias

        #Dark
        if not self.darks:
            raise ValueError("No darks available!")
        if exptime not in self.darks:
            raise KeyError(f"No dark for exptime={exptime}")
        calibrated -= (self.darks[exptime] - self.bias)

        #Flat
        if not self.flats:
            raise ValueError("No flats available!")
        if filter_used not in self.flats:
            raise KeyError(f"No flat for filter={filter_used}")
        calibrated /= self.flats[filter_used]

        return calibrated

    def apply_self(self, metadata_dir, calibrated_dir, subject_name = '', warn=True):
        """
        Apply bias, dark, and flat calibration to files in metadata.
        Updates self.metadata for tracking.
        """
        #Suppress warnings for cleaner output
        if not warn:
            np.seterr(all='ignore')
        
        #Output directory creation,
        if warn:
            print(f"Successfully calibrated files will be saved to '{calibrated_dir}'")
        os.makedirs(calibrated_dir, exist_ok=True)

        for row in self.metadata.itertuples():
            raw_data = fits.getdata(row.FILENAME)
            header = fits.getheader(row.FILENAME)
            exptime = row.EXPTIME
            filter_used = row.FILTER

            try:
                calibrated_data = self.apply_array(raw_data, exptime, filter_used)
            except (ValueError, KeyError) as e:
                if warn:
                    print(f"Skipping calibration for {row.FILENAME}: {e}")
                self.metadata.loc[row.Index, ['CAL_FILENAME', 'CAL_STATUS']] = [np.nan, f"FAIL - {e}"]
                continue
            
            calibrated_filename = os.path.join(calibrated_dir,
                                           os.path.basename(row.FILENAME).replace('.fit', '_calibrated.fit'))
            hdu = fits.PrimaryHDU(data = calibrated_data,
                                  header = header)
            hdu.writeto(calibrated_filename, overwrite=True)
            self.metadata.loc[row.Index, ['CAL_FILENAME', 'CAL_STATUS']] = [calibrated_filename, "SUCCESS"]
            if warn:
                print(f"Calibrated {row.FILENAME} -> {calibrated_filename}")
    
        self.metadata.to_csv(f'{metadata_dir}{subject_name}_metadata.csv', index=False)
        if warn:
            print(f"Current metadata saved to {metadata_dir}{subject_name}_metadata.csv")
        
        return None

    # --- Friendly representation ---
    def __repr__(self):
        metadata_status = 'present' if not self.metadata.empty else 'missing'
        return (f"<CalibrationFrames: "
            f"bias={'yes' if self.bias is not None else 'no'}, "
            f"darks={list(self.darks.keys())}, "
            f"flats={list(self.flats.keys())}, "
            f"metadata={metadata_status}>")

