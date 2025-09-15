import pandas as pd
import matplotlib.pyplot as plt
import re
import scipy 
import configparser
from .utils import read_spectrum_file, get_genus, get_species, plot_wrapper

class Peak:
    def __init__(self, x, y):
        self.x_value = x
        self.y_value = y

    def __lt__(self, other):
        return self.x_value < other.x_value

    def __str__(self):
        return f"({self.x_value}, {self.y_value})"

    def __repr__(self):
        return f"({self.x_value}, {self.y_value})"

    def get_x(self):
        return self.x_value

    def get_y(self):
        return self.y_value

class PeakList:
    def set_parameters(self, prominence_threshold_min=None, prominence_threshold_max=None,
                      min_height_threshold_denominator=None, max_height_threshold_denominator=None,
                      min_distance_between_peaks=None, max_distance_between_peaks=None):
        if prominence_threshold_min is not None:
            self.prominence_threshold_min = prominence_threshold_min
        if prominence_threshold_max is not None:
            self.prominence_threshold_max = prominence_threshold_max
        if min_height_threshold_denominator is not None:
            self.min_height_threshold_denominator = min_height_threshold_denominator
        if max_height_threshold_denominator is not None:
            self.max_height_threshold_denominator = max_height_threshold_denominator
        if min_distance_between_peaks is not None:
            self.min_distance_between_peaks = min_distance_between_peaks
        if max_distance_between_peaks is not None:
            self.max_distance_between_peaks = max_distance_between_peaks

    def __init__(self, spectrum, prominence_threshold_min=0.15, prominence_threshold_max=0.40,
                 min_height_threshold_denominator=3.0, max_height_threshold_denominator=3.3,
                 min_distance_between_peaks=160, max_distance_between_peaks=160,
                 min_wavelength=None, max_wavelength=None): #todo delete unnecesary inputs
        self.prominence_threshold_min = spectrum.prominence_threshold_min
        self.prominence_threshold_max = spectrum.prominence_threshold_max
        self.min_height_threshold_denominator = spectrum.min_height_threshold_denominator
        self.max_height_threshold_denominator = spectrum.max_height_threshold_denominator
        self.min_distance_between_peaks = spectrum.min_distance_between_peaks
        self.max_distance_between_peaks = spectrum.max_distance_between_peaks
        self.spectrum = spectrum

    def get_spectrum(self):
        return self.spectrum

    def get_peaks(self, spectrum=None, min_wavelength=None, max_wavelength=None):
        if spectrum is None:
            spectrum = self.spectrum
        peaks = spectrum.get_peaks_as_object(min_wavelength, max_wavelength)
        x = []
        y = []
        for peak in peaks:
            if not (855 < peak.get_x() < 869):
                x.append(peak.get_x())
                y.append(peak.get_y())
        return x, y

    def plot_settings(self, min_wavelength=None, max_wavelength=None):
        self.spectrum.plot_settings()
        x_values, y_values = self.get_peaks(min_wavelength, max_wavelength)
        return plt.scatter(x_values, y_values, color="r")

    @plot_wrapper
    def plot(self, min_wavelength=None, max_wavelength=None):
        plot = self.plot_settings(min_wavelength, max_wavelength)
        return plot

    def get_minima(self, spectrum, min_wavelength=None, max_wavelength=None):
        df = spectrum.data
        if min_wavelength and max_wavelength:
            df = df[(df["wavelength"] < max_wavelength) & (df["wavelength"] > min_wavelength)]
        elif min_wavelength:
            df = df[df["wavelength"] > min_wavelength]
        elif max_wavelength:
            df = df[df["wavelength"] < max_wavelength]
        x = df["wavelength"].values
        y = df[spectrum.metadata["measuring_mode"]].values
        y_max = y.max()
        y_inverted = -y + y_max
        maximum_height = y_inverted.max() * 0.60
        minimum_height = 0
        peaks_funct = scipy.signal.find_peaks(y_inverted, distance=self.min_distance_between_peaks,
                                              prominence=self.prominence_threshold_min,
                                              height=(minimum_height, maximum_height))
        peaks_index = peaks_funct[0]
        x_values = x[peaks_index]
        y_values = y[peaks_index]
        return peaks_index, x_values, y_values

    def get_maxima(self, spectrum, min_wavelength=None, max_wavelength=None):
        df = spectrum.data
        if min_wavelength and max_wavelength:
            df2 = df[(df["wavelength"] < max_wavelength) & (df["wavelength"] > min_wavelength)]
        elif min_wavelength:
            df2 = df[df["wavelength"] > min_wavelength]
        elif max_wavelength:
            df2 = df[df["wavelength"] < max_wavelength]
        else:
            df2 = df
        x = df2["wavelength"].values
        y = df2[spectrum.metadata["measuring_mode"]].values
        min_height = y.max() / self.min_height_threshold_denominator
        min_distance = 50
        max_distance = 100.00
        width_t = 50.00
        peaks_funct = scipy.signal.find_peaks(y, height=min_height, distance=self.max_distance_between_peaks,
                                              prominence=self.prominence_threshold_max)
        peaks_index = peaks_funct[0]
        x_values = x[peaks_index]
        y_values = y[peaks_index]
        return peaks_index, x_values, y_values
        
class Spectrum:
    """
    This class represents the data and metadata for a spectrum file.

    Parameters are now loaded from a `.config` file instead of being passed directly.
    The configuration file must contain a `[Spectrum]` section with the following fields:

    - prominence_threshold_min (float)
    - prominence_threshold_max (float)
    - min_height_threshold_denominator (float)
    - max_height_threshold_denominator (float)
    - min_distance_between_peaks (int)
    - max_distance_between_peaks (int)
    - min_wavelength (float or None)
    - max_wavelength (float or None)
    - equipment (str, optional)

    Example:
        >>> spectrum = Spectrum("config.ini")
    """
    def __str__(self):
        return self.code

    def get_polarization(self):
        return self.polarization

    def get_name(self):
        return self.filename

    def get_filename(self):
        return self.filename
    
    def __init__(self, file_location, config_file: str, collection = None):
        debug = True
        #Load
        self.file_location = file_location
        self.collection = collection
        self.metadata, self.data = read_spectrum_file(file_location, debug = True)
        self.code = self.metadata.get("code", "na")
        self.genus = get_genus(self.code, collection)
        self.species = get_species(self.code, collection)
        self.measuring_mode = self.metadata.get("measuring_mode", "na")
        self.polarization = self.metadata.get("polarization", "na")
        self.filename = self.metadata.get("filename", "na")
        
        # Load config
        config = configparser.ConfigParser()
        config.read(config_file)

        if "Spectrum" not in config:
            raise ValueError("Config file must contain a [Spectrum] section.")

        cfg = config["Spectrum"]
        
        if debug:
            print("Spectrum config:", cfg)
        # Parse values with fallback
        
        self.prominence_threshold_min = cfg.getfloat("prominence_threshold_min", 0.15)
        self.prominence_threshold_max = cfg.getfloat("prominence_threshold_max", 0.40)
        self.min_height_threshold_denominator = cfg.getfloat("min_height_threshold_denominator", 3.0)
        self.max_height_threshold_denominator = cfg.getfloat("max_height_threshold_denominator", 3.3)
        self.min_distance_between_peaks = cfg.getint("min_distance_between_peaks", 160)
        self.max_distance_between_peaks = cfg.getint("max_distance_between_peaks", 160)
        self.min_wavelength = cfg.getfloat("min_wavelength", fallback=450)
        self.max_wavelength = cfg.getfloat("max_wavelength", fallback=1100)
        self.equipment = cfg.get("equipment", None)
        
        print("min_wavelength", self.min_wavelength, "max_wavelength", self.max_wavelength, "code:", self.code)
        # PeakList initialization
        self.peaklist = PeakList(
            self,
            self.prominence_threshold_min,
            self.prominence_threshold_max,
            self.min_height_threshold_denominator,
            self.max_height_threshold_denominator,
            self.min_distance_between_peaks,
            self.max_distance_between_peaks,
            self.min_wavelength,
            self.max_wavelength,
        )
        
    def set_parameters(self, prominence_threshold_min=None, prominence_threshold_max=None,
                      min_height_threshold_denominator=None, max_height_threshold_denominator=None,
                      min_distance_between_peaks=None, max_distance_between_peaks=None):
        if prominence_threshold_min is not None:
            self.prominence_threshold_min = prominence_threshold_min
        if prominence_threshold_max is not None:
            self.prominence_threshold_max = prominence_threshold_max
        if min_height_threshold_denominator is not None:
            self.min_height_threshold_denominator = min_height_threshold_denominator
        if max_height_threshold_denominator is not None:
            self.max_height_threshold_denominator = max_height_threshold_denominator
        if min_distance_between_peaks is not None:
            self.min_distance_between_peaks = min_distance_between_peaks
        if max_distance_between_peaks is not None:
            self.max_distance_between_peaks = max_distance_between_peaks
        
        # PeakList initialization
        self.peaklist.set_parameters(
        prominence_threshold_min, 
        prominence_threshold_max,
        min_height_threshold_denominator, 
        max_height_threshold_denominator,
        min_distance_between_peaks, 
        max_distance_between_peaks)
        
    def filter_wavelengths(self, df):
        debug = True
        if debug:
            print("Filtering wavelengths")
            
        if self.min_wavelength:
            df = df[df["wavelength"] > self.min_wavelength]
        if self.max_wavelength:
            df = df[df["wavelength"] < self.max_wavelength]
        if self.min_wavelength and self.max_wavelength:
            df = df[(df["wavelength"] < self.max_wavelength)&(df["wavelength"] > self.min_wavelength)]
        return df

    def plot_settings(self):
        def title_maker(measuring_mode, code, genus=None, species=None):
            measuring_modes = {"A": "absorptance", "T": "transmittance", "R": "reflectance"}
            title = f"{measuring_modes.get(measuring_mode, measuring_mode)} for "
            if genus == "na" and species == "na":
                title += f"code = {code}"
            else:
                title += f"{self.genus} {self.species}, code {self.code}"
            return title
        measuring_mode = self.metadata["measuring_mode"]
        df = self.filter_wavelengths(self.data)
        x = df["wavelength"]
        y = df[measuring_mode]
        plt.plot(x, y)
        plot = plt.title(title_maker(measuring_mode, self.code, self.genus, self.species))
        return plot

    @plot_wrapper
    def plot(self, plot_maxima=False, plot_minima=False):
        plot = self.plot_settings()
        if plot_maxima:
            self.plot_maxima()
        if plot_minima:
            self.plot_minima()
        return

    def plot_maxima(self):
        min_i, x_values, y_values = self.peaklist.get_maxima(self, self.min_wavelength, self.max_wavelength)
        return plt.scatter(x_values, y_values, color="r")

    def plot_minima(self):
        min_i, x_values, y_values = self.peaklist.get_minima(self, self.min_wavelength, self.max_wavelength)
        return plt.scatter(x_values, y_values, color="r")

    def get_normalized_spectrum(self):
        df_unfiltered = self.data[["wavelength", self.measuring_mode]].copy()  # Explicit copy
        #limit to valid range
        df = self.filter_wavelengths(df_unfiltered)
        max_value = df[self.measuring_mode].max()
        if max_value == 0:
            warnings.warn(
                f"Cannot normalize spectrum for {self.get_filename()}: Maximum value is zero.",
                UserWarning
            )
            return df  # Return unchanged copy if normalization is impossible
        df[self.measuring_mode] = df[self.measuring_mode] / max_value
        return df

    def get_maxima(self, min_wavelength=None, max_wavelength=None):
        return self.peaklist.get_maxima(self, self.min_wavelength, self.max_wavelength)

    def get_minima(self, min_wavelength=None, max_wavelength=None):
        return self.peaklist.get_minima(self, self.min_wavelength, self.max_wavelength)

    def get_critical_points(self, min_wavelength=None, max_wavelength=None):
        return self.peaklist.get_peaks(self, self.min_wavelength, self.max_wavelength)

    def set_dataframe(self, df):
        self.data = df

    def get_dataframe(self):
        return self.data

    def get_data(self):
        return self.data

    def get_metadata(self):
        return self.metadata

    def get_code(self):
        return self.code

    def get_collection(self):
        return self.collection

    def get_species(self):
        return self.species

    def get_genus(self):
        return self.genus

    def __lt__(self, other):
        def alphanum_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
        return alphanum_key(self.code) < alphanum_key(other.code)

    def get_peaks_as_object(self, get_maxima=True, get_minima=True, min_wavelength=None, max_wavelength=None):
        import scipy
        x = self.data["wavelength"].values
        y = self.data[self.metadata["measuring_mode"]].values
        min_height = y.max() / self.max_height_threshold_denominator
        
        peaks = []
        if get_maxima:
            max_i, max_x_values, max_y_values = self.peaklist.get_maxima(self, min_wavelength, max_wavelength)
            for x_val, y_val in zip(max_x_values, max_y_values):
                peaks.append(Peak(x_val, y_val))
        if get_minima:
            min_i, min_x_values, min_y_values = self.peaklist.get_minima(self, min_wavelength, max_wavelength)
            for x_val, y_val in zip(min_x_values, min_y_values):
                peaks.append(Peak(x_val, y_val))
        return sorted(peaks)