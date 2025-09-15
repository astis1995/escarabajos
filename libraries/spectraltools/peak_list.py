import scipy.signal
import matplotlib.pyplot as plt
from .utils import plot_wrapper
from .spectrum import Spectrum
from .peak import Peak


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
                 min_wavelength=None, max_wavelength=None):
        self.prominence_threshold_min = prominence_threshold_min
        self.prominence_threshold_max = prominence_threshold_max
        self.min_height_threshold_denominator = min_height_threshold_denominator
        self.max_height_threshold_denominator = max_height_threshold_denominator
        self.min_distance_between_peaks = min_distance_between_peaks
        self.max_distance_between_peaks = max_distance_between_peaks
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
        debug = True
        df = spectrum.data
        if debug:
            print("min", min_wavelength, "max", max_wavelength)
        #filter spectrum to range
        if min_wavelength and max_wavelength:
            df = df[(df["wavelength"] < max_wavelength) & (df["wavelength"] > min_wavelength)]
        elif min_wavelength:
            df = df[df["wavelength"] > min_wavelength]
        elif max_wavelength:
            df = df[df["wavelength"] < max_wavelength]
        #extract x and y 
        x = df["wavelength"].values
        y = df[spectrum.metadata["measuring_mode"]].values
        y_max = y.max()
        #invert spectrum to get minima
        y_inverted = -y + y_max
        maximum_height = y_inverted.max() #* 0.60
        minimum_height = 0
        peaks_funct = scipy.signal.find_peaks(y_inverted, distance=self.min_distance_between_peaks,
                                              prominence=self.prominence_threshold_min,
                                              height=(minimum_height, maximum_height))
        peaks_index = peaks_funct[0]
        x_values = x[peaks_index]
        y_values = y[peaks_index]
        return peaks_index, x_values, y_values

    def get_maxima(self, spectrum, min_wavelength=None, max_wavelength=None):
        debug = True
        if debug:
            print("min", min_wavelength, "max", max_wavelength)
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