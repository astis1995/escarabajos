import scipy.signal
import matplotlib.pyplot as plt
from .utils import plot_wrapper
from .spectrum import Spectrum
from .peak import Peak


class PeakList:
    def set_parameters(self, prominence_threshold_for_min=None, prominence_threshold_for_max=None,
                      height_bottom_threshold_for_minimun=None, height_bottom_threshold_for_maximum=None,
                      smallest_distance_between_peaks_for_min=None, smallest_distance_between_peaks_for_max=None):
        if prominence_threshold_for_min is not None:
            self.prominence_threshold_for_min = prominence_threshold_for_min
        if prominence_threshold_for_max is not None:
            self.prominence_threshold_for_max = prominence_threshold_for_max
        if height_bottom_threshold_for_minimun is not None:
            self.height_bottom_threshold_for_minimun = height_bottom_threshold_for_minimun
        if height_bottom_threshold_for_maximum is not None:
            self.height_bottom_threshold_for_maximum = height_bottom_threshold_for_maximum
        if smallest_distance_between_peaks_for_min is not None:
            self.smallest_distance_between_peaks_for_min = smallest_distance_between_peaks_for_min
        if smallest_distance_between_peaks_for_max is not None:
            self.smallest_distance_between_peaks_for_max = smallest_distance_between_peaks_for_max

    def __init__(self, spectrum, prominence_threshold_for_min=0.15, prominence_threshold_for_max=0.40,
                 height_bottom_threshold_for_minimun=3.0, height_bottom_threshold_for_maximum=3.3,
                 smallest_distance_between_peaks_for_min=160, smallest_distance_between_peaks_for_max=160,
                 min_wavelength=None, max_wavelength=None):
        self.prominence_threshold_for_min = prominence_threshold_for_min
        self.prominence_threshold_for_max = prominence_threshold_for_max
        self.height_bottom_threshold_for_minimun = height_bottom_threshold_for_minimun
        self.height_bottom_threshold_for_maximum = height_bottom_threshold_for_maximum
        self.smallest_distance_between_peaks_for_min = smallest_distance_between_peaks_for_min
        self.smallest_distance_between_peaks_for_max = smallest_distance_between_peaks_for_max
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
        debug = False
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
        peaks_funct = scipy.signal.find_peaks(y_inverted, distance=self.smallest_distance_between_peaks_for_min,
                                              prominence=self.prominence_threshold_for_min,
                                              height=(minimum_height, maximum_height))
        peaks_index = peaks_funct[0]
        x_values = x[peaks_index]
        y_values = y[peaks_index]
        return peaks_index, x_values, y_values

    def get_maxima(self, spectrum, min_wavelength=None, max_wavelength=None):
        debug = False
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
        min_height = y.max() / self.height_bottom_threshold_for_minimun
        min_distance = 50
        max_distance = 100.00
        width_t = 50.00
        peaks_funct = scipy.signal.find_peaks(y, height=min_height, distance=self.smallest_distance_between_peaks_for_max,
                                              prominence=self.prominence_threshold_for_max)
        peaks_index = peaks_funct[0]
        x_values = x[peaks_index]
        y_values = y[peaks_index]
        return peaks_index, x_values, y_values