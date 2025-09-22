import pandas as pd
import matplotlib.pyplot as plt
import re
import scipy 
import configparser
from .utils import read_spectrum_file, get_genus, get_species, plot_wrapper
import numpy as np
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
    

    def __init__(self, spectrum): #todo delete unnecesary inputs
        
        self.spectrum = spectrum


    def get_spectrum(self):
        return self.spectrum

    def get_peaks(self, spectrum=None, min_wavelength=None, max_wavelength=None, debug=False):
        if spectrum is None:
            spectrum = self.spectrum

        # Get raw peaks
        peaks = spectrum.get_peaks_as_object(min_wavelength, max_wavelength)

        # Collect x, y values
        x = [peak.get_x() for peak in peaks]
        y = [peak.get_y() for peak in peaks]

        # --- Filtro adicional personalizado (ejemplo que ya tenías) ---
        # Elimina picos en un rango "prohibido" (855–869 nm)
        mask = [(not (855 < xi < 869)) for xi in x]
        x = np.array(x)[mask]
        y = np.array(y)[mask]

        # --- Filtro final por min/max rango ---
        if min_wavelength is not None or max_wavelength is not None:
            mask = np.ones_like(x, dtype=bool)
            if min_wavelength is not None:
                mask &= x >= min_wavelength
            if max_wavelength is not None:
                mask &= x <= max_wavelength

            if debug and not np.all(mask):
                print(f"⚠️ get_peaks: removed {np.sum(~mask)} peaks outside wavelength range.")

            x = x[mask]
            y = y[mask]

        return x, y


    def plot_settings(self, min_wavelength=None, max_wavelength=None):
        self.spectrum.plot_settings()
        x_values, y_values = self.get_peaks(min_wavelength, max_wavelength)
        return plt.scatter(x_values, y_values, color="r")

    @plot_wrapper
    def plot(self, min_wavelength=None, max_wavelength=None):
        plot = self.plot_settings(min_wavelength, max_wavelength)
        return plot

    def get_minima(self, spectrum,
                   height_bottom_threshold_for_minimum,
                   min_wavelength=None, 
                   max_wavelength=None, 
                   height_top_threshold_for_minimum=None,
                   smallest_distance_between_peaks_for_min=None,
                   prominence_threshold_for_min=None,
                   debug=False):
        debug=True
        df_unfiltered = spectrum.data
        #print("Spectrum get minima ", df.info)
        # --- Filtrado por rango ---
        if debug:
            print("min wav", min_wavelength, "max_wav", max_wavelength)
        if (min_wavelength) and (max_wavelength):
            df = df_unfiltered[(df_unfiltered["wavelength"] < max_wavelength) & (df_unfiltered["wavelength"] > min_wavelength)]
        elif min_wavelength:
            df = df_unfiltered[df_unfiltered["wavelength"] > min_wavelength]
        elif max_wavelength:
            df = df_unfiltered[df_unfiltered["wavelength"] < max_wavelength]
        else:
            df = df_unfiltered.copy()

        x = df["wavelength"].values
        y = df[spectrum.metadata["measuring_mode"]].values

        # --- Verificación de vacíos ---
        if y.size == 0:
            if debug:
                print("⚠️ get_minima: no data points after filtering.")
            return np.array([]), np.array([]), np.array([])

        # --- Invertir señal ---
        y_max = y.max()
        y_inverted = -y + y_max

        # --- Calcular thresholds ---
        max_val = y_inverted.max()
        maximum_height = max_val * height_top_threshold_for_minimum if height_top_threshold_for_minimum else max_val
        minimum_height = max_val * height_bottom_threshold_for_minimum if height_bottom_threshold_for_minimum else 0.0

        if debug:
            print(f"""get_minima: max_val={max_val}, 
                  bottom_thr={minimum_height}, top_thr={maximum_height}, 
                  {prominence_threshold_for_min=}{min_wavelength=}{max_wavelength=}""")
            
        # --- Encontrar mínimos (picos de y_inverted) ---
        peaks_index, properties = scipy.signal.find_peaks(
            y_inverted,
            distance=smallest_distance_between_peaks_for_min,
            prominence=prominence_threshold_for_min,
            height=(minimum_height, maximum_height)
        )

        # --- Si no hay picos ---
        if peaks_index.size == 0:
            if debug:
                print("⚠️ get_minima: no minima found.")
            return np.array([]), np.array([]), np.array([])

        # --- Resultados ---
        x_values = x[peaks_index]
        y_values = y[peaks_index]   # Nota: devuelvo los valores originales (no invertidos)
        
         # --- Filtro final: descartar puntos fuera del rango ---
        if min_wavelength is not None or max_wavelength is not None:
            mask = np.ones_like(x_values, dtype=bool)
            if min_wavelength is not None:
                mask &= x_values >= min_wavelength
            if max_wavelength is not None:
                mask &= x_values <= max_wavelength

            x_values = x_values[mask]
            y_values = y_values[mask]

            if debug and not np.all(mask):
                print(f"⚠️ get_minima: removed {np.sum(~mask)} minima outside wavelength range.")
        return peaks_index, x_values, y_values


    def get_maxima(self, spectrum,
                   height_bottom_threshold_for_maximum,
                   min_wavelength=None, 
                   max_wavelength=None,                                   
                   height_top_threshold_for_maximum=None, 
                   smallest_distance_between_peaks_for_max=None,
                   prominence_threshold_for_max=None):
        debug = False
        df = spectrum.data
        if debug:
            print("df peaklist get max", df)
            print("min", min_wavelength, "max", max_wavelength)
        # --- Filtrado por rango de longitud de onda ---
        if (min_wavelength) and (max_wavelength):
            df2 = df[(df["wavelength"] < max_wavelength) & (df["wavelength"] > min_wavelength)]
        elif min_wavelength:
            df2 = df[df["wavelength"] > min_wavelength]
        elif max_wavelength:
            df2 = df[df["wavelength"] < max_wavelength]
        else:
            df2 = df.copy()

        if debug:
            print("after filtering get_maxima: Dataframe summary:\n", df2.info())

        # --- Extraer arrays ---
        x = df2["wavelength"].values
        y = df2[spectrum.metadata["measuring_mode"]].values

        # --- Verificación de vacíos ---
        if y.size == 0:
            if debug:
                print("⚠️ get_maxima: no data points after filtering.")
            # Devuelve arrays vacíos en vez de fallar
            return np.array([]), np.array([]), np.array([])

        # --- Umbral de altura ---
        min_height = y.max() * height_bottom_threshold_for_maximum
        if debug:
            print("Spectrum/get_maxima/height_bottom_threshold_for_maximum:",
                  height_bottom_threshold_for_maximum,
                  "→ min_height =", min_height)

        # --- Encontrar picos ---
        peaks_index, properties = scipy.signal.find_peaks(
            y,
            height=min_height,
            distance=smallest_distance_between_peaks_for_max,
            prominence=prominence_threshold_for_max
        )

        if peaks_index.size == 0:
            if debug:
                print("⚠️ get_maxima: no peaks found.")
            return np.array([]), np.array([]), np.array([])

        x_values = x[peaks_index]
        y_values = y[peaks_index]

        return peaks_index, x_values, y_values

        
class Spectrum:
    """
    This class represents the data and metadata for a spectrum file.

    Parameters are now loaded from a `.config` file instead of being passed directly.
    The configuration file must contain a `[Spectrum]` section with the following fields:

    - prominence_threshold_for_min (float)
    - prominence_threshold_for_max (float)
    - height_bottom_threshold_for_minimum (float)
    - height_bottom_threshold_for_maximum (float)
    - smallest_distance_between_peaks_for_min (int)
    - smallest_distance_between_peaks_for_max (int)
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
        debug = False
        #Load
        self.file_location = file_location
        self.collection = collection
        self.metadata, self.data = read_spectrum_file(file_location)
        self.code = self.metadata.get("code", "na")
        self.genus = get_genus(self.code, collection)
        self.species = get_species(self.code, collection)
        self.measuring_mode = self.metadata.get("measuring_mode", "na")
        self.polarization = self.metadata.get("polarization", "na")
        self.filename = self.metadata.get("filename", "na")
        
        # PeakList initialization
        self.peaklist = PeakList(spectrum = self)
        
        
    
    def print_parameters(self):
        debug = False
        params = [
            "prominence_threshold_for_min",
            "prominence_threshold_for_max",
            "height_bottom_threshold_for_minimum",
            "height_bottom_threshold_for_maximum",
            "smallest_distance_between_peaks_for_min",
            "smallest_distance_between_peaks_for_max",
        ]

        for param in params:
            if hasattr(self, param):
                value = getattr(self, param)
                if debug:
                    print(f"{param}: {value}")
            else:
                if debug:
                    print(f"{param}: (not set)")
                
    
        
    def filter_wavelengths(self, df, min_wavelength, max_wavelength):
        debug = False
        if debug:
            print("Filtering wavelengths")
            
        if min_wavelength:
            df = df[df["wavelength"] > min_wavelength]
        if max_wavelength:
            df = df[df["wavelength"] < max_wavelength]
        if min_wavelength and max_wavelength:
            df = df[(df["wavelength"] < max_wavelength)&(df["wavelength"] > min_wavelength)]
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

    def get_normalized_spectrum(self, min_wavelength, max_wavelength):
        df_unfiltered = self.data[["wavelength", self.measuring_mode]].copy()  # Explicit copy
        #limit to valid range
        if min_wavelength or max_wavelength:
            df = self.filter_wavelengths(df_unfiltered, min_wavelength, max_wavelength)
        else:
            df = df_unfiltered.copy()
        max_value = df[self.measuring_mode].max()
        if max_value == 0:
            warnings.warn(
                f"Cannot normalize spectrum for {self.get_filename()}: Maximum value is zero.",
                UserWarning
            )
            return df  # Return unchanged copy if normalization is impossible
        df[self.measuring_mode] = df[self.measuring_mode] / max_value
        return df

    def get_maxima(self, min_wavelength, 
                        max_wavelength, 
                        height_bottom_threshold_for_maximum, 
                        height_top_threshold_for_maximum, 
                        smallest_distance_between_peaks_for_max): 
        return self.peaklist.get_maxima(self, min_wavelength= min_wavelength, 
                                        max_wavelength= max_wavelength, 
                                        height_bottom_threshold_for_maximum = height_bottom_threshold_for_maximum, 
                                        height_top_threshold_for_maximum = height_top_threshold_for_maximum, 
                                        smallest_distance_between_peaks_for_max= smallest_distance_between_peaks_for_max,
                                        )

    def get_minima(self, min_wavelength, 
                        max_wavelength, 
                        height_bottom_threshold_for_minimum , 
                        height_top_threshold_for_minimum , 
                        smallest_distance_between_peaks_for_min):
                            return self.peaklist.get_minima(self, min_wavelength= min_wavelength, 
                                                            max_wavelength= max_wavelength, 
                                                            height_bottom_threshold_for_minimum=height_bottom_threshold_for_minimum, 
                                                            height_top_threshold_for_minimum=height_top_threshold_for_minimum, 
                                                            smallest_distance_between_peaks_for_min=smallest_distance_between_peaks_for_min)

    def get_critical_points(self, min_wavelength=None, max_wavelength=None, 
                                                    height_bottom_threshold_for_maximum = None, 
                                                    height_top_threshold_for_maximum = None, 
                                                    height_bottom_threshold_for_minimum = None, 
                                                    height_top_threshold_for_minimum = None):
        return self.peaklist.get_peaks(self, min_wavelength, max_wavelength, height_bottom_threshold_for_maximum, 
                                                                             height_top_threshold_for_maximum, 
                                                                             height_bottom_threshold_for_minimum, 
                                                                             height_top_threshold_for_minimum)

    def set_dataframe(self, df):
        self.data = df

    def get_dataframe(self):
        return self.data
    
    def get_equipment(self):
        print("Getting equipment. Spectrum")
        try:
            equip = self.metadata["equipment"]
        except Exception as e: 
            print(e)
            equip = "na"
        return equip
        
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

    def get_peaks_as_object(self, get_maxima=True, get_minima=True, min_wavelength=None, max_wavelength=None, height_bottom_threshold_for_maximum =None, top_height_threshold_maximum=None):
        import scipy
        x = self.data["wavelength"].values
        y = self.data[self.metadata["measuring_mode"]].values
        
        
        peaks = []
        if get_maxima:
            min_height = y.max() * self.height_bottom_threshold_for_maximum
            max_i, max_x_values, max_y_values = self.peaklist.get_maxima(self, height_bottom_threshold_for_maximum, top_height_threshold_maximum)
            for x_val, y_val in zip(max_x_values, max_y_values):
                peaks.append(Peak(x_val, y_val))
        if get_minima:
            min_height = y.max() * self.height_bottom_threshold_for_minimum
            min_i, min_x_values, min_y_values = self.peaklist.get_minima(self, bottom_height_threshold_minimum, top_height_threshold_minimum)
            for x_val, y_val in zip(min_x_values, min_y_values):
                peaks.append(Peak(x_val, y_val))
        return sorted(peaks)