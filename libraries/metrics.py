#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import configparser
import seaborn as sns
debug = False
from spectraltools import Specimen_Collection, Spectrum, create_path_if_not_exists


class Metric():
    """Abstract class for metrics. Supports comparison, naming, and string representation."""
    
    name = "Metric"
    
    def __init__(self, spectrum=None, config_file = None, collection_list = None, plot=False):
        self.spectrum = spectrum
        self.plot = plot
        self.metric_value = None

        # Generic defaults
        self.prominence_threshold_for_min = None
        self.prominence_threshold_for_max = None
        self.height_bottom_threshold_for_minimum = None
        self.height_bottom_threshold_for_maximum = None
        self.smallest_distance_between_peaks_for_min = None
        self.smallest_distance_between_peaks_for_max = None

        self.uv_vis_min_wavelength = None
        self.uv_vis_max_wavelength = None
        self.ir_min_wavelength = None
        self.ir_max_wavelength = None
        
        
    def get_metric_value(self):
        debug = False
        if debug:
            print("Getting metric value...")
        return self.metric_value

    def set_metric_value(self):
        return 0.0

    @classmethod
    def get_name(cls):
        return cls.name

    @staticmethod
    def description():
        return "No description yet"

    def __lt__(self, other):
        return self.metric_value < other.metric_value

    def __repr__(self):
        return f'{self.name} value: {self.metric_value:.4f} for {self.spectrum.genus} {self.spectrum.species}. File: {self.spectrum.filename}'
    
    
        
    

# -------------------------------------------------------------------------
# Metrics (examples modified to read config_file)
# -------------------------------------------------------------------------

class Gamma_First_Two_Peaks_Simple(Metric):
    name = "Gamma_First_Two_Peaks_Simple"

    def __init__(self, spectrum, config_file=None, collection_list=None, plot=False):
        self.plot = plot
        self.spectrum = spectrum
        self._load_config(config_file)
        self.collection_list = collection_list
        
        self.metric_value = self.set_metric_value(spectrum)
        

    def set_metric_value(self, spectrum):
        """
        Compute ratio of second peak height to first peak height.
        If fewer than two peaks are found, returns NaN.
        """
        
        debug = False
        if debug:
            print("Gamma_First_Two_Peaks_Simple, setting metric value")
        try:
            max_i, max_x, max_y = spectrum.get_maxima(
            height_bottom_threshold_for_maximum=self.height_bottom_threshold_for_maximum, 
            min_wavelength=self.start_wavelength, 
            max_wavelength=self.end_wavelength, 
            height_top_threshold_for_maximum=self.height_top_threshold_for_maximum, 
            smallest_distance_between_peaks_for_max=self.smallest_distance_between_peaks_for_max)
            if debug:
                print("height_bottom_threshold_for_maximum",height_bottom_threshold_for_maximum)
                print("max_i, max_x, max_y",max_i, max_x, max_y)
        except Exception as e:
            warnings.warn(
                f"{self.name}: Could not get maxima for {spectrum.code} "
                f"({spectrum.genus} {spectrum.species}): {e}"
            )
            return np.nan
        if not(len(max_y) >= 2 and max_y[0] != 0):
                print("len(max_y):", len(max_y),"max_y[0]",max_y[0])
        if len(max_y) >= 2 and max_y[0] != 0:
            ratio = max_y[1] / max_y[0]

            if debug or self.plot:
                df = spectrum.get_normalized_spectrum(min_wavelength= self.start_wavelength, max_wavelength = self.end_wavelength)
                x = df["wavelength"].values
                y = df[spectrum.metadata["measuring_mode"]].values

                plt.figure(figsize=(10, 6))
                plt.plot(x, y, label="Normalized Spectrum")
                plt.axvline(max_x[0], color="r", linestyle="--", label="First Peak")
                plt.axvline(max_x[1], color="g", linestyle="--", label="Second Peak")
                plt.title(
                    f"{spectrum.genus} {spectrum.species} (code: {spectrum.code})\n"
                    f"Ratio = {ratio:.3f}"
                )
                plt.xlabel("Wavelength (nm)")
                plt.ylabel(spectrum.metadata["measuring_mode"])
                plt.legend()
                plt.title(f"{spectrum.genus} {spectrum.species} {spectrum.collection.sex_lookup(spectrum.code, self.collection_list)} {spectrum.code}. {spectrum.get_equipment()}")
                plt.grid()
                plt.show()

            return ratio

        warnings.warn(
            f"{self.name}: Less than two peaks found for {spectrum.code} "
            f"({spectrum.genus} {spectrum.species}) — returning NaN."
        )
        return np.nan
        
    def print_parameters(self):
        
        print(f"""{self.visible_range_start_wavelength=}
        {self.start_wavelength=}
        {self.start_wavelength=}
        {self.end_wavelength=}
        {self.prominence_threshold_for_min=}
        {self.prominence_threshold_for_max=}
        {self.height_bottom_threshold_for_minimum=}
        {self.height_bottom_threshold_for_maximum=}
        {self.smallest_distance_between_peaks_for_min=}
        {self.smallest_distance_between_peaks_for_max=}
        {self.height_top_threshold_for_maximum}""") 
    def _load_config(self, config_file):
        debug = False
        config = configparser.ConfigParser()

        if not os.path.isfile(config_file):
            warnings.warn(f"Config file {config_file} not found. Using defaults.", UserWarning)
            return {}

        try:
            config.read(config_file)
        except Exception as e:
            warnings.warn(f"Failed to read config file {config_file}: {e}. Using defaults.", UserWarning)
            return {}

        if debug:
            print("config_file:", config_file)
            print("Available sections:", config.sections())

        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section in {config_file}, using defaults.", UserWarning)
            return {}

        cfg = config[self.name]
        
        # wavelength ranges
        self.visible_range_start_wavelength = cfg.getfloat("visible_range_start_wavelength", fallback=None)
        self.start_wavelength = cfg.getfloat("start_wavelength", fallback=None)
        self.end_wavelength = cfg.getfloat("end_wavelength", fallback=None)
        

        # thresholds for prominence and height
        self.prominence_threshold_for_min = cfg.getfloat("prominence_threshold_for_min", fallback=None)
        self.prominence_threshold_for_max = cfg.getfloat("prominence_threshold_for_max", fallback=None)
        self.height_bottom_threshold_for_minimum = cfg.getfloat("height_bottom_threshold_for_minimum", fallback=None)
        self.height_bottom_threshold_for_maximum = cfg.getfloat("height_bottom_threshold_for_maximum", fallback=None)

        # distance between peaks
        self.smallest_distance_between_peaks_for_min = cfg.getfloat("smallest_distance_between_peaks_for_min", fallback=None)
        self.smallest_distance_between_peaks_for_max = cfg.getfloat("smallest_distance_between_peaks_for_max", fallback=None)
        self.height_top_threshold_for_maximum  = cfg.getfloat("height_top_threshold_for_maximum", fallback=None)
        if debug:
            self.print_parameters()
        
        
class Gamma_First_Two_Peaks(Metric):
    name = "Gamma_First_Two_Peaks"
    
    def __init__(self, spectrum, config_file, collection_list,plot = None):
        # Call parent constructor 
        self.plot = plot
        
        self.spectrum = spectrum
        self._load_config(config_file)
        self.collection_list = collection_list
        self.metric_value = self.set_metric_value(spectrum)
        if debug:
            self.print_parameters()
        

    import os
    import configparser
    import warnings
    
    def print_parameters(self):
        
        print(f"""{self.visible_range_start_wavelength=}
        {self.start_wavelength=}
        {self.start_wavelength=}
        {self.end_wavelength=}
        {self.prominence_threshold_for_min=}
        {self.prominence_threshold_for_max=}
        {self.height_bottom_threshold_for_minimum=}
        {self.height_bottom_threshold_for_maximum=}
        {self.smallest_distance_between_peaks_for_min=}
        {self.smallest_distance_between_peaks_for_max=}
        {self.height_top_threshold_for_maximum}""")
        
    def _load_config(self, config_file):
        debug = False
        if debug:
            print("STARTING _load_config")
            print("config_file", config_file)
        
        config = configparser.ConfigParser()

        if not os.path.isfile(config_file):
            warnings.warn(f"Config file {config_file} not found. Using defaults.", UserWarning)
            return {}

        try:
            config.read(config_file)
        except Exception as e:
            warnings.warn(f"Failed to read config file {config_file}: {e}. Using defaults.", UserWarning)
            return {}

        if debug:
            print("config_file:", config_file)
            print("Available sections:", config.sections())

        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section in {config_file}, using defaults.", UserWarning)
            return {}

        cfg = config[self.name]
        if debug:
            print("Gamma_First_Two_Peaks_Golden cfg", cfg)
        
        # wavelength ranges
        self.visible_range_start_wavelength = cfg.getfloat("visible_range_start_wavelength", fallback=None)
        self.start_wavelength = cfg.getfloat("start_wavelength", fallback=None)
        self.end_wavelength = cfg.getfloat("end_wavelength", fallback=None)

        # thresholds for prominence and height
        self.prominence_threshold_for_min = cfg.getfloat("prominence_threshold_for_min", fallback=None)
        self.prominence_threshold_for_max = cfg.getfloat("prominence_threshold_for_max", fallback=None)
        self.height_bottom_threshold_for_minimum = cfg.getfloat("height_bottom_threshold_for_minimum", fallback=None)
        self.height_bottom_threshold_for_maximum = cfg.getfloat("height_bottom_threshold_for_maximum", fallback=None)

        # distance between peaks
        self.smallest_distance_between_peaks_for_min = cfg.getfloat("smallest_distance_between_peaks_for_min", fallback=None)
        self.smallest_distance_between_peaks_for_max = cfg.getfloat("smallest_distance_between_peaks_for_max", fallback=None)
        self.height_top_threshold_for_maximum  = cfg.getfloat("height_top_threshold_for_maximum", fallback=None)
        if debug:
            self.print_parameters()


    def set_metric_value(self, spectrum):
        debug = True
        if debug:
            print("Gamma_First_Two_Peaks: Getting metric value")
            print("Spectrum parameters", self.print_parameters())
        max_i, max_x, max_y = spectrum.get_maxima(
            height_bottom_threshold_for_maximum=self.height_bottom_threshold_for_maximum, 
            min_wavelength=self.start_wavelength, 
            max_wavelength=self.end_wavelength, 
            height_top_threshold_for_maximum=self.height_top_threshold_for_maximum, 
            smallest_distance_between_peaks_for_max=self.smallest_distance_between_peaks_for_max)
        if debug:
            print("max_i, max_x, max_y", max_i, max_x, max_y)
        if len(max_y) >= 2 and max_y[0] != 0:
            value = max_y[1] / max_y[0]
            if debug or self.plot:
                df = spectrum.get_normalized_spectrum(min_wavelength= self.start_wavelength, max_wavelength = self.end_wavelength)
                x = df["wavelength"].values
                y = df[spectrum.metadata["measuring_mode"]].values
                plt.figure(figsize=(10, 6))
                plt.plot(x, y, label="Normalized Spectrum")
                plt.axvline(max_x[0], color="r", linestyle="--", label="First Peak")
                plt.axvline(max_x[1], color="g", linestyle="--", label="Second Peak")
                plt.title(f"{spectrum.genus} {spectrum.species} {spectrum.collection.sex_lookup(spectrum.code, self.collection_list)} {spectrum.code}. {spectrum.get_equipment()}")
                plt.xlabel("Wavelength (nm)")
                plt.ylabel(spectrum.metadata["measuring_mode"])
                plt.legend()
                plt.grid()
                plt.show()
            return value
        return np.nan

class Gamma_First_Two_Peaks_Golden(Metric):
    name = "Gamma_First_Two_Peaks_Golden"
    
    def __init__(self, spectrum, config_file, collection_list,plot = None):
        # Call parent constructor 
        #super().__init__(spectrum, config_file, collection_list, plot)
        self.plot = plot
        self.spectrum = spectrum
        self._load_config(config_file) #XXXXXXXXXXXXXXXXXXXXXXXXX
        self.collection_list = collection_list
        self.metric_value = self.set_metric_value(spectrum)
        

    import os
    import configparser
    import warnings

    def _load_config(self, config_file):
        debug = False
        if debug:
            print("STARTING _load_config")
            print("config_file", config_file)
        
        config = configparser.ConfigParser()

        if not os.path.isfile(config_file):
            warnings.warn(f"Config file {config_file} not found. Using defaults.", UserWarning)
            return {}

        try:
            config.read(config_file)
        except Exception as e:
            warnings.warn(f"Failed to read config file {config_file}: {e}. Using defaults.", UserWarning)
            return {}

        if debug:
            print("config_file:", config_file)
            print("Available sections:", config.sections())

        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section in {config_file}, using defaults.", UserWarning)
            return {}

        cfg = config[self.name]
        if debug:
            print("Gamma_First_Two_Peaks_Golden cfg", cfg)
        
        # wavelength ranges
        self.visible_range_start_wavelength = cfg.getfloat("visible_range_start_wavelength", fallback=None)
        self.start_wavelength = cfg.getfloat("start_wavelength", fallback=None)
        self.end_wavelength = cfg.getfloat("end_wavelength", fallback=None)

        # thresholds for prominence and height
        self.prominence_threshold_for_min = cfg.getfloat("prominence_threshold_for_min", fallback=None)
        self.prominence_threshold_for_max = cfg.getfloat("prominence_threshold_for_max", fallback=None)
        self.height_bottom_threshold_for_minimum = cfg.getfloat("height_bottom_threshold_for_minimum", fallback=None)
        self.height_bottom_threshold_for_maximum = cfg.getfloat("height_bottom_threshold_for_maximum", fallback=None)

        # distance between peaks
        self.smallest_distance_between_peaks_for_min = cfg.getfloat("smallest_distance_between_peaks_for_min", fallback=None)
        self.smallest_distance_between_peaks_for_max = cfg.getfloat("smallest_distance_between_peaks_for_max", fallback=None)
        self.height_top_threshold_for_maximum  = cfg.getfloat("height_top_threshold_for_maximum", fallback=None)
        if debug:
            self.print_parameters()

    def print_parameters(self):
        
        print(f"""{self.visible_range_start_wavelength=}
        {self.start_wavelength=}
        {self.start_wavelength=}
        {self.end_wavelength=}
        {self.prominence_threshold_for_min=}
        {self.prominence_threshold_for_max=}
        {self.height_bottom_threshold_for_minimum=}
        {self.height_bottom_threshold_for_maximum=}
        {self.smallest_distance_between_peaks_for_min=}
        {self.smallest_distance_between_peaks_for_max=}
        {self.height_top_threshold_for_maximum}""")
        
    def set_metric_value(self, spectrum):
        debug = False
        if debug or self.plot:
            print("Gamma_First_Two_Peaks_Golden Getting metric value") #height_bottom_threshold_for_maximum, top_height_threshold_maximum
        if debug:    
            print("Gamma_First_Two_Peaks_Golden/set_metric_value: self.height_bottom_threshold_for_maximum", self.height_bottom_threshold_for_maximum)
        
            print("spectrum info", spectrum.data)
        max_i, max_x, max_y = spectrum.get_maxima( min_wavelength = self.start_wavelength, 
                                                  max_wavelength = self.end_wavelength,
                                                  height_top_threshold_for_maximum = self.height_top_threshold_for_maximum,
                                                  height_bottom_threshold_for_maximum = self.height_bottom_threshold_for_maximum, 
                                                  smallest_distance_between_peaks_for_max = self.smallest_distance_between_peaks_for_max)
        if debug:
            print("max_i, max_x, max_y", max_i, max_x, max_y)
        if len(max_y) >= 2 and max_y[0] != 0:
            value = max_y[1] / max_y[0]
            if debug or self.plot:
                df = spectrum.get_normalized_spectrum(min_wavelength= self.start_wavelength, max_wavelength = self.end_wavelength)
                x = df["wavelength"].values
                y = df[spectrum.metadata["measuring_mode"]].values
                plt.figure(figsize=(10, 6))
                plt.plot(x, y, label="Normalized Spectrum")
                plt.axvline(max_x[0], color="r", linestyle="--", label="First Peak")
                plt.axvline(max_x[1], color="g", linestyle="--", label="Second Peak")
                plt.title(f"{self.name}: {spectrum.genus} {spectrum.species} (code: {spectrum.code})")
                plt.xlabel("Wavelength (nm)")
                plt.ylabel(spectrum.metadata["measuring_mode"])
                plt.legend()
                plt.grid()
                plt.show()
            return value
        return np.nan

class Gamma_First_Two_Peaks_Silver(Metric):
    name = "Gamma_First_Two_Peaks_Silver"
    
    def __init__(self, spectrum, config_file, collection_list,plot = None):
        # Call parent constructor 
        #super().__init__(spectrum, config_file, collection_list, plot)
        self.plot = plot
        self.spectrum = spectrum
        self._load_config(config_file) #XXXXXXXXXXXXXXXXXXXXXXXXX
        self.collection_list = collection_list
        self.metric_value = self.set_metric_value(spectrum)
        

    import os
    import configparser
    import warnings

    def _load_config(self, config_file):
        debug = True
        if debug:
            print("STARTING _load_config")
            print("config_file", config_file)
        
        config = configparser.ConfigParser()

        if not os.path.isfile(config_file):
            warnings.warn(f"Config file {config_file} not found. Using defaults.", UserWarning)
            return {}

        try:
            config.read(config_file)
        except Exception as e:
            warnings.warn(f"Failed to read config file {config_file}: {e}. Using defaults.", UserWarning)
            return {}

        if debug:
            print("config_file:", config_file)
            print("Available sections:", config.sections())

        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section in {config_file}, using defaults.", UserWarning)
            return {}

        cfg = config[self.name]
        if debug:
            print("Gamma_First_Two_Peaks_Golden cfg", cfg)
        
        # wavelength ranges
        self.visible_range_start_wavelength = cfg.getfloat("visible_range_start_wavelength", fallback=None)
        self.start_wavelength = cfg.getfloat("start_wavelength", fallback=None)
        self.end_wavelength = cfg.getfloat("end_wavelength", fallback=None)

        # thresholds for prominence and height
        self.prominence_threshold_for_min = cfg.getfloat("prominence_threshold_for_min", fallback=None)
        self.prominence_threshold_for_max = cfg.getfloat("prominence_threshold_for_max", fallback=None)
        self.height_bottom_threshold_for_minimum = cfg.getfloat("height_bottom_threshold_for_minimum", fallback=None)
        self.height_bottom_threshold_for_maximum = cfg.getfloat("height_bottom_threshold_for_maximum", fallback=None)

        # distance between peaks
        self.smallest_distance_between_peaks_for_min = cfg.getfloat("smallest_distance_between_peaks_for_min", fallback=None)
        self.smallest_distance_between_peaks_for_max = cfg.getfloat("smallest_distance_between_peaks_for_max", fallback=None)
        self.height_top_threshold_for_maximum  = cfg.getfloat("height_top_threshold_for_maximum", fallback=None)
        if debug:
            self.print_parameters()

    def print_parameters(self):
        
        print(f"""{self.visible_range_start_wavelength=}
        {self.visible_range_end_wavelength=}
        {self.start_wavelength=}
        {self.end_wavelength=}
        {self.prominence_threshold_for_min=}
        {self.prominence_threshold_for_max=}
        {self.height_bottom_threshold_for_minimum=}
        {self.height_bottom_threshold_for_maximum=}
        {self.smallest_distance_between_peaks_for_min=}
        {self.smallest_distance_between_peaks_for_max=}
        {self.height_top_threshold_for_maximum}""")
        
    def set_metric_value(self, spectrum):
        debug = True
        if debug or self.plot:
            print("Gamma_First_Two_Peaks_Silver Getting metric value") #height_bottom_threshold_for_maximum, top_height_threshold_maximum
        if debug:    
            print("Gamma_First_Two_Peaks_Silver/set_metric_value: self.height_bottom_threshold_for_maximum", self.height_bottom_threshold_for_maximum)
        
            print("spectrum info", spectrum.data)
            print("self.smallest_distance_between_peaks_for_max", self.smallest_distance_between_peaks_for_max)
        max_i, max_x, max_y = spectrum.get_maxima( min_wavelength = self.start_wavelength, 
                                                  max_wavelength = self.end_wavelength,
                                                  height_top_threshold_for_maximum = self.height_top_threshold_for_maximum,
                                                  height_bottom_threshold_for_maximum = self.height_bottom_threshold_for_maximum, 
                                                  smallest_distance_between_peaks_for_max = self.smallest_distance_between_peaks_for_max)
        if debug:
            print("max_i, max_x, max_y", max_i, max_x, max_y)
        if len(max_y) >= 2 and max_y[0] != 0:
            value = max_y[1] / max_y[0]
            if debug or self.plot:
                df = spectrum.get_normalized_spectrum(min_wavelength= self.start_wavelength, max_wavelength = self.end_wavelength)
                x = df["wavelength"].values
                y = df[spectrum.metadata["measuring_mode"]].values
                plt.figure(figsize=(10, 6))
                plt.plot(x, y, label="Normalized Spectrum")
                plt.axvline(max_x[0], color="r", linestyle="--", label="First Peak")
                plt.axvline(max_x[1], color="g", linestyle="--", label="Second Peak")
                plt.title(f"{self.name}: {spectrum.genus} {spectrum.species} (code: {spectrum.code})")
                plt.xlabel("Wavelength (nm)")
                plt.ylabel(spectrum.metadata["measuring_mode"])
                plt.legend()
                plt.grid()
                plt.show()
            return value
        return np.nan
        
class Gamma_Arbitrary_Limits_Silver(Metric):
    """This gamma metric calculates the ratio between the maximum in the IR range
    and the maximum in the visible range. Wavelength ranges are loaded from the 
    [Gamma_Arbitrary_Limits] section of the config file."""
    debug = False
    name = "Gamma_Arbitrary_Limits_Silver"

    def __init__(self, spectrum, config_file: str, collection_list,plot = None):
        # Call parent constructor 
        self.plot = plot
        
        self.spectrum = spectrum
        self._load_config(config_file)
        self.collection_list = collection_list
        self.metric_value = self.set_metric_value(spectrum)
        
        
    def _load_config(self, config_file):
        debug = False
        if debug:
            print("config_file",config_file)
        config = configparser.ConfigParser()
        config.read(config_file)
        if debug:
            print("Available sections:", config.sections())
            pass
        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section, using defaults.", UserWarning)
            cfg = {}
        else:
            cfg = config[self.name]
            if debug:
                print("cfg",cfg)

        # Read values, fallback to defaults if missing
        self.start_wavelength = float(cfg.get("start_wavelength", None))
        self.end_wavelength = float(cfg.get("end_wavelength", None))
        self.uv_vis_min_wavelength = float(cfg.get("uv_vis_min_wavelength", None))
        self.uv_vis_max_wavelength = float(cfg.get("uv_vis_max_wavelength", None))
        self.ir_min_wavelength = float(cfg.get("ir_min_wavelength", None))
        self.ir_max_wavelength = float(cfg.get("ir_max_wavelength", None))

    def set_metric_value(self, spectrum, debug=False):
        debug = False
        def get_maximum_in_range(spectrum, min_wavelength, max_wavelength):
            measuring_mode = spectrum.metadata["measuring_mode"]
            df = spectrum.data
            subset = df[(df["wavelength"] > min_wavelength) & (df["wavelength"] < max_wavelength)]
            if subset.empty:
                return None, 0.0
            idx = subset[measuring_mode].idxmax()
            return df.loc[idx, "wavelength"], df.loc[idx, measuring_mode]
        
        
        
        uv_vis_x, uv_vis_max = get_maximum_in_range(spectrum, self.uv_vis_min_wavelength, self.uv_vis_max_wavelength)
        
        ir_x, ir_max = get_maximum_in_range(spectrum, self.ir_min_wavelength, self.ir_max_wavelength)
        
        if debug:
            print("UV VIS MIN",self.uv_vis_min_wavelength, "UV VIS MAX:", self.uv_vis_max_wavelength )
            print("POINT:", uv_vis_x, uv_vis_max)
            print("ir_min_wavelength",self.uv_vis_min_wavelength, "ir_max_wavelength:", self.uv_vis_max_wavelength )
            print("POINT:", ir_x, ir_max)
        
        if (debug or self.plot) and uv_vis_x and ir_x:
            df = spectrum.get_normalized_spectrum(min_wavelength= self.start_wavelength, max_wavelength = self.end_wavelength)
            plt.figure(figsize=(10, 6))
            plt.plot(df["wavelength"], df[spectrum.metadata["measuring_mode"]], label="Normalized Spectrum")
            plt.title(f"{spectrum.genus} {spectrum.species} {spectrum.collection.sex_lookup(spectrum.code, self.collection_list)} {spectrum.code}. {spectrum.get_equipment()}")
            plt.xlabel("wavelength"); plt.ylabel("Reflectance");
            plt.axvline(uv_vis_x, color="b", linestyle="--", label="Visible Peak")
            plt.axvline(ir_x, color="m", linestyle="--", label="IR Peak")
            plt.legend();
            plt.grid()
            plt.show()

        if ir_max == 0:
            return np.nan
        return uv_vis_max / ir_max
    
    def print_parameters(self):
        
        print(f"""{self.visible_range_start_wavelength=}
        {self.start_wavelength=}
        {self.start_wavelength=}
        {self.end_wavelength=}
        {self.prominence_threshold_for_min=}
        {self.prominence_threshold_for_max=}
        {self.height_bottom_threshold_for_minimum=}
        {self.height_bottom_threshold_for_maximum=}
        {self.smallest_distance_between_peaks_for_min=}
        {self.smallest_distance_between_peaks_for_max=}
        {self.height_top_threshold_for_maximum}""")

    @staticmethod
    def description():
        return (
            "This algorithm calculates the ratio between the highest reflectance "
            "peak in the visible range (uv_vis_min_wavelength–uv_vis_max_wavelength) "
            "and the maximum peak in the IR range (ir_min_wavelength–ir_max_wavelength)."
        )

class Gamma_Arbitrary_Limits(Metric):
    """This gamma metric calculates the ratio between the maximum in the IR range
    and the maximum in the visible range. Wavelength ranges are loaded from the 
    [Gamma_Arbitrary_Limits] section of the config file."""
    debug = False
    name = "Gamma_Arbitrary_Limits"

    def __init__(self, spectrum, config_file: str, collection_list,plot = None):
        # Call parent constructor 
        self.plot = plot
        
        self.spectrum = spectrum
        self._load_config(config_file)
        self.collection_list = collection_list
        self.metric_value = self.set_metric_value(spectrum)
        
        
    def _load_config(self, config_file):
        debug = False
        
        config = configparser.ConfigParser()
        config.read(config_file)
        if debug:
            #print("Available sections:", config.sections())
            pass
        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section, using defaults.", UserWarning)
            cfg = {}
        else:
            cfg = config[self.name]
            if debug:
                print("cfg",cfg)

        # Read values, fallback to defaults if missing
        self.uv_vis_min_wavelength = float(cfg.get("uv_vis_min_wavelength", None))
        self.uv_vis_max_wavelength = float(cfg.get("uv_vis_max_wavelength", None))
        self.ir_min_wavelength = float(cfg.get("ir_min_wavelength", None))
        self.ir_max_wavelength = float(cfg.get("ir_max_wavelength", None))

    def set_metric_value(self, spectrum, debug=False):
        debug = False
        def get_maximum_in_range(spectrum, min_wavelength, max_wavelength):
            measuring_mode = spectrum.metadata["measuring_mode"]
            df = spectrum.data
            subset = df[(df["wavelength"] > min_wavelength) & (df["wavelength"] < max_wavelength)]
            if subset.empty:
                return None, 0.0
            idx = subset[measuring_mode].idxmax()
            return df.loc[idx, "wavelength"], df.loc[idx, measuring_mode]
        if debug:
            
            print("UV VIS MIN",self.uv_vis_min_wavelength, "UV VIS MAX:", self.uv_vis_max_wavelength )
        uv_vis_x, uv_vis_max = get_maximum_in_range(spectrum, self.uv_vis_min_wavelength, self.uv_vis_max_wavelength)
        if debug:
            print("POINT:", uv_vis_x, uv_vis_max)
            print("ir_min_wavelength",self.uv_vis_min_wavelength, "ir_max_wavelength:", self.uv_vis_max_wavelength )
        ir_x, ir_max = get_maximum_in_range(spectrum, self.ir_min_wavelength, self.ir_max_wavelength)
        if debug:    
            print("POINT:", ir_x, ir_max)
        if debug and uv_vis_x and ir_x:
            df = spectrum.get_normalized_spectrum(min_wavelength= self.start_wavelength, max_wavelength = self.end_wavelength)
            plt.figure(figsize=(10, 6))
            plt.plot(df["wavelength"], df[spectrum.metadata["measuring_mode"]], label="Normalized Spectrum")
            plt.title(f"{spectrum.genus} {spectrum.species} {spectrum.collection.sex_lookup(spectrum.code, self.collection_list)} {spectrum.code}. {spectrum.get_equipment()}")
            plt.xlabel("wavelength"); plt.ylabel("Reflectance");
            plt.axvline(uv_vis_x, color="b", linestyle="--", label="Visible Peak")
            plt.axvline(ir_x, color="m", linestyle="--", label="IR Peak")
            plt.legend(); 
            plt.grid()
            plt.show()

        if ir_max == 0:
            return np.nan
        return uv_vis_max / ir_max


    @staticmethod
    def description():
        return (
            "This algorithm calculates the ratio between the highest reflectance "
            "peak in the visible range (uv_vis_min_wavelength–uv_vis_max_wavelength) "
            "and the maximum peak in the IR range (ir_min_wavelength–ir_max_wavelength)."
        )

class Gamma_Arbitrary_Limits_Golden(Metric):
    """This gamma metric calculates the ratio between the maximum in the IR range
    and the maximum in the visible range. Wavelength ranges are loaded from the 
    [Gamma_Arbitrary_Limits] section of the config file."""
    debug = False
    name = "Gamma_Arbitrary_Limits_Golden"

    def __init__(self, spectrum, config_file: str, collection_list,plot = None):
        # Call parent constructor 
        self.plot = plot
        
        self.spectrum = spectrum
        self._load_config(config_file)
        self.collection_list = collection_list
        self.metric_value = self.set_metric_value(spectrum)
        
        
    def _load_config(self, config_file):
        debug = False
        
        config = configparser.ConfigParser()
        config.read(config_file)
        if debug:
            #print("Available sections:", config.sections())
            pass
        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section, using defaults.", UserWarning)
            cfg = {}
        else:
            cfg = config[self.name]
            if debug:
                print("cfg",cfg)

        # Read values, fallback to defaults if missing
        self.uv_vis_min_wavelength = float(cfg.get("uv_vis_min_wavelength", None))
        self.uv_vis_max_wavelength = float(cfg.get("uv_vis_max_wavelength", None))
        self.ir_min_wavelength = float(cfg.get("ir_min_wavelength", None))
        self.ir_max_wavelength = float(cfg.get("ir_max_wavelength", None))

    def set_metric_value(self, spectrum, debug=False):
        debug = False
        def get_maximum_in_range(spectrum, min_wavelength, max_wavelength):
            measuring_mode = spectrum.metadata["measuring_mode"]
            df = spectrum.data
            subset = df[(df["wavelength"] > min_wavelength) & (df["wavelength"] < max_wavelength)]
            if subset.empty:
                return None, 0.0
            idx = subset[measuring_mode].idxmax()
            return df.loc[idx, "wavelength"], df.loc[idx, measuring_mode]
        
        if debug:
            print("UV VIS MIN",self.uv_vis_min_wavelength, "UV VIS MAX:", self.uv_vis_max_wavelength )
        uv_vis_x, uv_vis_max = get_maximum_in_range(spectrum, self.uv_vis_min_wavelength, self.uv_vis_max_wavelength)
        if debug:
            print("POINT:", uv_vis_x, uv_vis_max)
            print("ir_min_wavelength",self.uv_vis_min_wavelength, "ir_max_wavelength:", self.uv_vis_max_wavelength )
        ir_x, ir_max = get_maximum_in_range(spectrum, self.ir_min_wavelength, self.ir_max_wavelength)
        if debug:
            print("POINT:", ir_x, ir_max)
        if debug and uv_vis_x and ir_x:
            df = spectrum.get_normalized_spectrum(min_wavelength= self.start_wavelength, max_wavelength = self.end_wavelength)
            plt.figure(figsize=(10, 6))
            plt.plot(df["wavelength"], df[spectrum.metadata["measuring_mode"]], label="Normalized Spectrum")
            plt.title(f"{spectrum.genus} {spectrum.species} {spectrum.collection.sex_lookup(spectrum.code, self.collection_list)} {spectrum.code}. {spectrum.get_equipment()}")
            plt.xlabel("wavelength"); plt.ylabel("Reflectance");
            plt.axvline(uv_vis_x, color="b", linestyle="--", label="Visible Peak")
            plt.axvline(ir_x, color="m", linestyle="--", label="IR Peak")
            plt.legend(); 
            plt.grid()
            plt.show()

        if ir_max == 0:
            return np.nan
        return uv_vis_max / ir_max


    @staticmethod
    def description():
        return (
            "This algorithm calculates the ratio between the highest reflectance "
            "peak in the visible range (uv_vis_min_wavelength–uv_vis_max_wavelength) "
            "and the maximum peak in the IR range (ir_min_wavelength–ir_max_wavelength)."
        )
        
import numpy as np
import matplotlib.pyplot as plt
import configparser, warnings
from scipy.optimize import curve_fit

class Gamma_Gaussian_Fit_Silver(Metric):
    """This gamma metric calculates the ratio between Gaussian-fitted maximums 
    in the visible and IR wavelength ranges. Both ranges are defined in the config file."""
    
    name = "Gamma_Gaussian_Fit_Silver"

    def __init__(self, spectrum, config_file: str, collection_list,plot = None):
        # Call parent constructor 
        #super().__init__(spectrum, config_file, collection_list, plot)
        self.plot = plot
        self.spectrum = spectrum
        self._load_config(config_file)
        self.collection_list = collection_list
        self.metric_value = self.set_metric_value(spectrum)

    def _load_config(self, config_file):
        debug = False
        if debug:
            print("STARTING _load_config")
            print("config_file", config_file)
            
        config = configparser.ConfigParser()

        if not os.path.isfile(config_file):
            warnings.warn(f"Config file {config_file} not found. Using defaults.", UserWarning)
            return {}

        try:
            config.read(config_file)
        except Exception as e:
            warnings.warn(f"Failed to read config file {config_file}: {e}. Using defaults.", UserWarning)
            return {}

        if debug:
            print("config_file:", config_file)
            print("Available sections:", config.sections())

        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section in {config_file}, using defaults.", UserWarning)
            return {}

        cfg = config[self.name]
        if debug:
            print("Gamma_First_Two_Peaks_Golden cfg", cfg)
        
        # wavelength ranges
        self.visible_range_start_wavelength = cfg.getfloat("visible_range_start_wavelength", fallback=None)
        self.start_wavelength = cfg.getfloat("start_wavelength", fallback=None)
        self.end_wavelength = cfg.getfloat("end_wavelength", fallback=None)
        self.uv_vis_min_wavelength = cfg.getfloat("uv_vis_min_wavelength", fallback=None)
        self.uv_vis_max_wavelength = cfg.getfloat("uv_vis_max_wavelength", fallback=None)
        self.ir_min_wavelength = cfg.getfloat("ir_min_wavelength", fallback=None)
        self.ir_max_wavelength = cfg.getfloat("ir_max_wavelength", fallback=None)

        # thresholds for prominence and height
        self.prominence_threshold_for_min = cfg.getfloat("prominence_threshold_for_min", fallback=None)
        self.prominence_threshold_for_max = cfg.getfloat("prominence_threshold_for_max", fallback=None)
        self.height_bottom_threshold_for_minimum = cfg.getfloat("height_bottom_threshold_for_minimum", fallback=None)
        self.height_bottom_threshold_for_maximum = cfg.getfloat("height_bottom_threshold_for_maximum", fallback=None)

        # distance between peaks
        self.smallest_distance_between_peaks_for_min = cfg.getfloat("smallest_distance_between_peaks_for_min", fallback=None)
        self.smallest_distance_between_peaks_for_max = cfg.getfloat("smallest_distance_between_peaks_for_max", fallback=None)
        self.height_top_threshold_for_maximum  = cfg.getfloat("height_top_threshold_for_maximum", fallback=None)
        if debug:
            self.print_parameters()
        

    def print_parameters(self):
        print(f"""{self.visible_range_start_wavelength=}
        {self.start_wavelength=}
        {self.end_wavelength=}
        {self.uv_vis_min_wavelength=}
        {self.uv_vis_max_wavelength=}
        {self.ir_min_wavelength=}
        {self.ir_max_wavelength=}
        {self.prominence_threshold_for_min=}
        {self.prominence_threshold_for_max=}
        {self.height_bottom_threshold_for_minimum=}
        {self.height_bottom_threshold_for_maximum=}
        {self.smallest_distance_between_peaks_for_min=}
        {self.smallest_distance_between_peaks_for_max=}
        {self.height_top_threshold_for_maximum}""")

    @staticmethod
    def _gaussian(x, a, mu, sigma, c):
        """Gaussian with offset"""
        return a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c

    def _fit_gaussian_peak(self, df, min_wavelength, max_wavelength, measuring_mode):
        """
        Extract subset of spectrum, fit Gaussian, and return:
            peak position (mu), peak height (ymax), fit x-values, fit y-values.
        If Gaussian fitting fails, fallback to quadratic (Taylor 2nd degree).
        """
        if debug:
            print("range _fit_gaussian_peak: ", min_wavelength, max_wavelength)
        subset = df[(df["wavelength"] > min_wavelength) & (df["wavelength"] < max_wavelength)]
        if subset.empty:
            warnings.warn(
                f"No data in range {min_wavelength}-{max_wavelength} "
                f"for {getattr(self.spectrum, 'code', 'unknown')} "
                f"({getattr(self.spectrum, 'genus', 'unknown')} "
                f"{getattr(self.spectrum, 'species', 'unknown')})"
            )
            return None, 0.0, None, None

        xdata = subset["wavelength"].values
        ydata = subset[measuring_mode].values

        # Initial guesses
        mu_guess = xdata[np.argmax(ydata)]
        a_guess = ydata.max() - ydata.min()
        sigma_guess = (max_wavelength - min_wavelength) / 10.0
        c_guess = ydata.min()

        try:
            # Try Gaussian fit
            popt, _ = curve_fit(
                self._gaussian,
                xdata,
                ydata,
                p0=[a_guess, mu_guess, sigma_guess, c_guess],
                maxfev=10000
            )
            a, mu, sigma, c = popt
            ymax = self._gaussian(mu, *popt)

            # Gaussian curve for plotting
            xfit = np.linspace(min_wavelength, max_wavelength, 300)
            yfit = self._gaussian(xfit, *popt)

            return mu, ymax, xfit, yfit

        except Exception as e:
            warnings.warn(
                f"Gaussian fit failed in range {min_wavelength}-{max_wavelength} "
                f"for {getattr(self.spectrum, 'code', 'unknown')} "
                f"({getattr(self.spectrum, 'genus', 'unknown')} "
                f"{getattr(self.spectrum, 'species', 'unknown')}): {e}. "
                f"Falling back to quadratic approximation."
            )

            try:
                # Quadratic (Taylor 2nd order) fit
                coeffs = np.polyfit(xdata, ydata, 2)  # ax^2 + bx + c
                poly = np.poly1d(coeffs)

                # Peak (vertex of parabola) = -b/(2a)
                if coeffs[0] != 0:
                    mu_quad = -coeffs[1] / (2 * coeffs[0])
                    ymax_quad = poly(mu_quad)
                else:
                    mu_quad = mu_guess
                    ymax_quad = ydata.max()

                xfit = np.linspace(min_wavelength, max_wavelength, 300)
                yfit = poly(xfit)

                return mu_quad, ymax_quad, xfit, yfit

            except Exception as e2:
                warnings.warn(
                    f"Quadratic fallback also failed in range {min_wavelength}-{max_wavelength} "
                    f"for {getattr(self.spectrum, 'code', 'unknown')} "
                    f"({getattr(self.spectrum, 'genus', 'unknown')} "
                    f"{getattr(self.spectrum, 'species', 'unknown')}): {e2}. "
                    f"Returning raw max instead."
                )
                return mu_guess, ydata.max(), None, None



    def set_metric_value(self, spectrum):
        debug = False
        measuring_mode = spectrum.metadata["measuring_mode"]
        df = spectrum.data
        if debug:
            print(f"{self.uv_vis_min_wavelength=}{self.uv_vis_max_wavelength=}{self.ir_min_wavelength=}{self.ir_max_wavelength=}")
        # Fit visible and IR peaks
        uv_vis_x, uv_vis_max, uv_xfit, uv_yfit = self._fit_gaussian_peak(
            df, self.uv_vis_min_wavelength, self.uv_vis_max_wavelength, measuring_mode
        )
        ir_x, ir_max, ir_xfit, ir_yfit = self._fit_gaussian_peak(
            df, self.ir_min_wavelength, self.ir_max_wavelength, measuring_mode
        )

        if debug or self.plot:
            df_norm = spectrum.get_normalized_spectrum(min_wavelength= self.start_wavelength, max_wavelength = self.end_wavelength)
            plt.figure(figsize=(10, 6))
            plt.plot(df_norm["wavelength"], df_norm[measuring_mode], label="Normalized Spectrum")

            # Add Gaussian fits if available
            if uv_xfit is not None:
                plt.plot(uv_xfit, uv_yfit / max(uv_yfit), "b--", label="Visible Gaussian Fit (norm.)")
            if ir_xfit is not None:
                plt.plot(ir_xfit, ir_yfit / max(ir_yfit), "m--", label="IR Gaussian Fit (norm.)")

            plt.axvline(uv_vis_x, color="b", linestyle=":", label="Visible Peak")
            plt.axvline(ir_x, color="m", linestyle=":", label="IR Peak")

            plt.title(f"{spectrum.genus} {spectrum.species} {spectrum.collection.sex_lookup(spectrum.code, self.collection_list)} {spectrum.code}. {spectrum.get_equipment()}")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Reflectance (normalized)")
            plt.legend()
            plt.grid()
            plt.show()

        if ir_max == 0:
            return np.nan
        return uv_vis_max / ir_max

    @staticmethod
    def description():
        return (
            "This algorithm calculates the ratio between Gaussian-fitted peaks "
            "in the visible and IR wavelength ranges, and plots the fits alongside "
            "the normalized spectrum."
        )
        
class Gamma_Gaussian_Fit(Metric):
    """This gamma metric calculates the ratio between Gaussian-fitted maximums 
    in the visible and IR wavelength ranges. Both ranges are defined in the config file."""
    debug = False
    name = "Gamma_Gaussian_Fit"

    def __init__(self, spectrum, config_file: str, collection_list,plot = None):
        # Call parent constructor 
        #super().__init__(spectrum, config_file, collection_list, plot)
        self.plot = plot
        self.spectrum = spectrum
        self._load_config(config_file)
        self.collection_list = collection_list
        self.metric_value = self.set_metric_value(spectrum)

    def _load_config(self, config_file):
        debug = False
        if debug:
            print("STARTING _load_config")
            print("config_file", config_file)
            
        config = configparser.ConfigParser()

        if not os.path.isfile(config_file):
            warnings.warn(f"Config file {config_file} not found. Using defaults.", UserWarning)
            return {}

        try:
            config.read(config_file)
        except Exception as e:
            warnings.warn(f"Failed to read config file {config_file}: {e}. Using defaults.", UserWarning)
            return {}

        if debug:
            print("config_file:", config_file)
            print("Available sections:", config.sections())

        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section in {config_file}, using defaults.", UserWarning)
            return {}

        cfg = config[self.name]
        if debug:
            print("Gamma_First_Two_Peaks_Golden cfg", cfg)
        
        # wavelength ranges
        self.visible_range_start_wavelength = cfg.getfloat("visible_range_start_wavelength", fallback=None)
        self.start_wavelength = cfg.getfloat("start_wavelength", fallback=None)
        self.end_wavelength = cfg.getfloat("end_wavelength", fallback=None)
        self.uv_vis_min_wavelength = cfg.getfloat("uv_vis_min_wavelength", fallback=None)
        self.uv_vis_max_wavelength = cfg.getfloat("uv_vis_max_wavelength", fallback=None)
        self.ir_min_wavelength = cfg.getfloat("ir_min_wavelength", fallback=None)
        self.ir_max_wavelength = cfg.getfloat("ir_max_wavelength", fallback=None)

        # thresholds for prominence and height
        self.prominence_threshold_for_min = cfg.getfloat("prominence_threshold_for_min", fallback=None)
        self.prominence_threshold_for_max = cfg.getfloat("prominence_threshold_for_max", fallback=None)
        self.height_bottom_threshold_for_minimum = cfg.getfloat("height_bottom_threshold_for_minimum", fallback=None)
        self.height_bottom_threshold_for_maximum = cfg.getfloat("height_bottom_threshold_for_maximum", fallback=None)

        # distance between peaks
        self.smallest_distance_between_peaks_for_min = cfg.getfloat("smallest_distance_between_peaks_for_min", fallback=None)
        self.smallest_distance_between_peaks_for_max = cfg.getfloat("smallest_distance_between_peaks_for_max", fallback=None)
        self.height_top_threshold_for_maximum  = cfg.getfloat("height_top_threshold_for_maximum", fallback=None)
        if debug:
            self.print_parameters()
        

    def print_parameters(self):
        print(f"""{self.visible_range_start_wavelength=}
        {self.start_wavelength=}
        {self.end_wavelength=}
        {self.uv_vis_min_wavelength=}
        {self.uv_vis_max_wavelength=}
        {self.ir_min_wavelength=}
        {self.ir_max_wavelength=}
        {self.prominence_threshold_for_min=}
        {self.prominence_threshold_for_max=}
        {self.height_bottom_threshold_for_minimum=}
        {self.height_bottom_threshold_for_maximum=}
        {self.smallest_distance_between_peaks_for_min=}
        {self.smallest_distance_between_peaks_for_max=}
        {self.height_top_threshold_for_maximum}""")

    @staticmethod
    def _gaussian(x, a, mu, sigma, c):
        """Gaussian with offset"""
        return a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c

    def _fit_gaussian_peak(self, df, min_wavelength, max_wavelength, measuring_mode):
        """
        Extract subset of spectrum, fit Gaussian, and return:
            peak position (mu), peak height (ymax), fit x-values, fit y-values.
        If Gaussian fitting fails, fallback to quadratic (Taylor 2nd degree).
        """
        if debug:
            print("range _fit_gaussian_peak: ", min_wavelength, max_wavelength)
        subset = df[(df["wavelength"] > min_wavelength) & (df["wavelength"] < max_wavelength)]
        if subset.empty:
            warnings.warn(
                f"No data in range {min_wavelength}-{max_wavelength} "
                f"for {getattr(self.spectrum, 'code', 'unknown')} "
                f"({getattr(self.spectrum, 'genus', 'unknown')} "
                f"{getattr(self.spectrum, 'species', 'unknown')})"
            )
            return None, 0.0, None, None

        xdata = subset["wavelength"].values
        ydata = subset[measuring_mode].values

        # Initial guesses
        mu_guess = xdata[np.argmax(ydata)]
        a_guess = ydata.max() - ydata.min()
        sigma_guess = (max_wavelength - min_wavelength) / 10.0
        c_guess = ydata.min()

        try:
            # Try Gaussian fit
            popt, _ = curve_fit(
                self._gaussian,
                xdata,
                ydata,
                p0=[a_guess, mu_guess, sigma_guess, c_guess],
                maxfev=10000
            )
            a, mu, sigma, c = popt
            ymax = self._gaussian(mu, *popt)

            # Gaussian curve for plotting
            xfit = np.linspace(min_wavelength, max_wavelength, 300)
            yfit = self._gaussian(xfit, *popt)

            return mu, ymax, xfit, yfit

        except Exception as e:
            warnings.warn(
                f"Gaussian fit failed in range {min_wavelength}-{max_wavelength} "
                f"for {getattr(self.spectrum, 'code', 'unknown')} "
                f"({getattr(self.spectrum, 'genus', 'unknown')} "
                f"{getattr(self.spectrum, 'species', 'unknown')}): {e}. "
                f"Falling back to quadratic approximation."
            )

            try:
                # Quadratic (Taylor 2nd order) fit
                coeffs = np.polyfit(xdata, ydata, 2)  # ax^2 + bx + c
                poly = np.poly1d(coeffs)

                # Peak (vertex of parabola) = -b/(2a)
                if coeffs[0] != 0:
                    mu_quad = -coeffs[1] / (2 * coeffs[0])
                    ymax_quad = poly(mu_quad)
                else:
                    mu_quad = mu_guess
                    ymax_quad = ydata.max()

                xfit = np.linspace(min_wavelength, max_wavelength, 300)
                yfit = poly(xfit)

                return mu_quad, ymax_quad, xfit, yfit

            except Exception as e2:
                warnings.warn(
                    f"Quadratic fallback also failed in range {min_wavelength}-{max_wavelength} "
                    f"for {getattr(self.spectrum, 'code', 'unknown')} "
                    f"({getattr(self.spectrum, 'genus', 'unknown')} "
                    f"{getattr(self.spectrum, 'species', 'unknown')}): {e2}. "
                    f"Returning raw max instead."
                )
                return mu_guess, ydata.max(), None, None



    def set_metric_value(self, spectrum):
        debug = False
        measuring_mode = spectrum.metadata["measuring_mode"]
        df = spectrum.data
        if debug:
            print(f"{self.uv_vis_min_wavelength=}{self.uv_vis_max_wavelength=}{self.ir_min_wavelength=}{self.ir_max_wavelength=}")
        # Fit visible and IR peaks
        uv_vis_x, uv_vis_max, uv_xfit, uv_yfit = self._fit_gaussian_peak(
            df, self.uv_vis_min_wavelength, self.uv_vis_max_wavelength, measuring_mode
        )
        ir_x, ir_max, ir_xfit, ir_yfit = self._fit_gaussian_peak(
            df, self.ir_min_wavelength, self.ir_max_wavelength, measuring_mode
        )

        if debug or self.plot:
            df_norm = spectrum.get_normalized_spectrum(min_wavelength= self.start_wavelength, max_wavelength = self.end_wavelength)
            plt.figure(figsize=(10, 6))
            plt.plot(df_norm["wavelength"], df_norm[measuring_mode], label="Normalized Spectrum")

            # Add Gaussian fits if available
            if uv_xfit is not None:
                plt.plot(uv_xfit, uv_yfit / max(uv_yfit), "b--", label="Visible Gaussian Fit (norm.)")
            if ir_xfit is not None:
                plt.plot(ir_xfit, ir_yfit / max(ir_yfit), "m--", label="IR Gaussian Fit (norm.)")

            plt.axvline(uv_vis_x, color="b", linestyle=":", label="Visible Peak")
            plt.axvline(ir_x, color="m", linestyle=":", label="IR Peak")

            plt.title(f"{spectrum.genus} {spectrum.species} {spectrum.collection.sex_lookup(spectrum.code, self.collection_list)} {spectrum.code}. {spectrum.get_equipment()}")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Reflectance (normalized)")
            plt.legend()
            plt.grid()
            plt.show()

        if ir_max == 0:
            return np.nan
        return uv_vis_max / ir_max

    @staticmethod
    def description():
        return (
            "This algorithm calculates the ratio between Gaussian-fitted peaks "
            "in the visible and IR wavelength ranges, and plots the fits alongside "
            "the normalized spectrum."
        )


class Gamma_Area_Under_Curve_Naive(Metric):
    name = "Gamma_Area_Under_Curve_Naive"
    debug = False
    def __init__(self, spectrum, config_file: str, collection_list,plot = None):
        # Call parent constructor 
        self.plot = plot
        
        self.spectrum = spectrum
        self._load_config(config_file)
        self.collection_list = collection_list
        self.metric_value = self.set_metric_value(spectrum)
        
        
    def _load_config(self, config_file):
        debug = False
        if debug:
            print("READING:", config_file)
        config = configparser.ConfigParser()
        config.read(config_file)
        if debug:
            print("Available sections:", config.sections())
        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section, using defaults.", UserWarning)
            cfg = {}
        else:
            cfg = config[self.name] 
            if debug:
                print("cfg",cfg)
        self.visible_start_wavelength = cfg.getfloat("visible_start_wavelength", None)
        self.visible_end_wavelength = cfg.getfloat("visible_end_wavelength", None)
        self.ir_start_wavelength = cfg.getfloat("ir_start_wavelength", None)
        self.ir_end_wavelength = cfg.getfloat("ir_end_wavelength", None)
        self.start_wavelength = cfg.getfloat("start_wavelength", None)
        self.end_wavelength = cfg.getfloat("end_wavelength", None)
        self.height_top_threshold_for_minimum = cfg.getfloat("height_top_threshold_for_minimum", None)
        self.smallest_distance_between_peaks_for_min = cfg.getfloat("smallest_distance_between_peaks_for_min", None)
        if debug:
            self.print_parameters()
    
    def print_parameters(self):
        print(f"""{self.visible_start_wavelength=}
        {self.visible_end_wavelength=}
        {self.ir_start_wavelength=}
        {self.ir_end_wavelength=}
        {self.start_wavelength=}
        {self.end_wavelength=}""")
    def set_metric_value(self, spectrum, debug=False):
        debug = False
        df = spectrum.get_normalized_spectrum(min_wavelength= self.start_wavelength, max_wavelength = self.end_wavelength)

        def get_area_under_curve(start, finish):
            subset = df[(df["wavelength"] >= start) & (df["wavelength"] <= finish)]
            wavelengths = subset["wavelength"].values
            heights = subset[spectrum.measuring_mode].values
            return np.trapz(heights, wavelengths), subset

        area_uv, subset_uv = get_area_under_curve(self.visible_start_wavelength, self.visible_end_wavelength)
        area_ir, subset_ir = get_area_under_curve(self.ir_start_wavelength, self.ir_end_wavelength)

        if debug or self.plot:
            plt.figure(figsize=(10, 6))
            plt.plot(df["wavelength"], df[spectrum.measuring_mode], label="Normalized Spectrum")
            plt.fill_between(subset_uv["wavelength"], subset_uv[spectrum.measuring_mode], alpha=0.5, color="skyblue", label="Visible Area")
            plt.fill_between(subset_ir["wavelength"], subset_ir[spectrum.measuring_mode], alpha=0.5, color="orange", label="IR Area")
            plt.legend();
            plt.title(f"{spectrum.genus} {spectrum.species} {spectrum.collection.sex_lookup(spectrum.code, self.collection_list)} {spectrum.code}. {spectrum.get_equipment()}")
            plt.xlabel("wavelength"); plt.ylabel("Reflectance");
            plt.grid()
            plt.show()

        return area_ir / area_uv if area_uv != 0 else np.nan

class Gamma_Area_Under_Curve_Naive_Golden(Metric):
    name = "Gamma_Area_Under_Curve_Naive_Golden"
    debug = False
    def __init__(self, spectrum, config_file: str, collection_list,plot = None):
        # Call parent constructor 
        self.plot = plot
        self.spectrum = spectrum
        self._load_config(config_file)
        self.collection_list = collection_list
        self.metric_value = self.set_metric_value(spectrum)
        
        
    def _load_config(self, config_file):
        debug = False
        if debug:
            print("READING:", config_file)
        config = configparser.ConfigParser()
        config.read(config_file)
        if debug:
            print("Available sections:", config.sections())
        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section, using defaults.", UserWarning)
            cfg = {}
        else:
            cfg = config[self.name] 
            if debug:
                print("cfg",cfg)
        self.start_wavelength = cfg.getfloat("start_wavelength", None)
        self.end_wavelength = cfg.getfloat("end_wavelength", None)
        self.visible_start_wavelength = cfg.getfloat("visible_start_wavelength", None)
        self.visible_end_wavelength = cfg.getfloat("visible_end_wavelength", None)
        self.ir_start_wavelength = cfg.getfloat("ir_start_wavelength", None)
        self.ir_end_wavelength = cfg.getfloat("ir_end_wavelength", None)
        

    def set_metric_value(self, spectrum, debug=False):
        debug = False
        df = spectrum.get_normalized_spectrum(min_wavelength= self.start_wavelength, max_wavelength = self.end_wavelength)

        def get_area_under_curve(start, finish):
            subset = df[(df["wavelength"] >= start) & (df["wavelength"] <= finish)]
            wavelengths = subset["wavelength"].values
            heights = subset[spectrum.measuring_mode].values
            return np.trapz(heights, wavelengths), subset

        area_uv, subset_uv = get_area_under_curve(self.visible_start_wavelength, self.visible_end_wavelength)
        area_ir, subset_ir = get_area_under_curve(self.ir_start_wavelength, self.ir_end_wavelength)
        

        if area_uv and area_ir:
            if area_uv <= 0.0:
                warnings.warn(
                    f"Gamma_Area_Under_Curve_Naive_Golden: area_uv is zero or less ({area_uv})",
                    RuntimeWarning
                )
                metric_value = np.nan
            else:
                metric_value = area_ir / area_uv
                return metric_value

        if debug or self.plot:
            plt.figure(figsize=(10, 6))
            plt.plot(df["wavelength"], df[spectrum.measuring_mode], label="Normalized Spectrum")

            # Fill areas
            plt.fill_between(
                subset_uv["wavelength"], 
                subset_uv[spectrum.measuring_mode], 
                alpha=0.5, color="skyblue", label="Visible Area"
            )
            plt.fill_between(
                subset_ir["wavelength"], 
                subset_ir[spectrum.measuring_mode], 
                alpha=0.5, color="orange", label="IR Area"
            )

            # Metric value
            metric_val = area_ir / area_uv if area_uv != 0 else np.nan
            plt.text(
                0.98, 0.95, f"Metric = {metric_val:.4f}",
                ha="right", va="top",
                transform=plt.gca().transAxes,
                fontsize=12, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
            )

            plt.legend(loc="best")
            plt.title(
                f"{spectrum.genus} {spectrum.species} "
                f"{spectrum.collection.sex_lookup(spectrum.code, self.collection_list)} "
                f"{spectrum.code}. {spectrum.get_equipment()}"
            )
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Reflectance")
            plt.grid()
            plt.show()

class Gamma_Area_Under_Curve_Naive_Silver(Metric):
    name = "Gamma_Area_Under_Curve_Naive_Silver"
    debug = False
    def __init__(self, spectrum, config_file: str, collection_list,plot = None):
        # Call parent constructor 
        self.plot = plot
        self.spectrum = spectrum
        self._load_config(config_file)
        self.collection_list = collection_list
        self.metric_value = self.set_metric_value(spectrum)
        
        
    def _load_config(self, config_file):
        debug = False
        if debug:
            print("READING:", config_file)
        config = configparser.ConfigParser()
        config.read(config_file)
        if debug:
            print("Available sections:", config.sections())
        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section, using defaults.", UserWarning)
            cfg = {}
        else:
            cfg = config[self.name] 
            if debug:
                print("cfg",cfg)
        self.start_wavelength = cfg.getfloat("start_wavelength", None)
        self.end_wavelength = cfg.getfloat("end_wavelength", None)
        self.visible_start_wavelength = cfg.getfloat("visible_start_wavelength", None)
        self.visible_end_wavelength = cfg.getfloat("visible_end_wavelength", None)
        self.ir_start_wavelength = cfg.getfloat("ir_start_wavelength", None)
        self.ir_end_wavelength = cfg.getfloat("ir_end_wavelength", None)
        

    def set_metric_value(self, spectrum, debug=False):
        debug = False
        df = spectrum.get_normalized_spectrum(min_wavelength= self.start_wavelength, max_wavelength = self.end_wavelength)

        def get_area_under_curve(start, finish):
            subset = df[(df["wavelength"] >= start) & (df["wavelength"] <= finish)]
            wavelengths = subset["wavelength"].values
            heights = subset[spectrum.measuring_mode].values
            return np.trapz(heights, wavelengths), subset

        area_uv, subset_uv = get_area_under_curve(self.visible_start_wavelength, self.visible_end_wavelength)
        area_ir, subset_ir = get_area_under_curve(self.ir_start_wavelength, self.ir_end_wavelength)
        

        if area_uv and area_ir:
            if area_uv <= 0.0:
                warnings.warn(
                    f"Gamma_Area_Under_Curve_Naive_Golden: area_uv is zero or less ({area_uv})",
                    RuntimeWarning
                )
                metric_value = np.nan
            else:
                metric_value = area_ir / area_uv
                return metric_value

        if debug or self.plot:
            plt.figure(figsize=(10, 6))
            plt.plot(df["wavelength"], df[spectrum.measuring_mode], label="Normalized Spectrum")

            # Fill areas
            plt.fill_between(
                subset_uv["wavelength"], 
                subset_uv[spectrum.measuring_mode], 
                alpha=0.5, color="skyblue", label="Visible Area"
            )
            plt.fill_between(
                subset_ir["wavelength"], 
                subset_ir[spectrum.measuring_mode], 
                alpha=0.5, color="orange", label="IR Area"
            )

            # Metric value
            metric_val = area_ir / area_uv if area_uv != 0 else np.nan
            plt.text(
                0.98, 0.95, f"Metric = {metric_val:.4f}",
                ha="right", va="top",
                transform=plt.gca().transAxes,
                fontsize=12, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
            )

            plt.legend(loc="best")
            plt.title(
                f"{spectrum.genus} {spectrum.species} "
                f"{spectrum.collection.sex_lookup(spectrum.code, self.collection_list)} "
                f"{spectrum.code}. {spectrum.get_equipment()}"
            )
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Reflectance")
            plt.grid()
            plt.show()
            
class Gamma_Area_Under_Curve_First_Min_Cut(Metric):
    name = "Gamma_Area_Under_Curve_First_Min_Cut"
    debug = False

    def __init__(self, spectrum, config_file: str, collection_list,plot):
        # Call parent constructor 
        self.plot = plot
        self.spectrum = spectrum
        self._load_config(config_file)
        self.collection_list = collection_list
        self.metric_value = self.set_metric_value(spectrum)
        
        
        
    def _load_config(self, config_file):
        debug = False
        config = configparser.ConfigParser()
        config.read(config_file)
        if debug:
            print("Available sections:", config.sections())
        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section, using defaults.", UserWarning)
            cfg = {}
        else:
            cfg = config[self.name] 
            if debug:
                print("_load_config: cfg",cfg)
            
        # wavelength ranges
        self.visible_range_start_wavelength = cfg.getfloat("visible_range_start_wavelength", fallback=450.0)
        self.start_wavelength = cfg.getfloat("start_wavelength", fallback=None)
        self.end_wavelength = cfg.getfloat("end_wavelength", fallback=None)

        # thresholds for prominence and height
        self.prominence_threshold_for_min = cfg.getfloat("prominence_threshold_for_min", fallback=None)
        self.prominence_threshold_for_max = cfg.getfloat("prominence_threshold_for_max", fallback=None)
        self.height_bottom_threshold_for_minimum = cfg.getfloat("height_bottom_threshold_for_minimum", fallback=None)
        self.height_bottom_threshold_for_maximum = cfg.getfloat("height_bottom_threshold_for_maximum", fallback=None)
        self.height_top_threshold_for_maximum = cfg.getfloat("height_top_threshold_for_maximum", fallback=None)
        self.height_top_threshold_for_minimum = cfg.getfloat("height_top_threshold_for_minimum", fallback=None)
        
        # distance between peaks
        self.smallest_distance_between_peaks_for_min = cfg.getfloat("smallest_distance_between_peaks_for_min", fallback=None)
        self.smallest_distance_between_peaks_for_max = cfg.getfloat("smallest_distance_between_peaks_for_max", fallback=None)

    def set_metric_value(self, spectrum):
        debug = False
        if debug:
            print("Gamma_Area_Under_Curve_First_Min_Cut min",self.start_wavelength,"max",self.end_wavelength)
        df = spectrum.get_normalized_spectrum(min_wavelength= self.start_wavelength, max_wavelength = self.end_wavelength)
        if debug:
            print("normalized:", df)
        x = df["wavelength"].values
        y = df["%R"].values 
        if debug:
            print("get_maxima")
        max_i, max_xs, _ = spectrum.get_maxima(height_bottom_threshold_for_maximum = self.height_bottom_threshold_for_maximum, 
                                               min_wavelength = self.start_wavelength, 
                                               max_wavelength = self.end_wavelength, 
                                               height_top_threshold_for_maximum = self.height_top_threshold_for_maximum, 
                                               smallest_distance_between_peaks_for_max = self.smallest_distance_between_peaks_for_max)
        if debug:
        
            print("max_i, max_xs, _ ", max_i, max_xs, _ )
            print("get_minima")
        min_i, min_xs, _ = spectrum.get_minima(height_bottom_threshold_for_minimum = self.height_bottom_threshold_for_minimum,
                                               min_wavelength = self.start_wavelength, 
                                               max_wavelength = self.end_wavelength,
                                               height_top_threshold_for_minimum = self.height_top_threshold_for_minimum,
                                               smallest_distance_between_peaks_for_min = self.smallest_distance_between_peaks_for_min
                                               )
        if debug:
            print(" min_i, min_xs, _ ",  min_i, min_xs, _  )
        if len(max_xs) < 1:
            return np.nan

        first_max_x = max_xs[0]
        second_max_x = max_xs[1] if len(max_xs) > 1 else x.max()

        min_in_between_x = next((m for m in min_xs if first_max_x <= m <= second_max_x), None)
        if not min_in_between_x:
            min_in_between_x = next((m for m in min_xs if m > second_max_x), None)
        if not min_in_between_x:
            return np.nan
        if min_in_between_x <= self.visible_range_start_wavelength:
            min_in_between_x = start_wavelength

        min_after_second_max_x = next((m for m in min_xs if m > min_in_between_x and m > second_max_x), None)
        if not min_after_second_max_x:
            return np.nan

        def get_area(start, end):
            subset = df[(df["wavelength"] >= start) & (df["wavelength"] <= end)]
            return np.trapz(subset[spectrum.metadata["measuring_mode"]], subset["wavelength"]) if not subset.empty else 0

        area_uv = get_area(self.visible_range_start_wavelength, min_in_between_x)
        area_ir = get_area(min_in_between_x, min_after_second_max_x)
        gamma = area_ir / area_uv if area_uv != 0 else np.nan
        print("GAMMA", gamma)
        if debug or self.plot:
            print("PLOTTING")
            # Optional plotting for debugging
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, label="Normalized Spectrum")
            plt.axvline(x=first_max_x, color='r', linestyle='--', label='First Max')
            plt.axvline(x=second_max_x, color='g', linestyle='--', label='Second Max')
            plt.axvline(x=min_in_between_x, color='b', linestyle='--', label='First Min')
            plt.axvline(x=min_after_second_max_x, color='m', linestyle='--', label='Second Min')
            plt.title(f"Spectrum {spectrum.get_filename()}")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel(spectrum.metadata["measuring_mode"])
            plt.title(f"{spectrum.genus} {spectrum.species} {spectrum.collection.sex_lookup(spectrum.code, self.collection_list)} {spectrum.code}. {spectrum.get_equipment()}")
            plt.legend()
            plt.show()
            plt.grid()
            plt.fill_between(df["wavelength"], y, where=(df["wavelength"] >= self.visible_range_start_wavelength) & (df["wavelength"] <= min_in_between_x),
                 alpha=0.5, color="skyblue", label="Visible Area")
            plt.fill_between(df["wavelength"], y, where=(df["wavelength"] >= min_in_between_x) & (df["wavelength"] <= min_after_second_max_x),
                             alpha=0.5, color="orange", label="IR Area")

            
        return gamma


# -------------------------------------------------------------------------
# Metrics without wavelength parameters (but still accept config_file)
# -------------------------------------------------------------------------

import numpy as np
import warnings

class Gamma_Vector_Relative_Reflectance_old(Metric):
    name = "Gamma_Vector_Relative_Reflectance"
    
    def __init__(self, spectrum, config_file: str, collection_list, plot=None):
        # Call parent constructor
        self.plot = plot
        self.spectrum = spectrum
        self._load_config(config_file)
        self.collection_list = collection_list
        self.metric_value = self.set_metric_value(spectrum)
    
    def _load_config(self, config_file):
        debug = False
        config = configparser.ConfigParser()
        config.read(config_file)
        if debug:
            print("Available sections:", config.sections())
        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section, using defaults.", UserWarning)
            cfg = {}
        else:
            cfg = config[self.name] 
            if debug:
                print("_load_config: cfg",cfg)
            
        # wavelength ranges
        self.visible_range_start_wavelength = cfg.getfloat("visible_range_start_wavelength", fallback=450.0)
        self.start_wavelength = cfg.getfloat("start_wavelength", fallback=None)
        self.end_wavelength = cfg.getfloat("end_wavelength", fallback=None)

        # thresholds for prominence and height
        self.prominence_threshold_for_min = cfg.getfloat("prominence_threshold_for_min", fallback=None)
        self.prominence_threshold_for_max = cfg.getfloat("prominence_threshold_for_max", fallback=None)
        self.height_bottom_threshold_for_minimum = cfg.getfloat("height_bottom_threshold_for_minimum", fallback=None)
        self.height_bottom_threshold_for_maximum = cfg.getfloat("height_bottom_threshold_for_maximum", fallback=None)
        self.height_top_threshold_for_maximum = cfg.getfloat("height_top_threshold_for_maximum", fallback=None)
        self.height_top_threshold_for_minimum = cfg.getfloat("height_top_threshold_for_minimum", fallback=None)
        
        # distance between peaks
        self.smallest_distance_between_peaks_for_min = cfg.getfloat("smallest_distance_between_peaks_for_min", fallback=None)
        self.smallest_distance_between_peaks_for_max = cfg.getfloat("smallest_distance_between_peaks_for_max", fallback=None)
        
    def set_metric_value(self, spectrum):
        _, _, max_y = spectrum.get_maxima(height_bottom_threshold_for_maximum = self.height_bottom_threshold_for_maximum, 
                                               min_wavelength = self.start_wavelength, 
                                               max_wavelength = self.end_wavelength, 
                                               height_top_threshold_for_maximum = self.height_top_threshold_for_maximum, 
                                               smallest_distance_between_peaks_for_max = self.smallest_distance_between_peaks_for_max)

        if max_y is None or len(max_y) == 0:
            warnings.warn(
                f"[Gamma_Vector_Relative_Reflectance] No maxima found for spectrum "
                f"{getattr(spectrum, 'filename', 'unknown')}. Returning NaN vector.",
                UserWarning
            )
            # Return a placeholder so shape is consistent
            return np.array([np.nan])

        # Normalize by the first maximum
        return np.array(max_y) / max_y[0]


class Gamma_Vector_Relative_Reflectance(Metric):
    name = "Gamma_Vector_Relative_Reflectance"
    
    def __init__(self, spectrum, config_file: str, collection_list, plot=None):
        self.plot = plot
        self.spectrum = spectrum
        self._load_config(config_file)
        self.collection_list = collection_list
        self.metric_value = self.set_metric_value(spectrum)
    
    def _load_config(self, config_file):
        debug = False
        config = configparser.ConfigParser()
        config.read(config_file)
        if debug:
            print("Available sections:", config.sections())
        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section, using defaults.", UserWarning)
            cfg = {}
        else:
            cfg = config[self.name] 
            if debug:
                print("_load_config: cfg",cfg)
            
        # wavelength ranges
        self.visible_range_start_wavelength = cfg.getfloat("visible_range_start_wavelength", fallback=450.0)
        self.start_wavelength = cfg.getfloat("start_wavelength", fallback=None)
        self.end_wavelength = cfg.getfloat("end_wavelength", fallback=None)

        # thresholds for prominence and height
        self.prominence_threshold_for_min = cfg.getfloat("prominence_threshold_for_min", fallback=None)
        self.prominence_threshold_for_max = cfg.getfloat("prominence_threshold_for_max", fallback=None)
        self.height_bottom_threshold_for_minimum = cfg.getfloat("height_bottom_threshold_for_minimum", fallback=None)
        self.height_bottom_threshold_for_maximum = cfg.getfloat("height_bottom_threshold_for_maximum", fallback=None)
        self.height_top_threshold_for_maximum = cfg.getfloat("height_top_threshold_for_maximum", fallback=None)
        self.height_top_threshold_for_minimum = cfg.getfloat("height_top_threshold_for_minimum", fallback=None)
        
        # distance between peaks
        self.smallest_distance_between_peaks_for_min = cfg.getfloat("smallest_distance_between_peaks_for_min", fallback=None)
        self.smallest_distance_between_peaks_for_max = cfg.getfloat("smallest_distance_between_peaks_for_max", fallback=None)
        
    def set_metric_value(self, spectrum, debug=False):
        _, _, max_y = spectrum.get_maxima(
            height_bottom_threshold_for_maximum=self.height_bottom_threshold_for_maximum, 
            min_wavelength=self.start_wavelength, 
            max_wavelength=self.end_wavelength, 
            height_top_threshold_for_maximum=self.height_top_threshold_for_maximum, 
            smallest_distance_between_peaks_for_max=self.smallest_distance_between_peaks_for_max
        )

        if max_y is None or len(max_y) == 0:
            warnings.warn(
                f"[Gamma_Vector_Relative_Reflectance] No maxima found for spectrum "
                f"{getattr(spectrum, 'filename', 'unknown')}. Returning NaN vector.",
                UserWarning
            )
            return np.array([np.nan])

        import matplotlib.pyplot as plt

        # --- Debug/Plot logic ---
        if debug or self.plot:
            df = spectrum.get_normalized_spectrum(
                min_wavelength=self.start_wavelength, 
                max_wavelength=self.end_wavelength
            )
            
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Left axis: normalized spectrum
            ax1.plot(df["wavelength"], df[spectrum.measuring_mode], label="Normalized Spectrum", color="blue")
            ax1.set_xlabel("Wavelength (nm)")
            ax1.set_ylabel("Normalized Reflectance", color="blue")
            ax1.tick_params(axis="y", labelcolor="blue")
            ax1.grid()

            # Right axis: maxima values
            ax2 = ax1.twinx()
            for y in max_y:
                ax2.axhline(y, linestyle="--", alpha=0.6, color="red", label="Maxima" if y == max_y[0] else "")
            ax2.set_ylabel("Absolute Reflectance (raw maxima)", color="red")
            ax2.tick_params(axis="y", labelcolor="red")

            # Title + legend
            fig.suptitle(
                f"{self.name}: {spectrum.genus} {spectrum.species} "
                f"{spectrum.collection.sex_lookup(spectrum.code, self.collection_list)} "
                f"{spectrum.code}. {spectrum.get_equipment()}"
            )

            fig.legend(loc="upper right")
            plt.show()






class Wavelength_Vector(Metric):
    name = "Wavelength_Vector"

    def __init__(self, spectrum, config_file: str, collection_list,plot = None):
        # Call parent constructor 
        
        self.plot = plot
        self.spectrum = spectrum
        self._load_config(config_file)
        self.collection_list = collection_list
        self.metric_value = self.set_metric_value(spectrum)
    def set_metric_value(self, spectrum):
        _, max_x, _ = spectrum.get_maxima(height_bottom_threshold_for_maximum = self.height_bottom_threshold_for_maximum, 
                                               min_wavelength = self.start_wavelength, 
                                               max_wavelength = self.end_wavelength, 
                                               height_top_threshold_for_maximum = self.height_top_threshold_for_maximum, 
                                               smallest_distance_between_peaks_for_max = self.smallest_distance_between_peaks_for_max)
        return np.array(list(max_x))
        
    def _load_config(self, config_file):
        debug = False
        config = configparser.ConfigParser()
        config.read(config_file)
        if debug:
            print("Available sections:", config.sections())
        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section, using defaults.", UserWarning)
            cfg = {}
        else:
            cfg = config[self.name] 
            if debug:
                print("_load_config: cfg",cfg)
            
        # wavelength ranges
        self.visible_range_start_wavelength = cfg.getfloat("visible_range_start_wavelength", fallback=450.0)
        self.start_wavelength = cfg.getfloat("start_wavelength", fallback=None)
        self.end_wavelength = cfg.getfloat("end_wavelength", fallback=None)

        # thresholds for prominence and height
        self.prominence_threshold_for_min = cfg.getfloat("prominence_threshold_for_min", fallback=None)
        self.prominence_threshold_for_max = cfg.getfloat("prominence_threshold_for_max", fallback=None)
        self.height_bottom_threshold_for_minimum = cfg.getfloat("height_bottom_threshold_for_minimum", fallback=None)
        self.height_bottom_threshold_for_maximum = cfg.getfloat("height_bottom_threshold_for_maximum", fallback=None)
        self.height_top_threshold_for_maximum = cfg.getfloat("height_top_threshold_for_maximum", fallback=None)
        self.height_top_threshold_for_minimum = cfg.getfloat("height_top_threshold_for_minimum", fallback=None)
        
        # distance between peaks
        self.smallest_distance_between_peaks_for_min = cfg.getfloat("smallest_distance_between_peaks_for_min", fallback=None)
        self.smallest_distance_between_peaks_for_max = cfg.getfloat("smallest_distance_between_peaks_for_max", fallback=None)


class Critical_Points(Metric):
    name = "Critical_Points"
    
    def __init__(self, spectrum, config_file: str, collection_list,plot = None):
        # Call parent constructor 
        self.plot = plot
        self.spectrum = spectrum
        self._load_config(config_file)
        self.collection_list = collection_list
        self.metric_value = self.set_metric_value(spectrum)
        
        
    def set_metric_value(self, spectrum, debug=False):
        _, min_x, min_y = spectrum.get_minima(
            height_bottom_threshold_for_minimum=self.height_bottom_threshold_for_minimum,
            min_wavelength=self.start_wavelength, 
            max_wavelength=self.end_wavelength,
            height_top_threshold_for_minimum=self.height_top_threshold_for_minimum,
            smallest_distance_between_peaks_for_min=self.smallest_distance_between_peaks_for_min
        )
        _, max_x, max_y = spectrum.get_maxima(
            height_bottom_threshold_for_maximum=self.height_bottom_threshold_for_maximum, 
            min_wavelength=self.start_wavelength, 
            max_wavelength=self.end_wavelength, 
            height_top_threshold_for_maximum=self.height_top_threshold_for_maximum, 
            smallest_distance_between_peaks_for_max=self.smallest_distance_between_peaks_for_max
        )

        if debug or self.plot:
            df = spectrum.get_normalized_spectrum(
                min_wavelength=self.start_wavelength, 
                max_wavelength=self.end_wavelength
            )
            plt.figure(figsize=(10, 6))
            plt.plot(df["wavelength"], df[spectrum.metadata["measuring_mode"]], label="Spectrum")
            for x in max_x: plt.axvline(x, color="r", linestyle="--", alpha=0.7, label="Max")
            for x in min_x: plt.axvline(x, color="b", linestyle="--", alpha=0.7, label="Min")
            plt.title(f"{self.name}: {spectrum.genus} {spectrum.species} "
                      f"{spectrum.collection.sex_lookup(spectrum.code, self.collection_list)} "
                      f"{spectrum.code}. {spectrum.get_equipment()}")
            plt.xlabel("Wavelength"); plt.ylabel("Reflectance")
            plt.legend(); plt.grid(); plt.show()

        return np.array([np.concatenate((min_x, max_x)), np.concatenate((min_y, max_y))])

        
    def _load_config(self, config_file):
        debug = False
        config = configparser.ConfigParser()
        config.read(config_file)
        if debug:
            print("Available sections:", config.sections())
        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section, using defaults.", UserWarning)
            cfg = {}
        else:
            cfg = config[self.name] 
            if debug:
                print("_load_config: cfg",cfg)
            
        # wavelength ranges
        self.visible_range_start_wavelength = cfg.getfloat("visible_range_start_wavelength", fallback=450.0)
        self.start_wavelength = cfg.getfloat("start_wavelength", fallback=None)
        self.end_wavelength = cfg.getfloat("end_wavelength", fallback=None)

        # thresholds for prominence and height
        self.prominence_threshold_for_min = cfg.getfloat("prominence_threshold_for_min", fallback=None)
        self.prominence_threshold_for_max = cfg.getfloat("prominence_threshold_for_max", fallback=None)
        self.height_bottom_threshold_for_minimum = cfg.getfloat("height_bottom_threshold_for_minimum", fallback=None)
        self.height_bottom_threshold_for_maximum = cfg.getfloat("height_bottom_threshold_for_maximum", fallback=None)
        self.height_top_threshold_for_maximum = cfg.getfloat("height_top_threshold_for_maximum", fallback=None)
        self.height_top_threshold_for_minimum = cfg.getfloat("height_top_threshold_for_minimum", fallback=None)
        
        # distance between peaks
        self.smallest_distance_between_peaks_for_min = cfg.getfloat("smallest_distance_between_peaks_for_min", fallback=None)
        self.smallest_distance_between_peaks_for_max = cfg.getfloat("smallest_distance_between_peaks_for_max", fallback=None)

class Critical_Points_Golden(Metric):
    name = "Critical_Points_Golden"
    
    def __init__(self, spectrum, config_file: str, collection_list,plot = None):
        
        self.plot = plot
        self.spectrum = spectrum
        self._load_config(config_file)
        self.collection_list = collection_list
        self.metric_value = self.set_metric_value(spectrum)
        
        
    def set_metric_value(self, spectrum, debug=False):
        _, min_x, min_y = spectrum.get_minima(
            height_bottom_threshold_for_minimum=self.height_bottom_threshold_for_minimum,
            min_wavelength=self.start_wavelength, 
            max_wavelength=self.end_wavelength,
            height_top_threshold_for_minimum=self.height_top_threshold_for_minimum,
            smallest_distance_between_peaks_for_min=self.smallest_distance_between_peaks_for_min
        )
        _, max_x, max_y = spectrum.get_maxima(
            height_bottom_threshold_for_maximum=self.height_bottom_threshold_for_maximum, 
            min_wavelength=self.start_wavelength, 
            max_wavelength=self.end_wavelength, 
            height_top_threshold_for_maximum=self.height_top_threshold_for_maximum, 
            smallest_distance_between_peaks_for_max=self.smallest_distance_between_peaks_for_max
        )

        if debug or self.plot:
            df = spectrum.get_normalized_spectrum(
                min_wavelength=self.start_wavelength, 
                max_wavelength=self.end_wavelength
            )
            plt.figure(figsize=(10, 6))
            plt.plot(df["wavelength"], df[spectrum.metadata["measuring_mode"]], label="Spectrum")
            for x in max_x: plt.axvline(x, color="r", linestyle="--", alpha=0.7, label="Max")
            for x in min_x: plt.axvline(x, color="b", linestyle="--", alpha=0.7, label="Min")
            plt.title(f"{self.name}: {spectrum.genus} {spectrum.species} "
                      f"{spectrum.collection.sex_lookup(spectrum.code, self.collection_list)} "
                      f"{spectrum.code}. {spectrum.get_equipment()}")
            plt.xlabel("Wavelength"); plt.ylabel("Reflectance")
            plt.legend(); plt.grid(); plt.show()

        return np.array([np.concatenate((min_x, max_x)), np.concatenate((min_y, max_y))])

    def _load_config(self, config_file):
        debug = False
        config = configparser.ConfigParser()
        config.read(config_file)
        if debug:
            print("Available sections:", config.sections())
        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section, using defaults.", UserWarning)
            cfg = {}
        else:
            cfg = config[self.name] 
            if debug:
                print("_load_config: cfg",cfg)
            
        # wavelength ranges
        self.visible_start_wavelength = cfg.getfloat("visible_start_wavelength", fallback=None)
        self.visible_end_wavelength = cfg.getfloat("visible_end_wavelength", fallback=None)
        self.ir_start_wavelength = cfg.getfloat("ir_start_wavelength", fallback=None)
        self.ir_end_wavelength = cfg.getfloat("ir_end_wavelength", fallback=None)
        self.start_wavelength = cfg.getfloat("start_wavelength", fallback=None)
        self.end_wavelength = cfg.getfloat("end_wavelength", fallback=None)

        # thresholds for prominence and height
        self.prominence_threshold_for_min = cfg.getfloat("prominence_threshold_for_min", fallback=None)
        self.prominence_threshold_for_max = cfg.getfloat("prominence_threshold_for_max", fallback=None)
        self.height_bottom_threshold_for_minimum = cfg.getfloat("height_bottom_threshold_for_minimum", fallback=None)
        self.height_bottom_threshold_for_maximum = cfg.getfloat("height_bottom_threshold_for_maximum", fallback=None)
        self.height_top_threshold_for_maximum = cfg.getfloat("height_top_threshold_for_maximum", fallback=None)
        self.height_top_threshold_for_minimum = cfg.getfloat("height_top_threshold_for_minimum", fallback=None)
        
        # distance between peaks
        self.smallest_distance_between_peaks_for_min = cfg.getfloat("smallest_distance_between_peaks_for_min", fallback=None)
        self.smallest_distance_between_peaks_for_max = cfg.getfloat("smallest_distance_between_peaks_for_max", fallback=None)
        if debug:
            self.print_parameters()
        
    
    def print_parameters(self):
        print(f"""{self.visible_start_wavelength=}
        {self.visible_end_wavelength=}
        {self.ir_start_wavelength=}
        {self.ir_end_wavelength=}
        {self.start_wavelength=}
        {self.end_wavelength=}""")
        
class Critical_Points_Silver(Metric):
    name = "Critical_Points_Silver"
    
    def __init__(self, spectrum, config_file: str, collection_list,plot = None):
        
        self.plot = plot
        self.spectrum = spectrum
        self._load_config(config_file)
        self.collection_list = collection_list
        self.metric_value = self.set_metric_value(spectrum)
        
        
    def set_metric_value(self, spectrum, debug=False):
        _, min_x, min_y = spectrum.get_minima(
            height_bottom_threshold_for_minimum=self.height_bottom_threshold_for_minimum,
            min_wavelength=self.start_wavelength, 
            max_wavelength=self.end_wavelength,
            height_top_threshold_for_minimum=self.height_top_threshold_for_minimum,
            smallest_distance_between_peaks_for_min=self.smallest_distance_between_peaks_for_min
        )
        _, max_x, max_y = spectrum.get_maxima(
            height_bottom_threshold_for_maximum=self.height_bottom_threshold_for_maximum, 
            min_wavelength=self.start_wavelength, 
            max_wavelength=self.end_wavelength, 
            height_top_threshold_for_maximum=self.height_top_threshold_for_maximum, 
            smallest_distance_between_peaks_for_max=self.smallest_distance_between_peaks_for_max
        )

        if debug or self.plot:
            df = spectrum.get_normalized_spectrum(
                min_wavelength=self.start_wavelength, 
                max_wavelength=self.end_wavelength
            )
            plt.figure(figsize=(10, 6))
            plt.plot(df["wavelength"], df[spectrum.metadata["measuring_mode"]], label="Spectrum")
            for x in max_x: plt.axvline(x, color="r", linestyle="--", alpha=0.7, label="Max")
            for x in min_x: plt.axvline(x, color="b", linestyle="--", alpha=0.7, label="Min")
            plt.title(f"{self.name}: {spectrum.genus} {spectrum.species} "
                      f"{spectrum.collection.sex_lookup(spectrum.code, self.collection_list)} "
                      f"{spectrum.code}. {spectrum.get_equipment()}")
            plt.xlabel("Wavelength"); plt.ylabel("Reflectance")
            plt.legend(); plt.grid(); plt.show()

        return np.array([np.concatenate((min_x, max_x)), np.concatenate((min_y, max_y))])

    def _load_config(self, config_file):
        debug = False
        config = configparser.ConfigParser()
        config.read(config_file)
        if debug:
            print("Available sections:", config.sections())
        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section, using defaults.", UserWarning)
            cfg = {}
        else:
            cfg = config[self.name] 
            if debug:
                print("_load_config: cfg",cfg)
            
        # wavelength ranges
        self.visible_start_wavelength = cfg.getfloat("visible_start_wavelength", fallback=None)
        self.visible_end_wavelength = cfg.getfloat("visible_end_wavelength", fallback=None)
        self.ir_start_wavelength = cfg.getfloat("ir_start_wavelength", fallback=None)
        self.ir_end_wavelength = cfg.getfloat("ir_end_wavelength", fallback=None)
        self.start_wavelength = cfg.getfloat("start_wavelength", fallback=None)
        self.end_wavelength = cfg.getfloat("end_wavelength", fallback=None)

        # thresholds for prominence and height
        self.prominence_threshold_for_min = cfg.getfloat("prominence_threshold_for_min", fallback=None)
        self.prominence_threshold_for_max = cfg.getfloat("prominence_threshold_for_max", fallback=None)
        self.height_bottom_threshold_for_minimum = cfg.getfloat("height_bottom_threshold_for_minimum", fallback=None)
        self.height_bottom_threshold_for_maximum = cfg.getfloat("height_bottom_threshold_for_maximum", fallback=None)
        self.height_top_threshold_for_maximum = cfg.getfloat("height_top_threshold_for_maximum", fallback=None)
        self.height_top_threshold_for_minimum = cfg.getfloat("height_top_threshold_for_minimum", fallback=None)
        
        # distance between peaks
        self.smallest_distance_between_peaks_for_min = cfg.getfloat("smallest_distance_between_peaks_for_min", fallback=None)
        self.smallest_distance_between_peaks_for_max = cfg.getfloat("smallest_distance_between_peaks_for_max", fallback=None)
        if debug:
            self.print_parameters()
        
    def print_parameters(self):
        print(f"""{self.visible_start_wavelength=}
        {self.visible_end_wavelength=}
        {self.ir_start_wavelength=}
        {self.ir_end_wavelength=}
        {self.ir_end_wavelength=}
        {self.prominence_threshold_for_min=}
        {self.prominence_threshold_for_max=}
        {self.end_wavelength=}""")
        
class Minimum_Points(Metric):
    name = "Minimum_Points"

    def __init__(self, spectrum, config_file: str, collection_list, plot = None):
        # Call parent constructor 
        
        
        self.spectrum = spectrum
        self._load_config(config_file)
        self.collection_list = collection_list
        self.metric_value = self.set_metric_value(spectrum)
        self.plot = plot
        
    def set_metric_value(self, spectrum):
        debug = False
        _, min_x, min_y = spectrum.get_minima(height_bottom_threshold_for_minimum = self.height_bottom_threshold_for_minimum,
                                               min_wavelength = self.start_wavelength, 
                                               max_wavelength = self.end_wavelength,
                                               height_top_threshold_for_minimum = self.height_top_threshold_for_minimum,
                                               smallest_distance_between_peaks_for_min = self.smallest_distance_between_peaks_for_min
                                               )
        if debug or self.plot:
            df = spectrum.get_normalized_spectrum(min_wavelength= self.start_wavelength, max_wavelength = self.end_wavelength)
            plt.figure(figsize=(10, 6))
            plt.plot(df["wavelength"], df[spectrum.metadata["measuring_mode"]], label="Spectrum")
            plt.title(f"{spectrum.genus} {spectrum.species} {spectrum.collection.sex_lookup(spectrum.code, self.collection_list)} {spectrum.code}. {spectrum.get_equipment()}")
            plt.xlabel("wavelength"); plt.ylabel("Reflectance");
            for x in max_x: plt.axvline(x, color="r", linestyle="--", alpha=0.7)
            for x in min_x: plt.axvline(x, color="b", linestyle="--", alpha=0.7)
            plt.legend(); 
            plt.grid()
            plt.show()
        return np.array([min_x, min_y])
        
        
    def _load_config(self, config_file):
        debug = False
        config = configparser.ConfigParser()
        config.read(config_file)
        if debug:
            print("Available sections:", config.sections())
        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section, using defaults.", UserWarning)
            cfg = {}
        else:
            cfg = config[self.name] 
            if debug:
                print("_load_config: cfg",cfg)
            
        # wavelength ranges
        self.visible_range_start_wavelength = cfg.getfloat("visible_range_start_wavelength", fallback=450.0)
        self.start_wavelength = cfg.getfloat("start_wavelength", fallback=None)
        self.end_wavelength = cfg.getfloat("end_wavelength", fallback=None)

        # thresholds for prominence and height
        self.prominence_threshold_for_min = cfg.getfloat("prominence_threshold_for_min", fallback=None)
        self.prominence_threshold_for_max = cfg.getfloat("prominence_threshold_for_max", fallback=None)
        self.height_bottom_threshold_for_minimum = cfg.getfloat("height_bottom_threshold_for_minimum", fallback=None)
        self.height_bottom_threshold_for_maximum = cfg.getfloat("height_bottom_threshold_for_maximum", fallback=None)
        self.height_top_threshold_for_minimum = cfg.getfloat("height_bottom_threshold_for_minimum", fallback=None)
        self.height_top_threshold_for_maximum = cfg.getfloat("height_bottom_threshold_for_maximum", fallback=None)

        # distance between peaks
        self.smallest_distance_between_peaks_for_min = cfg.getfloat("smallest_distance_between_peaks_for_min", fallback=None)
        self.smallest_distance_between_peaks_for_max = cfg.getfloat("smallest_distance_between_peaks_for_max", fallback=None)

class Maximum_Points(Metric):
    name = "Maximum_Points"

    def __init__(self, spectrum, config_file: str, collection_list,plot = None):
        
        
        self.spectrum = spectrum
        self._load_config(config_file)
        self.collection_list = collection_list
        self.metric_value = self.set_metric_value(spectrum)
        self.plot = plot
        
    def set_metric_value(self, spectrum, debug=False):
        _, max_x, max_y = spectrum.get_maxima(
            height_bottom_threshold_for_maximum=self.height_bottom_threshold_for_maximum, 
            min_wavelength=self.start_wavelength, 
            max_wavelength=self.end_wavelength, 
            height_top_threshold_for_maximum=self.height_top_threshold_for_maximum, 
            smallest_distance_between_peaks_for_max=self.smallest_distance_between_peaks_for_max
        )

        if debug or self.plot:
            df = spectrum.get_normalized_spectrum(
                min_wavelength=self.start_wavelength, 
                max_wavelength=self.end_wavelength
            )
            plt.figure(figsize=(10, 6))
            plt.plot(df["wavelength"], df[spectrum.metadata["measuring_mode"]], label="Spectrum")
            for x in max_x: plt.axvline(x, color="r", linestyle="--", alpha=0.7, label="Max")
            plt.title(f"{self.name}: {spectrum.genus} {spectrum.species} "
                      f"{spectrum.collection.sex_lookup(spectrum.code, self.collection_list)} "
                      f"{spectrum.code}. {spectrum.get_equipment()}")
            plt.xlabel("Wavelength"); plt.ylabel("Reflectance")
            plt.legend(); plt.grid(); plt.show()

        return np.array([max_x, max_y])

    def _load_config(self, config_file):
        debug = False
        config = configparser.ConfigParser()
        config.read(config_file)
        if debug:
            print("Available sections:", config.sections())
        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section, using defaults.", UserWarning)
            cfg = {}
        else:
            cfg = config[self.name] 
            if debug:
                print("_load_config: cfg",cfg)
            
        # wavelength ranges
        self.visible_range_start_wavelength = cfg.getfloat("visible_range_start_wavelength", fallback=450.0)
        self.start_wavelength = cfg.getfloat("start_wavelength", fallback=None)
        self.end_wavelength = cfg.getfloat("end_wavelength", fallback=None)

        # thresholds for prominence and height
        self.prominence_threshold_for_min = cfg.getfloat("prominence_threshold_for_min", fallback=None)
        self.prominence_threshold_for_max = cfg.getfloat("prominence_threshold_for_max", fallback=None)
        self.height_bottom_threshold_for_minimum = cfg.getfloat("height_bottom_threshold_for_minimum", fallback=None)
        self.height_bottom_threshold_for_maximum = cfg.getfloat("height_bottom_threshold_for_maximum", fallback=None)
        self.height_top_threshold_for_minimum = cfg.getfloat("height_bottom_threshold_for_minimum", fallback=None)
        self.height_top_threshold_for_maximum = cfg.getfloat("height_bottom_threshold_for_maximum", fallback=None)

        # distance between peaks
        self.smallest_distance_between_peaks_for_min = cfg.getfloat("smallest_distance_between_peaks_for_min", fallback=None)
        self.smallest_distance_between_peaks_for_max = cfg.getfloat("smallest_distance_between_peaks_for_max", fallback=None)

class Minimum_Points_Normalized(Metric):
    name = "Minimum_Points_Normalized"

    def __init__(self, spectrum, config_file: str, collection_list,plot = None):
        
        
        self.spectrum = spectrum
        self._load_config(config_file)
        self.collection_list = collection_list
        self.metric_value = self.set_metric_value(spectrum)
        self.plot = plot
        
    def set_metric_value(self, spectrum, debug=False):
        _, min_x, min_y = spectrum.get_minima(
            height_bottom_threshold_for_minimum=self.height_bottom_threshold_for_minimum,
            min_wavelength=self.start_wavelength, 
            max_wavelength=self.end_wavelength,
            height_top_threshold_for_minimum=self.height_top_threshold_for_minimum,
            smallest_distance_between_peaks_for_min=self.smallest_distance_between_peaks_for_min
        )

        if debug or self.plot:
            df = spectrum.get_normalized_spectrum(
                min_wavelength=self.start_wavelength, 
                max_wavelength=self.end_wavelength
            )
            plt.figure(figsize=(10, 6))
            plt.plot(df["wavelength"], df[spectrum.metadata["measuring_mode"]], label="Spectrum")
            for x in min_x: plt.axvline(x, color="b", linestyle="--", alpha=0.7, label="Min")
            plt.title(f"{self.name}: {spectrum.genus} {spectrum.species} "
                      f"{spectrum.collection.sex_lookup(spectrum.code, self.collection_list)} "
                      f"{spectrum.code}. {spectrum.get_equipment()}")
            plt.xlabel("Wavelength"); plt.ylabel("Reflectance")
            plt.legend(); plt.grid(); plt.show()

        return np.array([min_x, min_y / min_y[0]]) if len(min_y) > 0 else np.array([[], []])

    def _load_config(self, config_file):
        debug = False
        config = configparser.ConfigParser()
        config.read(config_file)
        if debug:
            print("Available sections:", config.sections())
        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section, using defaults.", UserWarning)
            cfg = {}
        else:
            cfg = config[self.name] 
            if debug:
                print("_load_config: cfg",cfg)
            
        # wavelength ranges
        self.visible_range_start_wavelength = cfg.getfloat("visible_range_start_wavelength", fallback=450.0)
        self.start_wavelength = cfg.getfloat("start_wavelength", fallback=None)
        self.end_wavelength = cfg.getfloat("end_wavelength", fallback=None)

        # thresholds for prominence and height
        self.prominence_threshold_for_min = cfg.getfloat("prominence_threshold_for_min", fallback=None)
        self.prominence_threshold_for_max = cfg.getfloat("prominence_threshold_for_max", fallback=None)
        self.height_bottom_threshold_for_minimum = cfg.getfloat("height_bottom_threshold_for_minimum", fallback=None)
        self.height_bottom_threshold_for_maximum = cfg.getfloat("height_bottom_threshold_for_maximum", fallback=None)
        self.height_top_threshold_for_minimum = cfg.getfloat("height_bottom_threshold_for_minimum", fallback=None)
        self.height_top_threshold_for_maximum = cfg.getfloat("height_bottom_threshold_for_maximum", fallback=None)

        # distance between peaks
        self.smallest_distance_between_peaks_for_min = cfg.getfloat("smallest_distance_between_peaks_for_min", fallback=None)
        self.smallest_distance_between_peaks_for_max = cfg.getfloat("smallest_distance_between_peaks_for_max", fallback=None)

class Maximum_Points_Normalized(Metric):
    name = "Maximum_Points_Normalized"

    def __init__(self, spectrum, config_file: str, collection_list,plot = None):
        # Call parent constructor 
        
        
        self.spectrum = spectrum
        self._load_config(config_file)
        self.collection_list = collection_list
        self.metric_value = self.set_metric_value(spectrum)
        self.plot = plot
        
    def set_metric_value(self, spectrum, debug=False):
        _, max_x, max_y = spectrum.get_maxima(
            height_bottom_threshold_for_maximum=self.height_bottom_threshold_for_maximum, 
            min_wavelength=self.start_wavelength, 
            max_wavelength=self.end_wavelength, 
            height_top_threshold_for_maximum=self.height_top_threshold_for_maximum, 
            smallest_distance_between_peaks_for_max=self.smallest_distance_between_peaks_for_max
        )

        if debug or self.plot:
            df = spectrum.get_normalized_spectrum(
                min_wavelength=self.start_wavelength, 
                max_wavelength=self.end_wavelength
            )
            plt.figure(figsize=(10, 6))
            plt.plot(df["wavelength"], df[spectrum.metadata["measuring_mode"]], label="Spectrum")
            for x in max_x: plt.axvline(x, color="r", linestyle="--", alpha=0.7, label="Max")
            plt.title(f"{self.name}: {spectrum.genus} {spectrum.species} "
                      f"{spectrum.collection.sex_lookup(spectrum.code, self.collection_list)} "
                      f"{spectrum.code}. {spectrum.get_equipment()}")
            plt.xlabel("Wavelength"); plt.ylabel("Reflectance")
            plt.legend(); plt.grid(); plt.show()

        return np.array([max_x, max_y / max_y[0]]) if len(max_y) > 0 else np.array([[], []])

    def _load_config(self, config_file):
        debug = False
        config = configparser.ConfigParser()
        config.read(config_file)
        if debug:
            print("Available sections:", config.sections())
        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section, using defaults.", UserWarning)
            cfg = {}
        else:
            cfg = config[self.name] 
            if debug:
                print("_load_config: cfg",cfg)
            
        # wavelength ranges
        self.visible_range_start_wavelength = cfg.getfloat("visible_range_start_wavelength", fallback=450.0)
        self.start_wavelength = cfg.getfloat("start_wavelength", fallback=None)
        self.end_wavelength = cfg.getfloat("end_wavelength", fallback=None)

        # thresholds for prominence and height
        self.prominence_threshold_for_min = cfg.getfloat("prominence_threshold_for_min", fallback=None)
        self.prominence_threshold_for_max = cfg.getfloat("prominence_threshold_for_max", fallback=None)
        self.height_bottom_threshold_for_minimum = cfg.getfloat("height_bottom_threshold_for_minimum", fallback=None)
        self.height_bottom_threshold_for_maximum = cfg.getfloat("height_bottom_threshold_for_maximum", fallback=None)
        self.height_top_threshold_for_minimum = cfg.getfloat("height_bottom_threshold_for_minimum", fallback=None)
        self.height_top_threshold_for_maximum = cfg.getfloat("height_bottom_threshold_for_maximum", fallback=None)

        # distance between peaks
        self.smallest_distance_between_peaks_for_min = cfg.getfloat("smallest_distance_between_peaks_for_min", fallback=None)
        self.smallest_distance_between_peaks_for_max = cfg.getfloat("smallest_distance_between_peaks_for_max", fallback=None)

# -------------------------------------------------------------------------
# Testbench and helpers (now also pass config_file)
# -------------------------------------------------------------------------

class Metric_Testbench():
    """Test metrics on selected spectra and generate a boxplot."""

    def __init__(self, Metric, config_file: str, spectra, collection_list):
        if not spectra:
            raise ValueError("No spectra to evaluate")
        self.collection_list = collection_list
        self.metric_class = Metric
        self.config_file = config_file
        self.spectra = spectra
        self.test_df, self.boxplot_path = self.get_boxplot()
        
    
    def get_boxplot(self):
        metric_list = []
        for spectrum in self.spectra:
            try:
                metric = self.metric_class(spectrum = spectrum, config_file = self.config_file, collection_list = self.collection_list, plot = False)
                if metric.metric_value is not None and not np.isnan(metric.metric_value):
                    metric_list.append(metric)
                else:
                    warnings.warn(f"Skipping {spectrum.get_filename()} — metric value is NaN.")
            except Exception as e:
                print(f"Error computing metric for {getattr(spectrum, 'filename', 'unknown')}: {e}")

        if not metric_list:
            warnings.warn("No valid metrics computed — boxplot cannot be generated.")
            return pd.DataFrame(), None

        metric_df = pd.DataFrame(
            [
                {
                    "species": m.spectrum.species,
                    "genus": m.spectrum.genus,
                    "metric": m.metric_value,
                    "code": m.spectrum.code,
                    "filename": m.spectrum.filename,
                }
                for m in metric_list
            ]
        )

        # Sanity checks
        if metric_df.empty:
            warnings.warn("Metric DataFrame is empty — no boxplot generated.")
            return metric_df, None

        if metric_df["species"].isna().all():
            warnings.warn("Species column is empty/NaN — cannot group boxplot.")
            return metric_df, None

        # Safe plotting with seaborn
        try:
            plt.figure(figsize=(12, 12))
            ax = sns.boxplot(
                data=metric_df,
                x="species", y="metric",
                showfliers=False,
                palette="Set3"
            )
            # Overlay individual points (jittered to avoid overlap)
            sns.stripplot(
                data=metric_df,
                x="species", y="metric",
                color="black", size=5, jitter=True, alpha=0.7
            )

            plt.xticks(rotation=90)
            plt.title(f"Metric: {self.metric_class.get_name()}")
            plt.grid(True, axis="y")

            # Save to file
            path = os.path.join("report_location", "report_images", "gamma_image")
            create_path_if_not_exists(path)
            filename = os.path.join(path, f"{self.metric_class.get_name()}.jpeg")
            plt.savefig(filename, bbox_inches="tight")
            plt.show()
            plt.close()
            return metric_df, filename
        except Exception as e:
            warnings.warn(f"Boxplot generation failed: {e}")
            return metric_df, None
        
    def get_boxplot_old(self):
        metric_list = []
        for spectrum in self.spectra:
            try:
                metric = self.metric_class(spectrum, self.config_file, self.collection_list)
                if metric.metric_value is not None and not np.isnan(metric.metric_value):
                    metric_list.append(metric)
                else:
                    warnings.warn(f"Skipping {spectrum.get_filename()} — metric value is NaN.")
            except Exception as e:
                print(f"Error computing metric for {getattr(spectrum, 'filename', 'unknown')}: {e}")

        if not metric_list:
            warnings.warn("No valid metrics computed — boxplot cannot be generated.")
            return pd.DataFrame(), None

        metric_df = pd.DataFrame(
            [
                {
                    "species": m.spectrum.species,
                    "genus": m.spectrum.genus,
                    "metric": m.metric_value,
                    "code": m.spectrum.code,
                    "filename": m.spectrum.filename,
                }
                for m in metric_list
            ]
        )

        # Sanity checks
        if metric_feature_and_label_extractorempty:
            warnings.warn("Metric DataFrame is empty — no boxplot generated.")
            return metric_df, None

        if metric_df["species"].isna().all():
            warnings.warn("Species column is empty/NaN — cannot group boxplot.")
            return metric_df, None

        # Safe plotting
        try:
            ax = metric_df.boxplot(
                column=["metric"], by=["species"],
                rot=90, grid=True, figsize=(12, 12), showfliers=False
            )
            fig = ax.figure
            plt.title(f"Metric: {self.metric_class.get_name()}")
            path = os.path.join("report_location", "report_images", "gamma_image")
            create_path_if_not_exists(path)
            filename = os.path.join(path, f"{self.metric_class.get_name()}.jpeg")
            fig.savefig(filename)
            return metric_df, filename
        except Exception as e:
            warnings.warn(f"Boxplot generation failed: {e}")
            return metric_df, None



def get_aggregated_data(metric_class, config_file: str, filtered_spectra):
    metric_list = []
    for spectrum in filtered_spectra:
        try:
            metric = metric_class(spectrum, config_file)
            metric_list.append(metric)
        except Exception as e:
            print(e)

    metric_df = pd.DataFrame(columns=["species", "genus", "metric", "code", "filename"])
    for index, metric in enumerate(metric_list):
        metric_df.loc[index, "species"] = metric.spectrum.species
        metric_df.loc[index, "metric"] = metric.metric_value

    return metric_df.groupby('species')['metric'].agg(['mean', 'std'])


def save_aggregated_data(metric_class, config_file: str, filtered_spectra, agregated_data_location):
    grouped_stats = get_aggregated_data(metric_class, config_file, filtered_spectra)
    path_location = os.path.join(agregated_data_location, "metric_avg_std")
    create_path_if_not_exists(path_location)
    path_and_filename = os.path.join(path_location, f'{metric_class.get_name()}')
    grouped_stats.to_csv(path_and_filename, index=True, sep="\t")
    return path_and_filename


def feature_and_label_extractor(Metric, config_file: str, spectra, collection_list, debug, plot = False):
    debug = False
    features, labels, codes = [], [], []
    for spectrum in spectra:
        if debug:
            #spectrum.plot()
            pass
        metric = Metric(spectrum, config_file, collection_list,plot)
        
        features.append(metric.get_metric_value())
        labels.append(spectrum.get_species())
        codes.append(spectrum.code)
    return [codes, features, labels]
