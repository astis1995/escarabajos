#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import configparser
DEBUG = True
from spectraltools import Specimen_Collection, Spectrum, create_path_if_not_exists


class Metric():
    """Abstract class for metrics. Supports comparison, naming, and string representation."""
    debug = True
    name = "Metric"

    def get_metric_value(self):
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

class Gamma_First_Two_Peaks(Metric):
    name = "Gamma_First_Two_Peaks"
    debug = True
    def __init__(self, spectrum, config_file: str):
        self.spectrum = spectrum
        self._load_config(config_file)
        self.metric_value = self.set_metric_value(spectrum)

    def _load_config(self, config_file):
        config = configparser.ConfigParser()
        #("config_file",config_file)
        config.read(config_file)
        #print("Available sections:", config.sections())
        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section, using defaults.", UserWarning)
            cfg = {}
        else:
            cfg = config[self.name]
            print("cfg",cfg)

    def set_metric_value(self, spectrum, debug=False):
        max_i, max_x, max_y = spectrum.get_maxima()
        if len(max_y) >= 2 and max_y[0] != 0:
            value = max_y[1] / max_y[0]
            if debug:
                df = spectrum.get_normalized_spectrum()
                x = df["wavelength"].values
                y = df[spectrum.metadata["measuring_mode"]].values
                plt.figure(figsize=(10, 6))
                plt.plot(x, y, label="Normalized Spectrum")
                plt.axvline(max_x[0], color="r", linestyle="--", label="First Peak")
                plt.axvline(max_x[1], color="g", linestyle="--", label="Second Peak")
                plt.title(f"{spectrum.genus} {spectrum.species} (code: {spectrum.code})")
                plt.xlabel("Wavelength (nm)")
                plt.ylabel(spectrum.metadata["measuring_mode"])
                plt.legend()
                plt.show()
            return value
        return np.nan


class Gamma_Arbitrary_Limits_Silver(Metric):
    """This gamma metric calculates the ratio between the maximum in the IR range
    and the maximum in the visible range. Wavelength ranges are loaded from the 
    [Gamma_Arbitrary_Limits] section of the config file."""
    debug = True
    name = "Gamma_Arbitrary_Limits"

    def __init__(self, spectrum, config_file: str):
        self.spectrum = spectrum
        self._load_config(config_file)
        self.metric_value = self.set_metric_value(spectrum)

    def _load_config(self, config_file):
        debug = True
        
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
            print("cfg",cfg)

        # Read values, fallback to defaults if missing
        self.uv_vis_min_wavelength = float(cfg.get("uv_vis_min_wavelength", 450))
        self.uv_vis_max_wavelength = float(cfg.get("uv_vis_max_wavelength", 800))
        self.ir_min_wavelength = float(cfg.get("ir_min_wavelength", self.uv_vis_max_wavelength))
        self.ir_max_wavelength = float(cfg.get("ir_max_wavelength", 1500))

    def set_metric_value(self, spectrum, debug=False):
        debug = True
        def get_maximum_in_range(spectrum, min_wavelength, max_wavelength):
            measuring_mode = spectrum.metadata["measuring_mode"]
            df = spectrum.data
            subset = df[(df["wavelength"] > min_wavelength) & (df["wavelength"] < max_wavelength)]
            if subset.empty:
                return None, 0.0
            idx = subset[measuring_mode].idxmax()
            return df.loc[idx, "wavelength"], df.loc[idx, measuring_mode]
        
        print("UV VIS MIN",self.uv_vis_min_wavelength, "UV VIS MAX:", self.uv_vis_max_wavelength )
        uv_vis_x, uv_vis_max = get_maximum_in_range(spectrum, self.uv_vis_min_wavelength, self.uv_vis_max_wavelength)
        print("POINT:", uv_vis_x, uv_vis_max)
        print("ir_min_wavelength",self.uv_vis_min_wavelength, "ir_max_wavelength:", self.uv_vis_max_wavelength )
        ir_x, ir_max = get_maximum_in_range(spectrum, self.ir_min_wavelength, self.ir_max_wavelength)
        print("POINT:", ir_x, ir_max)
        if debug and uv_vis_x and ir_x:
            df = spectrum.get_normalized_spectrum()
            plt.figure(figsize=(10, 6))
            plt.plot(df["wavelength"], df[spectrum.metadata["measuring_mode"]], label="Normalized Spectrum")
            plt.axvline(uv_vis_x, color="b", linestyle="--", label="Visible Peak")
            plt.axvline(ir_x, color="m", linestyle="--", label="IR Peak")
            plt.legend(); plt.show()

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

class Gamma_Arbitrary_Limits(Metric):
    """This gamma metric calculates the ratio between the maximum in the IR range
    and the maximum in the visible range. Wavelength ranges are loaded from the 
    [Gamma_Arbitrary_Limits] section of the config file."""
    debug = True
    name = "Gamma_Arbitrary_Limits"

    def __init__(self, spectrum, config_file: str):
        self.spectrum = spectrum
        self._load_config(config_file)
        self.metric_value = self.set_metric_value(spectrum)

    def _load_config(self, config_file):
        debug = True
        
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
            print("cfg",cfg)

        # Read values, fallback to defaults if missing
        self.uv_vis_min_wavelength = float(cfg.get("uv_vis_min_wavelength", 450))
        self.uv_vis_max_wavelength = float(cfg.get("uv_vis_max_wavelength", 800))
        self.ir_min_wavelength = float(cfg.get("ir_min_wavelength", self.uv_vis_max_wavelength))
        self.ir_max_wavelength = float(cfg.get("ir_max_wavelength", 1500))

    def set_metric_value(self, spectrum, debug=False):
        debug = True
        def get_maximum_in_range(spectrum, min_wavelength, max_wavelength):
            measuring_mode = spectrum.metadata["measuring_mode"]
            df = spectrum.data
            subset = df[(df["wavelength"] > min_wavelength) & (df["wavelength"] < max_wavelength)]
            if subset.empty:
                return None, 0.0
            idx = subset[measuring_mode].idxmax()
            return df.loc[idx, "wavelength"], df.loc[idx, measuring_mode]
        
        print("UV VIS MIN",self.uv_vis_min_wavelength, "UV VIS MAX:", self.uv_vis_max_wavelength )
        uv_vis_x, uv_vis_max = get_maximum_in_range(spectrum, self.uv_vis_min_wavelength, self.uv_vis_max_wavelength)
        print("POINT:", uv_vis_x, uv_vis_max)
        print("ir_min_wavelength",self.uv_vis_min_wavelength, "ir_max_wavelength:", self.uv_vis_max_wavelength )
        ir_x, ir_max = get_maximum_in_range(spectrum, self.ir_min_wavelength, self.ir_max_wavelength)
        print("POINT:", ir_x, ir_max)
        if debug and uv_vis_x and ir_x:
            df = spectrum.get_normalized_spectrum()
            plt.figure(figsize=(10, 6))
            plt.plot(df["wavelength"], df[spectrum.metadata["measuring_mode"]], label="Normalized Spectrum")
            plt.axvline(uv_vis_x, color="b", linestyle="--", label="Visible Peak")
            plt.axvline(ir_x, color="m", linestyle="--", label="IR Peak")
            plt.legend(); plt.show()

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



class Gamma_Area_Under_Curve_Naive(Metric):
    name = "Gamma_Area_Under_Curve_Naive"
    debug = True
    def __init__(self, spectrum, config_file: str):
        self.spectrum = spectrum
        self._load_config(config_file)
        self.metric_value = self.set_metric_value(spectrum)

    def _load_config(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        print("Available sections:", config.sections())
        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section, using defaults.", UserWarning)
            cfg = {}
        else:
            cfg = config[self.name] 
            print("cfg",cfg)
        self.visible_start_wavelength = cfg.getfloat("visible_start_wavelength", fallback=450)
        self.visible_end_wavelength = cfg.getfloat("visible_end_wavelength", fallback=800)
        self.ir_start_wavelength = cfg.getfloat("ir_start_wavelength", fallback=800)
        self.ir_end_wavelength = cfg.getfloat("ir_end_wavelength", fallback=1500)

    def set_metric_value(self, spectrum, debug=False):
        debug = True
        df = spectrum.get_normalized_spectrum()

        def get_area_under_curve(start, finish):
            subset = df[(df["wavelength"] >= start) & (df["wavelength"] <= finish)]
            wavelengths = subset["wavelength"].values
            heights = subset[spectrum.measuring_mode].values
            return np.trapz(heights, wavelengths), subset

        area_uv, subset_uv = get_area_under_curve(self.visible_start_wavelength, self.visible_end_wavelength)
        area_ir, subset_ir = get_area_under_curve(self.ir_start_wavelength, self.ir_end_wavelength)

        if debug:
            plt.figure(figsize=(10, 6))
            plt.plot(df["wavelength"], df[spectrum.measuring_mode], label="Normalized Spectrum")
            plt.fill_between(subset_uv["wavelength"], subset_uv[spectrum.measuring_mode], alpha=0.5, color="skyblue", label="Visible Area")
            plt.fill_between(subset_ir["wavelength"], subset_ir[spectrum.measuring_mode], alpha=0.5, color="orange", label="IR Area")
            plt.legend(); plt.show()

        return area_ir / area_uv if area_uv != 0 else np.nan



class Gamma_Area_Under_Curve_First_Min_Cut(Metric):
    name = "Gamma_Area_Under_Curve_First_Min_Cut"
    debug = True

    def __init__(self, spectrum, config_file: str):
        self.spectrum = spectrum
        self._load_config(config_file)
        self.metric_value = self.set_metric_value(spectrum)
        
        #apply configuration to spectrum
        self.apply_configuration_to_spectrum()

    def apply_configuration_to_spectrum(self):
        self.spectrum.set_parameters( 
        prominence_threshold_min=self.prominence_threshold_min, 
        prominence_threshold_max=self.prominence_threshold_max,
        min_height_threshold_denominator=self.min_height_threshold_denominator, 
        max_height_threshold_denominator=self.max_height_threshold_denominator,
        min_distance_between_peaks=self.min_distance_between_peaks, 
        max_distance_between_peaks=self.max_distance_between_peaks)
        
    def _load_config(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        print("Available sections:", config.sections())
        if self.name not in config:
            warnings.warn(f"Config missing [{self.name}] section, using defaults.", UserWarning)
            cfg = {}
        else:
            cfg = config[self.name] 
            print("cfg",cfg)
            
        # wavelength ranges
        self.visible_range_start_wavelength = cfg.getfloat("visible_range_start_wavelength", fallback=450.0)
        self.start_wavelength = cfg.getfloat("start_wavelength", fallback=None)
        self.end_wavelength = cfg.getfloat("end_wavelength", fallback=None)

        # thresholds for prominence and height
        self.prominence_threshold_min = cfg.getfloat("prominence_threshold_min", fallback=None)
        self.prominence_threshold_max = cfg.getfloat("prominence_threshold_max", fallback=None)
        self.min_height_threshold_denominator = cfg.getfloat("min_height_threshold_denominator", fallback=None)
        self.max_height_threshold_denominator = cfg.getfloat("max_height_threshold_denominator", fallback=None)

        # distance between peaks
        self.min_distance_between_peaks = cfg.getfloat("min_distance_between_peaks", fallback=None)
        self.max_distance_between_peaks = cfg.getfloat("max_distance_between_peaks", fallback=None)

    def set_metric_value(self, spectrum, start_wavelength=None, end_wavelength=None, debug=False):
        debug = True
        df = spectrum.get_normalized_spectrum()
        x = df["wavelength"].values
        y = df["%R"].values 
        
        if not start_wavelength:
            start_wavelength = self.visible_range_start_wavelength
        subset_df = df[df["wavelength"] >= start_wavelength] if not end_wavelength else df[
            (df["wavelength"] >= start_wavelength) & (df["wavelength"] <= end_wavelength)
        ]
        if subset_df.empty:
            return np.nan

        max_i, max_xs, _ = spectrum.get_maxima()
        min_i, min_xs, _ = spectrum.get_minima()
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
        
        if debug:
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
            plt.legend()
            plt.show()
            plt.fill_between(df["wavelength"], y, where=(df["wavelength"] >= self.visible_range_start_wavelength) & (df["wavelength"] <= min_in_between_x),
                 alpha=0.5, color="skyblue", label="Visible Area")
            plt.fill_between(df["wavelength"], y, where=(df["wavelength"] >= min_in_between_x) & (df["wavelength"] <= min_after_second_max_x),
                             alpha=0.5, color="orange", label="IR Area")

            
        return gamma


# -------------------------------------------------------------------------
# Metrics without wavelength parameters (but still accept config_file)
# -------------------------------------------------------------------------

class Gamma_Vector_Relative_Reflectance(Metric):
    name = "Gamma_Vector_Relative_Reflectance"

    def __init__(self, spectrum, config_file: str):
        self.spectrum = spectrum
        self.metric_value = self.set_metric_value(spectrum)

    def set_metric_value(self, spectrum):
        _, _, max_y = spectrum.get_maxima()
        return np.array(list(max_y / max_y[0]))




class Wavelength_Vector(Metric):
    name = "Wavelength_Vector"

    def __init__(self, spectrum, config_file: str):
        self.spectrum = spectrum
        self.metric_value = self.set_metric_value(spectrum)

    def set_metric_value(self, spectrum):
        _, max_x, _ = spectrum.get_maxima()
        return np.array(list(max_x))


class Critical_Points(Metric):
    name = "Critical_Points"
    
    def __init__(self, spectrum, config_file: str):
        self.spectrum = spectrum
        self.metric_value = self.set_metric_value(spectrum)

    def set_metric_value(self, spectrum):
        debug = True
        _, min_x, min_y = spectrum.get_minima()
        _, max_x, max_y = spectrum.get_maxima()
        if debug:
            df = spectrum.get_normalized_spectrum()
            plt.figure(figsize=(10, 6))
            plt.plot(df["wavelength"], df[spectrum.metadata["measuring_mode"]], label="Spectrum")
            for x in max_x: plt.axvline(x, color="r", linestyle="--", alpha=0.7)
            for x in min_x: plt.axvline(x, color="b", linestyle="--", alpha=0.7)
            plt.legend(); plt.show()
        return np.array([np.concatenate((min_x, max_x)), np.concatenate((min_y, max_y))])


class Minimum_Points(Metric):
    name = "Minimum_Points"

    def __init__(self, spectrum, config_file: str):
        self.spectrum = spectrum
        self.metric_value = self.set_metric_value(spectrum)

    def set_metric_value(self, spectrum):
        debug = True
        _, min_x, min_y = spectrum.get_minima()
        if debug:
            df = spectrum.get_normalized_spectrum()
            plt.figure(figsize=(10, 6))
            plt.plot(df["wavelength"], df[spectrum.metadata["measuring_mode"]], label="Spectrum")
            for x in max_x: plt.axvline(x, color="r", linestyle="--", alpha=0.7)
            for x in min_x: plt.axvline(x, color="b", linestyle="--", alpha=0.7)
            plt.legend(); plt.show()
        return np.array([min_x, min_y])


class Maximum_Points(Metric):
    name = "Maximum_Points"

    def __init__(self, spectrum, config_file: str):
        self.spectrum = spectrum
        self.metric_value = self.set_metric_value(spectrum)

    def set_metric_value(self, spectrum):
        debug = True
        _, max_x, max_y = spectrum.get_maxima()
        if debug:
            df = spectrum.get_normalized_spectrum()
            plt.figure(figsize=(10, 6))
            plt.plot(df["wavelength"], df[spectrum.metadata["measuring_mode"]], label="Spectrum")
            for x in max_x: plt.axvline(x, color="r", linestyle="--", alpha=0.7)
            
            plt.legend(); plt.show()
        return np.array([max_x, max_y])


class Minimum_Points_Normalized(Metric):
    name = "Minimum_Points_Normalized"

    def __init__(self, spectrum, config_file: str):
        self.spectrum = spectrum
        self.metric_value = self.set_metric_value(spectrum)

    def set_metric_value(self, spectrum):
        _, min_x, min_y = spectrum.get_minima()
        return np.array([min_x, min_y / min_y[0]])


class Maximum_Points_Normalized(Metric):
    name = "Maximum_Points_Normalized"

    def __init__(self, spectrum, config_file: str):
        self.spectrum = spectrum
        self.metric_value = self.set_metric_value(spectrum)

    def set_metric_value(self, spectrum):
        _, max_x, max_y = spectrum.get_maxima()
        return np.array([max_x, max_y / max_y[0]])


# -------------------------------------------------------------------------
# Testbench and helpers (now also pass config_file)
# -------------------------------------------------------------------------

class Metric_Testbench():
    """Test metrics on selected spectra and generate a boxplot."""

    def __init__(self, Metric, config_file: str, spectra):
        if not spectra:
            raise ValueError("No spectra to evaluate")
        self.metric_class = Metric
        self.config_file = config_file
        self.spectra = spectra
        self.test_df, self.boxplot_path = self.get_boxplot()

    def get_boxplot(self):
        metric_list = []
        for spectrum in self.spectra:
            try:
                metric = self.metric_class(spectrum, self.config_file)
                metric_list.append(metric)
            except Exception as e:
                print(e)

        metric_df = pd.DataFrame(columns=["species", "genus", "metric", "code", "filename"])
        for index, metric in enumerate(metric_list):
            metric_df.loc[index, "species"] = metric.spectrum.species
            metric_df.loc[index, "genus"] = metric.spectrum.genus
            metric_df.loc[index, "metric"] = metric.metric_value
            metric_df.loc[index, "code"] = metric.spectrum.code
            metric_df.loc[index, "filename"] = metric.spectrum.filename

        ax = metric_df.boxplot(column=["metric"], by=["species"], rot=90, grid=True, figsize=(12, 12), showfliers=False)
        fig = ax.figure
        plt.title(f" Metric: {self.metric_class.get_name()} ")
        path = os.path.join("report_location", "report_images", "gamma_image")
        create_path_if_not_exists(path)
        filename = os.path.join(path, f"{self.metric_class.get_name()}.jpeg")
        fig.savefig(filename)

        return metric_df, filename


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


def feature_and_label_extractor(Metric, config_file: str, spectra, debug=DEBUG):
    debug = False
    features, labels, codes = [], [], []
    for spectrum in spectra:
        if debug:
            #spectrum.plot()
            pass
        metric = Metric(spectrum, config_file)
        features.append(metric.get_metric_value())
        labels.append(spectrum.get_species())
        codes.append(spectrum.code)
    return [codes, features, labels]
