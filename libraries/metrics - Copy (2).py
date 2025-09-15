#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from spectraltools import Specimen_Collection, Spectrum, create_path_if_not_exists
import warnings
import configparser

class Metric():
    """This is an abstract class that represents every metric, allows it to be compared, have a description and a name.
    This is useful when using it in the report methods """
    name = "Metric"
    
    def get_metric_value(self):
        return (self.metric_value)

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
    

# In[2]:
# In[ ]:


class Gamma_First_Two_Peaks(Metric):
    """This gamma metric calculates the ratio between the second and first peak."""
    name = "Gamma_First_Two_Peaks"

    def __init__(self, spectrum):
        self.spectrum = spectrum
        self.metric_value = self.set_metric_value(spectrum)
        
    def set_metric_value(self, spectrum):
        # Get list of maxima
        max_i, max_x, max_y = spectrum.get_maxima()
        # Check if there are at least two peaks and the first peak is non-zero
        if len(max_y) >= 2 and max_y[0] != 0:
            metric_value = max_y[1] / max_y[0]
        else:
            warnings.warn(
                f"Cannot compute Gamma_First_Two_Peaks for {spectrum.get_filename()}: "
                f"Fewer than two peaks found (found {len(max_y)}) or first peak is zero (max_y={max_y}).",
                UserWarning
            )
            metric_value = np.nan
        return metric_value

    @staticmethod
    def description():
        return f"""This algorithm calculates the ratio between the second and first reflectance peak."""

    def __repr__(self):
        return f'Gamma first two peaks {self.metric_value:.4f} for {self.spectrum.genus} {self.spectrum.species} in {self.spectrum.filename}'


class Gamma_Arbitrary_Limits(Metric):
    """This gamma metric calculates the ratio between the maximum in the IR range and the maximum in the visible range. Ranges are static."""
    uv_vis_min_wavelength, uv_vis_max_wavelength = 450.00, 800.00
    ir_min_wavelength = uv_vis_max_wavelength
    ir_max_wavelength = 1500.00
    name = "Gamma_Arbitrary_Limits"
       
        
    def set_metric_value(self, spectrum):
        
        def get_maximum_in_range(spectrum, min_wavelength, max_wavelength):
            measuring_mode = spectrum.metadata["measuring_mode"]
            df = spectrum.data
            max_value = df[(df["wavelength"] > min_wavelength) & (df["wavelength"]  < max_wavelength) ].max()
            #print(f"max value \n {max_value}")
            wavelength, measure = max_value["wavelength"], max_value[measuring_mode]
            return wavelength, measure

        uv_vis_wavelength, uv_vis_max = get_maximum_in_range(spectrum, Gamma_Arbitrary_Limits.uv_vis_min_wavelength, Gamma_Arbitrary_Limits.uv_vis_max_wavelength)
        ir_wavelength, ir_max = get_maximum_in_range(spectrum, Gamma_Arbitrary_Limits.ir_min_wavelength, Gamma_Arbitrary_Limits.ir_max_wavelength)
        metric_value_return = (uv_vis_max / ir_max)*1.00
        #print(metric_value_return)
        return metric_value_return

    def __init__(self, spectrum):
        self.spectrum = spectrum
        self.metric_value = self.set_metric_value(spectrum)
        
    
    @staticmethod
    def description():
        return f"""This algorithm calculates the ratio between the highest reflectance peak in the visible range (Between {Gamma_Arbitrary_Limits.uv_vis_min_wavelength} nm and {Gamma_Arbitrary_Limits.uv_vis_max_wavelength} nm) and the maximum peak in the IR range up to {Gamma_Arbitrary_Limits.ir_max_wavelength} nm. Beyond {Gamma_Arbitrary_Limits.ir_max_wavelength} nm the internal structure's reflectance generates unwanted noise."""
    def __repr__(self):
        return  f'Gamma arbitrary limits, value: {self.metric_value:.4f} for {self.spectrum.genus} {self.spectrum.species}. File: {self.spectrum.filename}'


# In[3]:


def feature_and_label_extractor(Metric, config_file, spectra, debug = False):
    if debug:
        print("config_file", config_file)
        print("spectra length", len(spectra))
        print("first spectra:\n", spectra[0])
    features = []
    labels = []
    codes = []
    #get code, label and feature for each spectrum
    for spectrum in spectra:
        if debug:
            spectrum.plot()
        metric = Metric(spectrum, config_file)
        feature = metric.get_metric_value()
        label = spectrum.get_species()
        code = spectrum.code
        codes.append(code)
        features.append(feature)
        labels.append(label)
        
    data = [codes, features, labels]
    
    return data
    


# In[4]:


class Gamma_Area_Under_Curve_Naive(Metric):
    """This method calculates the ratio between the area under the curve for the spectrum between {Gamma_Area_Under_Curve_Naive.visible_start_wavelength} 
        and {Gamma_Area_Under_Curve_Naive.visible_end_wavelength} nm (visible range) and between {Gamma_Area_Under_Curve_Naive.ir_start_wavelength} nm and 
        {GammaAreaUnderCurveNaive.ir_end_wavelength} nm (infrared range)."""
    visible_start_wavelength = 450
    visible_end_wavelength = ir_start_wavelength = 800
    ir_end_wavelength = 1500
    name = "Gamma_Area_Under_Curve_Naive"
    
    def __init__(self, spectrum):
        self.spectrum = spectrum
        self.metric_value = self.set_metric_value(spectrum)
   
    def description():
        return f"""This method calculates the ratio between the area under the curve for the spectrum between {Gamma_Area_Under_Curve_Naive.visible_start_wavelength} 
        and {Gamma_Area_Under_Curve_Naive.visible_end_wavelength} nm (visible range) and between {Gamma_Area_Under_Curve_Naive.ir_start_wavelength} nm and 
        {GammaAreaUnderCurveNaive.ir_end_wavelength} nm (infrared range)."""


    def set_metric_value(self, spectrum):

        def get_area_under_curve(spectrum, start_wavelength, finish_wavelength):
            # Subset the DataFrame to the range of interest
            subset_df = df[(df['wavelength'] >= start_wavelength) & (df['wavelength'] <= finish_wavelength)]

            # Extract the wavelengths and heights as arrays
            wavelengths = subset_df['wavelength'].values
            heights = subset_df[spectrum.measuring_mode].values

            # Calculate the area under the curve using the trapezoidal rule
            area_under_curve = np.trapz(heights, wavelengths)

            #print("Area under the curve:", area_under_curve)
            return area_under_curve

        import numpy as np
        df = spectrum.get_normalized_spectrum()
        area_uv_visible = get_area_under_curve(spectrum, Gamma_Area_Under_Curve_Naive.visible_start_wavelength, Gamma_Area_Under_Curve_Naive.visible_end_wavelength)
        area_ir = get_area_under_curve(spectrum, Gamma_Area_Under_Curve_Naive.ir_start_wavelength, Gamma_Area_Under_Curve_Naive.ir_end_wavelength)
        metric = area_ir/area_uv_visible
        return metric


# In[5]:

import configparser
import warnings
import numpy as np
import matplotlib.pyplot as plt

class Gamma_Area_Under_Curve_First_Min_Cut(Metric):
    """
    This method calculates the area for the visible region (starting at a configurable wavelength,
    default = 450 nm) and ending in the first minimum between the maximum in the visible range and 
    the maximum in the IR range. Then it calculates the area of the IR range up to the second minimum. 
    The ratio between these two areas is the gamma value.

    Parameters are loaded from a `.config` file under the section `[GammaMetric]`.

    Expected config keys:
    - visible_range_start_wavelength (float, default=450)
    - start_wavelength (float, optional, overrides visible start)
    - end_wavelength (float, optional, max wavelength limit)
    """

    name = "Gamma_Area_Under_Curve_First_Min_Cut"
    debug = True

    def __init__(self, spectrum, config_file: str):
        self.spectrum = spectrum
        self.config_file = config_file

        # Load config
        config = configparser.ConfigParser()
        config.read(config_file)

        if "GammaMetric" not in config:
            raise ValueError("Config file must contain a [GammaMetric] section.")

        cfg = config["GammaMetric"]

        # Load ranges from config (with fallbacks)
        self.visible_range_start_wavelength = cfg.getfloat("visible_range_start_wavelength", fallback=450)
        self.start_wavelength = cfg.getfloat("start_wavelength", fallback=None)
        self.end_wavelength = cfg.getfloat("end_wavelength", fallback=None)

        # Compute metric value
        self.metric_value = self.set_metric_value(
            spectrum,
            start_wavelength=self.start_wavelength,
            end_wavelength=self.end_wavelength,
            debug=self.debug
        )

    def set_metric_value(self, spectrum, start_wavelength=None, end_wavelength=None, debug=False):
        # Get normalized spectrum
        df = spectrum.get_normalized_spectrum()
        x = df["wavelength"].values
        y = df[spectrum.metadata["measuring_mode"]].values

        # Subset between limits
        if not start_wavelength:
            start_wavelength = self.visible_range_start_wavelength
        if not end_wavelength:
            subset_df = df[df["wavelength"] >= start_wavelength]
        else:
            subset_df = df[(df["wavelength"] >= start_wavelength) & (df["wavelength"] <= end_wavelength)]

        # Warn if empty
        if subset_df.empty:
            warnings.warn(
                f"No data points after visible range start {start_wavelength} for {spectrum.get_filename()}. Returning NaN.",
                UserWarning
            )
            return np.nan

        # Get maxima and minima
        max_i, max_xs, max_ys = spectrum.get_maxima()
        min_i, min_xs, min_ys = spectrum.get_minima()

        if len(max_xs) < 1:
            warnings.warn(
                f"No maxima found for {spectrum.get_filename()}. Returning NaN.",
                UserWarning
            )
            return np.nan

        first_max_x = max_xs[0]
        second_max_x = max_xs[1] if len(max_xs) > 1 else x.max()

        # Find first minimum between first and second maxima
        min_in_between_x = None
        for min_x in min_xs:
            if first_max_x <= min_x <= second_max_x:
                min_in_between_x = min_x
                break

        if min_in_between_x is None:
            for min_x in min_xs:
                if min_x > second_max_x:
                    min_in_between_x = min_x
                    break

        if min_in_between_x is None:
            warnings.warn(
                f"No minimum found between maxima or after second maximum for {spectrum.get_filename()}. Returning NaN.",
                UserWarning
            )
            return np.nan

        # Ensure minimum is > visible start
        if min_in_between_x <= self.visible_range_start_wavelength:
            min_in_between_x = start_wavelength

        # Find second minimum
        min_after_second_max_x = None
        for min_x in min_xs:
            if min_x > min_in_between_x and min_x > second_max_x:
                min_after_second_max_x = min_x
                break

        if min_after_second_max_x is None:
            warnings.warn(
                f"No second minimum found after first minimum {min_in_between_x} for {spectrum.get_filename()}. Returning NaN.",
                UserWarning
            )
            return np.nan

        # Compute areas
        def get_area_under_curve(df, start_wavelength, finish_wavelength):
            subset_df = df[(df['wavelength'] >= start_wavelength) & (df['wavelength'] <= finish_wavelength)]
            if subset_df.empty:
                warnings.warn(
                    f"No data points between {start_wavelength} and {finish_wavelength} for {spectrum.get_filename()}. Returning 0.",
                    UserWarning
                )
                return 0.0
            wavelengths = subset_df['wavelength'].values
            heights = subset_df[spectrum.metadata["measuring_mode"]].values
            return np.trapz(heights, wavelengths)

        area_uv_visible = get_area_under_curve(df, self.visible_range_start_wavelength, min_in_between_x)
        area_ir = get_area_under_curve(df, min_in_between_x, min_after_second_max_x)

        if area_uv_visible == 0:
            warnings.warn(
                f"UV-visible area is zero for {spectrum.get_filename()}. Returning NaN.",
                UserWarning
            )
            return np.nan

        gamma = area_ir / area_uv_visible

        if debug:
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

        return gamma

    @staticmethod
    def description():
        return f"""This algorithm calculates the area for the visible region (starting at a configurable wavelength)
        and ending in the first minima between the maximum in the visible range and the maximum in the IR range.
        Then calculates the area of the IR range up to the second minimum. The ratio between these two areas is the gamma value."""

    def __repr__(self):
        return f'Gamma area under curve first min cut {self.metric_value:.4f} for {self.spectrum.genus} {self.spectrum.species} in {self.spectrum.filename}'

# In[6]:


class Gamma_Vector_Relative_Reflectance(Metric):
    """This gamma metric calculates a vector with all the relative heights with respect to the first peak."""
    name = "Gamma_Vector_Relative_Reflectance"

    def __init__(self, spectrum):
        self.spectrum = spectrum
        self.metric_value = self.set_metric_value(spectrum)
        
    def set_metric_value(self, spectrum):
        #get list of maxima and minima
        max_i, max_x, max_y = spectrum.get_maxima()
        
        #Divide every peak over first peak
        metric_value = list(max_y/max_y[0])
        return np.array(metric_value)

    @staticmethod
    def description():
        return f"""This gamma metric calculates a vector with all the relative heights with respect to the first peak"""

    def __repr__(self):
        return f'Gamma vector relative reflectance: {self.metric_value} for {self.spectrum.genus} {self.spectrum.species} in {self.spectrum.filename}'


# In[7]:


class Wavelength_Vector(Metric):
    """This gamma metric calculates a vector with each peak's wavelength."""
    name = "Wavelength_Vector"

    def __init__(self, spectrum):
        self.spectrum = spectrum
        self.metric_value = self.set_metric_value(spectrum)
        
    def set_metric_value(self, spectrum):
        #get list of maxima and minima
        max_i, max_x, max_y = spectrum.get_maxima()
        
        #Divide every peak over first peak
        metric_value = list(max_x)
        return np.array(metric_value)

    @staticmethod
    def description():
        return f"""This metric calculates a vector with each peak's wavelength."""

    def __repr__(self):
        return f'Vector wavelength : {self.metric_value} for {self.spectrum.genus} {self.spectrum.species} in {self.spectrum.filename}'


# In[8]:


class Critical_Points(Metric):
    """This metric returns a vector with each critical point wavelength and relative reflectance."""
    name = "Critical_Points"

    def __init__(self, spectrum):
        self.spectrum = spectrum
        self.metric_value = self.set_metric_value(spectrum)
        
    def set_metric_value(self, spectrum):
       
        min_i, min_x, min_y = spectrum.get_minima() 

        max_i, max_x, max_y = spectrum.get_maxima() 
        #print(max_x, max_y, min_x, min_y)
        metric_value = [np.concatenate((min_x, max_x)), np.concatenate((min_y ,max_y))]
        
        return np.array(metric_value)
     

    @staticmethod
    def description():
        return f"""This metric returns a vector with each critical point wavelength and relative reflectance."""

    def __repr__(self):
        return f'Critical_Points : {self.metric_value} for {self.spectrum.genus} {self.spectrum.species} in {self.spectrum.filename}'


# In[9]:


class Minimum_Points(Metric):
    """This metric returns a vector with each minimum wavelength and absolute reflectance."""
    name = "Minimum_Points"

    def __init__(self, spectrum):
        self.spectrum = spectrum
        self.metric_value = self.set_metric_value(spectrum)
        
    def set_metric_value(self, spectrum):
       
        #get list of maxima and minima
        min_i, min_x, min_y = spectrum.get_minima() 
        
        metric_value = [min_x, min_y]
        
        return np.array(metric_value)

    @staticmethod
    def description():
        return f"""This metric returns a vector with each minimum's wavelength and reflectance."""

    def __repr__(self):
        return f'Minimum_Points : {self.metric_value} for {self.spectrum.genus} {self.spectrum.species} in {self.spectrum.filename}'


# In[10]:


class Maximum_Points(Metric):
    """This metric returns a vector with each minimum wavelength and absolute reflectance."""
    name = "Maximum_Points"

    def __init__(self, spectrum):
        self.spectrum = spectrum
        self.metric_value = self.set_metric_value(spectrum)
        
    def set_metric_value(self, spectrum):
       
        #get list of maxima and minima
        max_i, max_x, max_y = spectrum.get_maxima() 
        
        metric_value = [max_x, max_y]
        
        #first maximum is metric_value[0][1]
        return np.array(metric_value)

    @staticmethod
    def description():
        return f"""This metric returns a vector with each maximum's wavelength and reflectance."""

    def __repr__(self):
        return f'Maximum_Points : {self.metric_value} for {self.spectrum.genus} {self.spectrum.species} in {self.spectrum.filename}'


# In[11]:


class Minimum_Points_Normalized(Metric):
    """This metric returns a vector with each minimum wavelength and relative reflectance."""
    name = "Minimum_Points_Normalized"

    def __init__(self, spectrum):
        self.spectrum = spectrum
        self.metric_value = self.set_metric_value(spectrum)
        
    def set_metric_value(self, spectrum):
       
        #get list of maxima and minima
        min_i, min_x, min_y = spectrum.get_minima() 

        min_y = min_y/min_y[0]
        
        metric_value = [min_x, min_y]
        
        return np.array(metric_value)


    @staticmethod
    def description():
        return f"""This metric returns a vector with each minimum's wavelength and relative reflectance."""

    def __repr__(self):
        return f'Minimum_Points_Normalized : {self.metric_value} for {self.spectrum.genus} {self.spectrum.species} in {self.spectrum.filename}'


# In[12]:


class Maximum_Points_Normalized(Metric):
    """This metric returns a vector with each minimum wavelength and relative reflectance."""
    name = "Maximum_Points_Normalized"

    def __init__(self, spectrum):
        self.spectrum = spectrum
        self.metric_value = self.set_metric_value(spectrum)
        
    def set_metric_value(self, spectrum):
       
        #get list of maxima and minima
        max_i, max_x, max_y = spectrum.get_maxima() 

        max_y = max_y/max_y[0]
        
        metric_value = [max_x, max_y]
        
        return np.array(metric_value)

    @staticmethod
    def description():
        return f"""This metric returns a vector with each maximum's wavelength and relative reflectance."""

    def __repr__(self):
        return f'Maximum_Points_Normalized : {self.metric_value} for {self.spectrum.genus} {self.spectrum.species} in {self.spectrum.filename}'




class Metric_Testbench():
    """This class tests the metrics for the selected spectra and creates a boxplot for the species selected.
    Returns the path to the boxplot image"""
    
    #Calculate gammas
    
    def __init__(self, Metric, filtered_spectra):
        if not filtered_spectra:
            raise ValueError("No spectra to evaluate")
        self.metric_class = Metric
        self.spectra = filtered_spectra
        self.test_df, self.boxplot_path = self.get_boxplot()
        
    def get_boxplot(self):
        
        filtered_spectra = self.spectra
        Metric = self.metric_class
        
        metric_list = []

        for spectrum in filtered_spectra:
            #print(spectrum.get_normalized_spectrum())
            try:
                metric = Metric(spectrum)
                metric_list.append(metric)
            except Exception as e:
                print(e)
        sorted(metric_list)

        metric_df = pd.DataFrame(columns=["species", "genus", "gamma", "code", "filename"])

        #add specimen information to the gammas
        for index, metric in enumerate(metric_list):
            metric_df.loc[index,"species"] = metric.spectrum.species
            metric_df.loc[index,"genus"] = metric.spectrum.genus
            metric_df.loc[index,"metric"] = metric.metric_value
            metric_df.loc[index,"code"] = metric.spectrum.code
            metric_df.loc[index,"filename"] = metric.spectrum.filename

        #print(gamma_df)
        
        #finally, information is presented as a boxplot and saved
        ax = metric_df.boxplot(column=["metric"], by=["species"], ax=None, fontsize=None, rot=90, grid=True, figsize=(4*3, 4*3), layout=None, return_type=None, backend=None, showfliers=False)
        fig = ax.figure
        plt.title(f" Metric: {Metric.get_name() }. Collections: {collection_names}. \n Metric values for C. resplendens, C. kalinini and C. cupreomarginata.")
        
        path= os.path.join(report_location, "report_images", "gamma_image")
        create_path_if_not_exists(path)
        filename = os.path.join(path, f"{Metric.get_name()} "+ collection_names + f"-{current_date}" +".jpeg") 
        fig.savefig(filename)
        
        return metric_df, filename

def get_aggregated_data(metric_class,filtered_spectra):
    #Calculate gammas
    metric_list = []

    for spectrum in filtered_spectra:
        
        try:
            metric = metric_class(spectrum)
            metric_list.append(metric)
            #print(metric_list)
        except Exception as e:
            print(e)
            
    #Order the list
    sorted(metric_list)
    
    #Create a dataframe
    metric_df = pd.DataFrame(columns=["species", "genus", "gamma", "code", "filename"])

    #add specimen information to the metric
    for index, metric in enumerate(metric_list):
        metric_df.loc[index,"species"] = metric.spectrum.species
        metric_df.loc[index,"metric"] = metric.metric_value

    #get info on df
    grouped_stats = metric_df.groupby('species')['metric'].agg(['mean', 'std'])
    #print(grouped_stats)
    grouped_stats = grouped_stats
    #print(grouped_stats)
    return grouped_stats
    
def save_aggregated_data(metric_class,filtered_spectra, agregated_data_location):

    #calculate aggregated data 
    grouped_stats = get_aggregated_data(metric_class,filtered_spectra)
    
    #save information
    path_location = os.path.join(agregated_data_location, "metric_avg_std")
    create_path_if_not_exists(path_location)
    path_and_filename = os.path.join( path_location, f'{metric_class.get_name()}')
    grouped_stats.to_csv( path_and_filename, index=True, sep = "\t")
    
    #return path
    return path_and_filename

def read_aggregated_data(agregated_data_location):

    folder_path = agregated_data_location

    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    #print(file_list)
    
    # Print the list of files
    dataframes = {}
    
    for file_name in file_list:
        full_path = os.path.join(folder_path, file_name)
        
        df = pd.read_csv(full_path, sep="\t", header = 0)
        dataframes[file_name]= df
        
    #return path
    return dataframes