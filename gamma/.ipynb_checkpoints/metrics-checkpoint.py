#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from spectraltools import Specimen_Collection, Spectrum, create_path_if_not_exists

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
        #get list of maxima and minima
        max_i, max_x, max_y = spectrum.get_maxima()
        #Divide second peak over first peak
        metric_value = max_y[1]/max_y[0]
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
        print(metric_value_return)
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


def feature_and_label_extractor(Metric, spectra):
    features = []
    labels = []
    codes = []
    #get code, label and feature for each spectrum
    for spectrum in spectra:
        #spectrum.plot()
        metric = Metric(spectrum)
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
    #this is a subclass of Gamma
    #get_gamma_factor must be redefined
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


class  Gamma_Area_Under_Curve_First_Min_Cut(Metric):
    #this is a subclass of Gamma
    #get_gamma_factor must be redefined
    visible_range_start_wavelength = 450
    name = "Gamma_Area_Under_Curve_First_Min_Cut"

    def __init__(self, spectrum):
        self.spectrum = spectrum
        self.metric_value = self.set_metric_value(spectrum)
        
    def description():
        return f"""This algorithm calculates the area for the visible region (starting at {Gamma_Area_First_Min_Cut.visible_range_start_wavelength} 
        and ending in the first minima between the maximum in the visible range and the maximum in the IR range. 
        Then calculates the area of the IR range up to the second minumum. The ratio between these two areas is the gamma value."""

    def set_metric_value(self, spectrum):

        def get_area_under_curve(spectrum, start_wavelength, finish_wavelength):
            # Assuming your DataFrame is named df and has columns 'wavelength' and 'height'
            # Let's say you have start_wavelength and finish_wavelength variables for the range you want to integrate over
            # Subset the DataFrame to the range of interest
            subset_df = df[(df['wavelength'] >= start_wavelength) & (df['wavelength'] <= finish_wavelength)]

            # Extract the wavelengths and heights as arrays
            wavelengths = subset_df['wavelength'].values
            heights = subset_df[spectrum.measuring_mode].values

            # Calculate the area under the curve using the trapezoidal rule
            area_under_curve = np.trapz(heights, wavelengths)

            # print("Area under the curve:", area_under_curve)
            return area_under_curve

        import numpy as np

        #test_spectrum = filtered_spectra[0]
        #get the highest data recorded
        max_value = spectrum.data[spectrum.measuring_mode].max()
        #get maxima and minima
        x = spectrum.data["wavelength"].values
        y = spectrum.data[spectrum.measuring_mode].values

        #get x and y positions of maxima and minima
        max_i, max_xs, max_ys = spectrum.get_maxima()
        min_i, min_xs, min_ys= spectrum.get_minima()

        #get x locations of first and second maxima and the minimum in between
        first_max_x = max_xs[0]
        try:
            second_max_x = max_xs[1]
        except Exception as e:
            second_max_x = x.max()
            print(e)
        try:
            second_max_y = max_ys[1]
        except Exception as e:
            second_max_y = 0
            print(e)

        min_in_between_i = 0
        min_in_between_x =0
        min_in_between_y =0
        #get the location of the minimum in between
        for index in min_i:
            #print("index")
            if first_max_x <= x[index] <= second_max_x:
                min_in_between_i = index
                min_in_between_x = x[index]
                min_in_between_y = y[index]
                break
        #If we cant find a first minimum (it could be because there is a small minimum not large enough to be detected.
        #we are going to find the next one 
        if min_in_between_i ==0:
            for index in min_i:
                #print("index")
                if second_max_x <= x[index] :
                    min_in_between_i = index
                    min_in_between_x = x[index]
                    min_in_between_y = y[index]
                    break
            
        #second minimum
        #get the location of the second minimum
        min_after_second_max_i = 0
        min_after_second_max_x = 0
        min_after_second_max_y = 0
        for index in min_i:
           
            #check if the second min is greater than the first min_in_between too
            if (second_max_x  <= x[index]) & (x[min_in_between_i]  < x[index]): 
                min_after_second_max_i = index
                min_after_second_max_x = x[index]
                min_after_second_max_y = y[index]
                break
 
        x_values = [first_max_x, min_in_between_x, second_max_x, min_after_second_max_x]
        y_values = [max_ys[0]/max_value, min_in_between_y/max_value, second_max_y/max_value, min_after_second_max_y/max_value]
        #get the normalized spectrum
        df = spectrum.get_normalized_spectrum()
        #plot
        x = df["wavelength"].values
        y =df[spectrum.measuring_mode].values

        #modify y to have last value equal to first one
        y_mod = y
        y_mod[-1] = y_mod[0]
        
        #split x, y LEFT
        #print(f"fmi: {min_in_between_i}")
        x_left = x[:min_in_between_i]
        y_left = y[:min_in_between_i]
        #print(f"{spectrum}")
        #print(f"{y=}")
        #print(f"{y_left=}")
        #set last one to zero for picture to be displaye properly
        y_left[-1] = y_left[0]
        

        #split x, y RIGHT
        #print(f"min_after_second_max_i: {min_after_second_max_i}")
        x_right = x[min_in_between_i:min_after_second_max_i]
        y_right = y[min_in_between_i:min_after_second_max_i]
        #set last one to zero for picture to be displaye properly
        y_right[0] = y_right[-1] = y_left[0]
        
        #show figure

        area_uv_visible = get_area_under_curve(spectrum, Gamma_Area_Under_Curve_First_Min_Cut.visible_range_start_wavelength, min_in_between_x)
        area_ir = get_area_under_curve(spectrum, min_in_between_x, min_after_second_max_x)
        gamma = area_ir/area_uv_visible
        #print(f"gamma: {gamma}")
        return gamma


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
    """This metric returns a vector with each minimum wavelength and relative reflectance."""
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
    """This metric returns a vector with each minimum wavelength and relative reflectance."""
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

    print(file_list)
    
    # Print the list of files
    dataframes = {}
    
    for file_name in file_list:
        full_path = os.path.join(folder_path, file_name)
        
        df = pd.read_csv(full_path, sep="\t", header = 0)
        dataframes[file_name]= df
        
    #return path
    return dataframes