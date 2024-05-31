#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import math
import re
import scipy


#decorator
def plot_wrapper(func):
    def wrapper(*args, **kwargs):
        #print("Something is happening before the function is called.")
        plt.figure(figsize=(10, 5))
        result = func(*args, **kwargs)
        #print("Something is happening after the function is called.")
        plt.grid(True)
        plt.show()
        return result
    return wrapper

class Specimen_Collection:
    """This class represents a physical collection of specimens"""
    def read_collection(self,collection_path):
        with open(collection_path, encoding= "latin1") as f:
          df = pd.read_csv(f, sep="\t", decimal =",", header=0, encoding="iso-8859-1")
          return df

    def __init__(self, name, data_folder_path, metadata_path, quality):
        self.name = name
        self.data_folder_path = data_folder_path
        self.metadata = self.read_collection(metadata_path)
        self.quality = quality
        self.description = "No description"

    def set_description(self, description):
        self.description = description
    
    def get_name(self):
        return self.name
    
    def get_metadata(self):
        return self.metadata
    
    def get_data_folder_path(self):
        return self.data_folder_path
    
    def get_data_filenames(self):
        
        folder_path = self.get_data_folder_path()
        
        #list files in folder
        file_list = os.listdir(folder_path)

        #file extension
        file_extension = ".txt"

        # filters a list of strings to create a new list containing only the elements that end with file_extension
        def filter_substring_elements(path_strings, substring):
            filtered_paths = [path for path in path_strings if substring in path]
            return filtered_paths

        #full path list
        filtered_list = [os.path.join(folder_path, path) for path in filter_substring_elements(file_list, file_extension)]
        
        return filtered_list
        
    def read_spectrum(self, file_path, collection):
        
        spectrum = Spectrum(file_path, collection)
        
        return spectrum
    
    def get_spectra(self):
        filenames = self.get_data_filenames()
        spectra = []
        for filename in filenames:
            spectrum = Spectrum(filename, self)
            spectra.append(spectrum)
        return spectra



# In[6]:


#test collection class
def test_collection_class():
    angsol_collection = Specimen_Collection("ANGSOL", angsol_collection_path, angsol_collection_metadata, "HIGH")
    #print(f"{angsol_collection.get_metadata()=} \n" )
    #print(f"{angsol_collection.get_data_folder_path()=} \n" )   
    #print(f"{angsol_collection.get_data_filenames()=} \n" )
#test_collection_class()


# In[7]:


def create_path_if_not_exists(path):
        # Check if the path already exists
        if not os.path.exists(path):
            # Create the directory and any missing parent directories
            os.makedirs(path)
            print(f"Directory '{path}' created successfully.")
        else:
            print(f"Directory '{path}' already exists.")


# In[8]:


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
    prominence_threshold_min = 0.15
    prominence_threshold_max = 0.40
    min_height_threshold_denominator = 3.0
    max_height_threshold_denominator = 3.3
    min_distance_between_peaks = 50 #nm
    max_distance_between_peaks = 50 #min distance 

    def set_parameters(prominence_threshold_min = None, prominence_threshold_max = None, min_height_threshold_denominator = None, max_height_threshold_denominator = None,
    min_distance_between_peaks = None, max_distance_between_peaks = None ):
        if prominence_threshold_min:
            self.prominence_threshold_min = prominence_threshold_min
        if prominence_threshold_max:
            self.prominence_threshold_max = prominence_threshold_max
        if min_height_threshold_denominator:
            self.min_height_threshold_denominator = min_height_threshold_denominator
        if max_height_threshold_denominator:
            self.max_height_threshold_denominator = max_height_threshold_denominator
        if min_distance_between_peaks:
            self.min_distance_between_peaks = min_distance_between_peaks
        if max_distance_between_peaks:
            self.max_distance_between_peaks = max_distance_between_peaks
        
            
    def __init__(self, spectrum):
        self.spectrum = spectrum
        self.peaks = self.get_peaks()

    def get_spectrum(self):
        return self.spectrum

    def get_peaks_as_object(self):
        import scipy
        #get info
        x = self.spectrum.data["wavelength"].values
        y = self.spectrum.data[self.spectrum.metadata["measuring_mode"]].values
        
        #parameters
        min_height = y.max()/self.max_height_threshold_denominator
        
        width_t = 50.00

        #get peaks

        min_i, min_x_values, min_y_values = self.get_minima()
        max_i, max_x_values, max_y_values = self.get_maxima()

        peaks = []

        #print("peak called")
        for i in zip(max_x_values, max_y_values):
            max_peak = Peak(i[0], i[1])
            peaks.append(max_peak)
        for i in zip(min_x_values, min_y_values):
            min_peak = Peak(i[0], i[1])
            peaks.append(min_peak)

        peaks = sorted(peaks)
        return peaks
        
    def get_peaks(self):
        peaks = self.get_peaks_as_object()
        x = []
        y = []
        for peak in peaks: 
            if not ((peak.get_x() > 855)&(peak.get_x() < 869)):  
                x.append(peak.get_x())
                y.append(peak.get_y())
        
        return x, y


    def plot_settings(self): 
        self.spectrum.plot_settings()
        x_values, y_values = self.get_peaks()
            
        return plt.scatter(x_values, y_values, color="r")
        
    @plot_wrapper    
    def plot(self):
        self.plot_settings()

    
    def get_minima(self):
        """This method returns the index, x values and y values of every minimum in a spectrum"""
        
        #Get minimum
        spectrum = self.get_spectrum()
        #get wavelength and height of measurements
        x = spectrum.data["wavelength"].values
        y = spectrum.data[spectrum.metadata["measuring_mode"]].values
        
        #reflect plot across x axis and displace it upwards
        y_max = y.max()
        y_inverted = -y + y_max

        #maximum height
        # This prevents  minima less than 40% 
        maximum_height = y_inverted.max() * 0.60
        minimum_height = 0
        #get minima
        peaks_funct = scipy.signal.find_peaks(y_inverted, distance= self.min_distance_between_peaks, prominence=self.prominence_threshold_min, height = (minimum_height, maximum_height))
        peaks_index = peaks_funct[0]
        x_values = x[peaks_index]
        y_values = y[peaks_index]

        return peaks_index, x_values, y_values

    def get_maxima(self):
        """This method returns the index, x values and y values of every maxima in a spectrum"""
        spectrum = self.get_spectrum()
        #get wavelength and height of measurements
        x = spectrum.data["wavelength"].values
        y = spectrum.data[spectrum.metadata["measuring_mode"]].values
    
        #define minimum height and min distance between minima
        min_height = y.max()/self.min_height_threshold_denominator
        min_distance = 50 #nm
        max_distance = 100.00
        width_t = 50.00
    
        #get maxima
        peaks_funct = scipy.signal.find_peaks(y, height= min_height, distance= self.max_distance_between_peaks, prominence= self.prominence_threshold_max)
        peaks_index = peaks_funct[0] #indices
        x_values = x[peaks_index]   #x values
        y_values = y[peaks_index]    #y values
    
        return peaks_index, x_values, y_values


# In[9]:


#Spectrum class
class Spectrum:
    """This class represents the data and metadata for a L1050 file.
    It provides the maxima and minima and a """
    #These variables delimit the thresholds used to determine if a point can be considered a maximum or minimum
    
        
    
    def get_metadata_and_dataframe(self, file_location):
         #definitions
        #Logic to read ASCII data
        import os
        import pandas as pd
        import re

        def get_sample_code_from_filename(row_str, file_location):
            #print("string")
            #print(file_location)
            filename = os.path.basename(file_location)
            re1 = r"([a-zA-Z\d]+)(?:-\d)*(?:.Sample)*.(?:txt)*(?:ASC)*"
            #Names are in the form CODE-MEASUREMENTNUMBER.TXT
            p = re.compile(re1)
            m = p.match(filename)
            # print(f"match filename: {m}")
            if m:
                # print(f"group 1: {m.group(1)}")
                return(m.group(1))
            return get_sample_code(row_str)

        def get_sample_code(row_str):
            #Tries to get the sample code from the file, if it does not match
            #it tries to get it from the filename.
            # print("string")
            # print(row_str)
            re1 = r"([a-zA-Z\d]+)(?:-\d)*(?:.Sample)*.(?:txt)*(?:ASC)*"
            #Names are in the form CODE-MEASUREMENTNUMBER.TXT
            p = re.compile(re1)
            m = p.match(row_str)
            # print(f"match: {m}")
            if m:
                return(m.group(1))
            else:
                ""

        def responses(str):
            re1 = "\d+/(\d+,\d+) \d+,\d+/(\d+,\d+)"
            p = re.compile(re1)
            m= p.match(str)
            if m:
                return m.group(1),m.group(2)
            else:
                return "",""
        def attenuator_settings(str):
            re1 = "S:(\d+,\d+) R:(\d+,\d+)"
            p = re.compile(re1)
            m= p.match(str)
            if m:
                return m.group(1),m.group(2)
            else:
                return "",""
        def slit_pmt_aperture(str):
            re1 = "\d+/servo \d+,\d+/(\d+,\d+)"
            p = re.compile(re1)
            m= p.match(str)
            if m:
                return m.group(1)
            else:
                return ""
        #Initializa metadata dict
        metadata = {}

        #Read header
        lines = []
        with open(file_location, encoding= "latin1") as myfile:
            lines = myfile.readlines()[0:90]
        metadata["header"] = "".join(lines)


        #read_metadata
        f = open(file_location, encoding= "latin1")

        df = pd.DataFrame()
        with f as data_file:
            for index, row in enumerate(data_file): #0-89

                row_str = row.strip()
                if index +1 == 3: #Filename and extension
                    metadata["filename"]= row_str
                    metadata["code"] = get_sample_code_from_filename(row_str, file_location)
                if index + 1 == 4: #date DD/MM/YYYY
                    metadata["date"]= row_str
                if index + 1 == 5:#Time HH:MM:SS.SS
                    metadata["time"]= row_str
                if index + 1 == 8:#user
                    metadata["user"]= row_str
                if index + 1 == 9:#description
                    metadata["description"]= row_str
                if index + 1 == 10:#minimum wavelength
                    metadata["minimum_wavelength"]= row_str
                if index + 1 == 12:#equipment name
                    metadata["equipment"]= row_str
                if index + 1 == 13:#equipment series
                    metadata["series"]= row_str
                if index + 1 == 14:#data visualizer version, equipment version, date and time
                    metadata["software"]= row_str
                if index + 1 == 21:#Operating mode
                    metadata["operating_mode"]= row_str
                if index + 1 == 22: #Number of cycles
                    metadata["cycles"]= row_str
                if index + 1 == 32: #range/servo
                    metadata["slit_pmt"]= slit_pmt_aperture(row_str)
                if index + 1 == 33:
                    metadata["response_ingaas"], metadata["response_pmt"]= responses(row_str)
                if index + 1 == 35: #pmt gain, if 0 is automatic
                    metadata["pmt_gain"]= row_str
                if index + 1 == 36: #InGaAs detector gain
                    metadata["ingaas_gain"]= row_str
                if index + 1 == 42:#monochromator wavelength nm
                    metadata["monochromator_change"]= row_str
                if index + 1 == 43:#lamp change wavelength
                    metadata["lamp_change"]= row_str
                if index + 1 == 44:#pmt wavelength
                    metadata["pmt_change"]= row_str
                if index + 1 == 45:#beam selector
                    metadata["beam_selector"]= row_str
                if index + 1 == 46:
                    metadata["cbm"]= row_str
                if index + 1 == 47: #cbd status, on/off
                    metadata["cbd_status"]= row_str
                if index + 1 == 48: #attenuator percentage
                    metadata["attenuator_sample"], metadata["attenuator_reference"]= attenuator_settings(row_str)
                if index + 1 == 49:
                    metadata["polarizer"]= row_str
                if index + 1 == 80:
                    metadata["units"]= row_str
                if index + 1 == 81:
                    metadata["measuring_mode"]= row_str
                if index + 1 == 84:
                    metadata["maximum_wavelength"]= row_str
                if index + 1 == 85:
                    metadata["step"]= row_str
                if index + 1 == 86:
                    metadata["number_of_datapoints"]= row_str
                if index + 1 == 88:
                    metadata["maximum_measurement"]= row_str
                if index + 1 == 89:
                    metadata["minimum_measurement"]= row_str
                if index +1 == 90:
                    break
            df = pd.read_csv(f, sep="\t", decimal =".", names=["wavelength", metadata["measuring_mode"]]).dropna()
            df = df[df["wavelength"]<2000].reset_index()
            df["wavelength"],df[metadata["measuring_mode"]] = df["wavelength"].astype(float), df[metadata["measuring_mode"]].astype(float)
            return metadata, df
    
    def __str__(self):
        return self.code

    def __init__(self, file_location, collection):

        import re

        def get_genus(code, collection):
            #print("get_genus")
            
            #variables
            collection_name = collection.get_name()
            collection_metadata = collection.get_metadata()
            
            #Locate specimen
            specimen= collection_metadata.loc[collection_metadata["code"]==code]

            if specimen.empty:
                print(f"No data for {code} in collection {collection_name}")
                return ""
            #print("not mt")
            # print(f"specimen genus {specimen}")
            result = specimen.iloc[0]["genus"]
            #print(f"genus, type{type(result)}")
            
            if isinstance(result,str):
                return result
            else:
                return str(result)

        def get_species(code, collection):
            #print("get_species")
            # print(f"code: {code}")
            
            #variables
            collection_name = collection.get_name()
            collection_metadata = collection.get_metadata()
            
            #Locate specimen
            specimen= collection_metadata.loc[collection_metadata["code"]==code]

            if specimen.empty:
                print(f"No data for {code} in collection {collection_name}")
                result = ""
            #print("not mt")
            #print(f"specimen species {specimen}")
            result = str(specimen.iloc[0]["species"])
            #print(f"species, type{type(result)}")
            if isinstance(result,str):
                return result
            else:
                return str(result)

        #attributes
        self.file_location = file_location
        self.collection = collection
        
        self.metadata, self.data = self.get_metadata_and_dataframe(file_location)
        self.code = self.metadata["code"]
        self.filename =  self.metadata["filename"]
        self.genus = get_genus(self.code, collection)
        self.species = get_species(self.code, collection)
        self.measuring_mode = self.metadata["measuring_mode"]

    def plot_settings(self):
        measuring_mode = self.metadata["measuring_mode"]
        
        df = self.data
        
        x = df["wavelength"]
        y = df[measuring_mode]
        
        plt.plot(x, y)
        plot = plt.title(f"{measuring_mode} for {self.genus} {self.species}, code {self.code}")
        
        return plot
        
    @plot_wrapper    
    def plot(self):
        self.plot_settings()
        
    def get_normalized_spectrum(self):
        df = self.data[["wavelength", self.measuring_mode]]
        max_value = df[self.measuring_mode].max()
        df.loc[:,self.measuring_mode] = df.loc[:,self.measuring_mode]/max_value
        return df
        
    def get_maxima(self):
        maxima_list = PeakList(self).get_maxima()
        return maxima_list
        
    def get_minima(self):
        minima_list = PeakList(self).get_minima()
        return minima_list
    
    def get_critical_points(self):
        peaks = PeakList(self).get_peaks()
        return peaks    

    def get_data(self):
        return self.data

    def get_code(self):
        return self.code
        
    def get_collection(self):
        return self.collection
    def get_species(self):
        return self.species
    def get_genus(self):
        return self.genus

    
####



# In[10]:


def test_spectrum_class():
    angsol_collection = Specimen_Collection("ANGSOL", angsol_collection_path, angsol_collection_metadata, "HIGH")
    filenames = angsol_collection.get_data_filenames()
    spectra = angsol_collection.get_spectra()
    for spectrum in spectra:
        spectrum.plot()
#test_spectrum_class()


# In[11]:


def test_peak_class():
    angsol_collection = Specimen_Collection("ANGSOL", angsol_collection_path, angsol_collection_metadata, "HIGH")
    filenames = angsol_collection.get_data_filenames()
    spectra = angsol_collection.get_spectra()
    for spectrum in spectra:
        peaklist1 = PeakList(spectrum)
        #peaklist1.plot()
        #print(peaklist1)
        #print(peaklist1.plot())
#test_peak_class()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




