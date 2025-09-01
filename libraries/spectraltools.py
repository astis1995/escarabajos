#!/usr/bin/env python
# coding: utf-8

# In[1]:
"""Spectral Tools library:
This library allows us to read CRAIC and L1050 files.


#This script requires the file spectraltools.py to work an the following code:
import sys
import os
# Add the external folder to the system path
current_dir = os.getcwd()
external_folder_path = os.path.abspath(os.path.join(current_dir, '../libraries'))
sys.path.append(external_folder_path)
import spectraltools


"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import math
import re
import scipy
import warnings
from pathlib import Path

#CONSTANTS
min_craic_wv = 420 
max_craic_wv = 950
min_craic_wv_polarized = 420 
max_craic_wv_polarized = 700
#most common regex should go first
regex_dict = {
    "l1050_filename_regex" : "([a-zA-Z]+\\d{4})-*_*(\\d)*(?:.Sample)*.(?:txt)*(?:ASC)*", #code #reading
    "craic_data_comma_regex" : "\\d*.\\d*,\\d*.\\d*\n",
    "craic_data_tab_regex" : "\\d*.\\d*\t\\d*.\\d*",
    "craic_filename_regex_0" :  "(\\d+?).csv", #codes
    "craic_filename_regex_1" :  "(\\d+?)([RLOT])(\\d+).csv", #code #polarization #reading
    "craic_filename_regex_2" : "([a-zA-Z\\d]+)_([RLOT]).csv", # "code #polarization
    "craic_filename_regex_3" :  "(\\d+?)-(\\d+)*.csv", #code #reading
    ###Estudio-spectral-escarabajos
    "craic_filename_regex_4": "(\\d+?)([RLOT])(\\d+)-(?:(elytrum)*(pronotum)*(escutelo)*).csv", #code #polarization #reading-#location
    "craic_filename_regex_5": "(\\d+?)-variacion(\\d+).csv", #code #polarization #reading-#location
    "craic_filename_regex_6" : "(\\d+)(?:(escutelo)*(pronoto)*)([RLOT])(\\d+)", 
    "macraspis-blue-1": "(\\d+)-macraspis-blue-average([TOLR]).csv", #reading pol
    "sinnumero-rojo": "sinnumero-rojo-(\\d+)", #reading
    "macraspis-green": "(\\d+)-macraspis-green-average([LRTO]).csv", #reading polarization,
    "macraspis-chrysis": r"1-macraspis-chrysis-average([LR](tot)).csv", #reading polarization,
    "calomacraspis-1": "calomacraspis-(?:(elytrum)*(pronotum)*(escutelo)*)-std([LRTO]).csv", #location, polarization,
    "calomacraspis-2": "calomacraspis-(?:(elytrum)*(pronotum)*(escutelo)*(scutellum)*)([LRTO]).csv", #location, polarization,
    "calomacraspis-2": "calomacraspis-(?:(elytrum)*(pronotum)*(escutelo)*(scutellum)*)([LRTO]).csv", #location, polarization,
    "calomacraspis-2": "calomacraspis-(?:(elytrum)*(pronotum)*(escutelo)*(scutellum)*)([LRTO]).csv", #location, polarization,
    "cupreomarginata-1": "cupreo-average([LROT]).csv", #location, polarization,
    "cupreomarginata-2": "cupreo([LROT])-std.csv", #location, polarization,
    "cupreomarginata-3": "cupreoT-averageL.csv", #location, polarization,
    "cupreomarginata-4": "(ojo)-ccupreomarginata-izquierdo-([LRTO]).csv", #location, polarization,
    "resplendens-CVG": "resplendensCVG([LOTR](total)*)CP.csv", #location, polarization,
    }


#COLECCIONS

#plot rainbow
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import colorsys

def integer_generator(start=0):
    if(start%10==0):
        print(f"{start=}")
    while True:
        yield start
        start += 1
gen = integer_generator()

def get_contrasting_color():
    index = next(gen)
    # Ensure the index is a non-negative integer
    if index < 0:
        raise ValueError("Index must be a non-negative integer.")
    
    # Number of colors to generate
    total_colors = 360  # Using full circle of hue (0-360)
    
    # Calculate the hue based on the index
    hue = (index * 137.508) % total_colors  # Use golden angle for good contrast
    saturation = 0.8  # Keep saturation high for vibrant colors
    lightness = 0.5   # Keep lightness moderate

    # Convert HSL to RGB
    r, g, b = colorsys.hls_to_rgb(hue / 360.0, lightness, saturation)
    
    # Convert RGB values to 0-255 range
    return (r , g , b )


def draw_rainbow_background():
    """
    Plots wavelength vs intensity with a custom background gradient.

    Parameters:
        longitudes_de_onda (array-like): Wavelengths in nm.
        intensidad (array-like): Intensity values.
    """
    # Create the figure and axes
    fig, ax = plt.subplots()

    # Create a custom colormap from violet to red
    colors = [
        "#8A2BE2",  # Violet
        "#0000FF",  # Blue
        "#00FF00",  # Green
        "#FFFF00",  # Yellow
        "#FFA500",  # Orange
        "#FF0000"   # Red
    ]
    custom_cmap = LinearSegmentedColormap.from_list("violet_to_red", colors, N=256)

    # Create a gradient for the background
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    # Show the background with the custom colormap
    ax.imshow(gradient, aspect='auto', cmap=custom_cmap, extent=[380, 750, 0, 200], alpha=0.15)

    # Plot the data# Plot the data
    #df = dataframe
    #x = df["wavelength"]
    #y = df[metadata["measuring_mode"]]
    
    #if kind == "scatter":
    #    scatter = ax.scatter(x, y, color=color, alpha=alpha, label=label)
    #    scatter.set_sizes([s])
    #else:
    #    ax.plot(df["wavelength"], df[metadata["measuring_mode"]], color=get_contrasting_color(), label=label)

    


# Call the function


#decorator
def plot_wrapper(func):
    def wrapper(*args, **kwargs):
        #pri*nt("Something is happening before the function is called.")
        plt.figure(figsize=(10, 5))
        result = func(*args, **kwargs)
        #pri*nt("Something is happening after the function is called.")
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

    def __init__(self, name = None, data_folder_path= None, metadata_path= None, quality= None):
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

    def get_codes(self):
        codes = set(self.metadata["code"].values)
        codes = list(map(str, codes))
        #print(f"{codes=}")
        return codes

    def get_species():
        species = set(self.metadata["species"])
        return species

    def get_genera():
        genera = set(self.metadata["genus"])
        return genera

    def get_data_folder_path(self):
        return self.data_folder_path

    def get_data_filenames(self):
        """Gets every filename under data_folder_path with the extension in file_extension"""
        folder_path = self.get_data_folder_path()

        #list files in folder
        file_list = os.listdir(folder_path)

        #file extension
        file_extension = [".txt", ".csv"]

        # filters a list of strings to create a new list containing only the elements that end with file_extension
        def filter_substring_elements(path_strings, substring_list):
            filtered_paths = []
            for extension in substring_list:
                filtered_paths += [path for path in path_strings if (extension in path)]
            return filtered_paths

        #full path list
        filtered_list = [os.path.join(folder_path, path) for path in filter_substring_elements(file_list, file_extension)]

        return filtered_list

    def read_spectrum(self, file_path, collection, min_wavelength = None, max_wavelength = None):

        spectrum = Spectrum(file_path, collection, min_wavelength, max_wavelength)

        return spectrum

    def get_spectra(self, min_wavelength = None, max_wavelength = None):
        filenames = self.get_data_filenames()
        spectra = []
        #print(f"data filenames: {filenames}")
        for filename in filenames:
            spectrum = Spectrum(filename, self, min_wavelength, max_wavelength)
            if spectrum:
                spectra.append(spectrum)
            else:
                print(f"The following filename {filename} was not converted into an spectrum. Check it")
        return spectra

    def genus_lookup(self, code, collection_list):
        """
        Returns the genus for the given code from the collection list.
        
        Parameters:
        - code (str): The code to search for.
        - collection_list (list): List of collections to search in.
        
        Returns:
        - genus (str): The genus associated with the code, or "na" if not found.
        """
        for collection in collection_list:
            codes = list(collection.get_codes())
            if code in codes:
                metadata = collection.get_metadata()
                try:
                    genus = list(metadata.loc[metadata["code"].astype(str) == code, "genus"].values)[0]
                    return genus
                except Exception as e:
                    print(f"Error retrieving genus: {e}")
                    return "na"
        print(f"The provided code ({code}) is not in the collection list.")
        return "na"


    def species_lookup(self, code, collection_list):
        """
        Returns the species for the given code from the collection list.
        
        Parameters:
        - code (str): The code to search for.
        - collection_list (list): List of collections to search in.
        
        Returns:
        - species (str): The species associated with the code, or "na" if not found.
        """
        for collection in collection_list:
            codes = list(collection.get_codes())
            if code in codes:
                metadata = collection.get_metadata()
                try:
                    species = list(metadata.loc[metadata["code"].astype(str) == code, "species"].values)[0]
                    return species
                except Exception as e:
                    print(f"Error retrieving species: {e}")
                    return "na"
        print(f"The provided code ({code}) is not in the collection list.")
        return "na"
    
    
    def genus_species_lookup(self, code, collection_list):
        """
        Returns both genus and species for the given code from the collection list.
        
        Parameters:
        - code (str): The code to search for.
        - collection_list (list): List of collections to search in.
        
        Returns:
        - tuple: (genus, species) associated with the code, or ("na", "na") if not found.
        """
        genus = genus_lookup(code, collection_list)
        species = species_lookup(code, collection_list)
        
        return genus, species

    def collection_lookup(self, code, collection_list):
        
        #pri*nt("Collection lookup")
        #convert code to int
        code = str(code)
        
        
        for collection in collection_list:
            codes = list(collection.get_codes())
            codes = [str(num) for num in codes]
            
            #pri*nt(f"{type(codes)=}")
            #pri*nt(f"{type(code)=}")
            #pri*nt(f"{code in codes=}")
            
            if code in codes:
                #pri*nt(f"{collection}")
                return collection
            
                
                
        #raise ValueError(err_msj)
        #warnings.warn(f"{err_msj}", UserWarning)
        err_msj = f"The provided code ({code}) is not in the collection list:\n {collection_list} \n. Returning None instead"
        print(err_msj)
        ##logging.error(f'An error occurred: {err_msj}')
        return None
    def __str__(self):
        try:
            return self.name
        except:
            return "None"
    def __repr__(self):
        try:
            return self.name
        except AttributeError:
            return "None"




#test collection class
def test_collection_class():
    angsol_collection = Specimen_Collection("ANGSOL", angsol_collection_path, angsol_collection_metadata, "HIGH")
    #pri*nt(f"{angsol_collection.get_metadata()=} \n" )
    #pri*nt(f"{angsol_collection.get_data_folder_path()=} \n" )
    #pri*nt(f"{angsol_collection.get_data_filenames()=} \n" )
#test_collection_class()


def create_path_if_not_exists(path):
        # Check if the path already exists
        if not os.path.exists(path):
            # Create the directory and any missing parent directories
            os.makedirs(path)
            #pri*nt(f"Directory '{path}' created successfully.")
        else:
            pass
            #pri*nt(f"Directory '{path}' already exists.")


# In[8]:
def check_fluorometer_file(file):
    #check if it is a hidden file
    if Path(file).name.startswith("."):
        return False
    #check if it is a .asc file
    if not Path(file).name.endswith(".asc"):
        return False
    #check if it is a CRAIC file
    #pri*nt(f"{file=}")
    with open(file) as f:
        first_line = (f.readline())
        #pri*nt(f"{first_line=}")
        regex = r"PE FL       SUBTECH     SPECTRUM    ASCII       PEDS        4.00"
        match = re.search(regex, first_line)
        if match:
            return True
        else:
            return False
            
def check_CRAIC_file(file):
    #check if it is a hidden file
    if Path(file).name.startswith("."):
        return False
    #check if it is a .csv file
    if not Path(file).name.endswith(".csv"):
        return False
    #check if it is a CRAIC file
    #pri*nt(f"{file=}")
    with open(file) as f:
        first_line = (f.readline())
        #pri*nt(f"{first_line=}")
        regex = r"Time1=\d*ms:Average1=\d*:Objective=\d*X:Aperture=\d*: \(\d*/\d*/\d* \d*:\d*:\d* (?:AM)*(?:PM)*\)"
        match = re.search(regex, first_line)
        if match:
            return True
        else:
            return False

def check_l1050_file(file):
    #check if it is a hidden file
    if Path(file).name.startswith("."):
        return False
    if not (Path(file).name.endswith(".txt") or Path(file).name.endswith(".ASC")) :
        return False
    with open(file) as f:
        #pri*nt(f"{file=}")
        first_line = (f.readline())
        #pri*nt(f"{first_line=}")
        regex = r"PE UV       SUBTECH     SPECTRUM    ASCII       PEDS        .*"
        match = re.search(regex, first_line)
        if match:
            return True
        else:
            return False

def check_empty_CRAIC_file(f):
    try:
        if pd.read_csv(f, sep="	", decimal =".", names=["wavelength", "measuring_mode"], skiprows = 9).empty:
            return True
    except:
        pass
    try:
        if pd.read_csv(f, sep=",", decimal =".", names=["wavelength", "measuring_mode"], skiprows = 9).empty:
            return True
    except:
        pass
    return False


    
def check_empty_l1050_file(f):
    try:
        if pd.read_csv(f, sep="	", decimal =".", names=["wavelength", "measuring_mode"], skiprows = 90).empty:
            return True
    except:
        pass
    return False

def read_l1050_file(file):
    """
    This function takes filepath for a l1050 file, reads it and returns its metadata and dataframe
    """
    return get_metadata_and_dataframe_l1050(file)

def read_CRAIC_file(file):
    return get_metadata_and_dataframe_CRAIC(file)

def read_fluorometer_file(file):
    return get_metadata_and_dataframe_fluorometer(file)
    
#read_spectrum_file method
    
def read_spectrum_file(file):
    # Check if the file is an L1050 type
    if check_l1050_file(file):
        #pri*nt("l1050 file")
        return read_l1050_file(file)

    # Check if the file is a CRAIC type
    elif check_CRAIC_file(file):
        #pri*nt("CRAIC file")
        return read_CRAIC_file(file)
        
    elif check_fluorometer_file(file):
        #print("Fluorometer file")
        return read_fluorometer_file(file)
    # Raise an exception if the file type is unknown
    else:
        #pri*nt(f"{file=}")
        
        #raise ValueError("The file is neither a valid L1050 nor CRAIC file.")
        print(f"The file {file} is neither a valid L1050 nor CRAIC file.")
        return (None, None)
def get_metadata_from_filename(file_location):
            """Returns the code and polarization from filename. Examples:
            BIOUCR0001_L code: BIOUCR0001 polarization: L
            BIOUCR0001_R code: BIOUCR0001 polarization: R
            BIOUCR0001_O code: BIOUCR0001 polarization: O (no polarization)
            BIOUCR0001_0 code: BIOUCR0001 polarization: 0 (degrees)
            BIOUCR0001_0 code: BIOUCR0001 polarization: 90 (degrees)
            1037298L2 code: 1037298 polarization L """
            basename = Path(file_location).name
            code , polarization = get_metadata_from_basename(basename)
            return code, polarization
    
def get_metadata_from_basename(basename):
            """Returns the code and polarization from filename. Examples:
            BIOUCR0001_L code: BIOUCR0001 polarization: L
            BIOUCR0001_R code: BIOUCR0001 polarization: R
            BIOUCR0001_O code: BIOUCR0001 polarization: O (no polarization)
            BIOUCR0001_0 code: BIOUCR0001 polarization: 0 (degrees)
            BIOUCR0001_0 code: BIOUCR0001 polarization: 90 (degrees)
            1037298L2 code: 1037298 polarization L """

            #initialize
            code = "na"
            
            #pri*nt(file_location)
            
            #re1 = r"([a-zA-Z\d]+)_(R)*(L)*(O)*.csv"

            info = get_info_from_format(basename)
            
            code = info["code"]
            polarization = info["polarization"]
            
            if not polarization:
                polarization = "T"
            return code, polarization

def first_line(str):
    #pri*nt(f"{str=}")
    #re1 = r"Time1=(\d)*ms:Average1=(\d)*:Objective=(\d)*X:Aperture=(\d)*: (((\d)*/(\d)*/(\d)*) ((\d)*:(\d)*:(\d)* (AM)*(PM)*))"
    #re1 = "Time1=43ms:Average1=10:Objective=10X:Aperture=1: (3/5/2024 8:54:50 AM)"
    re1 = r"Time1=(\d*)ms:Average1=(\d*).*:Objective=(\d*X):Aperture=(\d*): \((\d*/\d*/\d*) (\d*:\d*:\d* (AM)*(PM)*)\)"
    p = re.compile(re1)
    m= p.match(str)
    if m:
        #pri*nt("match!")
        time1 = m.group(1)
        #pri*nt(f"{time1=}")
        average1 = m.group(2)
        #pri*nt(f"{average1=}")
        objective = m.group(3)
        #pri*nt(f"{objective=}")
        aperture = m.group(4)
        #pri*nt(f"{aperture=}")
        date = m.group(5)
        #pri*nt(f"{date=}")
        time = m.group(6)
        #pri*nt(f"{time=}")
        return time1, average1, objective, aperture, date, time
    else:
        return "",""
        
def measuring_mode(str):
    """Changes the measuring mode of CRAIC files to the standard notation"""
    if (str != ""):
        if str == "Reflectance":
            return "%R"
        elif str == "Transmittance":
            return "%T"
        elif str == "Fluorescence":
            return "%F"
        if str == "Absorptance":
            return "%A"
    else:
        return ""
        
def average_2(str):
    """Reads CRAIC files' average_2 data"""
    re1 = r"Avg2: (\d*.\d*)"
    p = re.compile(re1)
    m= p.match(str)
    if m:
        return m.group(1)
    else:
        return ""
def integration_time1(str):
    """Reads CRAIC files' integration_time1 data"""
    re1 = r"Int.Time1:(\d*.\d*)"
    p = re.compile(re1)
    m= p.match(str)
    if m:
        return m.group(1)
    else:
        return ""
def integration_time2(str):
    """Reads CRAIC files' integration_time2 data"""
    re1 = r"Int.Time2:(\d*.\d*)"
    p = re.compile(re1)
    m= p.match(str)
    if m:
        return m.group(1)
    else:
        return ""

def get_format(string_line):
    
    #pri*nt(f"{string_line}")
    format_type = None
    for element in regex_dict:
        #print(f"{element=}")
        #print(f"{regex_dict[element]=}")
        if re.fullmatch(regex_dict[element], string_line):
            format_type = element
            #pri*nt(f"{format_type=}")
            return format_type
    return None
def get_CRAIC_info_from_filename(file):
    basename = Path(file).name
    #pri*nt(f"{basename}")
    
    return get_info_from_format(basename)
def get_info_from_format(string_line):
    #print(f"3 {string_line=}")
    format_type = get_format(string_line)
    #print(f"3 {format_type=}")
    info = {
        "code": None,
        "polarization": None,
        "reading" : None,
        "location" : None,
        "genus":None,
        "species":None,
        "original": None
    }
    #print(f"{string_line=}")
    
    if format_type == "craic_filename_regex_0":
        regex = regex_dict[format_type]
        #pri*nt(f"{regex=}")
        p = re.compile(regex)
        m = p.fullmatch(string_line)
        if m:
            info["code"] = m.group(1)
            info["polarization"] = "T"
            info["reading"] = 0 #zero means that it is an average
            info["original"] = False #it is an average
            #pri*nt(f"{string_line=}{format_type=}{m=}{m.groups()=}")
            return info
    if format_type == "craic_filename_regex_3":
        regex = regex_dict[format_type]
        #pri*nt(f"{regex=}")
        p = re.compile(regex)
        m = p.fullmatch(string_line)
        if m:
            info["code"] = m.group(1)
            info["polarization"] = None
            info["reading"] = m.group(2)
            info["original"] = True #it an original file
            #pri*nt(f"{string_line=}{format_type=}{m=}{m.groups()=}")
            
            return info
    if format_type == "craic_filename_regex_1":
        regex = regex_dict[format_type]
        p = re.compile(regex)
        m = p.fullmatch(string_line)
        #pri*nt(f"{m}")
        if m:
            info["code"] = m.group(1)
            info["polarization"] = m.group(2)
            info["reading"] = m.group(3)
            info["original"] = True #it an original file
            return info
    if format_type == "craic_filename_regex_2":
        regex = regex_dict[format_type]
        #pri*nt(f"{regex=}")
        p = re.compile(regex)
        m = p.fullmatch(string_line)
        if m:
            info["code"] = m.group(1)
            info["polarization"] = m.group(2)
            info["reading"] = None
            info["original"] = True #it an original file
            #pri*nt(f"{string_line=}{format_type=}{m=}{m.groups()=}")
            return info
            
    if format_type == "craic_filename_regex_4":
        regex = regex_dict[format_type]
        #pri*nt(f"{regex=}")
        p = re.compile(regex)
        m = p.fullmatch(string_line)
        if m:
            info["code"] = m.group(1)
            info["polarization"] = m.group(2)
            info["reading"] = m.group(3)
            info["location"] = m.group(4)
            info["original"] = True #it an original file
            #pri*nt(f"{string_line=}{format_type=}{m=}{m.groups()=}")
            return info
    if format_type == "craic_filename_regex_5":
        regex = regex_dict[format_type]
        #pri*nt(f"{regex=}")
        p = re.compile(regex)
        m = p.fullmatch(string_line)
        if m:
            info["code"] = m.group(1)
            info["polarization"] = "T"
            info["reading"] = m.group(2)
            info["location"] = None
            info["original"] = True #it an original file
            #pri*nt(f"{string_line=}{format_type=}{m=}{m.groups()=}")
            return info
    if format_type == "craic_filename_regex_6":
        #"craic_filename_regex_6" : "(\d+)(?:(escutelo)*(pronoto)*)([RLOT])(\d+)"
        regex = regex_dict[format_type]
        #pri*nt(f"{regex=}")
        p = re.compile(regex)
        m = p.fullmatch(string_line)
        if m:
            info["code"] = m.group(1)
            info["polarization"] = m.group(3)
            info["reading"] = m.group(4)
            info["location"] = m.group(2)
            info["original"] = True #it an original file
            #pri*nt(f"{string_line=}{format_type=}{m=}{m.groups()=}")
            return info
            
            
    if format_type == "sinnumero-rojo":
        # "sinnumero-rojo": "sinnumero-rojo-(\d+)"
        regex = regex_dict[format_type]
        #pri*nt(f"{regex=}")
        p = re.compile(regex)
        m = p.fullmatch(string_line)
        if m:
            info["code"] = "sinnumero-rojo"
            info["polarization"] = "T"
            info["reading"] = m.group(1)
            info["location"] = None
            info["species"] = "boucardi"
            info["genus"] = "Chrysina"
            info["original"] = True #it an original file
            #pri*nt(f"{string_line=}{format_type=}{m=}{m.groups()=}")
            return info
    if format_type == "macraspis-blue-1":
        # "macraspis-blue-1": "(\d+)-macraspis-blue-average([TOLR]).csv"
        regex = regex_dict[format_type]
        #pri*nt(f"{regex=}")
        p = re.compile(regex)
        m = p.fullmatch(string_line)
        if m:
            info["code"] = "sinnumero-rojo"
            info["polarization"] = m.group(2)
            info["reading"] = m.group(1)
            info["location"] = None
            info["species"] = "sp."
            info["genus"] = "Macraspis"
            info["color"] = "blue"
            info["original"] = True #it an original file
            #pri*nt(f"{string_line=}{format_type=}{m=}{m.groups()=}")
            return info
    if format_type == "macraspis-green":
        # "macraspis-blue-1": "(\d+)-macraspis-blue-average([TOLR]).csv"
        regex = regex_dict[format_type]
        #pri*nt(f"{regex=}")
        p = re.compile(regex)
        m = p.fullmatch(string_line)
        if m:
            info["code"] = "macraspis-green"
            info["polarization"] = m.group(2)
            info["reading"] = m.group(1)
            info["location"] = None
            info["species"] = "sp."
            info["genus"] = "Macraspis"
            info["color"] = "green"
            info["original"] = True #it an original file
            #pri*nt(f"{string_line=}{format_type=}{m=}{m.groups()=}")
            return info
            
    #macraspis-green
    if format_type == "l1050_filename_regex":
        regex = regex_dict[format_type]
        #pri*nt(f"{regex=}")
        p = re.compile(regex)
        m = p.fullmatch(string_line)
        if m:
            info["code"] = m.group(1)
            info["polarization"] = None
            info["reading"] =  m.group(2)
            info["original"] = True #it an original file
            #pri*nt(f"{string_line=}{format_type=}{m=}{m.groups()=}")
            return info
    return info
    
def get_code_from_format(string_line):
    info = get_info_from_format(string_line)
    code = info["code"]
    #print(f"{string_line=} {info=} {code=}")
    return str(code)
    
def get_code_from_filename(file):
    basename = Path(file).name
    #print(f"1 {basename=}")
    code = get_code_from_format(basename)
    #print(f" 1 {code=}")
    return code
    
def get_codes_from_filenames(files):
    codes = []
    for file in files:
        code = get_code_from_filename(file)
        if code:
            codes.append(code)
    codes = set(codes)
    codes = sorted(codes)
    return codes


    
def get_polarization_from_format(string_line):
    info = get_info_from_format(string_line)
    polarization = info["polarization"]
    return polarization
    
def get_reading_from_format(string_line):
    reading = get_info_from_format(string_line)
    polarization = info["reading"]
    return reading


            
####################################################################################################################################################

def get_metadata_and_dataframe_CRAIC(file_location):
        """Reads CRAIC files' dataframe and metadata"""
        #definitions
    
        #Inicializa metadata dict
        metadata = {}

        #formatting
        formatting = ""
    
        #Read header
        first_data_line = ""
        lines = []
        header = []
    
        with open(file_location, encoding= "latin1") as myfile:
            
            lines = myfile.readlines()[0:] #reads lines from 1 to n
            #pri*nt(f"4. get_metadata_and_dataframe_CRAIC: {lines=}")
            #check the format of the first line
            first_data_line = lines[9]
            #print(f"4. get_metadata_and_dataframe_CRAIC: {first_data_line=}")
            #header
            header = "".join(lines[0:9])
        metadata["header"] = header

        #check CRAIC format
        
        format_type = get_format(first_data_line)
        

        #pri*nt(f"File: {file_location}")
        f = open(file_location, encoding= "latin1")

        df = pd.DataFrame()

        #The file will open on the first line, the following for loop will iterate over each line until finished
        #after the break statement, the file will be in the nth line, from which the dataframe will be read
        with f as data_file:
            for index, row in enumerate(data_file): #0-89
                row_str = row.strip()
                #pri*nt(f"{row_str=}")
                if index +1 == 1: #First line
                    metadata["time1"], metadata["average1"], metadata["objective"], metadata["aperture"], metadata["date"], metadata["time"] =first_line(row_str)
                if index + 1 == 3: #Mode(reflectance, transmittance, absorptance, fluorescence)
                    metadata["measuring_mode"]= measuring_mode(row_str)
                    #pri*nt(f"{measuring_mode(row_str)=}")
                if index + 1 == 7:#average2
                    metadata["average2"]= average_2(row_str)
                if index + 1 == 8:#int. Time1
                    metadata["integration_time1"]= integration_time1(row_str)
                if index + 1 == 9:#int. Time 2
                    metadata["integration_time2"]= integration_time2(row_str)
                    break
        f = open(file_location, encoding= "latin1")

        """This section reads the dataframe"""

        with f as data_file:
            #print(f"{file_location=}")
            #try reading using tabs as separator
            #df = pd.DataFrame([])

            #print(f"5 {format_type=}")
            #try reading using comma as separator
            if format_type == "craic_data_comma_regex":
                df = pd.read_csv(file_location, sep=",", decimal =".", names=["wavelength", metadata["measuring_mode"]], skiprows = 9).dropna()
            #If nothing works show warning
            if format_type == "craic_data_tab_regex":
                df = pd.read_csv(file_location, sep="\t", decimal =".", names=["wavelength", metadata["measuring_mode"]], skiprows = 9).dropna()
            if df.empty:
                warnings.warn(f"Dataframe is empty. File: {file_location}", UserWarning)
                pass #debug

            #print(f"get_metadata_and_: {df=}")
            #wavelength is always measured in ms
            metadata["filename"]= file_location
            metadata["units"]= "nm"
            #get additional metadata info
            #from filename
            metadata["polarization"] = "T"
            metadata["code"], metadata["polarization"]= get_metadata_from_filename(file_location)
            #pri*nt(f"{metadata['code']=}")
            #pri*nt(f"{metadata['polarization']=}")
            #from data analysis
            metadata["minimum_wavelength"]= df["wavelength"].min()
            metadata["maximum_wavelength"]= df["wavelength"].max()
            #pri*nt(df["wavelength"].diff())
            metadata["step"]= np.round(np.mean(df["wavelength"].diff()),8) #8 significant figures
            #pri*nt(f"{metadata["step"]=}")
            metadata["number_of_datapoints"]= len(df[metadata["measuring_mode"]])
            metadata["maximum_measurement"]=  df[metadata["measuring_mode"]].max()
            metadata["minimum_measurement"]= df[metadata["measuring_mode"]].min()
            metadata["equipment"] = "CRAIC"
            metadata["genus"] = "na"
            metadata["species"] = "na"
            #filter spurious wavelengths
            
            df["wavelength"],df[metadata["measuring_mode"]] = df["wavelength"].astype(float), df[metadata["measuring_mode"]].astype(float)
            if metadata["polarization"] in ["R", "L"]:
                #filter between 420 and 700 nm
                df = df[(df["wavelength"]> min_craic_wv_polarized) &(df["wavelength"]<max_craic_wv_polarized)].reset_index()
            else:
                df = df[(df["wavelength"]> min_craic_wv) &(df["wavelength"]<max_craic_wv)].reset_index()
            df = df.drop("index", axis=1)

            return metadata, df


def get_metadata_and_dataframe_l1050(file_location):
        #definitions
        #Logic to read ASCII data
        import os
        import pandas as pd
        import re

        def get_sample_code_from_filename(row_str, file_location):
            #pri*nt("string")
            #pri*nt(file_location)
            filename = os.path.basename(file_location)
            re1 = r"([a-zA-Z\d]+)(?:-\d)*(?:.Sample)*.(?:txt)*(?:ASC)*"
            #Names are in the form CODE-MEASUREMENTNUMBER.TXT
            p = re.compile(re1)
            m = p.match(filename)
            # pri*nt(f"match filename: {m}")
            if m:
                # pri*nt(f"group 1: {m.group(1)}")
                return(m.group(1))
            return get_sample_code(row_str)

        def get_sample_code(row_str):
            #Tries to get the sample code from the file, if it does not match
            #it tries to get it from the filename.
            # pri*nt("string")
            # pri*nt(row_str)
            re1 = r"([a-zA-Z\d]+)(?:-\d)*(?:.Sample)*.(?:txt)*(?:ASC)*"
            #Names are in the form CODE-MEASUREMENTNUMBER.TXT
            p = re.compile(re1)
            m = p.match(row_str)
            # pri*nt(f"match: {m}")
            if m:
                return(m.group(1))
            else:
                ""

        def responses(str):
            re1 = r"\d+/(\d+,\d+) \d+,\d+/(\d+,\d+)"
            p = re.compile(re1)
            m= p.match(str)
            if m:
                return m.group(1),m.group(2)
            else:
                return "",""
        def attenuator_settings(str):
            re1 = r"S:(\d+,\d+) R:(\d+,\d+)"
            p = re.compile(re1)
            m= p.match(str)
            if m:
                return m.group(1),m.group(2)
            else:
                return "",""
        def slit_pmt_aperture(str):
            re1 = r"\d+/servo \d+,\d+/(\d+,\d+)"
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
        #pri*nt(f"File: {file_location}")
        f = open(file_location, encoding= "latin1")

        df = pd.DataFrame()
        with f as data_file:
            for index, row in enumerate(data_file): #0-89

                row_str = row.strip()
                if index +1 == 3: #Filename and extension
                    metadata["filename"]= file_location
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
            #normally l1050 spectrum does not have polarization
            metadata["genus"] = "na"
            metadata["species"] = "na"
            metadata["polarization"] = "T"
            df = pd.read_csv(f, sep="\t", decimal =",", names=["wavelength", metadata["measuring_mode"]]).dropna()
            #pri*nt(df) #debug
            df["wavelength"],df[metadata["measuring_mode"]] = df["wavelength"].astype(float), df[metadata["measuring_mode"]].astype(float)
            df = df[df["wavelength"]<2000].reset_index()
            df = df.drop("index", axis=1)
            return metadata, df


import os
import pandas as pd
import re

def get_metadata_and_dataframe_fluorometer(file_location):
    # Definitions
    def get_sample_code_from_filename(row_str, file_location):
        filename = os.path.basename(file_location)
        re1 = r"([a-zA-Z\d]+)(?:-\d)*(?:.Sample)*.(?:txt)*(?:ASC)*"
        p = re.compile(re1)
        m = p.match(filename)
        if m:
            return m.group(1)
        return get_sample_code(row_str)

    def get_sample_code(row_str):
        re1 = r"([a-zA-Z\d]+)(?:-\d)*(?:.Sample)*.(?:txt)*(?:ASC)*"
        p = re.compile(re1)
        m = p.match(row_str)
        if m:
            return m.group(1)
        else:
            return ""

    # Initialize metadata dict
    metadata = {}

    # Read header (first 51 lines)
    with open(file_location, encoding="latin1") as myfile:
        lines = myfile.readlines()[0:51]
    metadata["header"] = "".join(lines)
    metadata["measuring_mode"] = "F"
    # Read metadata
    with open(file_location, encoding="latin1") as f:
        df = pd.DataFrame()
        for index, row in enumerate(f):  # Read the first 51 lines
            row_str = row.strip()

            if index + 1 == 3:  # Filename and extension
                metadata["filename"] = file_location
                metadata["code"] = get_sample_code_from_filename(row_str, file_location)
            if index + 1 == 4:  # Date DD/MM/YYYY
                metadata["date"] = row_str
            if index + 1 == 5:  # Time HH:MM:SS.SS
                metadata["time"] = row_str
            if index + 1 == 6:  # User (NANOMATERIALES(ND0002))
                metadata["user"] = row_str
            if index + 1 == 7:  # Description (Metodo 2, modificado)
                metadata["description"] = row_str
            if index + 1 == 10:  # Excitation Wavelength (nm)
                metadata["excitation_wavelength"] = row_str
            if index + 1 == 11:  # Measurement Method
                metadata["method"] = row_str
            if index + 1 == 12:  # Instrument Name (SPECTROMETER/DATA SYSTEM)
                metadata["instrument_name"] = row_str
            if index + 1 == 13:  # Instrument Serial Number
                metadata["instrument_serial"] = row_str
            if index + 1 == 14:  # Software Version or Date Code
                metadata["software_version"] = row_str

            # Instrument Parameters (lines 16-38)
            param_dict = {
                "Excitation Wavelength": "excitation_wavelength",
                "Emission Start Wavelength": "emission_start_wavelength",
                "Emission End Wavelength": "emission_end_wavelength",
                "PMT Voltage": "pmt_voltage",
                "PMT Gain": "pmt_gain"
            }

            for key, value in param_dict.items():
                if key in row_str:
                    metadata[value] = row_str.split('=')[-1].strip()

            if index + 1 == 39:  # End of header information
                break

    # Load the measurement data (assuming it's tab-separated with a header)
    df = pd.read_csv(file_location, sep="\t", decimal=".", names=["wavelength", "F"], skiprows=51).dropna()
    df["wavelength"], df["F"] = df["wavelength"].astype(float), df["F"].astype(float)
    df = df[df["wavelength"] < 2000].reset_index(drop=True)

    return metadata, df




def get_genus(code, collection):
    #pri*nt("get_genus")
    
    #variables
    collection_name = collection.get_name()
    collection_metadata = collection.get_metadata()
    
    #Locate specimen
    #print(f"{type(code)=} {code=}")
    
    #print(f"{type(collection_metadata["code"])} {collection_metadata["code"]}")
    specimen = collection_metadata.loc[collection_metadata["code"].astype(str)==(code),"genus"]
    #pri*nt(f"Genus {specimen=}")
    
    if specimen.empty:
        err_msj = (f"No genus data for {code} in collection {collection_name}")
        ##logging.error(f'An error occurred: {err_msj}')
        print(err_msj)
        return ""
    #pri*nt("not mt")
    # pri*nt(f"specimen genus {specimen}")
    result = specimen.iloc[0]
    #pri*nt(f"genus, type{type(result)}")
    
    if isinstance(result,str):
        return result
    else:
        return str(result)
    
def get_species(code, collection):
    #pri*nt("get_species")
    # pri*nt(f"code: {code}")
    
    #variables
    collection_name = collection.get_name()
    collection_metadata = collection.get_metadata()
    
    #Locate specimen
    specimen = collection_metadata.loc[collection_metadata["code"].astype(str)==str(code),"species"]
    
    if specimen.empty:
        err_msj = (f"No species data for {code} in collection {collection_name}")
        
        ##logging.error(f'An error occurred: {err_msj}')
        print(err_msj)
        result = ""
    #pri*nt("not mt")
    #pri*nt(f"specimen species {specimen}")
    try:
        result = str(specimen.iloc[0])
    except Exception as e:
        err_msj = ("Update specimen in the corresponding collection, please")
        #logging.error(f'An error occurred: {err_msj}')
        print(err_msj)
        return "na"
    #pri*nt(f"species, type{type(result)}")
    if isinstance(result,str):
        return result
    else:
        return str(result)
                

#Spectrum class
class Spectrum:
    """This class represents the data and metadata for a L1050 file. """
    #

    def __str__(self):
        return self.code

    def get_polarization(self):
        return self.polarization
    def get_name(self):
            return self.filename
    def get_filename(self):
            return self.filename
        
    def __init__(self, file_location, collection=None, genus = None, 
                 species = None, 
                 prominence_threshold_min = 0.15, 
                 prominence_threshold_max = 0.40, 
                 min_height_threshold_denominator = 3.0, 
                 max_height_threshold_denominator = 3.3,
                 min_distance_between_peaks = 160, 
                 max_distance_between_peaks = 160, 
                 min_wavelength = None, 
                 max_wavelength = None, equipment = None):

        #attributes
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
        self.file_location = file_location
        self.collection = collection
        #read file
        self.metadata, self.data = read_spectrum_file(file_location)
        #pri*nt(self.metadata)

        self.filename =  file_location
        try:
            self.polarization = self.metadata["polarization"]
        except:
            self.polarization = "na"
        try:
            self.code = self.metadata["code"]
        except:
            self.code = "na"
        try:
            self.measuring_mode = self.metadata["measuring_mode"]
        except:
            self.measuring_mode = "na"
        try:
            if not genus:
                self.genus = get_genus(self.code, collection)
            else:
                self.genus = genus
        except:
            self.genus = "na"
        try:
            if not species:
                self.species = get_species(self.code, collection)
            else:
                self.species = species
        except:
            self.species = "na"

        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        #print("creating peaklist")
        self.peaklist = PeakList(self,
                                prominence_threshold_min, 
                                     prominence_threshold_max, 
                                     min_height_threshold_denominator, 
                                     max_height_threshold_denominator,
                                     min_distance_between_peaks, 
                                     max_distance_between_peaks,
                                     min_wavelength,
                                     max_wavelength,
                                     )
        
    def filter_wavelengths(self, df):
        #filter wavelengths
        if self.min_wavelength:
            df = df[df["wavelength"]>self.min_wavelength]
        if self.max_wavelength:
            df = df[df["wavelength"]<self.max_wavelength]
        if self.min_wavelength and self.max_wavelength:
            df = df[(df["wavelength"]<self.max_wavelength) & (df["wavelength"]>self.min_wavelength)]
        return df
        
    def plot_settings(self):
        def title_maker(measuring_mode, code, genus=None, species=None):
            measuring_modes = {"A": "absorptance", "T": "transmittance", "R": "reflectance"}
            title = f"{measuring_modes[measuring_mode]} for "
            if genus == "na" and species == "na":
                title += f"code = {code}"
            else:
                title += f"{self.genus} {self.species}, code {self.code}"
            return title
        measuring_mode = self.metadata["measuring_mode"]

        df = self.data

        #filter wavelengths
        df = self.filter_wavelengths(df)
        
        #print(f"{df}")
        x = df["wavelength"]
        y = df[measuring_mode]

        plt.plot(x, y)
        plot = plt.title( title_maker(measuring_mode, self.code, self.genus, self.species))
        #print(f"plot 1: {plot}")
        return plot

    @plot_wrapper
    def plot(self, plot_maxima = False, plot_minima = False):
        plot = self.plot_settings()

        if plot_maxima:
            self.plot_maxima()
        if plot_minima:
            self.plot_minima()
        #print(f"plot 2: {plot}")
        return 
    
    def plot_maxima(self):
        min_i, x_values, y_values = self.peaklist.get_maxima(self, self.min_wavelength , self.max_wavelength )
        return plt.scatter(x_values, y_values, color="r")
    
    def plot_minima(self):
        min_i, x_values, y_values = self.peaklist.get_minima(self, self.min_wavelength , self.max_wavelength )
        return plt.scatter(x_values, y_values, color="r")
    
    def get_normalized_spectrum(self):
        df = self.data[["wavelength", self.measuring_mode]]
        max_value = df[self.measuring_mode].max()
        df.loc[:,self.measuring_mode] = df.loc[:,self.measuring_mode]/max_value
        return df

    def get_maxima(self,min_wavelength = None, max_wavelength = None):
        maxima_list = self.peaklist.get_maxima(self, min_wavelength , max_wavelength )
        return maxima_list

    def get_minima(self, min_wavelength = None, max_wavelength = None):
        minima_list = self.peaklist.get_minima(self, min_wavelength , max_wavelength )
        return minima_list

    def get_critical_points(self, min_wavelength = None, max_wavelength = None):
        peaks = self.peaklist.get_peaks(min_wavelength, max_wavelength)
        return peaks

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
            import re
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
        return alphanum_key(self.code) < alphanum_key(other.code)
    
    def get_peaks_as_object(self, get_maxima = True, get_minima = True, min_wavelength = None, max_wavelength = None):
        import scipy
        #get info
        x = self.data["wavelength"].values
        y = self.data[self.metadata["measuring_mode"]].values

        #parameters
        min_height = y.max()/self.max_height_threshold_denominator

        width_t = 50.00

        #get peaks
        if get_minima:
            min_i, min_x_values, min_y_values = self.peaklist.get_minima(self, min_wavelength , max_wavelength )
        if get_maxima:
            max_i, max_x_values, max_y_values = self.peaklist.get_maxima(self, min_wavelength, max_wavelength )

        peaks = []

        #pri*nt("peak called")
        if get_maxima:
            for i in zip(max_x_values, max_y_values):
                max_peak = Peak(i[0], i[1])
                peaks.append(max_peak)
        if get_minima:
            for i in zip(min_x_values, min_y_values):
                min_peak = Peak(i[0], i[1])
                peaks.append(min_peak)

        peaks = sorted(peaks)
        return peaks

########################################################################################################################

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

    def set_parameters(self, prominence_threshold_min = None, prominence_threshold_max = None, min_height_threshold_denominator = None, max_height_threshold_denominator = None,
    min_distance_between_peaks = None, max_distance_between_peaks =None ):
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
            

    def __init__(self, spectrum, 
                 prominence_threshold_min = 0.15, 
                 prominence_threshold_max = 0.40, 
                 min_height_threshold_denominator = 3.0, 
                 max_height_threshold_denominator = 3.3,
                 min_distance_between_peaks = 160, 
                 max_distance_between_peaks = 160,
                 min_wavelength = None, 
                 max_wavelength = None):
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
            
        self.spectrum = spectrum

    def get_spectrum(self):
        return self.spectrum

   

    def get_peaks(self, spectrum =None, min_wavelength = None, max_wavelength = None):

        peaks = spectrum.get_peaks_as_object(min_wavelength, max_wavelength)
        x = []
        y = []
        #filter peaks far from the detector jump 
        for peak in peaks:
            if not ((peak.get_x() > 855)&(peak.get_x() < 869)):
                x.append(peak.get_x())
                y.append(peak.get_y())

        return x, y


    def plot_settings(self, min_wavelength = None, max_wavelength = None):
        self.spectrum.plot_settings()
        x_values, y_values = self.get_peaks(min_wavelength, max_wavelength )

        return plt.scatter(x_values, y_values, color="r")

    @plot_wrapper
    def plot(self, min_wavelength = None, max_wavelength = None):
        plot = self.plot_settings(min_wavelength , max_wavelength )
        return plot


    def get_minima(self, spectrum, min_wavelength = None, max_wavelength = None):
        """This method returns the index, x values and y values of every minimum in a spectrum"""

        #Get minimum
        #spectrum = self.get_spectrum()
        #get wavelength and height of measurements
        df = spectrum.data

        #filter wavelengths
        
        if min_wavelength and max_wavelength:
            df = df[(df["wavelength"]<max_wavelength) & (df["wavelength"]>min_wavelength)]
        elif min_wavelength:
            df = df[df["wavelength"]>min_wavelength]
        elif max_wavelength:
            df = df[df["wavelength"]<max_wavelength]
            
        #get wavelength and height of measurements
        x = df["wavelength"].values
        y = df[spectrum.metadata["measuring_mode"]].values

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

    def get_maxima(self, spectrum, min_wavelength = None, max_wavelength = None):
        """This method returns the index, x values and y values of every maxima in a spectrum"""
        #spectrum = self.get_spectrum( )
        #get wavelength and height of measurements
        df = spectrum.data

        #filter wavelengths
        if min_wavelength and max_wavelength:
            df2 = df[(df["wavelength"]<max_wavelength) & (df["wavelength"]>min_wavelength)]
        elif min_wavelength:
            df2 = df[df["wavelength"]>min_wavelength]
        elif max_wavelength:
            df2 = df[df["wavelength"]<max_wavelength]
        else:
            df2 = df
        
        x = df2["wavelength"].values
        y = df2[spectrum.metadata["measuring_mode"]].values

        #define minimum height and min distance between minima
        min_height = y.max()/self.min_height_threshold_denominator
        min_distance = 50 #nm
        max_distance = 100.00
        width_t = 50.00

        #get maxima
        #print(f"{min_height=}{self.max_distance_between_peaks=}{self.prominence_threshold_max=}")
        peaks_funct = scipy.signal.find_peaks(y, height= min_height, distance= self.max_distance_between_peaks, prominence= self.prominence_threshold_max)
        peaks_index = peaks_funct[0] #indices
        x_values = x[peaks_index]   #x values
        y_values = y[peaks_index]    #y values

        return peaks_index, x_values, y_values

