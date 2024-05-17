#!/usr/bin/env python
# coding: utf-8
# In[ ]:
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import math
import re
# This line of code allow us to access data in colab
# In[ ]:


class Specimen_Collection:
    """Define Specimen Collection class"""

    def read_collection(self, database_path):
        with open(database_path, encoding="latin1") as f:
            df = pd.read_csv(f, sep="\t", decimal=",",
                             header=0, encoding="iso-8859-1")
            return df

    def __init__(self, name, data_folder_path, metadata_path, quality):
        self.name = name
        self.data_folder_path = data_folder_path
        self.metadata = self.read_collection(metadata_path)
        self.quality = quality
        self.description = "No description"

    def set_description(self, description):
        self.description = description

# In[ ]:

# In[ ]:


class Spectrum:
    """This class reads L1050 files, saves its metadata and data and creates a
    spectrum object """

    def get_metadata_and_dataframe(file_location):
        """Reads metadata and dataframe info from a file location"""
        import os
        import pandas as pd
        import re

        def get_sample_code_from_filename(row_str, file_location):
            # print("string")
            # print(file_location)
            filename = os.path.basename(file_location)
            re1 = r"([a-zA-Z\d]+)(?:-\d)*(?:.Sample)*.(?:txt)*(?:ASC)*"
            # Names are in the form CODE-MEASUREMENTNUMBER.TXT
            p = re.compile(re1)
            m = p.match(filename)
            # print(f"match filename: {m}")
            if m:
                # print(f"group 1: {m.group(1)}")
                return (m.group(1))
            return get_sample_code(row_str)

        def get_sample_code(row_str):
            # Tries to get the sample code from the file, if it does not match
            # it tries to get it from the filename.
            # print("string")
            # print(row_str)
            re1 = r"([a-zA-Z\d]+)(?:-\d)*(?:.Sample)*.(?:txt)*(?:ASC)*"
            # Names are in the form CODE-MEASUREMENTNUMBER.TXT
            p = re.compile(re1)
            m = p.match(row_str)
            # print(f"match: {m}")
            if m:
                return (m.group(1))
            else:
                ""

        def responses(str):
            re1 = "\d+/(\d+,\d+) \d+,\d+/(\d+,\d+)"
            p = re.compile(re1)
            m = p.match(str)
            if m:
                return m.group(1), m.group(2)
            else:
                return "", ""

        def attenuator_settings(str):
            re1 = "S:(\d+,\d+) R:(\d+,\d+)"
            p = re.compile(re1)
            m = p.match(str)
            if m:
                return m.group(1), m.group(2)
            else:
                return "", ""

        def slit_pmt_aperture(str):
            re1 = "\d+/servo \d+,\d+/(\d+,\d+)"
            p = re.compile(re1)
            m = p.match(str)
            if m:
                return m.group(1)
            else:
                return ""
        # Initialize metadata dict
        metadata = {}

        # Read header
        lines = []
        with open(file_location, encoding="latin1") as myfile:
            lines = myfile.readlines()[0:90]
        metadata["header"] = "".join(lines)

        # read_metadata
        f = open(file_location, encoding="latin1")

        df = pd.DataFrame()
        with f as data_file:
            for index, row in enumerate(data_file):  # 0-89

                row_str = row.strip()
                if index + 1 == 3:  # Filename and extension
                    metadata["filename"] = row_str
                    metadata["code"] = get_sample_code_from_filename(
                        row_str, file_location)
                if index + 1 == 4:  # date DD/MM/YYYY
                    metadata["date"] = row_str
                if index + 1 == 5:  # Time HH:MM:SS.SS
                    metadata["time"] = row_str
                if index + 1 == 8:  # user
                    metadata["user"] = row_str
                if index + 1 == 9:  # description
                    metadata["description"] = row_str
                if index + 1 == 10:  # minimum wavelength
                    metadata["minimum_wavelength"] = row_str
                if index + 1 == 12:  # equipment name
                    metadata["equipment"] = row_str
                if index + 1 == 13:  # equipment series
                    metadata["series"] = row_str
                if index + 1 == 14:  # data visualizer version, equipment version, date and time
                    metadata["software"] = row_str
                if index + 1 == 21:  # Operating mode
                    metadata["operating_mode"] = row_str
                if index + 1 == 22:  # Number of cycles
                    metadata["cycles"] = row_str
                if index + 1 == 32:  # range/servo
                    metadata["slit_pmt"] = slit_pmt_aperture(row_str)
                if index + 1 == 33:
                    metadata["response_ingaas"], metadata["response_pmt"] = responses(
                        row_str)
                if index + 1 == 35:  # pmt gain, if 0 is automatic
                    metadata["pmt_gain"] = row_str
                if index + 1 == 36:  # InGaAs detector gain
                    metadata["ingaas_gain"] = row_str
                if index + 1 == 42:  # monochromator wavelength nm
                    metadata["monochromator_change"] = row_str
                if index + 1 == 43:  # lamp change wavelength
                    metadata["lamp_change"] = row_str
                if index + 1 == 44:  # pmt wavelength
                    metadata["pmt_change"] = row_str
                if index + 1 == 45:  # beam selector
                    metadata["beam_selector"] = row_str
                if index + 1 == 46:
                    metadata["cbm"] = row_str
                if index + 1 == 47:  # cbd status, on/off
                    metadata["cbd_status"] = row_str
                if index + 1 == 48:  # attenuator percentage
                    metadata["attenuator_sample"], metadata["attenuator_reference"] = attenuator_settings(
                        row_str)
                if index + 1 == 49:
                    metadata["polarizer"] = row_str
                if index + 1 == 80:
                    metadata["units"] = row_str
                if index + 1 == 81:
                    metadata["measuring_mode"] = row_str
                if index + 1 == 84:
                    metadata["maximum_wavelength"] = row_str
                if index + 1 == 85:
                    metadata["step"] = row_str
                if index + 1 == 86:
                    metadata["number_of_datapoints"] = row_str
                if index + 1 == 88:
                    metadata["maximum_measurement"] = row_str
                if index + 1 == 89:
                    metadata["minimum_measurement"] = row_str
                if index + 1 == 90:
                    break
            df = pd.read_csv(f, sep="\t", decimal=".", names=[
                             "wavelength", metadata["measuring_mode"]]).dropna()
            df = df[df["wavelength"] < 2000]
            df["wavelength"], df[metadata["measuring_mode"]] = df["wavelength"].astype(
                float), df[metadata["measuring_mode"]].astype(float)
            return metadata, df

    def __str__(self):
        return self.code

    def __init__(self, name, metadata, data, database):

        import re

        def get_genus(code, database):
            # print("get_genus")

            specimen = database.loc[database["code"] == code]

            if specimen.empty:
                print(f"No data for {code} in database {database}")
                return ""
            # print("not mt")
            # print(f"specimen genus {specimen}")
            result = specimen.iloc[0]["genus"]
            # print(f"genus, type{type(result)}")
            if isinstance(result, str):

                return result
            else:

                return str(result)

        def get_species(code, database):
            # print("get_species")
            # print(f"code: {code}")
            specimen = database.loc[database["code"] == code]

            if specimen.empty:
                print(f"No data for {code} in database {database}")
                result = ""
            # print("not mt")
            # print(f"specimen species {specimen}")
            result = str(specimen.iloc[0]["species"])
            # print(f"species, type{type(result)}")
            if isinstance(result, str):

                return result
            else:

                return str(result)

        # attributes
        self.name = name
        self.metadata = metadata
        self.code = metadata["code"]
        self.data = data
        self.database = database
        self.filename = metadata["filename"]
        self.genus = get_genus(self.code, database)
        self.species = get_species(self.code, database)
        self.measuring_mode = self.metadata["measuring_mode"]

    def plot(self):
        measuring_mode = self.metadata["measuring_mode"]
        return self.data.plot(x="wavelength", y=self.metadata["measuring_mode"], grid=True, markersize=3, title=f"{measuring_mode} for {self.genus} {self.species}, code {self.code}")

    def get_normalized_spectrum(self):
        df = self.data[["wavelength", self.measuring_mode]]
        max_value = df[self.measuring_mode].max()
        df[self.measuring_mode] = df[self.measuring_mode]/max_value
        return df


def read_spectrum(file_path, database):
    metadata, df = get_metadata_and_dataframe(file_path)
    # print(metadata)
    # print(df)
    spectrum = Spectrum(metadata["filename"], metadata, df, database)
    # print(spectrum.data)
    return spectrum


def read_spectra_from_folder(folder_path, database_metadata):

    # list files in folder
    file_list = os.listdir(folder_path)

    # file extension
    file_extension = ".txt"

    # filters a list of strings to create a new list containing only the elements that end with file_extension

    def filter_substring_elements(path_strings, substring):
        filtered_paths = [path for path in path_strings if substring in path]
        return filtered_paths

    # full path list
    filtered_list = [os.path.join(folder_path, path)
                     for path in filter_substring_elements(file_list, file_extension)]

    # read each element of filtered_list
    spectra = []

    for path in filtered_list:
        spectrum = read_spectrum(path, database_metadata)
        spectra.append(spectrum)

    return spectra


# In[ ]:


class Peak:
    """It represents a maximum in a transmittance/reflectance/absorptance graph"""

    def __init__(self, x, y):
        self.x_value = x
        self.y_value = y

    def __str__(self):
        return f"({self.x_value}, {self.y_value})"

    def __repr__(self):
        return f"({self.x_value}, {self.y_value})"


class PeakList:
    """It represents a list of all the peaks for a particular spectrum """

    def __init__(self, spectrum):
        self.spectrum = spectrum
        self.peaks = self.get_peaks()

    def get_peaks(self):
        # variables
        prominence_threshold_min = 0.15
        prominence_threshold_max = 0.60
        min_height_threshold_denominator = 3.0
        max_height_threshold_denominator = 2.5
        min_distance_between_peaks = 125  # nm

        import scipy

        # get info
        x = self.spectrum.data["wavelength"].values
        y = self.spectrum.data[self.spectrum.metadata["measuring_mode"]].values

        # parameters
        min_height = y.max()/max_height_threshold_denominator

        width_t = 50.00

        # get peaks

        peaks_funct = scipy.signal.find_peaks(
            y, height=min_height, distance=min_distance_between_peaks, prominence=prominence_threshold_max)

        # print(f"peaks_funct {peaks_funct}")
        peaks_index = peaks_funct[0]
        # print(f"peaks_index {peaks_index}")
        x_values = x[peaks_index]
        y_values = y[peaks_index]

        peaks = []
        for i in range(len(x_values)):
            peak = Peak(x_values[i], y_values[i])
            peaks.append(peak)
        return peaks

    def plot(self):
        self.spectrum.plot()
        x_values = []
        y_values = []
        for peak in self.get_peaks():
            x_values.append(peak.x_value)
            y_values.append(peak.y_value)
        return plt.scatter(x_values, y_values, color="r")
