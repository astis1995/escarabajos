import pandas as pd
import os
from .spectrum import Spectrum
from .utils import create_path_if_not_exists
DEBUG = False 

class Specimen_Collection:
    """This class represents a physical collection of specimens"""
    def read_collection(self, collection_path):
        debug = DEBUG
        
        with open(collection_path, encoding="latin1") as f:
            df = pd.read_csv(f, sep="\t", decimal=",", header=0, encoding="iso-8859-1")
            if debug:
                print(df)
            return df

    def __init__(self, name=None, data_folder_path=None, metadata_path=None, quality=None, config_file = None):
        self.name = name
        self.data_folder_path = data_folder_path
        self.metadata = self.read_collection(metadata_path) if metadata_path else pd.DataFrame()
        self.quality = quality
        self.description = "No description"
        self.config_file = config_file

    def set_description(self, description):
        self.description = description

    def get_name(self):
        return self.name

    def get_metadata(self):
        return self.metadata

    def get_codes(self):
        codes = set(self.metadata["code"].values)
        codes = list(map(str, codes))
        return codes

    def get_species(self):
        species = set(self.metadata["species"])
        return species

    def get_genera(self):
        genera = set(self.metadata["genus"])
        return genera

    def get_data_folder_path(self):
        return self.data_folder_path

    def get_data_filenames(self):
        """Gets every filename under data_folder_path with the extension in file_extension"""
        debug = DEBUG
        
        folder_path = self.get_data_folder_path()
        if debug:
            print("folder_path",folder_path )
            
        if not folder_path or not os.path.exists(folder_path):
            return []
        file_list = os.listdir(folder_path)
        if debug:
            print("file_list", file_list)
            
        file_extension = [".txt", ".csv", ".TXT"]
        filtered_paths = []
        for extension in file_extension:
            filtered_paths += [path for path in file_list if extension in path]
        
        if debug:
            print("filtered_paths", filtered_paths)
            
        full_paths = [os.path.join(folder_path, path) for path in filtered_paths]
        
        if debug:
            print("full_paths", full_paths)
        return full_paths

    def read_spectrum(self, file_path, min_wavelength=None, max_wavelength=None):
        return Spectrum(file_location = file_path, config_file = self.config_file,  collection = self )

    def get_spectra(self, min_wavelength=None, max_wavelength=None):
        debug = DEBUG
        
        filenames = self.get_data_filenames()
        
        if debug:
            print("\nfilenames: ", filenames)
        spectra = []
        for filename in filenames:
            if debug:
                print("Current: ", filename)
            spectrum = Spectrum(file_location = filename, config_file = self.config_file,  collection = self )
            if spectrum and not spectrum.data.empty:
                spectra.append(spectrum)
            else:
                print(f"The following filename {filename} was not converted into a spectrum. Check it")
        return spectra

    def genus_lookup(self, code, collection_list):
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
        genus = self.genus_lookup(code, collection_list)
        species = self.species_lookup(code, collection_list)
        return genus, species

    def collection_lookup(self, code, collection_list):
        code = str(code)
        for collection in collection_list:
            codes = [str(num) for num in collection.get_codes()]
            if code in codes:
                return collection
        print(f"The provided code ({code}) is not in the collection list.")
        return None

    def __str__(self):
        return self.name if self.name else "None"

    def __repr__(self):
        return self.name if self.name else "None"