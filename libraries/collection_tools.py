#!/usr/bin/env python
# coding: utf-8

# In[3]:


"""How to use:
collection_dict = get_collections()


"""
from datapath_selector import get_paths
from spectraltools import *

def get_collections_list():
    """OPTIONS: cicima_laptop, colaboratory, wfh, cicima_desktop
        """
    #get the paths for each collection
    collection_paths = get_paths()

    #load the collections into memory
    
    inbio_2018_2019_collection = Specimen_Collection("INBIO", collection_paths["2018_2019_inbio_collection_path"] , collection_paths["2018_2019_inbio_collection_metadata"] , "HIGH")
    angsol_collection = Specimen_Collection("ANGSOL", collection_paths["angsol_collection_path"] , collection_paths["angsol_collection_metadata"] , "HIGH")
    angsol_collection.set_description("ANGSOL collection has specimens that belong to Angel Solís. The confidence that we have about specimen identification is high.")
    
    cicimaucr_collection = Specimen_Collection("CICIMAUCR1", collection_paths["cicimaucr_collection_path"] , collection_paths["cicima_ucr_metadata"] , "HIGH")
    cicimaucr_collection_2 = Specimen_Collection("CICIMAUCR2", collection_paths["cicimaucr_collection_2_path"] , collection_paths["cicima_ucr_metadata"] , "HIGH")
    cicimaucr_collection_3 = Specimen_Collection("CICIMAUCR3", collection_paths["cicimaucr_collection_3_path"] , collection_paths["cicima_ucr_metadata"] , "HIGH")
    inbucr_collection = Specimen_Collection("INBUCR", collection_paths["inbucr_collection_path"] , collection_paths["inbucr_metadata"] , "HIGH")
    bioucr_collection = Specimen_Collection("BIOUCR", collection_paths["bioucr_collection_path"] , collection_paths["bioucr_metadata"] , "LOW")

    #return a dictionary with the collections
    collection_list = [inbio_2018_2019_collection, angsol_collection, cicimaucr_collection, cicimaucr_collection_2, cicimaucr_collection_3, inbucr_collection, bioucr_collection]
    return collection_list

def get_collections_dict():
    collection_list = get_collections_list()
    collection_dict = {item.get_name() : item for item in collection_list}
    
    return collection_dict

def get_specimen_info(code):
    """
    Returns the specified column and all other columns except 'code' for the given code from the collection list.

    Parameters:
    - code (str): The code to search for.
    - collection_list (list): List of collections to search in.
    - column_name (str): The name of the column to retrieve alongside other columns except 'code'.

    Returns:
    - result (DataFrame or str): A DataFrame containing the requested data or 'na' if not found.
    """
    
    collection_list = get_collections_list()
    
    for collection in collection_list:
        codes = list(collection.get_codes())
        if code in codes:
            metadata = collection.get_metadata()
            try:
                # Filter the row matching the given code
                filtered_row = metadata.loc[metadata["code"].astype(str) == code]

                if filtered_row.empty:
                    return "na"

                
                selected_columns = [col for col in metadata.columns]
                result = filtered_row[selected_columns]

                return result
            except Exception as e:
                print(f"Error retrieving data: {e}")
                return "na"

    print(f"The provided code ({code}) is not in the collection list.")
    return "na"


# In[ ]:




