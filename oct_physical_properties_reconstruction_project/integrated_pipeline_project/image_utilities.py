import os
import re
import numpy as np
from PIL import Image
from collections import defaultdict
from pathlib import Path

import os


def list_valid_files(base_folder, recursively=False, ignore_lst=None, ext = ".tif"):
    """
    Lists all .tif files in base_folder, optionally recursively, 
    while ignoring folders whose names appear in ignore_lst.
    
    Parameters:
    - base_folder (str): Path to the folder.
    - recursively (bool): If True, include subfolders.
    - ignore_lst (list): List of folder names to ignore (exact name match).
    
    Returns:
    - List[str]: Full paths to valid .tif files.
    """
    if ignore_lst is None:
        ignore_lst = []

    tif_files = []

    if recursively:
        for root, dirs, files in os.walk(base_folder):
            # Skip ignored folders
            if any(ignored in os.path.normpath(root).split(os.sep) for ignored in ignore_lst):
                continue
            for f in files:
                if f.lower().endswith(ext):
                    full_path = os.path.join(root, f)
                    if os.path.isfile(full_path):
                        tif_files.append(full_path)
    else:
        for f in os.listdir(base_folder):
            full_path = os.path.join(base_folder, f)
            if os.path.isdir(full_path) and os.path.basename(full_path) in ignore_lst:
                continue
            if f.lower().endswith(ext) and os.path.isfile(full_path):
                tif_files.append(full_path)

    return tif_files



def extract_identifiers(filename, re_name ):
    testing  = False
    if testing: print(f"current filename: ", filename)
    # Matches pattern: anything_XX-YY.tif where XX is vertical, YY is horizontal
    re_dict = {
                "single_scan_re" : r'(\d+)-(\d+).tif',
                "single_volume_scan" : r'.*-(\d+).tif',
                "volume_scan_re" : r'_(\d+)-(\d+)_(\d+)_(\d+)-(\d+).tif',
                "median_files_re" : r"(\d+)-(\d+)_med.tif",
                "average_files_re" : r"(\d+)-(\d+)_avg.tif",
                "gauss_files_re": r"gauss_and_curve2_(\d+)-(\d+)_med.png"
    
                }
    if (re_name == "gauss_files_re") or (re_name == "single_scan_re") or (re_name == "median_files_re") or (re_name == "average_files_re"):
            
        match = re.search(re_dict[re_name], filename)
        if match:
            v = int(match.group(1))
            h = int(match.group(2))
            return v, h
        return None, None
    elif re_name == "volume_scan_re":
        
        match = re.search(re_dict[re_name], filename)
        if match:
            v = int(match.group(1))
            h = int(match.group(2))
            focal_value = int(match.group(3))
            focus_value = int(match.group(4))
            v_3d = int(match.group(5))
            v_joined = v*512+v_3d
            return v_joined, h #todo change accordingly
        return None, None
    elif re_name == "single_volume_scan":
        
        match = re.search(re_dict[re_name], filename)
        if match:
            v = int(match.group(1))
            h = 1
            
            return v, h #todo change accordingly
        return None, None
    

def group_by_v_h(base_folder, re_name = "volume_scan_re",
                    ignore_lst = None, recursive = True, ext = ".tif"):
    testing  = False
    
    grouped = defaultdict(list)
    
    list_files = list_valid_files(base_folder, recursive, ignore_lst, ext )
    
    if testing: print(f"list_files in {base_folder}",list_files)
    for full_path in list_files:
        basename = Path(full_path).name
        v, h = extract_identifiers(basename, re_name)
        if testing:  print("identifiers, v, h", basename, v,h)
        if testing: print("v, h",v, h)
        if v is not None and h is not None:
            grouped[(v, h)].append(full_path)
    return grouped
