import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import warnings
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import colorsys

# Constants
min_craic_wv = 420
max_craic_wv = 950
min_craic_wv_polarized = 420
max_craic_wv_polarized = 700
regex_dict = {
    "l1050_filename_regex": r"([a-zA-Z]+\\d{4})-*_*(\\d)*(?:.Sample)*.(?:txt)*(?:ASC)*",
    "craic_data_comma_regex": r"\d*.\\d*,\\d*.\\d*\n",
    "craic_data_tab_regex": r"\d*.\\d*\t\\d*.\\d*",
    "craic_filename_regex_0": r"(\\d+?).csv",
    "craic_filename_regex_1": r"(\\d+?)([RLOT])(\\d+).csv",
    "craic_filename_regex_2": r"([a-zA-Z\\d]+)_([RLOT]).csv",
    "craic_filename_regex_3": r"(\\d+?)-(\\d+)*.csv",
    "craic_filename_regex_4": r"(\\d+?)([RLOT])(\\d+)-(?:(elytrum)*(pronotum)*(escutelo)*).csv",
    "craic_filename_regex_5": r"(\\d+?)-variacion(\\d+).csv",
    "craic_filename_regex_6": r"(\\d+)(?:(escutelo)*(pronoto)*)([RLOT])(\\d+)",
    "macraspis-blue-1": r"(\\d+)-macraspis-blue-average([TOLR]).csv",
    "sinnumero-rojo": r"sinnumero-rojo-(\\d+)",
    "macraspis-green": r"(\\d+)-macraspis-green-average([LRTO]).csv",
    "macraspis-chrysis": r"1-macraspis-chrysis-average([LR](tot)).csv",
    "calomacraspis-1": r"calomacraspis-(?:(elytrum)*(pronotum)*(escutelo)*)-std([LRTO]).csv",
    "calomacraspis-2": r"calomacraspis-(?:(elytrum)*(pronotum)*(escutelo)*(scutellum)*)([LRTO]).csv",
    "cupreomarginata-1": r"cupreo-average([LROT]).csv",
    "cupreomarginata-2": r"cupreo([LROT])-std.csv",
    "cupreomarginata-3": r"cupreoT-averageL.csv",
    "cupreomarginata-4": r"(ojo)-ccupreomarginata-izquierdo-([LRTO]).csv",
    "resplendens-CVG": r"resplendensCVG([LOTR](total)*)CP.csv",
    "avantes_avalight_filename_regex": r"([a-zA-Z\d\s]+)_([TBM])_(\d+SP)\.TXT",
    "avantes_avalight_filename_regex_complete_beetle": r"^(CICIMAUCR\d+)(?:\s+MAX\d*)?_(\d+SP)\.TXT$",
}

def integer_generator(start=0):
    if start % 10 == 0:
        print(f"{start=}")
    while True:
        yield start
        start += 1

gen = integer_generator()

def get_contrasting_color():
    index = next(gen)
    if index < 0:
        raise ValueError("Index must be a non-negative integer.")
    total_colors = 360
    hue = (index * 137.508) % total_colors
    saturation = 0.8
    lightness = 0.5
    r, g, b = colorsys.hls_to_rgb(hue / 360.0, lightness, saturation)
    return (r, g, b)

def draw_rainbow_background():
    fig, ax = plt.subplots()
    colors = ["#8A2BE2", "#0000FF", "#00FF00", "#FFFF00", "#FFA500", "#FF0000"]
    custom_cmap = LinearSegmentedColormap.from_list("violet_to_red", colors, N=256)
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect='auto', cmap=custom_cmap, extent=[380, 750, 0, 200], alpha=0.15)
    return ax

def plot_wrapper(func):
    def wrapper(*args, **kwargs):
        plt.figure(figsize=(10, 5))
        result = func(*args, **kwargs)
        plt.grid(True)
        plt.show()
        return result
    return wrapper

def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def check_fluorometer_file(file):
    if Path(file).name.startswith("."):
        return False
    if not Path(file).name.endswith(".asc"):
        return False
    with open(file) as f:
        first_line = f.readline()
        regex = r"PE FL       SUBTECH     SPECTRUM    ASCII       PEDS        4.00"
        match = re.search(regex, first_line)
        return bool(match)

def check_CRAIC_file(file):
    if Path(file).name.startswith("."):
        return False
    if not Path(file).name.endswith(".csv"):
        return False
    with open(file) as f:
        first_line = f.readline()
        regex = r"Time1=\d*ms:Average1=\d*:Objective=\d*X:Aperture=\d*: \(\d*/\d*/\d* \d*:\d*:\d* (?:AM)*(?:PM)*\)"
        match = re.search(regex, first_line)
        return bool(match)

def check_l1050_file(file):
    debug = False
    if debug:
        print(file)
        
    if Path(file).name.startswith("."):
        return False
    if not (Path(file).name.endswith(".txt") or Path(file).name.endswith(".ASC")):
        return False
    with open(file) as f:
        first_line = f.readline()
        regex = r"PE UV       SUBTECH     SPECTRUM    ASCII       PEDS        .*"
        match = re.search(regex, first_line)
        return bool(match)

import re
from pathlib import Path

def check_avantes_avalight_file(file ):
    """
    Checks whether a given file matches any 'avantes_avalight' regex in regex_dict.
    Returns True if a match is found, False otherwise.
    """
    
    filename = Path(file).name

    # Exclude hidden files and non-TXT files
    if filename.startswith("."):
        return False
    if not filename.upper().endswith(".TXT"):
        return False

    # Check against all regexes containing 'avantes_avalight'
    for key, pattern in regex_dict.items():
        if "avantes_avalight" in key:
            p = re.compile(pattern, re.IGNORECASE)
            if p.fullmatch(filename):
                return True

    return False

def check_empty_CRAIC_file(f):
    try:
        if pd.read_csv(f, sep="\t", decimal=".", names=["wavelength", "measuring_mode"], skiprows=9).empty:
            return True
    except:
        pass
    try:
        if pd.read_csv(f, sep=",", decimal=".", names=["wavelength", "measuring_mode"], skiprows=9).empty:
            return True
    except:
        pass
    return False

def check_empty_l1050_file(f):
    try:
        if pd.read_csv(f, sep="\t", decimal=",", names=["wavelength", "measuring_mode"], skiprows=90, encoding="latin1").empty:
            return True
    except:
        pass
    return False

def check_empty_avantes_avalight_file(f):
    try:
        if pd.read_csv(f, sep=";", decimal=",", names=["wavelength", "sample", "dark", "reference", "reflectance"], skiprows=7).empty:
            return True
    except:
        pass
    return False

def read_l1050_file(file):
    return get_metadata_and_dataframe_l1050(file)

def read_CRAIC_file(file):
    return get_metadata_and_dataframe_CRAIC(file)

def read_fluorometer_file(file):
    return get_metadata_and_dataframe_fluorometer(file)

def read_avantes_avalight_file(file):
    return get_metadata_and_dataframe_avantes_avalight(file)

def read_spectrum_file(file):
    debug = False
    if debug:
        print("read_spectrum_file: ",file)
        
    if check_l1050_file(file):
        return read_l1050_file(file)
    elif check_CRAIC_file(file):
        return read_CRAIC_file(file)
    elif check_fluorometer_file(file):
        return read_fluorometer_file(file)
    elif check_avantes_avalight_file(file):
        return read_avantes_avalight_file(file)
    else:
        print(f"The file {file} is neither a valid L1050, CRAIC, fluorometer, nor Avantes AvaLight file.")
        return None, None

def get_metadata_from_filename(file_location):
    basename = Path(file_location).name
    code, polarization = get_metadata_from_basename(basename)
    return code, polarization

def get_metadata_from_basename(basename):
    info = get_info_from_format(basename)
    code = info["code"]
    polarization = info["polarization"] or "T"
    return code, polarization

def first_line(str):
    re1 = r"Time1=(\d*)ms:Average1=(\d*).*:Objective=(\d*X):Aperture=(\d*): \((\d*/\d*/\d*) (\d*:\d*:\d* (AM)*(PM)*)\)"
    p = re.compile(re1)
    m = p.match(str)
    if m:
        return m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6)
    return "", "", "", "", "", ""

def measuring_mode(str):
    if str:
        if str == "Reflectance":
            return "%R"
        elif str == "Transmittance":
            return "%T"
        elif str == "Fluorescence":
            return "%F"
        elif str == "Absorptance":
            return "%A"
    return ""

def average_2(str):
    re1 = r"Avg2: (\d*.\d*)"
    p = re.compile(re1)
    m = p.match(str)
    return m.group(1) if m else ""

def integration_time1(str):
    re1 = r"Int.Time1:(\d*.\d*)"
    p = re.compile(re1)
    m = p.match(str)
    return m.group(1) if m else ""

def integration_time2(str):
    re1 = r"Int.Time2:(\d*.\d*)"
    p = re.compile(re1)
    m = p.match(str)
    return m.group(1) if m else ""

def get_format(string_line):
    for element in regex_dict:
        if re.fullmatch(regex_dict[element], string_line):
            return element
    return None

def get_CRAIC_info_from_filename(file):
    basename = Path(file).name
    return get_info_from_format(basename)

def get_info_from_format(string_line):
    debug = False
    if debug:
        print("Get info from format")
        print("String: ", string_line) 
    format_type = get_format(string_line)
    info = {
        "code": None,
        "polarization": None,
        "reading": None,
        "location": None,
        "genus": None,
        "species": None,
        "original": None
    }
    if format_type == "craic_filename_regex_0":
        p = re.compile(regex_dict[format_type])
        m = p.fullmatch(string_line)
        if m:
            info["code"] = m.group(1)
            info["polarization"] = "T"
            info["reading"] = 0
            info["original"] = False
            return info
    if format_type == "craic_filename_regex_3":
        p = re.compile(regex_dict[format_type])
        m = p.fullmatch(string_line)
        if m:
            info["code"] = m.group(1)
            info["polarization"] = None
            info["reading"] = m.group(2)
            info["original"] = True
            return info
    if format_type == "craic_filename_regex_1":
        p = re.compile(regex_dict[format_type])
        m = p.fullmatch(string_line)
        if m:
            info["code"] = m.group(1)
            info["polarization"] = m.group(2)
            info["reading"] = m.group(3)
            info["original"] = True
            return info
    if format_type == "craic_filename_regex_2":
        p = re.compile(regex_dict[format_type])
        m = p.fullmatch(string_line)
        if m:
            info["code"] = m.group(1)
            info["polarization"] = m.group(2)
            info["reading"] = None
            info["original"] = True
            return info
    if format_type == "craic_filename_regex_4":
        p = re.compile(regex_dict[format_type])
        m = p.fullmatch(string_line)
        if m:
            info["code"] = m.group(1)
            info["polarization"] = m.group(2)
            info["reading"] = m.group(3)
            info["location"] = m.group(4)
            info["original"] = True
            return info
    if format_type == "craic_filename_regex_5":
        p = re.compile(regex_dict[format_type])
        m = p.fullmatch(string_line)
        if m:
            info["code"] = m.group(1)
            info["polarization"] = "T"
            info["reading"] = m.group(2)
            info["location"] = None
            info["original"] = True
            return info
    if format_type == "craic_filename_regex_6":
        p = re.compile(regex_dict[format_type])
        m = p.fullmatch(string_line)
        if m:
            info["code"] = m.group(1)
            info["polarization"] = m.group(3)
            info["reading"] = m.group(4)
            info["location"] = m.group(2)
            info["original"] = True
            return info
    if format_type == "sinnumero-rojo":
        p = re.compile(regex_dict[format_type])
        m = p.fullmatch(string_line)
        if m:
            info["code"] = "sinnumero-rojo"
            info["polarization"] = "T"
            info["reading"] = m.group(1)
            info["location"] = None
            info["species"] = "boucardi"
            info["genus"] = "Chrysina"
            info["original"] = True
            return info
    if format_type == "macraspis-blue-1":
        p = re.compile(regex_dict[format_type])
        m = p.fullmatch(string_line)
        if m:
            info["code"] = "sinnumero-rojo"
            info["polarization"] = m.group(2)
            info["reading"] = m.group(1)
            info["location"] = None
            info["species"] = "sp."
            info["genus"] = "Macraspis"
            info["color"] = "blue"
            info["original"] = True
            return info
    if format_type == "macraspis-green":
        p = re.compile(regex_dict[format_type])
        m = p.fullmatch(string_line)
        if m:
            info["code"] = "macraspis-green"
            info["polarization"] = m.group(2)
            info["reading"] = m.group(1)
            info["location"] = None
            info["species"] = "sp."
            info["genus"] = "Macraspis"
            info["color"] = "green"
            info["original"] = True
            return info
    if format_type == "l1050_filename_regex":
        p = re.compile(regex_dict[format_type])
        m = p.fullmatch(string_line)
        if m:
            info["code"] = m.group(1)
            info["polarization"] = None
            info["reading"] = m.group(2)
            info["original"] = True
            return info
    if format_type == "avantes_avalight_filename_regex":
        p = re.compile(regex_dict[format_type])
        m = p.fullmatch(string_line)
        if m:
            info["code"] = m.group(1)
            info["measuring_location"] = m.group(2)
            info["spectrometer_name"] = m.group(3)
            info["polarization"] = "T"
            info["original"] = True
            return info
    if format_type == "avantes_avalight_filename_regex_complete_beetle":
        if debug:
            print("Regex match: avantes_avalight_filename_regex_complete_beetle")
        p = re.compile(regex_dict[format_type])
        m = p.fullmatch(string_line)
        if m:
            info["code"] = m.group(1)                 # specimen code, e.g. CICIMAUCR0016
            info["spectrometer_name"] = m.group(2)    # spectrometer, e.g. 7314920SP
            info["measuring_location"] = None         # not available in this filename
            info["polarization"] = "T"                # default assumption
            info["original"] = True
            
            if debug:
                print("Info: ", info)
            return info

    return info

def get_code_from_format(string_line):
    info = get_info_from_format(string_line)
    return str(info["code"]) if info["code"] else ""

def get_codes_from_filenames(files):
    codes = set(get_code_from_filename(file) for file in files if get_code_from_filename(file))
    return sorted(codes)

def get_polarization_from_format(string_line):
    info = get_info_from_format(string_line)
    return info["polarization"]

def get_reading_from_format(string_line):
    info = get_info_from_format(string_line)
    return info["reading"]

def get_metadata_and_dataframe_CRAIC(file_location):
    metadata = {}
    with open(file_location, encoding="latin1") as myfile:
        lines = myfile.readlines()[0:9]
        header = "".join(lines)
        first_data_line = lines[8] if len(lines) > 8 else ""
    metadata["header"] = header
    format_type = get_format(first_data_line)
    with open(file_location, encoding="latin1") as f:
        for index, row in enumerate(f):
            row_str = row.strip()
            if index + 1 == 1:
                metadata["time1"], metadata["average1"], metadata["objective"], metadata["aperture"], metadata["date"], metadata["time"] = first_line(row_str)
            if index + 1 == 3:
                metadata["measuring_mode"] = measuring_mode(row_str)
            if index + 1 == 7:
                metadata["average2"] = average_2(row_str)
            if index + 1 == 8:
                metadata["integration_time1"] = integration_time1(row_str)
            if index + 1 == 9:
                metadata["integration_time2"] = integration_time2(row_str)
                break
    df = pd.DataFrame()
    with open(file_location, encoding="latin1") as f:
        if format_type == "craic_data_comma_regex":
            df = pd.read_csv(f, sep=",", decimal=".", names=["wavelength", metadata["measuring_mode"]], skiprows=9, encoding="latin1").dropna()
        elif format_type == "craic_data_tab_regex":
            df = pd.read_csv(f, sep="\t", decimal=".", names=["wavelength", metadata["measuring_mode"]], skiprows=9, encoding="latin1").dropna()
        if df.empty:
            warnings.warn(f"Dataframe is empty. File: {file_location}", UserWarning)
    metadata["filename"] = file_location
    metadata["units"] = "nm"
    metadata["code"], metadata["polarization"] = get_metadata_from_filename(file_location)
    metadata["minimum_wavelength"] = df["wavelength"].min()
    metadata["maximum_wavelength"] = df["wavelength"].max()
    metadata["step"] = np.round(np.mean(df["wavelength"].diff()), 8)
    metadata["number_of_datapoints"] = len(df[metadata["measuring_mode"]])
    metadata["maximum_measurement"] = df[metadata["measuring_mode"]].max()
    metadata["minimum_measurement"] = df[metadata["measuring_mode"]].min()
    metadata["equipment"] = "CRAIC"
    metadata["genus"] = "na"
    metadata["species"] = "na"
    df["wavelength"], df[metadata["measuring_mode"]] = df["wavelength"].astype(float), df[metadata["measuring_mode"]].astype(float)
    if metadata["polarization"] in ["R", "L"]:
        df = df[(df["wavelength"] > min_craic_wv_polarized) & (df["wavelength"] < max_craic_wv_polarized)].reset_index(drop=True)
    else:
        df = df[(df["wavelength"] > min_craic_wv) & (df["wavelength"] < max_craic_wv)].reset_index(drop=True)
    return metadata, df

def get_metadata_and_dataframe_l1050(file_location):
    metadata = {}
    with open(file_location, encoding="latin1") as myfile:
        lines = myfile.readlines()[0:90]
    metadata["header"] = "".join(lines)
    with open(file_location, encoding="latin1") as f:
        for index, row in enumerate(f):
            row_str = row.strip()
            if index + 1 == 3:
                metadata["filename"] = file_location
                metadata["code"] = get_sample_code_from_filename(row_str, file_location)
            if index + 1 == 4:
                metadata["date"] = row_str
            if index + 1 == 5:
                metadata["time"] = row_str
            if index + 1 == 8:
                metadata["user"] = row_str
            if index + 1 == 9:
                metadata["description"] = row_str
            if index + 1 == 10:
                metadata["minimum_wavelength"] = row_str
            if index + 1 == 12:
                metadata["equipment"] = row_str
            if index + 1 == 13:
                metadata["series"] = row_str
            if index + 1 == 14:
                metadata["software"] = row_str
            if index + 1 == 21:
                metadata["operating_mode"] = row_str
            if index + 1 == 22:
                metadata["cycles"] = row_str
            if index + 1 == 32:
                metadata["slit_pmt"] = slit_pmt_aperture(row_str)
            if index + 1 == 33:
                metadata["response_ingaas"], metadata["response_pmt"] = responses(row_str)
            if index + 1 == 35:
                metadata["pmt_gain"] = row_str
            if index + 1 == 36:
                metadata["ingaas_gain"] = row_str
            if index + 1 == 42:
                metadata["monochromator_change"] = row_str
            if index + 1 == 43:
                metadata["lamp_change"] = row_str
            if index + 1 == 44:
                metadata["pmt_change"] = row_str
            if index + 1 == 45:
                metadata["beam_selector"] = row_str
            if index + 1 == 46:
                metadata["cbm"] = row_str
            if index + 1 == 47:
                metadata["cbd_status"] = row_str
            if index + 1 == 48:
                metadata["attenuator_sample"], metadata["attenuator_reference"] = attenuator_settings(row_str)
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
    metadata["genus"] = "na"
    metadata["species"] = "na"
    metadata["polarization"] = "T"
    metadata["equipment"] = "L1050"
    df = pd.read_csv(file_location, sep="\t", decimal=",", names=["wavelength", metadata["measuring_mode"]], skiprows=90, encoding="latin1").dropna()
    df["wavelength"], df[metadata["measuring_mode"]] = df["wavelength"].astype(float), df[metadata["measuring_mode"]].astype(float)
    df = df[df["wavelength"] < 2000].reset_index(drop=True)
    return metadata, df

def get_metadata_and_dataframe_fluorometer(file_location):
    def get_sample_code_from_filename(row_str, file_location):
        filename = os.path.basename(file_location)
        re1 = r"([a-zA-Z\d\s]+)(?:-\d)*(?:.Sample)*.(?:txt)*(?:ASC)*"
        p = re.compile(re1)
        m = p.match(filename)
        if m:
            return m.group(1)
        warnings.warn(f"Failed to extract sample code from filename: {file_location}. Missing sample code.", UserWarning)
        return get_sample_code(row_str)

    def get_sample_code(row_str):
        re1 = r"([a-zA-Z\d\s]+)(?:-\d)*(?:.Sample)*.(?:txt)*(?:ASC)*"
        p = re.compile(re1)
        m = p.match(row_str)
        if m:
            return m.group(1)
        warnings.warn(f"Failed to extract sample code from input: {row_str}. Missing sample code.", UserWarning)
        return "na"

    metadata = {}
    with open(file_location, encoding="latin1") as myfile:
        lines = myfile.readlines()[0:51]
    metadata["header"] = "".join(lines)
    metadata["measuring_mode"] = "F"
    metadata["equipment"] = "Fluorometer"
    with open(file_location, encoding="latin1") as f:
        for index, row in enumerate(f):
            row_str = row.strip()
            if index + 1 == 3:
                metadata["filename"] = file_location
                metadata["code"] = get_sample_code_from_filename(row_str, file_location)
            if index + 1 == 4:
                metadata["date"] = row_str
            if index + 1 == 5:
                metadata["time"] = row_str
            if index + 1 == 6:
                metadata["user"] = row_str
            if index + 1 == 7:
                metadata["description"] = row_str
            if index + 1 == 10:
                metadata["excitation_wavelength"] = row_str
            if index + 1 == 11:
                metadata["method"] = row_str
            if index + 1 == 12:
                metadata["instrument_name"] = row_str
            if index + 1 == 13:
                metadata["instrument_serial"] = row_str
            if index + 1 == 14:
                metadata["software_version"] = row_str
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
            if index + 1 == 39:
                break
    df = pd.read_csv(file_location, sep="\t", decimal=".", names=["wavelength", "F"], skiprows=51, encoding="latin1").dropna()
    df["wavelength"], df["F"] = df["wavelength"].astype(float), df["F"].astype(float)
    df = df[df["wavelength"] < 2000].reset_index(drop=True)
    return metadata, df

def get_metadata_and_dataframe_avantes_avalight(file_location):
    metadata = {}
    with open(file_location, encoding="latin1") as myfile:
        lines = myfile.readlines()[0:6]
    metadata["header"] = "".join(lines)
    with open(file_location, encoding="latin1") as f:
        for index, row in enumerate(f):
            row_str = row.strip()
            if index + 1 == 1:
                metadata["sample_name"] = row_str
            if index + 1 == 2:
                metadata["integration_time_ms"] = row_str.split(":")[1].strip()
            if index + 1 == 3:
                metadata["averaging_scans"] = row_str.split(":")[1].strip()
            if index + 1 == 4:
                metadata["smoothing_pixels"] = row_str.split(":")[1].strip()
            if index + 1 == 5:
                metadata["spectrometer_name"] = row_str.split(":")[1].strip()
            if index + 1 == 6:
                break
    metadata["filename"] = file_location
    metadata["measuring_mode"] = "%R"
    metadata["units"] = "nm"
    metadata["equipment"] = "Avantes AvaLight"
    metadata["genus"] = "na"
    metadata["species"] = "na"
    metadata["polarization"] = "T"
    
    basename = Path(file_location).name
    info = get_info_from_format(basename)
    metadata["code"] = info["code"]
    metadata["measuring_location"] = info.get("measuring_location", "na")
    metadata["spectrometer_name"] = info.get("spectrometer_name", metadata["spectrometer_name"])
    
    df = pd.read_csv(file_location, sep=";", decimal=",", 
                     names=["wavelength", "sample", "dark", "reference", "%R"], 
                     skiprows=7, encoding="latin1").dropna()
    df["wavelength"] = df["wavelength"].astype(float)
    df["%R"] = df["%R"].astype(float)
    
    df = df[df["wavelength"] > 350].reset_index(drop=True)
    
    metadata["minimum_wavelength"] = df["wavelength"].min()
    metadata["maximum_wavelength"] = df["wavelength"].max()
    metadata["step"] = np.round(np.mean(df["wavelength"].diff()), 8)
    metadata["number_of_datapoints"] = len(df["%R"])
    metadata["maximum_measurement"] = df["%R"].max()
    metadata["minimum_measurement"] = df["%R"].min()
    
    return metadata, df

def get_genus(code, collection):
    collection_name = collection.get_name()
    collection_metadata = collection.get_metadata()
    specimen = collection_metadata.loc[collection_metadata["code"].astype(str) == str(code), "genus"]
    if specimen.empty:
        print(f"Utils: No genus data for {code} in collection {collection_name}")
        return ""
    return str(specimen.iloc[0])

def get_species(code, collection):
    collection_name = collection.get_name()
    collection_metadata = collection.get_metadata()
    specimen = collection_metadata.loc[collection_metadata["code"].astype(str) == str(code), "species"]
    if specimen.empty:
        print(f"No species data for {code} in collection {collection_name}")
        return "na"
    try:
        return str(specimen.iloc[0])
    except Exception:
        print("Update specimen in the corresponding collection, please")
        return "na"

def responses(str):
    re1 = r"\d+/(\d+(?:,\d+)?) \d+,\d+/(\d+(?:,\d+)?)"
    p = re.compile(re1)
    m = p.match(str)
    if m:
        return m.group(1), m.group(2)
    warnings.warn(f"Failed to parse response settings from input: {str}. Missing response_ingaas and response_pmt.", UserWarning)
    return "na", "na"

def attenuator_settings(str):
    re1 = r"S:(\d+(?:,\d+)?) R:(\d+(?:,\d+)?)"
    p = re.compile(re1)
    m = p.match(str)
    if m:
        return m.group(1), m.group(2)
    warnings.warn(f"Failed to parse attenuator settings from input: {str}. Missing attenuator_sample and attenuator_reference.", UserWarning)
    return "na", "na"

def slit_pmt_aperture(str):
    re1 = r"\d+/servo \d+,\d+/(\d+(?:,\d+)?)"
    p = re.compile(re1)
    m = p.match(str)
    if m:
        return m.group(1)
    warnings.warn(f"Failed to parse slit PMT aperture from input: {str}. Missing slit_pmt.", UserWarning)
    return "na"

def get_sample_code_from_filename(row_str, file_location):
    filename = os.path.basename(file_location)
    re1 = r"([a-zA-Z\d\s]+)(?:-\d)*(?:.Sample)*.(?:txt)*(?:ASC)*"
    p = re.compile(re1)
    m = p.match(filename)
    if m:
        return m.group(1)
    warnings.warn(f"Failed to extract sample code from filename: {file_location}. Missing sample code.", UserWarning)
    return get_sample_code(row_str)

def get_sample_code(row_str):
    re1 = r"([a-zA-Z\d\s]+)(?:-\d)*(?:.Sample)*.(?:txt)*(?:ASC)*"
    p = re.compile(re1)
    m = p.match(row_str)
    if m:
        return m.group(1)
    warnings.warn(f"Failed to extract sample code from input: {row_str}. Missing sample code.", UserWarning)
    return "na"