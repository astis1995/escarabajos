import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

def pad_arrays(arrays, max_length=None, pad_value=0.0):
    """
    Pad a list of numpy arrays to the maximum length in the list (or specified max_length) with pad_value.
    
    Parameters:
    - arrays: List of numpy arrays (1D or 2D).
    - max_length: Maximum length to pad to (default: None, uses max length in arrays).
    - pad_value: Value to use for padding (default: 0.0).
    
    Returns:
    - Padded list of arrays.
    """
    if not arrays:
        return []
    
    # Determine if arrays are 1D (vectorial) or 2D (matricial)
    is_2d = any(arr.ndim == 2 for arr in arrays if isinstance(arr, np.ndarray))
    
    if is_2d:
        # For 2D arrays, find max rows and cols
        max_rows = max_length[0] if max_length and isinstance(max_length, (list, tuple)) else max(arr.shape[0] for arr in arrays if isinstance(arr, np.ndarray))
        max_cols = max_length[1] if max_length and isinstance(max_length, (list, tuple)) and len(max_length) > 1 else max(arr.shape[1] for arr in arrays if isinstance(arr, np.ndarray))
        padded = []
        for arr in arrays:
            if not isinstance(arr, np.ndarray):
                padded.append(np.full((max_rows, max_cols), pad_value))
                continue
            rows, cols = arr.shape
            padded_arr = np.full((max_rows, max_cols), pad_value)
            padded_arr[:rows, :cols] = arr
            padded.append(padded_arr)
    else:
        # For 1D arrays
        max_len = max_length if max_length and isinstance(max_length, (int)) else max(len(arr) for arr in arrays if isinstance(arr, np.ndarray))
        padded = []
        for arr in arrays:
            if not isinstance(arr, np.ndarray):
                padded.append(np.full(max_len, pad_value))
                continue
            padded_arr = np.full(max_len, pad_value)
            padded_arr[:len(arr)] = arr
            padded.append(padded_arr)
    
    return padded

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# ---------- Helpers ----------


def compute_outliers(series):
    """Return outliers using IQR rule."""
    q1, q3 = np.percentile(series, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return series[(series < lower) | (series > upper)]

# ---------- Scalar case ----------
def handle_scalar_old(df, Metric, debug=False):
    stats_df = df.groupby("species")["metric_value"].agg(
        ["mean", "std", "median", "min", "max", "count"]
    )
    stats_df["iqr"] = df.groupby("species")["metric_value"].apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25)
    )
    stats_df["cv"] = stats_df["std"] / stats_df["mean"]

    # --- Boxplot + individual points ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="species", y="metric_value", data=df, showfliers=False)
    sns.stripplot(
        x="species", y="metric_value", data=df,
        jitter=True, dodge=True, alpha=0.6, color="black", size=4
    )
    plt.title(f"Boxplot of {Metric.name} with individual points")
    plt.show()

    # --- Histograms + rugplot ---
    for species in df["species"].unique():
        plt.figure(figsize=(8, 6))
        sns.histplot(df[df["species"] == species]["metric_value"], kde=True, bins=15)
        sns.rugplot(
            df[df["species"] == species]["metric_value"], 
            height=0.05, color="red"
        )
        plt.title(f"{Metric.name} for {species} (histogram + rug)")
        plt.show()

    if debug:
        for species in df["species"].unique():
            outs = compute_outliers(df[df["species"] == species]["metric_value"])
            if not outs.empty:
                print(f"Outliers for {species}:", outs.tolist())
    return stats_df.round(4)

# ---------- Scalar case ----------
def handle_scalar(df, Metric, debug=False):
    # Drop missing metric values
    df = df.dropna(subset=["metric_value"])
    if df.empty:
        print(f"⚠️ {Metric.name}: No valid metric values after dropping NaNs.")
        return pd.DataFrame(columns=["mean", "std", "median", "min", "max", "count", "iqr", "cv"])

    # --- Statistics ---
    stats_df = df.groupby("species")["metric_value"].agg(
        ["mean", "std", "median", "min", "max", "count"]
    )
    stats_df["iqr"] = df.groupby("species")["metric_value"].apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25)
    )
    stats_df["cv"] = stats_df["std"] / stats_df["mean"]

    # --- Boxplot + individual points ---
    if not df.empty and df["species"].nunique() > 0:
        plt.figure(figsize=(8, 6))
        try:
            sns.boxplot(x="species", y="metric_value", data=df, showfliers=False)
            sns.stripplot(
                x="species", y="metric_value", data=df,
                jitter=True, dodge=True, alpha=0.6, color="black", size=4
            )
            plt.title(f"Boxplot of {Metric.name} with individual points")
            plt.show()
        except ValueError as e:
            print(f"⚠️ Skipping boxplot due to error: {e}")

    # --- Histograms + rugplot ---
    for species in df["species"].dropna().unique():
        subset = df[df["species"] == species]["metric_value"].dropna()
        if subset.empty:
            continue
        plt.figure(figsize=(8, 6))
        sns.histplot(subset, kde=True, bins=15)
        if len(subset) > 1:
            sns.rugplot(subset, height=0.05, color="red")
        plt.title(f"{Metric.name} for {species} (histogram + rug)")
        plt.show()

    # --- Outliers ---
    if debug:
        for species in df["species"].dropna().unique():
            subset = df[df["species"] == species]["metric_value"].dropna()
            outs = compute_outliers(subset)
            if not outs.empty:
                print(f"Outliers for {species}:", outs.tolist())

    return stats_df.round(4)




# ---------- Vector case ----------
def handle_vector(df, Metric, debug=False):
    stats_dict = {}

    # --- Ensure all metric values are numpy arrays ---
    df = df.copy()
    df["metric_value"] = df["metric_value"].apply(lambda v: np.array(v, dtype=float))

    # --- Pad all arrays across the entire dataset to same length ---
    values_all = df["metric_value"].tolist()
    padded_all = np.array(pad_arrays(values_all))  # Now guaranteed rectangular

    # --- Build global padded DataFrame for plotting ---
    padded_df = pd.DataFrame(
        padded_all, columns=[f"elem_{i}" for i in range(padded_all.shape[1])]
    )
    padded_df["species"] = df["species"].values

    # --- Stats per species ---
    for species, group in df.groupby("species"):
        values = group["metric_value"].tolist()
        padded = np.array(pad_arrays(values))   # pad within species
        value_df = pd.DataFrame(
            padded, columns=[f"elem_{i}" for i in range(padded.shape[1])]
        )
        stats = value_df.agg(["mean", "std", "median", "min", "max"])
        stats.loc["iqr"] = (
            np.percentile(value_df, 75, axis=0) - np.percentile(value_df, 25, axis=0)
        )
        stats.loc["cv"] = stats.loc["std"] / stats.loc["mean"]
        stats_dict[species] = stats

    stats_df = pd.concat(stats_dict, axis=1)
    stats_df.columns.names = ["species", "metric"]

    # --- Boxplot for first few elements + points ---
    melted = padded_df.melt(
        id_vars="species", var_name="element", value_name="value"
    )
    subset = melted[melted["element"].isin(["elem_0", "elem_1", "elem_2"])]

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x="species", y="value", hue="element",
        data=subset, showfliers=False
    )
    sns.stripplot(
        x="species", y="value", hue="element",
        data=subset, dodge=True, alpha=0.6, size=3, marker="o"
    )
    plt.title(f"{Metric.name} (first 3 elements) with individual points")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()

    return stats_df.round(4)




# ---------- Matrix case ----------
def handle_matrix(df, Metric, debug=False):
    df = df.copy()

    # Ensure each metric_value is a flattened 1D numpy array
    df["metric_value"] = df["metric_value"].apply(
        lambda m: np.array(m).flatten() if m is not None else np.array([])
    )

    # Pad across the entire dataset to ensure rectangular shape
    values_all = df["metric_value"].tolist()
    padded_all = np.array(pad_arrays(values_all))  # uniform shape

    # Replace original column with padded vectors
    df["metric_value"] = list(padded_all)

    # Reuse vector handler (now safe)
    return handle_vector(df, Metric, debug)


# ---------- Main wrapper ----------
def get_aggregated_data(Metric, spectra, config_file, collection_list, plot = False):
    """Dispatch scalar / vector / matrix metric aggregation."""
    debug = False
    if debug:
        print("Aggregating data")
    data = []
    for spectrum in spectra:
        try:
            species = spectrum.get_species()
            if species == "na":
                continue
            metric = Metric(spectrum, config_file, collection_list, plot = plot)
            value = metric.get_metric_value()
            data.append({"species": species, "metric_value": value, "filename": spectrum.get_filename()})
        except Exception as e:
            if debug: warnings.warn(f"Error computing metric for {spectrum.get_filename()}: {e}")

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    first_val = df["metric_value"].iloc[0]

    if np.isscalar(first_val):
        return handle_scalar(df, Metric, debug)
    elif isinstance(first_val, np.ndarray) and first_val.ndim == 1:
        return handle_vector(df, Metric, debug)
    elif isinstance(first_val, np.ndarray) and first_val.ndim == 2:
        return handle_matrix(df, Metric, debug)
    else:
        warnings.warn("Unsupported metric value type")
        return pd.DataFrame()
