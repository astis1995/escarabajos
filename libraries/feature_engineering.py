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

def get_aggregated_data(Metric, all_spectra, params = None, debug=False):
    """
    Compute aggregated statistics and visualizations for a given Metric across a list of spectra, grouped by species.
    Handles scalar, vectorial, and matricial metrics.
    
    Parameters:
    - Metric: A class with a name attribute and get_metric_value() method (returns scalar, 1D array, or 2D array).
    - all_spectra: List of Spectrum objects.
    - debug: Boolean to enable debug output (default: False).
    
    Returns:
    - stats_df: pandas DataFrame with per-species statistics (mean, std, median, min, max, count, IQR, CV).
    """
    # Initialize data storage
    data = []
    
    # Compute metric values for each spectrum
    for spectrum in all_spectra:
        try:
            species = spectrum.get_species()
            if species == "na":
                if debug:
                    warnings.warn(
                        f"Skipping spectrum {spectrum.get_filename()}: No species defined.",
                        UserWarning
                    )
                continue
            metric = Metric(spectrum, params)
            value = metric.get_metric_value()
            
            # Determine metric type
            if isinstance(value, (int, float)) and np.isfinite(value):
                # Scalar
                data.append({"species": species, "metric_value": value, "filename": spectrum.get_filename()})
            elif isinstance(value, np.ndarray):
                if value.ndim == 1:
                    # Vectorial
                    data.append({"species": species, "metric_value": value, "filename": spectrum.get_filename()})
                elif value.ndim == 2:
                    # Matricial
                    data.append({"species": species, "metric_value": value.flatten(), "filename": spectrum.get_filename()})
                else:
                    if debug:
                        warnings.warn(
                            f"Invalid metric value shape {value.shape} for {spectrum.get_filename()} "
                            f"(species: {species}). Skipping.",
                            UserWarning
                        )
                    continue
            else:
                if debug:
                    warnings.warn(
                        f"Invalid metric value type {type(value)} for {spectrum.get_filename()} "
                        f"(species: {species}). Skipping.",
                        UserWarning
                    )
                continue
        except Exception as e:
            if debug:
                warnings.warn(
                    f"Error computing metric for {spectrum.get_filename()}: {e}",
                    UserWarning
                )
    
    if not data:
        warnings.warn("No valid metric values computed. Returning empty DataFrame.", UserWarning)
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Determine metric type based on first valid metric_value
    first_value = df["metric_value"].iloc[0]
    is_scalar = isinstance(first_value, (int, float))
    is_vectorial = isinstance(first_value, np.ndarray) and first_value.ndim == 1
    is_matricial = isinstance(first_value, np.ndarray) and first_value.ndim > 1
    
    # Aggregate statistics
    if is_scalar:
        # Scalar: Compute statistics directly
        stats_df = df.groupby("species")["metric_value"].agg([
            "mean", "std", "median", "min", "max", "count",
            lambda x: np.percentile(x, 75) - np.percentile(x, 25)  # IQR
        ]).rename(columns={"<lambda_0>": "iqr"})
        stats_df["cv"] = stats_df["std"] / stats_df["mean"]
    else:
        # Vectorial or matricial: Pad arrays and compute statistics per element
        grouped = df.groupby("species")
        stats_dict = {}
        for species, group in grouped:
            values = group["metric_value"].tolist()
            # Find max length for padding (per species)
            max_length = max(len(v) for v in values if isinstance(v, np.ndarray))
            padded_values = pad_arrays(values, max_length=max_length, pad_value=0.0)
            # Convert to DataFrame for statistics
            value_df = pd.DataFrame(padded_values, columns=[f"elem_{i}" for i in range(max_length)])
            stats = value_df.agg([
                "mean", "std", "median", "min", "max", "count",
                lambda x: np.percentile(x, 75) - np.percentile(x, 25)
            ]).rename({"<lambda_0>": "iqr"})
            stats["cv"] = stats["std"] / stats["mean"]
            stats_dict[species] = stats
        # Combine statistics into a MultiIndex DataFrame
        stats_df = pd.concat(stats_dict, axis=1)
        stats_df.columns.names = ["species", "metric"]
    
    # Round numeric columns
    stats_df = stats_df.round(4)
    
    # Debug output
    if debug:
        print("Aggregated Statistics:")
        print(stats_df)
        print(f"Unique species: {list(df['species'].unique())}")
        print(f"Total valid spectra processed: {len(df)}")
    
    # Visualizations
    if is_scalar:
        # Scalar: Single boxplot and histogram
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="species", y="metric_value", data=df)
        plt.title(f"Boxplot of {Metric.name} by Species")
        plt.xticks(rotation=45)
        plt.xlabel("Species")
        plt.ylabel(f"{Metric.name} Value")
        plt.tight_layout()
        plt.show()
        
        # Histograms per species
        unique_species = df["species"].unique()
        n_species = len(unique_species)
        plt.figure(figsize=(12, 4 * n_species))
        for i, species in enumerate(unique_species, 1):
            plt.subplot(n_species, 1, i)
            sns.histplot(df[df["species"] == species]["metric_value"], kde=True)
            plt.title(f"Histogram of {Metric.name} for {species}")
            plt.xlabel(f"{Metric.name} Value")
            plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
        
        # Outlier detection
        outliers = []
        for species in unique_species:
            species_data = df[df["species"] == species]["metric_value"]
            q1 = np.percentile(species_data, 25)
            q3 = np.percentile(species_data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            species_outliers = species_data[(species_data < lower_bound) | (species_data > upper_bound)]
            for value in species_outliers:
                outlier_file = df[(df["species"] == species) & (df["metric_value"] == value)]["filename"].iloc[0]
                outliers.append({"species": species, "metric_value": value, "filename": outlier_file})
    else:
        # Vectorial or matricial: Boxplot per element, histogram for first few elements
        max_elements = 3  # Limit to 3 elements to match example outputs
        unique_species = df["species"].unique()
        max_length = max(len(v) for v in df["metric_value"] if isinstance(v, np.ndarray))
        # Create DataFrame with padded values
        padded_df = pd.DataFrame()
        for species in unique_species:
            species_values = df[df["species"] == species]["metric_value"].tolist()
            padded_values = pad_arrays(species_values, max_length=max_length, pad_value=0.0)
            temp_df = pd.DataFrame(padded_values, columns=[f"elem_{i}" for i in range(max_length)])
            temp_df["species"] = species
            padded_df = pd.concat([padded_df, temp_df], ignore_index=True)
        
        # Boxplot per element
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="species", y="value", hue="variable",
                    data=pd.melt(padded_df, id_vars=["species"], value_vars=[f"elem_{i}" for i in range(min(max_length, max_elements))]))
        plt.title(f"Boxplot of {Metric.name} Elements by Species")
        plt.xticks(rotation=45)
        plt.xlabel("Species")
        plt.ylabel(f"{Metric.name} Element Value")
        plt.legend(title="Element")
        plt.tight_layout()
        plt.show()
        
        # Histograms for first few elements
        plt.figure(figsize=(12, 4 * min(max_length, max_elements)))
        for i in range(min(max_length, max_elements)):
            plt.subplot(min(max_length, max_elements), 1, i + 1)
            for species in unique_species:
                sns.histplot(padded_df[padded_df["species"] == species][f"elem_{i}"], kde=True, label=species, stat="density")
            plt.title(f"Histogram of {Metric.name} Element {i} by Species")
            plt.xlabel(f"Element {i} Value")
            plt.ylabel("Density")
            plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Outlier detection for each element
        outliers = []
        for i in range(max_length):
            for species in unique_species:
                species_data = padded_df[padded_df["species"] == species][f"elem_{i}"].dropna()
                if species_data.empty:
                    continue
                q1 = np.percentile(species_data, 25)
                q3 = np.percentile(species_data, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                species_outliers = species_data[(species_data < lower_bound) | (species_data > upper_bound)]
                for value in species_outliers:
                    idx = padded_df[(padded_df["species"] == species) & (padded_df[f"elem_{i}"] == value)].index[0]
                    outlier_file = df.iloc[idx]["filename"]
                    outliers.append({"species": species, "element": i, "metric_value": value, "filename": outlier_file})
        
        # Heatmap for matricial metrics (mean values)
        if is_matricial:
            plt.figure(figsize=(12, 6))
            for species in unique_species:
                species_values = df[df["species"] == species]["metric_value"].tolist()
                mean_matrix = np.mean(pad_arrays(species_values, max_length=(max_length // max_length, max_length), pad_value=0.0), axis=0)
                plt.subplot(1, len(unique_species), list(unique_species).index(species) + 1)
                sns.heatmap(mean_matrix, annot=True, cmap="viridis")
                plt.title(f"Mean {Metric.name} Matrix for {species}")
                plt.xlabel("Column")
                plt.ylabel("Row")
            plt.tight_layout()
            plt.show()
    
    if outliers and debug:
        print("Outliers Detected:")
        outliers_df = pd.DataFrame(outliers)
        print(outliers_df.round(4))
    
    return stats_df