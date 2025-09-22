import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from pathlib import Path

# ---------- Helpers ----------
def create_path_if_not_exists(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def compute_outliers(series):
    """Return outliers using IQR rule."""
    q1, q3 = np.percentile(series, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return series[(series < lower) | (series > upper)]


# ---------- Scalar case ----------
def handle_scalar(df, Metric, save_path=None, debug=False):
    # Drop missing metric values
    df = df.dropna(subset=["metric_value"])
    if df.empty:
        print(f"⚠️ {Metric.name}: No valid metric values after dropping NaNs.")
        return pd.DataFrame(columns=["mean", "std", "median", "min", "max", "count", "iqr", "cv"]), None

    # --- Statistics ---
    stats_df = df.groupby("species")["metric_value"].agg(
        ["mean", "std", "median", "min", "max", "count"]
    )
    stats_df["iqr"] = df.groupby("species")["metric_value"].apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25)
    )
    stats_df["cv"] = stats_df["std"] / stats_df["mean"]

    # --- Boxplot + individual points ---
    filename = None
    if not df.empty and df["species"].nunique() > 0:
        plt.figure(figsize=(10, 7))
        try:
            ax = sns.boxplot(x="species", y="metric_value", data=df, showfliers=False, palette="Set3")
            sns.stripplot(
                x="species", y="metric_value", data=df,
                jitter=True, dodge=True, alpha=0.7, color="black", size=5
            )
            plt.xticks(rotation=90)
            plt.title(f"Boxplot of {Metric.name} with individual points")
            plt.grid(axis="y", linestyle="--", alpha=0.7)

            if save_path:
                create_path_if_not_exists(save_path)
                filename = os.path.join(save_path, f"{Metric.name}.jpeg")
                plt.savefig(filename, bbox_inches="tight")
            plt.show()
        except ValueError as e:
            print(f"⚠️ Skipping boxplot due to error: {e}")

    # --- Outliers (optional debug) ---
    if debug:
        for species in df["species"].dropna().unique():
            subset = df[df["species"] == species]["metric_value"].dropna()
            outs = compute_outliers(subset)
            if not outs.empty:
                print(f"Outliers for {species}:", outs.tolist())

    return stats_df.round(4), filename


# ---------- Vector case ----------
def handle_vector(df, Metric, save_path=None, debug=False):
    stats_dict = {}
    df = df.copy()
    df["metric_value"] = df["metric_value"].apply(lambda v: np.array(v, dtype=float))

    # Pad arrays globally
    max_len = max(len(arr) for arr in df["metric_value"])
    padded_all = np.array([np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan) for arr in df["metric_value"]])
    padded_df = pd.DataFrame(padded_all, columns=[f"elem_{i}" for i in range(max_len)])
    padded_df["species"] = df["species"].values

    # Stats per species
    for species, group in df.groupby("species"):
        values = group["metric_value"].tolist()
        padded = np.array([np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan) for arr in values])
        value_df = pd.DataFrame(padded, columns=[f"elem_{i}" for i in range(max_len)])
        stats = value_df.agg(["mean", "std", "median", "min", "max"])
        stats.loc["iqr"] = np.nanpercentile(value_df, 75, axis=0) - np.nanpercentile(value_df, 25, axis=0)
        stats.loc["cv"] = stats.loc["std"] / stats.loc["mean"]
        stats_dict[species] = stats

    stats_df = pd.concat(stats_dict, axis=1)
    stats_df.columns.names = ["species", "metric"]

    # Boxplot for first 3 elements
    melted = padded_df.melt(id_vars="species", var_name="element", value_name="value")
    subset = melted[melted["element"].isin(["elem_0", "elem_1", "elem_2"])]

    plt.figure(figsize=(12, 7))
    sns.boxplot(x="species", y="value", hue="element", data=subset, showfliers=False)
    sns.stripplot(x="species", y="value", hue="element", data=subset, dodge=True, alpha=0.6, size=3, marker="o")
    plt.title(f"{Metric.name} (first 3 elements) with individual points")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    filename = None
    if save_path:
        create_path_if_not_exists(save_path)
        filename = os.path.join(save_path, f"{Metric.name}_vector.jpeg")
        plt.savefig(filename, bbox_inches="tight")
    plt.show()

    return stats_df.round(4), filename


# ---------- Matrix case ----------
def handle_matrix(df, Metric, save_path=None, debug=False):
    df = df.copy()
    df["metric_value"] = df["metric_value"].apply(
        lambda m: np.array(m).flatten() if m is not None else np.array([])
    )

    max_len = max(len(arr) for arr in df["metric_value"])
    padded_all = np.array([np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan) for arr in df["metric_value"]])
    df["metric_value"] = list(padded_all)

    return handle_vector(df, Metric, save_path=save_path, debug=debug)


# ---------- Main wrapper ----------
def get_aggregated_data(Metric, spectra, config_file, collection_list, plot=False, save_path="report_location/report_images"):
    """Dispatch scalar / vector / matrix metric aggregation with plotting and saving."""
    debug = False
    data = []
    for spectrum in spectra:
        try:
            species = spectrum.get_species()
            if species == "na":
                continue
            metric = Metric(spectrum, config_file, collection_list, plot=plot)
            value = metric.get_metric_value()
            data.append({"species": species, "metric_value": value, "filename": spectrum.get_filename()})
        except Exception as e:
            if debug: warnings.warn(f"Error computing metric for {spectrum.get_filename()}: {e}")

    if not data:
        return pd.DataFrame(), None

    df = pd.DataFrame(data)
    first_val = df["metric_value"].iloc[0]

    if np.isscalar(first_val):
        return handle_scalar(df, Metric, save_path, debug)
    elif isinstance(first_val, np.ndarray) and first_val.ndim == 1:
        return handle_vector(df, Metric, save_path, debug)
    elif isinstance(first_val, np.ndarray) and first_val.ndim == 2:
        return handle_matrix(df, Metric, save_path, debug)
    else:
        warnings.warn("Unsupported metric value type")
        return pd.DataFrame(), None
