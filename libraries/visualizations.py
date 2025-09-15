import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def scatter_plot_3_variables(df_1, df_2, df_3, collection_names, report_location, image_counter, target_species=None):
    """
    Create a 3D scatter plot of three metric variables, colored by species.
    
    Parameters:
    - df_1, df_2, df_3: pandas DataFrames with columns ["code", metric_name, "species"].
    - collection_names: String for collection names (e.g., "ANGSOL_CICIMAUCR").
    - report_location: Path object for saving the plot (e.g., Path("reports/data_analysis")).
    - image_counter: Generator yielding image numbers.
    - target_species: List of species to include in colormap (default: None, uses unique species from data).
    
    Returns:
    - None (displays and saves the plot).
    """
    # Merge DataFrames
    joint_df = pd.merge(df_1, df_2, on=["code", "species"], how="inner")
    joint_df = pd.merge(joint_df, df_3, on=["code", "species"], how="inner")
    
    # Extract metric columns
    column_list = joint_df.columns.tolist()
    column_list = [x for x in column_list if x not in ["code", "species"]]
    if len(column_list) != 3:
        raise ValueError(f"Expected exactly 3 metric columns, got {len(column_list)}: {column_list}")
    
    x = joint_df[column_list[0]]
    y = joint_df[column_list[1]]
    z = joint_df[column_list[2]]
    species = joint_df["species"]
    
    # Create dynamic colormap based on target_species or unique species
    unique_species = list(set(species)) if target_species is None else target_species
    if not unique_species:
        raise ValueError("No species found in data or target_species.")
    
    # Generate colors using seaborn's husl palette
    color_palette = sns.color_palette("husl", n_colors=len(unique_species))
    colors = {species: color for species, color in zip(unique_species, color_palette)}
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points with color based on species
    for category in unique_species:
        indices = species == category
        ax.scatter(x[indices], y[indices], z[indices], c=[colors[category]], label=category, marker='o')
    
    # Set labels and title
    ax.set_xlabel(f'{column_list[0]}')
    ax.set_ylabel(f'{column_list[1]}')
    ax.set_zlabel(f'{column_list[2]}')
    
    # Add legend
    ax.legend()
    
    # Save image
    current_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    path = os.path.join(report_location, "report_images", current_date, "species classifier")
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path, f"{column_list[0]}_{column_list[1]}_{column_list[2]}_{collection_names}-{next(image_counter)}.png")
    plt.savefig(filename)
    
    plt.show()

def scatter_plot_2_variables(df_1, df_2, collection_names, report_location, image_counter, target_species=None):
    """
    Create a 2D scatter plot of two metric variables, colored by species.
    
    Parameters:
    - df_1, df_2: pandas DataFrames with columns ["code", metric_name, "species"].
    - collection_names: String for collection names (e.g., "ANGSOL_CICIMAUCR").
    - report_location: Path object for saving the plot (e.g., Path("reports/data_analysis")).
    - image_counter: Generator yielding image numbers.
    - target_species: List of species to include in colormap (default: None, uses unique species from data).
    
    Returns:
    - None (displays and saves the plot).
    """
    # Merge DataFrames
    joint_df = pd.merge(df_1, df_2, on=["code", "species"], how="inner")
    
    # Extract metric columns
    column_list = joint_df.columns.tolist()
    column_list = [x for x in column_list if x not in ["code", "species"]]
    if len(column_list) != 2:
        raise ValueError(f"Expected exactly 2 metric columns, got {len(column_list)}: {column_list}")
    
    x = joint_df[column_list[0]]
    y = joint_df[column_list[1]]
    species = joint_df["species"]
    
    # Create dynamic colormap based on target_species or unique species
    unique_species = list(set(species)) if target_species is None else target_species
    if not unique_species:
        raise ValueError("No species found in data or target_species.")
    
    # Generate colors using seaborn's husl palette
    color_palette = sns.color_palette("husl", n_colors=len(unique_species))
    colors = {species: color for species, color in zip(unique_species, color_palette)}
    
    # Create figure and 2D axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot points with color based on species
    for category in unique_species:
        indices = species == category
        ax.scatter(x[indices], y[indices], c=[colors[category]], label=category, marker='o')
    
    # Set labels and title
    ax.set_xlabel(f'{column_list[0]}')
    ax.set_ylabel(f'{column_list[1]}')
    
    # Add legend
    ax.legend()
    
    # Save image
    current_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    path = os.path.join(report_location, "report_images", current_date, "species classifier")
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path, f"{column_list[0]}_{column_list[1]}_{collection_names}-{next(image_counter)}.png")
    plt.savefig(filename)
    
    plt.show()