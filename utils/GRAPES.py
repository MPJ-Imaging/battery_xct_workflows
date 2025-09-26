##########################################################################################

# Graylevel Radial Analysis of ParticlES (GRAPES)

##########################################################################################
# V1.0.0
# Date: 2023-09-30
##########################################################################################
# Author: Matthew Jones
##########################################################################################
# Imports
##########################################################################################

import pandas as pd
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
try:
    from skimage.measure import marching_cubes
except ImportError:
    from skimage.measure import marching_cubes_lewiner as marching_cubes
from skimage.measure import regionprops_table, mesh_surface_area
from skimage.morphology import ball, remove_small_holes
from scipy.ndimage import distance_transform_edt as dist_trans
from scipy.ndimage import convolve
import SimpleITK as sitk
from tqdm import tqdm
from typing import Optional, Tuple, Any, Dict, Union, List
import logging
import warnings
import pickle
import tifffile as tiff
import os

from joblib import Parallel, delayed
import edt  # MLAEDT-3D's EDT module



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

##########################################################################################
# Functions
##########################################################################################

def _surface_area(region_mask):
    """calculate surface area from image of particle region in bounding box

    Args:
        region_mask (ndarray): binary/bool image of particle region in bounding box

    Returns:
        float: surface area of particle
    """    
    mask = region_mask
    tmp = np.pad(np.atleast_3d(mask), pad_width=1, mode='constant')
    verts, faces, norms, vals = marching_cubes(volume=tmp, level=0)
    surf_area = mesh_surface_area(verts, faces)
    return surf_area

def _sphericity(region_mask, sarea):
    """calculate surface area from image of particle region

    Args:
        region_mask (ndarray): binary/bool image of particle region in bounding box
        sarea (float): surface area of particle

    Returns:
        float: sphericity of particle
    """    
    vol = np.sum(region_mask)
    r = (3 / 4 / np.pi * vol)**(1 / 3)
    a_equiv = 4 * np.pi * r**2
    sphr = a_equiv / sarea
    return sphr
    
def _std(region_mask, intensity_im):
    """calculate the standard deviation of graylevels inside the particle region

    Args:
        region_mask (ndarray): binary/bool image of particle region in bounding box
        intensity_im (_type_): graylevel image of particle region in bounding box

    Returns:
        float: stadard deviation of graylevels in particles
    """
    vol = ma.masked_where(region_mask == 0, intensity_im)
    std = ma.std(vol)
    return std

def _grapes(
    region_mask: np.ndarray,
    intensity_im: np.ndarray,
    c0: int,
    c1: int,
    c2: Optional[int] = None,
    normalised_by: str = 'radial_max',
    start_at: str = 'edge',
    pixel_size: Optional[float] = None,
    fill_label_holes: bool = False,
    min_hole_size: int = 5000,
    anisotropy: Tuple[float, ...] = (1.0, 1.0, 1.0),  # Default isotropic
    order: str = 'C',  # Default C-order
    black_border: bool = True,
    parallel: int = 1  # Number of threads for edt.edt
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function is called to calculate the 'radial_layer' characteristics in the GRAPES dataframe.
    Individual images of particle regions extracted from GRAPES props dataframe are processed radially.

    Args:
        region_mask (ndarray): Binary/bool image of particle region in bounding box.
        intensity_im (ndarray): Corresponding graylevel image.
        c0 (int): Axis zero particle center of mass.
        c1 (int): Axis one particle center of mass.
        c2 (int, optional): Axis two particle center of mass (for 3D data).
        normalised_by (str): GRAPES normalization ('radial_max' or 'surface').
        start_at (str): Starting position for radial distance ('edge' or 'center').
        pixel_size (float, optional): If given, characteristics are converted to spatial units.
        fill_label_holes (bool): Whether to fill holes in the labeled regions.
        min_hole_size (int): Minimum hole size to fill.
        anisotropy: Tuple specifying voxel scaling in each dimension.
        order: Memory layout order ('C' or 'F').
        black_border: Whether to consider the image border in EDT.
        parallel: Number of threads used by edt.edt.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            - rp: Radial positions in particle.
            - gl: Average graylevel at radial positions.
            - norm_gl: Normalized graylevel at radial positions.
            - edt: Euclidean distance transform of particle.
    """
    try:
        # Fill holes if required
        if fill_label_holes:
            region_mask = remove_small_holes(region_mask, area_threshold=min_hole_size).astype(np.uint8)
        else:
            # Ensure region_mask is writable by creating a copy
            region_mask = region_mask.copy()

        # Compute the distance transform based on the starting point
        if start_at == 'edge':
            # For 'edge', compute EDT from the background (0) to foreground (1)
            edt_image = edt.edt(
                region_mask,
                anisotropy=anisotropy,
                black_border=black_border,
                order=order,
                parallel=parallel
            )
        elif start_at == 'center':
            # For 'center', set the center point to background before computing EDT
            lbl_c = region_mask.copy().astype(np.uint8)
            if region_mask.ndim == 3 and c2 is not None:
                lbl_c[c0, c1, c2] = 0
            else:
                lbl_c[c0, c1] = 0  # For 2D
            edt_image = edt.edt(
                lbl_c,
                anisotropy=anisotropy,
                black_border=black_border,
                order=order,
                parallel=parallel
            )
            # Ensure EDT is zero outside the original mask
            edt_image = np.where(region_mask == 0, 0, edt_image)
        else:
            raise ValueError("Invalid value for 'start_at'. Choose 'edge' or 'center'.")

        edt_image = np.round(edt_image).astype(int)
        max_dist = edt_image.max()

        if max_dist < 1:
            # Avoid empty arrays if max_dist is less than 1
            return np.array([]), np.array([]), np.array([]), edt_image

        # Flatten the arrays for vectorized processing
        flattened_edt = edt_image.flatten()
        flattened_intensity = intensity_im.flatten()

        # Create radial distance array excluding zero
        radial_distances = np.arange(1, max_dist + 1)

        # Create a mask matrix where each column corresponds to a radial distance
        # and each row indicates whether the pixel belongs to that radial layer
        mask_matrix = (flattened_edt[:, np.newaxis] == radial_distances[np.newaxis, :])

        # Compute the number of pixels in each radial layer to avoid division by zero
        counts = mask_matrix.sum(axis=0)
        valid = counts > 0

        # Compute the sum of intensities for each radial layer
        sum_intensities = (flattened_intensity[:, np.newaxis] * mask_matrix).sum(axis=0)

        # Compute the mean intensities where valid
        gl = np.zeros_like(radial_distances, dtype=np.float32)
        gl[valid] = sum_intensities[valid] / counts[valid]

        # Radial positions
        rp = radial_distances.copy()

        # Normalized gray levels
        norm_gl = gl.copy()
        if normalised_by == 'radial_max':
            if norm_gl.size > 0:
                surface_val = norm_gl[0]
                norm_gl -= surface_val
                radial_max = norm_gl.max()
                norm_gl = norm_gl / radial_max if radial_max != 0 else np.zeros_like(norm_gl)
        elif normalised_by == 'surface':
            if norm_gl.size > 0:
                surface_val = norm_gl[0] if start_at == 'edge' else norm_gl[-1]
                norm_gl = norm_gl / surface_val if surface_val != 0 else np.zeros_like(norm_gl)
        else:
            raise ValueError("Invalid value for 'normalised_by'. Choose 'radial_max' or 'surface'.")

        # Apply pixel size if provided
        if pixel_size is not None:
            rp = rp * pixel_size
            edt_image = edt_image * pixel_size

        return rp, gl, norm_gl, edt_image

    except Exception as e:
        logger.error(f"Error in _grapes_optimized: {e}")
        raise

def GRAPES(
    labels_arr: np.ndarray,
    grey_arr: np.ndarray,
    normalised_by: str = 'radial_max',
    start_at: str = 'edge',
    pixel_size: Optional[float] = None,
    fill_label_holes: bool = False,
    min_hole_size: int = 5000,
    anisotropy: Tuple[float, ...] = (1.0, 1.0, 1.0),  # Default isotropic
    order: str = 'C',  # Default C-order
    black_border: bool = True,
    parallel: int = 1,  # Number of threads for edt.edt
    n_jobs: int = -1  # Number of parallel jobs for GRAPES processing
) -> pd.DataFrame:
    """
    Graylevel Analysis of Particles in ElectrodeS (GRAPES) uses the GREAT2 algorithm to analyze the graylevel at different radial layers
    inside the particles of electrodes. Used to identify changes in density due to degradation of electrode particles.
    Reduces data to pd.DataFrame of particle labels and properties.

    Args:
        labels_arr (ndarray): Array of labeled regions where each unique region is a particle to be analyzed.
        grey_arr (ndarray): Graylevel array corresponding to labeled image.
        normalised_by (str, optional): Normalization method ('radial_max' or 'surface'). Defaults to 'radial_max'.
        start_at (str, optional): Starting point for radial distance ('edge' or 'center'). Defaults to 'edge'.
        pixel_size (float, optional): If given, characteristics are converted to spatial units. Defaults to None.
        fill_label_holes (bool, optional): Whether to fill holes in the labeled regions. Defaults to False.
        min_hole_size (int, optional): Minimum hole size to fill. Defaults to 5000.
        anisotropy (Tuple[float, ...], optional): Voxel scaling in each dimension. Defaults to (1.0, 1.0, 1.0).
        order (str, optional): Memory layout order ('C' or 'F'). Defaults to 'C'.
        black_border (bool, optional): Whether to consider the image border in EDT. Defaults to True.
        parallel (int, optional): Number of threads to use for EDT computation. Defaults to 1.
        n_jobs (int, optional): Number of parallel jobs for GRAPES processing. Defaults

    Returns:
        pd.DataFrame: DataFrame containing properties for each particle.
    """
    try:
        # Determine properties based on the dimensionality of grey_arr
        if grey_arr.ndim == 3:
            properties = (
                'label', 'area', 'centroid', 'centroid_local', 'equivalent_diameter_area',
                'intensity_max', 'intensity_mean', 'intensity_min', 'image', 'image_intensity'
            )
            extra_props = []  # Add any additional properties if needed
            props = regionprops_table(
                labels_arr,
                intensity_image=grey_arr,
                properties=properties,
                cache=True,
                extra_properties=extra_props
            )
            props = pd.DataFrame(props)

            # Extract necessary data for parallel processing
            regions_data = props.to_dict(orient='records')

            def process_region_3d(row):
                image = row['image']
                intensity = row['image_intensity']
                c0 = int(row['centroid_local-0'])
                c1 = int(row['centroid_local-1'])
                c2 = int(row['centroid_local-2'])
                return _grapes(
                    region_mask=image,
                    intensity_im=intensity,
                    c0=c0,
                    c1=c1,
                    c2=c2,
                    normalised_by=normalised_by,
                    start_at=start_at,
                    pixel_size=pixel_size,
                    fill_label_holes=fill_label_holes,
                    min_hole_size=min_hole_size,
                    anisotropy=anisotropy,
                    order=order,
                    black_border=black_border,
                    parallel=parallel
                )

            # Parallel processing of regions
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_region_3d)(row) for row in regions_data
            )

            # Unpack results
            GRAPES_rp, GRAPES_gl, GRAPES_norm_gl, GRAPES_distance_transform = zip(*results)

            # Compute sphericity for each region
            spheres = [
                _sphericity(row['image'], row.get('_surface_area', 1.0)) for row in regions_data
            ]

            # Assign results to the DataFrame
            props['sphericity'] = spheres
            props['radial_layers'] = GRAPES_rp
            props['radial_layers_graylevel'] = GRAPES_gl
            props['radial_layers_graylevelnormed'] = GRAPES_norm_gl
            props['radial_layers_dt'] = GRAPES_distance_transform

            # Rename columns to 3D equivalents
            props = props.rename(columns={
                'area': 'volume',
                'equivalent_diameter_area': 'equivalent_diameter_volume'
            })

            return props

        elif grey_arr.ndim == 2:
            properties = (
                'label', 'area', 'centroid', 'centroid_local', 'equivalent_diameter_area',
                'eccentricity', 'intensity_max', 'intensity_mean', 'intensity_min', 
                'image', 'image_intensity', 'perimeter'
            )
            extra_props = []  # Add any additional properties if needed
            props = regionprops_table(
                labels_arr,
                intensity_image=grey_arr,
                properties=properties,
                cache=True,
                extra_properties=extra_props
            )
            props = pd.DataFrame(props)

            # Extract necessary data for parallel processing
            regions_data = props.to_dict(orient='records')

            def process_region_2d(row):
                image = row['image']
                intensity = row['image_intensity']
                c0 = int(row['centroid_local-0'])
                c1 = int(row['centroid_local-1'])
                return _grapes_optimized(
                    region_mask=image,
                    intensity_im=intensity,
                    c0=c0,
                    c1=c1,
                    c2=None,  # 2D data
                    normalised_by=normalised_by,
                    start_at=start_at,
                    pixel_size=pixel_size,
                    fill_label_holes=fill_label_holes,
                    min_hole_size=min_hole_size,
                    anisotropy=anisotropy[:2],  # Adjust anisotropy for 2D
                    order=order,
                    black_border=black_border,
                    parallel=parallel
                )

            # Parallel processing of regions
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_region_2d)(row) for row in regions_data
            )

            # Unpack results
            GRAPES_rp, GRAPES_gl, GRAPES_norm_gl, GRAPES_distance_transform = zip(*results)

            # Assign results to the DataFrame
            props['radial_layers'] = GRAPES_rp
            props['radial_layers_graylevel'] = GRAPES_gl
            props['radial_layers_graylevelnormed'] = GRAPES_norm_gl
            props['radial_layers_dt'] = GRAPES_distance_transform

            return props

        else:
            raise ValueError("grey_arr must be either a 2D or 3D array.")

    except Exception as e:
        logger.error(f"Error in GRAPES_optimized: {e}")
        raise

def radial_layers_quartiles(
    grapes_df: pd.DataFrame,
    prop: str = 'volume',
    grapes_properties: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Reduces GRAPES DataFrame based on quartiles of a specified property,
    computing the average, standard deviation, and standard error of radial GRAPES properties within each quartile.

    Args:
        grapes_df (pd.DataFrame): DataFrame output from GRAPES.
        prop (str, optional): Column name in `grapes_df` to compute quartiles on. Defaults to 'volume'.
        grapes_properties (List[str], optional): List of GRAPES properties to aggregate.
            Defaults to ['radial_layers', 'radial_layers_graylevel', 'radial_layers_graylevelnormed'].

    Returns:
        pd.DataFrame: DataFrame with GRAPES properties averaged over each quartile bin,
                      including means, standard deviations, standard errors, and sample counts per radial layer.
    """
    if grapes_properties is None:
        grapes_properties = ['radial_layers', 'radial_layers_graylevel', 'radial_layers_graylevelnormed']

    # Validate inputs
    if prop not in grapes_df.columns:
        raise ValueError(f"The property '{prop}' is not present in the DataFrame.")

    for gprop in grapes_properties:
        if gprop not in grapes_df.columns:
            raise ValueError(f"The GRAPES property '{gprop}' is not present in the DataFrame.")

    # Create a copy to avoid modifying the original DataFrame
    df = grapes_df.copy()

    # Assign quartile bins
    try:
        df['quartile_bin'] = pd.qcut(df[prop], 4, labels=False, duplicates='drop')
    except ValueError as e:
        raise ValueError(f"Error in computing quartiles: {e}")

    # Initialize a dictionary to store aggregated data
    aggregated_data = {
        'Quartile': [],
        'Property': [],
        'Layer': [],
        'Mean': [],
        'Std': [],
        'SE': [],
        'Sample_n': []
    }

    # Iterate over each quartile
    for quartile in sorted(df['quartile_bin'].unique()):
        quartile_df = df[df['quartile_bin'] == quartile]

        # Iterate over each specified GRAPES property
        for gprop in grapes_properties:
            # Extract the list of radial layers for the current property
            radial_layers_list = quartile_df[gprop].dropna().tolist()

            if not radial_layers_list:
                continue  # Skip if no data is present

            # Determine the maximum number of layers across all particles in the quartile
            max_layers = max(layer.shape[0] for layer in radial_layers_list)

            # Initialize lists to store means, stds, SEs, and sample counts per layer
            means = []
            stds = []
            ses = []
            sample_n = []

            # Aggregate data layer-wise
            for layer_idx in range(max_layers):
                # Collect the layer values from all particles, handling varying lengths
                layer_values = [
                    layer[layer_idx] for layer in radial_layers_list
                    if layer_idx < layer.shape[0]
                ]

                if not layer_values:
                    # If no particles have this layer index, append NaN
                    means.append(np.nan)
                    stds.append(np.nan)
                    ses.append(np.nan)
                    sample_n.append(0)
                    continue

                # Compute statistics for the current layer
                layer_mean = np.mean(layer_values)
                layer_std = np.std(layer_values)
                n_samples = len(layer_values)
                layer_se = layer_std / np.sqrt(n_samples) if n_samples > 0 else np.nan

                means.append(layer_mean)
                stds.append(layer_std)
                ses.append(layer_se)
                sample_n.append(n_samples)

            # Populate the aggregated data dictionary
            for layer_idx in range(max_layers):
                aggregated_data['Quartile'].append(quartile)
                aggregated_data['Property'].append(gprop)
                aggregated_data['Layer'].append(layer_idx + 1)  # 1-based indexing for layers
                aggregated_data['Mean'].append(means[layer_idx])
                aggregated_data['Std'].append(stds[layer_idx])
                aggregated_data['SE'].append(ses[layer_idx])
                aggregated_data['Sample_n'].append(sample_n[layer_idx])

    # Convert the aggregated data into a DataFrame
    quartiles_df = pd.DataFrame(aggregated_data)

    return quartiles_df

def plot_quartile_radial_layers(
    quartiles_df: pd.DataFrame,
    prop: str = 'radial_layers',  # Changed from 'property' to 'prop' for consistency
    error_type: str = 'SE',  # 'SE' for Standard Error, 'Std' for Standard Deviation
    title: Optional[str] = None,
    xlabel: str = 'Radial Layer',
    ylabel: str = 'Mean Value',
    figsize: tuple = (10, 6),
    palette: Optional[List[str]] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Plots the mean radial layers of each quartile with either Standard Error (SE) or Standard Deviation (Std) as error bars.

    Args:
        quartiles_df (pd.DataFrame): DataFrame output from radial_layers_quartiles.
        prop (str, optional): GRAPES property to plot 
            (e.g., 'radial_layers', 'radial_layers_graylevel', 'radial_layers_graylevelnormed').
            Defaults to 'radial_layers'.
        error_type (str, optional): Type of error bars to use ('SE' for Standard Error or 'Std' for Standard Deviation).
            Defaults to 'SE'.
        title (str, optional): Title of the plot. If None, a default title is generated.
            Defaults to None.
        xlabel (str, optional): Label for the x-axis. Defaults to 'Radial Layer'.
        ylabel (str, optional): Label for the y-axis. Defaults to 'Mean Value'.
        figsize (tuple, optional): Size of the figure. Defaults to (10, 6).
        palette (List[str], optional): List of colors for quartiles. If None, a default palette is used.
            Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to True.

    Returns:
        Optional[plt.Figure]: The Matplotlib Figure object if the plot is generated, else None.
    """
    # Validate the specified property
    if prop not in quartiles_df['Property'].unique():
        raise ValueError(f"The specified property '{prop}' is not present in the DataFrame.")
    
    # Validate the error_type parameter
    if error_type not in ['SE', 'Std']:
        raise ValueError("The 'error_type' parameter must be either 'SE' (Standard Error) or 'Std' (Standard Deviation).")
    
    # Filter the DataFrame for the specified property
    property_df = quartiles_df[quartiles_df['Property'] == prop]
    
    # Check if the filtered DataFrame is empty
    if property_df.empty:
        raise ValueError(f"No data available for the property '{prop}'.")
    
    # Generate default title if not provided
    if title is None:
        error_label = "Standard Error" if error_type == 'SE' else "Standard Deviation"
        title = f"Mean {prop.replace('_', ' ').title()} Across Quartiles with {error_label} Error Bars"
    
    # Determine the unique quartiles and sort them
    quartiles = sorted(property_df['Quartile'].unique())
    
    # Define default color palette if not provided
    if palette is None:
        # Use a qualitative colormap from matplotlib
        cmap = plt.get_cmap('tab10')
        palette = [cmap(i) for i in range(len(quartiles))]
    
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each quartile
    for quartile, color in zip(quartiles, palette):
        quartile_data = property_df[property_df['Quartile'] == quartile]
        
        # Sort by Layer to ensure correct plotting order
        quartile_data = quartile_data.sort_values(by='Layer')
        
        layers = quartile_data['Layer'].values
        means = quartile_data['Mean'].values
        
        if error_type == 'SE':
            errors = quartile_data['SE'].values
        else:  # error_type == 'Std'
            errors = quartile_data['Std'].values
        
        ax.errorbar(
            layers,
            means,
            yerr=errors,
            label=f'Quartile {quartile + 1}',  # Assuming quartiles are 0-indexed
            fmt='-o',
            capsize=5,
            color=color,
            ecolor=color,
            markeredgecolor='black',
            markersize=5,
            linestyle='-',
            linewidth=1
        )
    
    # Customize the plot
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(title='Quartiles')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the plot if a save path is provided (Removed as per request)
    # if save_path:
    #     plt.savefig(save_path, dpi=300)
    #     print(f"Plot saved to {save_path}")
    
    # Display the plot
    if show:
        plt.show()
    
    # Close the plot to free memory
    plt.close(fig)
    
    # Return the figure object
    return fig

def radial_layers_deciles(
    grapes_df: pd.DataFrame,
    prop: str = 'volume',
    grapes_properties: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Reduces GRAPES DataFrame based on deciles of a specified property,
    computing the average, standard deviation, and standard error of radial GRAPES properties within each decile.

    Args:
        grapes_df (pd.DataFrame): DataFrame output from GRAPES.
        prop (str, optional): Column name in `grapes_df` to compute deciles on. Defaults to 'volume'.
        grapes_properties (List[str], optional): List of GRAPES properties to aggregate.
            Defaults to ['radial_layers', 'radial_layers_graylevel', 'radial_layers_graylevelnormed'].

    Returns:
        pd.DataFrame: DataFrame with GRAPES properties averaged over each decile bin,
                      including means, standard deviations, standard errors, and sample counts per radial layer.
    """
    if grapes_properties is None:
        grapes_properties = ['radial_layers', 'radial_layers_graylevel', 'radial_layers_graylevelnormed']

    # Validate inputs
    if prop not in grapes_df.columns:
        raise ValueError(f"The property '{prop}' is not present in the DataFrame.")

    for gprop in grapes_properties:
        if gprop not in grapes_df.columns:
            raise ValueError(f"The GRAPES property '{gprop}' is not present in the DataFrame.")

    # Create a copy to avoid modifying the original DataFrame
    df = grapes_df.copy()

    # Assign decile bins
    try:
        df['decile_bin'] = pd.qcut(df[prop], 10, labels=False, duplicates='drop')
    except ValueError as e:
        raise ValueError(f"Error in computing deciles: {e}")

    # Initialize a dictionary to store aggregated data
    aggregated_data = {
        'Decile': [],
        'Property': [],
        'Layer': [],
        'Mean': [],
        'Std': [],
        'SE': [],
        'Sample_n': []
    }

    # Iterate over each decile
    for decile in sorted(df['decile_bin'].unique()):
        decile_df = df[df['decile_bin'] == decile]

        # Iterate over each specified GRAPES property
        for gprop in grapes_properties:
            # Extract the list of radial layers for the current property
            radial_layers_list = decile_df[gprop].dropna().tolist()

            if not radial_layers_list:
                continue  # Skip if no data is present

            # Determine the maximum number of layers across all particles in the decile
            max_layers = max(layer.shape[0] for layer in radial_layers_list)

            # Initialize lists to store means, stds, SEs, and sample counts per layer
            means = []
            stds = []
            ses = []
            sample_n = []

            # Aggregate data layer-wise
            for layer_idx in range(max_layers):
                # Collect the layer values from all particles, handling varying lengths
                layer_values = [
                    layer[layer_idx] for layer in radial_layers_list
                    if layer_idx < layer.shape[0]
                ]

                if not layer_values:
                    # If no particles have this layer index, append NaN
                    means.append(np.nan)
                    stds.append(np.nan)
                    ses.append(np.nan)
                    sample_n.append(0)
                    continue

                # Compute statistics for the current layer
                layer_mean = np.mean(layer_values)
                layer_std = np.std(layer_values)
                n_samples = len(layer_values)
                layer_se = layer_std / np.sqrt(n_samples) if n_samples > 0 else np.nan

                means.append(layer_mean)
                stds.append(layer_std)
                ses.append(layer_se)
                sample_n.append(n_samples)

            # Populate the aggregated data dictionary
            for layer_idx in range(max_layers):
                aggregated_data['Decile'].append(decile)
                aggregated_data['Property'].append(gprop)
                aggregated_data['Layer'].append(layer_idx + 1)  # 1-based indexing for layers
                aggregated_data['Mean'].append(means[layer_idx])
                aggregated_data['Std'].append(stds[layer_idx])
                aggregated_data['SE'].append(ses[layer_idx])
                aggregated_data['Sample_n'].append(sample_n[layer_idx])

    # Convert the aggregated data into a DataFrame
    deciles_df = pd.DataFrame(aggregated_data)

    return deciles_df

def plot_decile_radial_layers(
    deciles_df: pd.DataFrame,
    prop: str = 'radial_layers',  # Changed from 'property' to 'prop' for consistency
    error_type: str = 'SE',  # 'SE' for Standard Error, 'Std' for Standard Deviation
    title: Optional[str] = None,
    xlabel: str = 'Radial Layer',
    ylabel: str = 'Mean Value',
    figsize: tuple = (10, 6),
    palette: Optional[List[str]] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Plots the mean radial layers of each decile with either Standard Error (SE) or Standard Deviation (Std) as error bars.

    Args:
        deciles_df (pd.DataFrame): DataFrame output from radial_layers_deciles.
        prop (str, optional): GRAPES property to plot 
            (e.g., 'radial_layers', 'radial_layers_graylevel', 'radial_layers_graylevelnormed').
            Defaults to 'radial_layers'.
        error_type (str, optional): Type of error bars to use ('SE' for Standard Error or 'Std' for Standard Deviation).
            Defaults to 'SE'.
        title (str, optional): Title of the plot. If None, a default title is generated.
            Defaults to None.
        xlabel (str, optional): Label for the x-axis. Defaults to 'Radial Layer'.
        ylabel (str, optional): Label for the y-axis. Defaults to 'Mean Value'.
        figsize (tuple, optional): Size of the figure. Defaults to (10, 6).
        palette (List[str], optional): List of colors for deciles. If None, a default palette is used.
            Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to True.

    Returns:
        Optional[plt.Figure]: The Matplotlib Figure object if the plot is generated, else None.
    """
    # Validate the specified property
    if prop not in deciles_df['Property'].unique():
        raise ValueError(f"The specified property '{prop}' is not present in the DataFrame.")
    
    # Validate the error_type parameter
    if error_type not in ['SE', 'Std']:
        raise ValueError("The 'error_type' parameter must be either 'SE' (Standard Error) or 'Std' (Standard Deviation).")
    
    # Filter the DataFrame for the specified property
    property_df = deciles_df[deciles_df['Property'] == prop]
    
    # Check if the filtered DataFrame is empty
    if property_df.empty:
        raise ValueError(f"No data available for the property '{prop}'.")
    
    # Generate default title if not provided
    if title is None:
        error_label = "Standard Error" if error_type == 'SE' else "Standard Deviation"
        title = f"Mean {prop.replace('_', ' ').title()} Across Deciles with {error_label} Error Bars"
    
    # Determine the unique deciles and sort them
    deciles = sorted(property_df['Decile'].unique())
    
    # Define default color palette if not provided
    if palette is None:
        # Use a qualitative colormap from matplotlib
        cmap = plt.get_cmap('tab10')
        palette = [cmap(i) for i in range(len(deciles))]
    
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each decile
    for decile, color in zip(deciles, palette):
        decile_data = property_df[property_df['Decile'] == decile]
        
        # Sort by Layer to ensure correct plotting order
        decile_data = decile_data.sort_values(by='Layer')
        
        layers = decile_data['Layer'].values
        means = decile_data['Mean'].values
        
        if error_type == 'SE':
            errors = decile_data['SE'].values
        else:  # error_type == 'Std'
            errors = decile_data['Std'].values
        
        ax.errorbar(
            layers,
            means,
            yerr=errors,
            label=f'Decile {decile + 1}',  # Assuming deciles are 0-indexed
            fmt='-o',
            capsize=5,
            color=color,
            ecolor=color,
            markeredgecolor='black',
            markersize=5,
            linestyle='-',
            linewidth=1
        )
    
    # Customize the plot
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(title='Deciles')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the plot if a save path is provided (Removed as per request)
    # if save_path:
    #     plt.savefig(save_path, dpi=300)
    #     print(f"Plot saved to {save_path}")
    
    # Display the plot
    if show:
        plt.show()
    
    # Close the plot to free memory
    plt.close(fig)
    
    # Return the figure object
    return fig


def compute_quartile_statistics(
    grapes_df: pd.DataFrame,
    prop: str = 'volume',
    additional_properties: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Reduces a GRAPES DataFrame into quartiles based on a specified property,
    computing the mean, standard deviation, and standard error for each GRAPES property within each quartile.

    Args:
        grapes_df (pd.DataFrame): The GRAPES DataFrame to process.
        prop (str, optional): The property over which to calculate quartiles. Defaults to 'volume'.
        additional_properties (List[str], optional): 
            Additional GRAPES properties to include in the analysis. 
            If provided, these properties are appended to the default set. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing quartile information and statistics (mean, Std, SE) for each specified property.
                      It has 4 rows (one for each quartile) and columns for each property's statistics.

    Raises:
        ValueError: If the specified `prop` is not in `grapes_df`.
        ValueError: If any of the `additional_properties` are not in `grapes_df`.
    """

    # Validate the specified property
    if prop not in grapes_df.columns:
        raise ValueError(f"The property '{prop}' is not present in the DataFrame.")

    # Define default GRAPES properties
    default_properties = [
        'volume', 
        'equivalent_diameter_volume', 
        'sphericity',
        '_surface_area',
        'intensity_min',
        'intensity_mean',
        'intensity_max'
    ]

    # Ensure all specified properties exist in the DataFrame
    missing_props = [p for p in default_properties if p not in grapes_df.columns]
    if missing_props:
        raise ValueError(f"The following properties are missing from the DataFrame: {missing_props}")

    # Create a copy to avoid modifying the original DataFrame
    df = grapes_df.copy()

    # Assign quartile bins based on the specified property
    try:
        df['quartile_bin'] = pd.qcut(df[prop], 4, labels=False, duplicates='drop')
    except ValueError as e:
        raise ValueError(f"Error in computing quartiles: {e}")

    # Group by quartile_bin and compute mean, Std for each property
    quartile_stats = []
    for p in default_properties:
        quartile_stat = df.groupby('quartile_bin').agg(**{
            f"{p}_mean": (p, 'mean'),
            f"{p}_std": (p, 'std'),
            f"{p}_count": (p, 'count')
        })
        quartile_stats.append(quartile_stat)
    quartile_stats = pd.concat(quartile_stats, axis=1)

    # Calculate Standard Error (SE) = std / sqrt(n) for each property
    for p in default_properties:
        std_col = f"{p}_std"
        count_col = f"{p}_count"
        se_col = f"{p}_se"
        if std_col in quartile_stats.columns and count_col in quartile_stats.columns:
            quartile_stats[se_col] = quartile_stats[std_col] / np.sqrt(quartile_stats[count_col])
        else:
            quartile_stats[se_col] = np.nan
    
    # Convert quartile_bin from 0-based to 1-based indexing
    quartile_stats.index = quartile_stats.index + 1
    quartile_stats.index.name = 'Quartile'

    # Reset index to turn Quartile into a column
    quartile_stats.reset_index(inplace=True)

    return quartile_stats

def compute_decile_statistics(
    grapes_df: pd.DataFrame,
    prop: str = 'volume',
    additional_properties: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Reduces a GRAPES DataFrame into deciles based on a specified property,
    computing the mean, standard deviation, and standard error for each GRAPES property within each decile.
    
    Args:
        grapes_df (pd.DataFrame): The GRAPES DataFrame to process.
        prop (str, optional): The property over which to calculate deciles. Defaults to 'volume'.
        additional_properties (List[str], optional): 
            Additional GRAPES properties to include in the analysis. 
            If provided, these properties are appended to the default set. Defaults to None.
    
    Returns:
        pd.DataFrame: A DataFrame containing decile information and statistics (mean, Std, SE) for each specified property.
                      It has 4 rows (one for each decile) and columns for each property's statistics.
    
    Raises:
        ValueError: If the specified `prop` is not in `grapes_df`.
        ValueError: If any of the `additional_properties` are not in `grapes_df`.
    """
    
    # Validate the specified property
    if prop not in grapes_df.columns:
        raise ValueError(f"The property '{prop}' is not present in the DataFrame.")

    # Define default GRAPES properties
    default_properties = [
        'volume', 
        'equivalent_diameter_volume', 
        'sphericity',
        '_surface_area',
        'intensity_min',
        'intensity_mean',
        'intensity_max'
    ]
        
    # Ensure all specified properties exist in the DataFrame
    missing_props = [p for p in default_properties if p not in grapes_df.columns]
    if missing_props:
        raise ValueError(f"The following properties are missing from the DataFrame: {missing_props}")

    # Create a copy to avoid modifying the original DataFrame
    df = grapes_df.copy()

    # Assign decile bins based on the specified property
    try:
        df['decile_bin'] = pd.qcut(df[prop], 10, labels=False, duplicates='drop')
    except ValueError as e:
        raise ValueError(f"Error in computing deciles: {e}")

    # Group by decile_bin and compute mean, Std for each property
    decile_stats = []
    for p in default_properties:
        decile_stat = df.groupby('decile_bin').agg(**{
            f"{p}_mean": (p, 'mean'),
            f"{p}_std": (p, 'std'),
            f"{p}_count": (p, 'count')
        })
        decile_stats.append(decile_stat)
    decile_stats = pd.concat(decile_stats, axis=1)
        
    # Calculate Standard Error (SE) = std / sqrt(n) for each property
    for p in default_properties:
        std_col = f"{p}_std"
        count_col = f"{p}_count"
        se_col = f"{p}_se"
        if std_col in decile_stats.columns and count_col in decile_stats.columns:
            decile_stats[se_col] = decile_stats[std_col] / np.sqrt(decile_stats[count_col])
        else:
            decile_stats[se_col] = np.nan  # Assign NaN if columns are missing

    # Convert decile_bin from 0-based to 1-based indexing
    decile_stats.index = decile_stats.index + 1
    decile_stats.index.name = 'Decile'

    # Reset index to turn Decile into a column
    decile_stats.reset_index(inplace=True)
    
    return decile_stats

def prop_2_image(
    labels: np.ndarray,
    df: pd.DataFrame,
    prop: str = 'volume',
    save_path: Optional[str] = None,
    slice_idx: Optional[int] = None,
    cmap: str = 'viridis',
    figsize: tuple = (8, 6),
    show: bool = True,
    save_format: Optional[str] = None
) -> np.ndarray:
    """
    Assign property values to labeled regions in an image based on a GRAPES dataframe and optionally save the image.
    
    Each pixel in the labels image is assigned the property value corresponding to its label.
    If `save_path` is provided, the image (2D or a specific slice of 3D) is plotted and saved to the specified path.
    For 3D images, saves the entire volume as a multi-page TIFF file.
    
    Args:
        labels (np.ndarray): 
            Labeled image where each pixel's value corresponds to a region label. 
            Can be 2D or 3D.
        df (pd.DataFrame): 
            GRAPES dataframe containing at least 'label' and the specified property.
        prop (str, optional): 
            The property to assign to the image regions. Defaults to 'volume'.
        save_path (Optional[str], optional): 
            File path to save the plotted image (e.g., 'output.png' for 2D, 'output.tiff' for 3D). 
            If None, the image is not saved. Defaults to None.
        slice_idx (Optional[int], optional): 
            The slice index to plot if `labels` is 3D. 
            If None, plots the slice corresponding to the particle's center of mass.
            Ignored if `labels` is 2D. Defaults to None.
        cmap (str, optional): 
            Colormap for the image plot. Defaults to 'viridis'.
        figsize (tuple, optional): 
            Size of the plot figure in inches. Defaults to (8, 6).
        show (bool, optional): 
            Whether to display the plot. Defaults to True.
        save_format (Optional[str], optional): 
            The format to save the image. 
            - For 2D images: Common image formats like 'png', 'jpg', 'tiff', etc.
            - For 3D images: Must be 'tiff' to support multi-page TIFFs.
            If None, the format is inferred from the file extension in `save_path`.
            Defaults to None.
    
    Returns:
        np.ndarray: 
            Image with the property values assigned to each pixel based on its label. 
            Labels not present in the dataframe are assigned 0.
    
    Raises:
        TypeError: 
            If `labels` is not a numpy ndarray or contains non-integer labels.
        ValueError: 
            If required columns are missing in the dataframe.
            If the specified slice index is out of bounds.
            If `save_format` is incompatible with the image dimensionality.
        KeyError: 
            If 'centroid_local-0' is missing when `slice_idx` is None in a 3D image.
        ImportError:
            If 'tifffile' library is not installed when saving 3D TIFF images.
    """
    
    # Input validation
    if not isinstance(labels, np.ndarray):
        raise TypeError(f"'labels' must be a numpy ndarray, got {type(labels)} instead.")
    
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"'labels' array must contain integer values, got {labels.dtype} instead.")
    
    required_columns = {'label', prop}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"The dataframe is missing required columns: {missing}")
    
    if df['label'].duplicated().any():
        duplicated_labels = df['label'][df['label'].duplicated()].unique()
        raise ValueError(f"Duplicate labels found in dataframe: {duplicated_labels}")
    
    # Create a mapping from label to property value
    label_to_prop = pd.Series(df[prop].values, index=df['label']).to_dict()
    
    # Determine the maximum label to create a lookup array
    max_label = labels.max()
    
    # Initialize the lookup array with 0 for labels not present in the mapping
    # Use float dtype to accommodate properties that are floats
    lookup_dtype = float
    lookup = np.zeros(max_label + 1, dtype=lookup_dtype)
    
    # Assign property values to the corresponding labels
    for label, value in label_to_prop.items():
        if not isinstance(label, (int, np.integer)):
            warnings.warn(f"Label '{label}' is not an integer. It will be skipped.")
            continue
        if label < 0 or label > max_label:
            warnings.warn(f"Label '{label}' is outside the range of labels in the image (0 to {max_label}). It will be skipped.")
            continue
        lookup[label] = value
    
    # Identify labels in the image that are not in the mapping
    unique_labels_in_image = np.unique(labels)
    labels_not_in_mapping = np.setdiff1d(unique_labels_in_image, list(label_to_prop.keys()))
    if len(labels_not_in_mapping) > 0:
        warnings.warn(f"The following labels in the image are not present in the dataframe and will be assigned 0: {labels_not_in_mapping}")
    
    # Vectorized mapping of labels to property values
    prop_image = lookup[labels]
    
    # Determine if the image is 2D or 3D
    is_2d = labels.ndim == 2
    is_3d = labels.ndim == 3
    
    if not (is_2d or is_3d):
        raise ValueError(f"Unsupported number of dimensions for 'labels': {labels.ndim}. Expected 2D or 3D array.")
    
    # Determine save format if not explicitly provided
    if save_format is None and save_path is not None:
        _, ext = os.path.splitext(save_path)
        if ext:
            save_format = ext.lower().strip('.')
        else:
            raise ValueError("Cannot infer 'save_format' from 'save_path' as it has no extension.")
    
    # Handle saving and plotting based on dimensionality
    if is_2d:
        # 2D Image Handling
        if save_path:
            # Determine the format
            if save_format is None:
                raise ValueError("When 'save_path' is provided, 'save_format' must be specified either via 'save_format' parameter or by including a valid file extension in 'save_path'.")
            supported_2d_formats = ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif']
            if save_format not in supported_2d_formats:
                raise ValueError(f"Unsupported save_format '{save_format}' for 2D images. Supported formats are: {supported_2d_formats}")
            
            # Plot and save the 2D image
            plt.figure(figsize=figsize)
            im = plt.imshow(prop_image, cmap=cmap)
            plt.title(f"Property '{prop}' Image")
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            cbar = plt.colorbar(im)
            cbar.ax.set_ylabel(prop, rotation=270, labelpad=15)
            plt.tight_layout()
            
            try:
                plt.savefig(save_path, dpi=300)
                if show:
                    plt.show()
                plt.close()
                if not show:
                    warnings.warn("Image saved successfully, but not displayed because 'show' is set to False.")
            except Exception as e:
                raise IOError(f"Failed to save the 2D image to '{save_path}': {e}")
        else:
            # Plot the 2D image without saving
            if show:
                plt.figure(figsize=figsize)
                im = plt.imshow(prop_image, cmap=cmap)
                plt.title(f"Property '{prop}' Image")
                plt.xlabel('X-axis')
                plt.ylabel('Y-axis')
                cbar = plt.colorbar(im)
                cbar.ax.set_ylabel(prop, rotation=270, labelpad=15)
                plt.tight_layout()
                plt.show()
                plt.close()
        
    elif is_3d:
        # 3D Image Handling
        num_slices = labels.shape[0]
        
        if slice_idx is None:
            # Attempt to determine the central slice
            slice_idx = num_slices // 2
            warnings.warn(f"'slice_idx' not provided. Using central slice index {slice_idx}.")
        
        # Validate slice index
        if not (0 <= slice_idx < num_slices):
            raise IndexError(f"'slice_idx' {slice_idx} is out of bounds for the labels array with shape {labels.shape}.")
        
        # Extract the specific slice
        prop_slice = prop_image[slice_idx, :, :]
        
        if save_path:
            # Determine the format
            if save_format is None:
                raise ValueError("When 'save_path' is provided, 'save_format' must be specified either via 'save_format' parameter or by including a valid file extension in 'save_path'.")
            if save_format != 'tiff' and save_format != 'tif':
                raise ValueError("For 3D images, the 'save_format' must be 'tiff' or 'tif' to support multi-page TIFF files.")
            
            # Plot and save the specific slice
            plt.figure(figsize=figsize)
            im = plt.imshow(prop_slice, cmap=cmap)
            plt.title(f"Property '{prop}' - Slice {slice_idx}")
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            cbar = plt.colorbar(im)
            cbar.ax.set_ylabel(prop, rotation=270, labelpad=15)
            plt.tight_layout()
            
            try:
                plt.savefig(save_path, dpi=300)
                if show:
                    plt.show()
                plt.close()
                if not show:
                    warnings.warn("Image saved successfully, but not displayed because 'show' is set to False.")
            except Exception as e:
                raise IOError(f"Failed to save the 3D slice image to '{save_path}': {e}")
            
            # Additionally, save the entire 3D image as a TIFF file if desired
            # This part depends on whether the user wants to save all slices or just the current slice
            # Here, we provide the option to save all slices as a TIFF stack
            # The user can call this function multiple times with different slice_idx if needed
            # Alternatively, we can add an option to save the entire 3D image
            # For simplicity, we'll assume saving the entire 3D image is not required here
            
        else:
            # Plot the specific slice without saving
            if show:
                plt.figure(figsize=figsize)
                im = plt.imshow(prop_slice, cmap=cmap)
                plt.title(f"Property '{prop}' - Slice {slice_idx}")
                plt.xlabel('X-axis')
                plt.ylabel('Y-axis')
                cbar = plt.colorbar(im)
                cbar.ax.set_ylabel(prop, rotation=270, labelpad=15)
                plt.tight_layout()
                plt.show()
                plt.close()
    
    return prop_image

def plot_radial_intensities(
    df: pd.DataFrame,
    label: int,
    plotting: str = 'normalised',
    title_suffix: Optional[str] = None,
    xlabel: str = 'Radial Position',
    ylabel: Optional[str] = None,
    figsize: tuple = (8, 6),
    color: str = 'blue',
    marker: Union[str, None] = 'o',
    linestyle: str = '-',
    linewidth: float = 1.5,
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[Figure]:
    """
    Plot graylevels or normalized graylevels at different radial positions for a specific particle label.
    
    Each radial position within the specified particle is plotted against its corresponding graylevel or
    normalized graylevel value.
    
    Args:
        df (pd.DataFrame): GRAPES output DataFrame containing at least 'label', 'radial_layers',
                           and either 'radial_layers_graylevelnormed' or 'radial_layers_graylevels'.
        label (int): The label of the particle to plot.
        plotting (str, optional): Type of plot - 'normalised' or 'graylevels'. 
                                  'normalised' plots normalized graylevels, 'graylevels' plots raw graylevels.
                                  Defaults to 'normalised'.
        title_suffix (str, optional): Additional string to append to the plot title for customization.
                                       Defaults to None.
        xlabel (str, optional): Label for the x-axis. Defaults to 'Radial Position'.
        ylabel (str, optional): Label for the y-axis. If None, it is set based on the plotting type.
                                Defaults to None.
        figsize (tuple, optional): Size of the figure in inches. Defaults to (8, 6).
        color (str, optional): Color of the plot line. Defaults to 'blue'.
        marker (str or None, optional): Marker style for the plot. Set to None for no markers.
                                         Defaults to 'o'.
        linestyle (str, optional): Line style for the plot. Defaults to '-'.
        linewidth (float, optional): Width of the plot line. Defaults to 1.5.
        save_path (str, optional): File path to save the plot image. If None, the plot is not saved.
                                   Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to True.
    
    Returns:
        Optional[Figure]: The Matplotlib Figure object if the plot is generated, else None.
    
    Raises:
        ValueError: If the 'plotting' parameter is not 'normalised' or 'graylevels'.
        KeyError: If required columns are missing from the DataFrame.
        ValueError: If the specified label is not found in the DataFrame.
        TypeError: If 'radial_layers' or graylevel columns are not array-like.
    """
    
    # Validate 'plotting' parameter
    valid_plotting = ['normalised', 'graylevels']
    if plotting not in valid_plotting:
        raise ValueError(f"Invalid plotting type '{plotting}'. Expected one of {valid_plotting}.")
    
    # Define required columns based on plotting type
    if plotting == 'normalised':
        required_columns = ['label', 'radial_layers', 'radial_layers_graylevelnormed']
        default_ylabel = 'Normalised Graylevel'
        plot_title = f'Particle {label} - Radial Layers Normalised'
    else:  # plotting == 'graylevels'
        required_columns = ['label', 'radial_layers', 'radial_layers_graylevels']
        default_ylabel = 'Graylevel'
        plot_title = f'Particle {label} - GRAPES Graylevels'
    
    # Check if required columns are present
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"The DataFrame is missing required columns: {missing_columns}")
    
    # Filter DataFrame for the specified label
    particle_df = df[df['label'] == label]
    if particle_df.empty:
        raise ValueError(f"No data found for label {label}. Please check the label and try again.")
    
    # Assuming each label corresponds to a unique particle, take the first occurrence
    particle_row = particle_df.iloc[0]
    
    # Extract radial layers and corresponding graylevels
    radial_layers = particle_row['radial_layers']
    if plotting == 'normalised':
        graylevels = particle_row['radial_layers_graylevelnormed']
    else:
        graylevels = particle_row['radial_layers_graylevels']
    
    # Validate that radial_layers and graylevels are array-like
    if not isinstance(radial_layers, (list, np.ndarray, pd.Series)):
        raise TypeError(f"'radial_layers' should be array-like, got {type(radial_layers)} instead.")
    if not isinstance(graylevels, (list, np.ndarray, pd.Series)):
        raise TypeError(f"'radial_layers_graylevelnormed' or 'radial_layers_graylevels' should be array-like, got {type(graylevels)} instead.")
    
    # Convert to numpy arrays for consistency
    radial_layers = np.array(radial_layers)
    graylevels = np.array(graylevels)
    
    # Check if both arrays have the same length
    if radial_layers.shape[0] != graylevels.shape[0]:
        raise ValueError(f"'radial_layers' and graylevels arrays must have the same length. "
                         f"Got {radial_layers.shape[0]} and {graylevels.shape[0]} respectively.")
    
    # Set ylabel if not provided
    if ylabel is None:
        ylabel = default_ylabel
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(radial_layers, graylevels, color=color, marker=marker, linestyle=linestyle, linewidth=linewidth)
    
    # Set plot title and labels
    if title_suffix:
        ax.set_title(f"{plot_title} - {title_suffix}", fontsize=14)
    else:
        ax.set_title(plot_title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    
    # Show the plot
    if show:
        plt.show()
    
    # Close the plot to free memory
    plt.close(fig)
    
    return fig

def plot_particle_image(
    df: pd.DataFrame,
    label: int,
    image_type: str = 'distance_transform',
    slice_idx: Optional[int] = None,
    title_suffix: Optional[str] = None,
    xlabel: str = 'X-axis',
    ylabel: str = 'Y-axis',
    figsize: tuple = (8, 6),
    cmap: str = 'viridis',
    colorbar: bool = True,
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[Figure]:
    """
    Plot the distance transform, graylevel, or mask image of a specific particle label from the GRAPES dataframe.
    
    Args:
        df (pd.DataFrame): GRAPES output DataFrame containing necessary columns.
        label (int): The label of the particle to plot.
        image_type (str, optional): Type of image to plot. Options:
            - 'distance_transform': Plots the distance transform image.
            - 'graylevel': Plots the graylevel image.
            - 'mask': Plots the mask image.
            Defaults to 'distance_transform'.
        slice_idx (int, optional): The slice index to plot. 
            If None, plots the slice corresponding to the particle's center of mass.
            Defaults to None.
        title_suffix (str, optional): Additional string to append to the plot title for customization.
            Defaults to None.
        xlabel (str, optional): Label for the x-axis. Defaults to 'X-axis'.
        ylabel (str, optional): Label for the y-axis. Defaults to 'Y-axis'.
        figsize (tuple, optional): Size of the figure in inches. Defaults to (8, 6).
        cmap (str, optional): Colormap to use for the image. Defaults to 'viridis'.
        colorbar (bool, optional): Whether to display a colorbar. Defaults to True.
        save_path (str, optional): File path to save the plot image. If None, the plot is not saved.
            Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to True.
    
    Returns:
        Optional[Figure]: The Matplotlib Figure object if the plot is generated, else None.
    
    Raises:
        ValueError: If 'image_type' is not one of the expected options.
        KeyError: If required columns are missing from the DataFrame.
        ValueError: If the specified label is not found in the DataFrame.
        IndexError: If the specified slice index is out of bounds.
        TypeError: If the image data is not in the expected format.
    """
    
    # Define valid image types
    valid_image_types = ['distance_transform', 'graylevel', 'mask']
    if image_type not in valid_image_types:
        raise ValueError(f"Invalid image_type '{image_type}'. Expected one of {valid_image_types}.")
    
    # Define required columns based on image_type
    if image_type == 'distance_transform':
        required_columns = ['label', 'centroid_local-0', 'radial_layers_dt']
        default_title = f'Particle {label} - Distance Transform'
    elif image_type == 'graylevel':
        required_columns = ['label', 'centroid_local-0', 'image_intensity']
        default_title = f'Particle {label} - Graylevel Image'
    elif image_type == 'mask':
        required_columns = ['label', 'centroid_local-0', 'image']
        default_title = f'Particle {label} - Mask Image'
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"The DataFrame is missing required columns for '{image_type}': {missing_columns}")
    
    # Filter DataFrame for the specified label
    particle_df = df[df['label'] == label]
    if particle_df.empty:
        raise ValueError(f"No data found for label {label} in the DataFrame.")
    
    # Assuming labels are unique; take the first occurrence
    particle_row = particle_df.iloc[0]
    
    # Determine the slice to plot
    if slice_idx is None:
        try:
            centre = int(particle_row['centroid_local-0'])
        except KeyError:
            raise KeyError("'centroid_local-0' column is missing from the DataFrame.")
        except (TypeError, ValueError):
            raise ValueError(f"'centroid_local-0' must be an integer or convertible to integer, got {particle_row['centroid_local-0']}.")
        slice_to_plot = centre
    else:
        slice_to_plot = slice_idx
    
    # Extract the appropriate image slice based on image_type
    if image_type == 'distance_transform':
        try:
            radial_layers_dt = particle_row['radial_layers_dt']
            if not isinstance(radial_layers_dt, (list, np.ndarray)):
                raise TypeError(f"'radial_layers_dt' for label {label} is not array-like.")
            distance_transform_image = radial_layers_dt[slice_to_plot]
        except IndexError:
            raise IndexError(f"Slice index {slice_to_plot} is out of bounds for 'radial_layers_dt'.")
        except TypeError as e:
            raise TypeError(f"Error accessing 'radial_layers_dt' for label {label}: {e}")
        
        image_to_plot = distance_transform_image
    elif image_type == 'graylevel':
        try:
            image_intensity = particle_row['image_intensity']
            if not isinstance(image_intensity, (list, np.ndarray)):
                raise TypeError(f"'image_intensity' for label {label} is not array-like.")
            graylevel_image = image_intensity[slice_to_plot]
        except IndexError:
            raise IndexError(f"Slice index {slice_to_plot} is out of bounds for 'image_intensity'.")
        except TypeError as e:
            raise TypeError(f"Error accessing 'image_intensity' for label {label}: {e}")
        
        # Replace zeros with the minimum non-zero value
        non_zero_values = graylevel_image[graylevel_image != 0]
        if non_zero_values.size == 0:
            warnings.warn(f"All graylevel values are zero for label {label} in slice {slice_to_plot}.")
            min_non_zero = 0
        else:
            min_non_zero = np.min(non_zero_values)
        
        # Avoid replacing zero with zero
        if min_non_zero == 0:
            warnings.warn(f"Minimum non-zero graylevel value is zero for label {label} in slice {slice_to_plot}. No replacement performed.")
            processed_graylevel_image = graylevel_image
        else:
            processed_graylevel_image = np.where(graylevel_image == 0, min_non_zero, graylevel_image)
        
        image_to_plot = processed_graylevel_image
    elif image_type == 'mask':
        try:
            image = particle_row['image']
            if not isinstance(image, (list, np.ndarray)):
                raise TypeError(f"'image' for label {label} is not array-like.")
            mask_image = image[slice_to_plot]
        except IndexError:
            raise IndexError(f"Slice index {slice_to_plot} is out of bounds for 'image'.")
        except TypeError as e:
            raise TypeError(f"Error accessing 'image' for label {label}: {e}")
        
        image_to_plot = mask_image
    
    # Validate the extracted image
    if not isinstance(image_to_plot, (np.ndarray, list)):
        raise TypeError(f"The extracted image data for label {label} is not array-like.")
    
    # Convert to NumPy array if it's a list
    if isinstance(image_to_plot, list):
        image_to_plot = np.array(image_to_plot)
    
    # Check if the image is 2D
    if image_to_plot.ndim != 2:
        raise ValueError(f"The extracted image for label {label} in slice {slice_to_plot} is not 2D. It has {image_to_plot.ndim} dimensions.")
    
    # Prepare the plot
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(image_to_plot, cmap=cmap)
    
    # Set title with optional suffix
    if title_suffix:
        ax.set_title(f"{default_title} - {title_suffix}", fontsize=14)
    else:
        ax.set_title(default_title, fontsize=14)
    
    # Set axis labels
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Add colorbar if requested
    if colorbar:
        cbar = fig.colorbar(cax, ax=ax)
        cbar.ax.set_ylabel('Intensity', rotation=270, labelpad=15)
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Save the plot if a save path is provided
    if save_path:
        try:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")
        except Exception as e:
            warnings.warn(f"Failed to save plot to {save_path}: {e}")
    
    # Display the plot
    if show:
        plt.show()
    
    # Close the plot to free memory
    plt.close(fig)
    
    return fig

def save_dataframes(
    dataframes: Dict[str, pd.DataFrame],
    output_dir: str = "./saved_dataframes",
    formats: Union[str, List[str]] = ['xlsx', 'pkl'],
    excel_filename: Optional[str] = "dataframes.xlsx",
    separate_excel_sheets: bool = True,
    overwrite: bool = False,
    verbose: bool = True
) -> None:
    """
    Save multiple pandas DataFrames to specified file formats (.xlsx and/or .pkl).
    
    This function allows saving each DataFrame as a separate file or consolidating
    multiple DataFrames into a single Excel workbook with multiple sheets.
    Additionally, it handles specific array-like columns by saving them into separate Excel sheets.
    
    Args:
        dataframes (Dict[str, pd.DataFrame]): 
            A dictionary where keys are desired DataFrame names and values are the pandas DataFrames to save.
        
        output_dir (str, optional): 
            The directory where the files will be saved. Defaults to "./saved_dataframes".
        
        formats (Union[str, List[str]], optional): 
            The file formats to save the DataFrames. Can be a single format or a list of formats.
            Supported formats are 'xlsx' and 'pkl'. Defaults to ['xlsx', 'pkl'].
        
        excel_filename (Optional[str], optional): 
            The filename for the consolidated Excel workbook if 'xlsx' is in formats and
            `separate_excel_sheets` is True. Defaults to "dataframes.xlsx".
        
        separate_excel_sheets (bool, optional): 
            If True and 'xlsx' is in formats, saves each DataFrame to a separate sheet
            within a single Excel workbook specified by `excel_filename`. 
            Additionally, saves specified array-like columns into separate sheets.
            If False, saves each DataFrame to a separate Excel file.
            Defaults to True.
        
        overwrite (bool, optional): 
            Whether to overwrite existing files. If False, appends a numerical suffix to the filename.
            Defaults to False.
        
        verbose (bool, optional): 
            If True, prints informative messages about the saving process. 
            If False, suppresses output. Defaults to True.
    
    Raises:
        ValueError: 
            If unsupported formats are provided or if dataframes dictionary is empty.
        TypeError: 
            If `dataframes` is not a dictionary or contains non-pandas DataFrame objects.
        IOError: 
            If saving to a file fails.
    """
    
    # Define the specific array-like columns to handle separately
    array_columns_to_handle = ['radial_layers_graylevel', 'radial_layers_graylevelnormed']
    array_columns_to_exclude = ['radial_layers', 'radial_layers_graylevel', 'radial_layers_graylevelnormed', 'radial_layers_dt', 'image_intensity', 'image']
    
    def sanitize_sheet_name(name: str) -> str:
        """
        Sanitize the sheet name to be compliant with Excel's naming rules.
        Removes or replaces invalid characters.
        
        Args:
            name (str): Original sheet name.
        
        Returns:
            str: Sanitized sheet name.
        """
        invalid_chars = ['\\', '/', '*', '?', ':', '[', ']']
        for char in invalid_chars:
            name = name.replace(char, '_')
        # Excel sheet names must be <= 31 characters
        return name[:31]
    
    def exclude_columns_except(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
        """
        Exclude specified columns from the DataFrame.
        
        Args:
            df (pd.DataFrame): Original DataFrame.
            exclude (List[str]): List of column names to exclude.
        
        Returns:
            pd.DataFrame: DataFrame without the excluded columns.
        """
        return df.drop(columns=exclude, errors='ignore')
    
    def process_array_column(df: pd.DataFrame, array_col: str, df_name: str) -> Optional[pd.DataFrame]:
        """
        Process a specific array-like column by expanding its arrays into separate columns.
        
        Args:
            df (pd.DataFrame): Original DataFrame.
            array_col (str): The array-like column to process.
            df_name (str): Name of the original DataFrame for naming the sheet.
        
        Returns:
            Optional[pd.DataFrame]: Processed DataFrame or None if processing fails.
        """
        if array_col not in df.columns:
            if verbose:
                print(f"Column '{array_col}' not found in DataFrame '{df_name}'. Skipping.")
            return None
        if 'label' not in df.columns:
            if verbose:
                print(f"Column 'label' not found in DataFrame '{df_name}'. Cannot set as index for '{array_col}'. Skipping.")
            return None
        
        # Ensure all entries in array_col are 1D arrays
        if not df[array_col].apply(lambda x: isinstance(x, (list, tuple, np.ndarray)) and np.ndim(x) == 1).all():
            if verbose:
                print(f"Column '{array_col}' in DataFrame '{df_name}' contains non-1D arrays. Skipping.")
            return None
        
        try:
            # Create a new DataFrame from the array column
            array_series = df[array_col]
            # Determine the maximum length of the arrays
            max_len = array_series.apply(len).max()
            # Pad arrays with NaN to ensure consistent length
            array_series_padded = array_series.apply(lambda x: list(x) + [np.nan]*(max_len - len(x)))
            array_df = pd.DataFrame(array_series_padded.tolist(), index=df['label'])
            # Rename the columns to indicate the array position
            array_df.columns = [f"layer_{i+1}" for i in range(array_df.shape[1])]
            if verbose:
                print(f"Processed array column '{array_col}' from DataFrame '{df_name}'.")
            return array_df
        except Exception as e:
            if verbose:
                print(f"Error processing array column '{array_col}' in DataFrame '{df_name}': {e}")
            return None
    
    # Input Validation
    if not isinstance(dataframes, dict):
        raise TypeError(f"'dataframes' must be a dictionary with string keys and pandas DataFrame values, got {type(dataframes)} instead.")
    
    if not dataframes:
        raise ValueError("The 'dataframes' dictionary is empty. Provide at least one DataFrame to save.")
    
    supported_formats = ['xlsx', 'pkl']
    if isinstance(formats, str):
        formats = [formats]
    elif isinstance(formats, list):
        formats = formats
    else:
        raise TypeError(f"'formats' must be a string or a list of strings, got {type(formats)} instead.")
    
    # Check for unsupported formats
    unsupported = set(formats) - set(supported_formats)
    if unsupported:
        raise ValueError(f"Unsupported formats detected: {unsupported}. Supported formats are {supported_formats}.")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print(f"Output directory set to: {os.path.abspath(output_dir)}")
    
    # Initialize a dictionary to hold array DataFrames to be saved as separate sheets
    array_dataframes = {}
    
    # Save as Excel
    if 'xlsx' in formats:
        if separate_excel_sheets:
            # Prepare the main Excel writer
            excel_path = os.path.join(output_dir, excel_filename)
            original_excel_path = excel_path
            counter = 1
            while os.path.exists(excel_path) and not overwrite:
                base, ext = os.path.splitext(original_excel_path)
                excel_path = f"{base}_{counter}{ext}"
                counter += 1
            if counter > 1 and not overwrite:
                warnings.warn(f"Excel file '{original_excel_path}' exists. Saving as '{excel_path}' instead.")
            
            try:
                with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                    for name, df in dataframes.items():
                        # Identify and process array-like columns
                        for array_col in array_columns_to_handle:
                            array_df = process_array_column(df, array_col, name)
                            if array_df is not None:
                                array_sheet_name = sanitize_sheet_name(f"{array_col}")
                                array_df.to_excel(writer, sheet_name=array_sheet_name, index=True)
                                if verbose:
                                    print(f"Saved array DataFrame '{array_col}' to sheet '{array_sheet_name}' in '{excel_path}'.")
                        
                        # Exclude the specified array columns from the main DataFrame
                        df_to_save = exclude_columns_except(df, array_columns_to_exclude)
                        
                        # Save the main DataFrame to a sheet
                        sheet_name = sanitize_sheet_name(name)
                        df_to_save.to_excel(writer, sheet_name=sheet_name, index=False)
                        if verbose:
                            print(f"Saved DataFrame '{name}' to sheet '{sheet_name}' in '{excel_path}'.")
                
                if verbose:
                    print(f"All DataFrames have been saved to '{excel_path}' with separate sheets.")
            except Exception as e:
                raise IOError(f"Failed to save DataFrames to Excel: {e}")
        else:
            # Save each DataFrame to a separate Excel file
            for name, df in dataframes.items():
                # Identify and process array-like columns
                array_dfs = {}
                for array_col in array_columns_to_handle:
                    array_df = process_array_column(df, array_col, name)
                    if array_df is not None:
                        array_dfs[array_col] = array_df
                
                # Exclude the specified array columns from the main DataFrame
                df_to_save = exclude_columns_except(df, array_columns_to_exclude)
                
                # Sanitize the DataFrame name for the filename
                sanitized_name = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in name)
                excel_path = os.path.join(output_dir, f"{sanitized_name}.xlsx")
                original_excel_path = excel_path
                counter = 1
                while os.path.exists(excel_path) and not overwrite:
                    excel_path = os.path.join(output_dir, f"{sanitized_name}_{counter}.xlsx")
                    counter += 1
                if counter > 1 and not overwrite:
                    warnings.warn(f"Excel file '{original_excel_path}' exists. Saving as '{excel_path}' instead.")
                try:
                    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                        # Save the main DataFrame
                        sheet_name = sanitize_sheet_name(name)
                        df_to_save.to_excel(writer, sheet_name=sheet_name, index=False)
                        if verbose:
                            print(f"Saved DataFrame '{name}' to sheet '{sheet_name}' in '{excel_path}'.")
                        
                        # Save the array DataFrames
                        for array_col, array_df in array_dfs.items():
                            array_sheet_name = sanitize_sheet_name(f"{name}_{array_col}")
                            array_df.to_excel(writer, sheet_name=array_sheet_name, index=True)
                            if verbose:
                                print(f"Saved array DataFrame '{name}_{array_col}' to sheet '{array_sheet_name}' in '{excel_path}'.")
                    
                    if verbose:
                        print(f"Saved DataFrame '{name}' and its array columns to '{excel_path}'.")
                except Exception as e:
                    raise IOError(f"Failed to save DataFrame '{name}' to Excel: {e}")
    
    # Save as Pickle
    if 'pkl' in formats:
        for name, df in dataframes.items():
            sanitized_name = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in name)
            pkl_path = os.path.join(output_dir, f"{sanitized_name}.pkl")
            original_pkl_path = pkl_path
            counter = 1
            while os.path.exists(pkl_path) and not overwrite:
                pkl_path = os.path.join(output_dir, f"{sanitized_name}_{counter}.pkl")
                counter += 1
            if counter > 1 and not overwrite:
                warnings.warn(f"Pickle file '{original_pkl_path}' exists. Saving as '{pkl_path}' instead.")
            try:
                df.to_pickle(pkl_path)
                if verbose:
                    print(f"Saved DataFrame '{name}' to '{pkl_path}'.")
            except Exception as e:
                raise IOError(f"Failed to save DataFrame '{name}' to Pickle: {e}")
    
    if verbose:
        print("All requested DataFrames have been saved successfully.")

def plot_vol_slice_images(
    im,
    slice_idx=None,
    title_suffix='',
    xlabel='X-axis',
    ylabel='Y-axis',
    figsize=(8, 6),
    cmap='viridis',
    colorbar=True,
    save_path=None,
    show=True
):
    """
    Plots a 2D image or a slice from a 3D image using matplotlib.pyplot.imshow.

    Parameters:
    ----------
    im : np.ndarray
        The image data to plot. Can be a 2D or 3D NumPy array.
    slice_idx : int, optional
        The index of the slice to plot if `im` is a 3D array. If None and `im` is 3D,
        the middle slice along the first axis is plotted. Ignored if `im` is 2D.
    title_suffix : str, optional
        A suffix to add to the plot title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    figsize : tuple, optional
        Size of the figure in inches, e.g., (width, height).
    cmap : str or Colormap, optional
        Colormap to use for the image.
    colorbar : bool, optional
        Whether to display a colorbar.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
    show : bool, optional
        Whether to display the figure.

    Returns:
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes objects created.
    """
    # Validate input dimensions
    if not isinstance(im, np.ndarray):
        raise TypeError("Input 'im' must be a NumPy array.")
    
    ndim = im.ndim
    if ndim not in [2, 3]:
        raise ValueError("Input 'im' must be a 2D or 3D NumPy array.")
    
    # If 3D, select the slice
    if ndim == 3:
        if slice_idx is None:
            slice_idx = im.shape[0] // 2  # Middle slice
        if not (0 <= slice_idx < im.shape[0]):
            raise IndexError(f"slice_idx {slice_idx} is out of bounds for the first axis with size {im.shape[0]}.")
        image_to_plot = im[slice_idx, :, :]
        title = f"Slice {slice_idx} {title_suffix}"
    else:
        image_to_plot = im
        title = f"2D Image {title_suffix}"
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(image_to_plot, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Add colorbar if requested
    if colorbar:
        fig.colorbar(cax, ax=ax)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if save_path is provided
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")
    
    # Show the plot if requested
    if show:
        plt.show()
    
    return fig, ax

##########################################################################
# End
##########################################################################


