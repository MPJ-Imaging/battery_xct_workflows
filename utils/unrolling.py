import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence, Tuple, Union, Mapping, Optional
from scipy.ndimage import center_of_mass as c_of_m
from scipy.ndimage import distance_transform_edt as dist_trans
from skimage.measure import label, find_contours, EllipseModel, ransac
from sklearn.linear_model import LinearRegression


def virtual_unroll(
    seg: np.ndarray,
    CoM: Optional[Union[Tuple[float, float], Sequence[float], np.ndarray]] = None,
    chunk_size: int = 75,
) -> pd.DataFrame:
    """
    Unroll a binary spiral segmentation into polar coordinates, split each winding
    into a separate labeled layer, and compute the local radial thickness.

    Parameters
    ----------
    seg : numpy.ndarray
        2D binary-like array (bool or 0/1) where nonzero/True pixels belong to the spiral.
        Shape: (H, W).
    CoM : tuple of float, sequence of float, or numpy.ndarray, optional
        Center of mass (y, x) used as the spiral origin. Accepts:
        - A length-2 tuple/list/array giving (y, x).
        - A 2D array mask (same shape as `seg`, i.e. the casing mask) from which center of mass will be computed
          via fitting an ellipse to the outer pixels.
        - If None (default), the center is computed from `seg` via `c_of_m(seg)`.
        The final center is cast to integer pixel indices (subpixels are discarded).
    chunk_size : int, optional
        Target number of pixels per angular chunk when downsampling the (angle, radius)
        samples for each layer. The number of chunks is
        `max(1, len(layer_samples) // chunk_size)`. Default is 75.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame where each row corresponds to one spiral layer with columns:
        - 'layer' : int
            Layer index (1 = innermost, increasing outward).
        - 'angular_positions' : list of float
            Mean angular positions (degrees) for each chunk in the layer.
        - 'radial_positions' : list of float
            Mean radial distances (pixels) for each chunk in the layer.
        - 'chunk_thkn' : list of float
            Radial thickness per chunk (max(radius) - min(radius)).

    Notes
    -----
    - Angles are defined using `arctan2(x - x0, y - y0)` They are converted to degrees in [0, 360).
    - A seam at 0° is zeroed from the segmentation before connected-component labeling,
      so each winding is separated into its own label.
    - Layers are relabeled so the innermost winding becomes layer 1 by reversing
      the label order.
    """
    if not isinstance(seg, np.ndarray):
        raise TypeError(f"`seg` must be a numpy.ndarray, got {type(seg)}.")
    if seg.ndim != 2:
        raise ValueError(f"`seg` must be 2D, got shape {seg.shape}.")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(f"`chunk_size` must be a positive int, got {chunk_size}.")

    seg_bool = seg.astype(bool)

    # Determine center of mass
    if CoM is None:
        com = c_of_m(seg_bool)
    elif isinstance(CoM, (tuple, list, np.ndarray)):
        arr = np.asarray(CoM)
        if arr.ndim == 1 and arr.size == 2:
            com = (float(arr[0]), float(arr[1]))
        elif arr.ndim == 2 and arr.shape == seg_bool.shape:
            contours = find_contours((arr).astype(float), 0.5)
            outer = max(contours, key=lambda c: 0.5*np.abs(np.sum(c[:,1]*np.roll(c[:,0],-1) - c[:,0]*np.roll(c[:,1],-1))))
            pts_xy = np.column_stack([outer[:,1], outer[:,0]])  # (x,y)

            model, inliers = ransac(
                pts_xy,
                EllipseModel,
                min_samples=5,
                residual_threshold=1.0,   
                max_trials=2000)
            
            xc, yc, a, b, phi = model.params
            com = (yc, xc)
        else:
            raise ValueError(
                "`CoM` must be length-2 (y, x) or a 2D mask of the same shape as `seg`."
            )
    else:
        raise TypeError(
            "`CoM` must be None, a length-2 sequence, or a numpy.ndarray of shape `seg.shape`."
        )
    com = (int(com[0]), int(com[1]))

    # Distance transform
    dist_seed = np.ones_like(seg_bool, dtype=np.uint16)
    if not (0 <= com[0] < seg_bool.shape[0] and 0 <= com[1] < seg_bool.shape[1]):
        raise ValueError(f"Center {com} is outside the image of shape {seg_bool.shape}.")
    dist_seed[com[0], com[1]] = 0
    dist_transform = dist_trans(dist_seed)

    # Angular coordinates
    yy, xx = np.indices(seg_bool.shape)
    y_in = yy - com[0]
    x_in = xx - com[1]
    angular_arr = np.rad2deg(np.arctan2(x_in, y_in))
    angular_arr = (angular_arr + 360.0) % 360.0

    # Seam cut
    seam_mask = np.isclose(angular_arr, 0.0)
    seg_for_label = np.where(seam_mask, 0, seg_bool)

    separated = label(seg_for_label, connectivity=2)
    max_lab = int(separated.max())
    if max_lab == 0:
        return pd.DataFrame(columns=["layer", "angular_positions", "radial_positions", "chunk_thkn"])

    label_seg = (max_lab + 1 - separated)
    label_seg = np.where(separated == 0, 0, label_seg)

    records = []
    for layer_id in range(1, int(label_seg.max()) + 1):
        mask = (label_seg == layer_id)
        if not np.any(mask):
            continue

        layer_angles = angular_arr[mask]
        layer_radii = dist_transform[mask]

        order = np.argsort(layer_angles)
        layer_angles = layer_angles[order]
        layer_radii = layer_radii[order]

        n = len(layer_angles)
        n_chunks = max(1, n // chunk_size)
        angle_chunks = np.array_split(layer_angles, n_chunks)
        radius_chunks = np.array_split(layer_radii, n_chunks)

        angulars = [float(np.mean(chunk)) for chunk in angle_chunks]
        radials = [float(np.mean(chunk)) for chunk in radius_chunks]
        chunk_thkn = [float(np.max(chunk) - np.min(chunk)) for chunk in radius_chunks]

        records.append(
            {
                "layer": layer_id,
                "angular_positions": angulars,
                "radial_positions": radials,
                "chunk_thkn": chunk_thkn,
            }
        )

    df = pd.DataFrame.from_records(records)
    return df

def add_linear_fit_errors(
    df: pd.DataFrame,
    angle_col: str = "angular_positions",
    radius_col: str = "radial_positions",
    errs_col: str = "errors",
    rmse_col: str = "rmse_lr",
    maxae_col: str = "maxae_lr",
    slope_col: str = "lr_slope",
    intercept_col: str = "lr_intercept",
    use_radians: bool = False,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Fit a linear model radius = a * angle + b for each layer and append error metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Per-layer DataFrame (one row per layer) with list-like columns for angles and radii,
        e.g. the output of `virtual_unroll`.
    angle_col : str, optional
        Column name containing angular samples (list/array of floats, degrees by default).
    radius_col : str, optional
        Column name containing radial samples (list/array of floats).
    errs_col : str, optional
        Column name containing absolute error of each chunk vs linear fit.   
    rmse_col : str, optional
        Output column name for root-mean-square error of the linear fit.
    maxae_col : str, optional
        Output column name for maximum absolute error of the linear fit.
    slope_col : str, optional
        Output column name for fitted slope (a).
    intercept_col : str, optional
        Output column name for fitted intercept (b).
    use_radians : bool, optional
        If True, convert angles from degrees to radians before fitting.
        (Your upstream angles are in degrees; leave False to keep behavior consistent.)
    copy : bool, optional
        If True, return a modified copy of `df`. If False, modify `df` in place.

    Returns
    -------
    pandas.DataFrame
        DataFrame with added columns: rmse_col, maxae_col, slope_col, intercept_col.

    Notes
    -----
    - This uses scikit-learn's LinearRegression with no regularization.
    - Rows with fewer than 2 points (or mismatched lengths) yield NaNs for outputs.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"`df` must be a pandas.DataFrame, got {type(df)}.")
    if angle_col not in df.columns or radius_col not in df.columns:
        raise KeyError(f"`df` must contain '{angle_col}' and '{radius_col}' columns.")

    out = df.copy() if copy else df

    # Prepare output columns with NaNs (handles any invalid rows gracefully)
    out[errs_col] =  pd.Series([None] * len(out), dtype=object)
    out[rmse_col] = np.nan
    out[maxae_col] = np.nan
    out[slope_col] = np.nan
    out[intercept_col] = np.nan

    lr = LinearRegression()

    for idx, row in out.iterrows():
        angles = row[angle_col]
        radii = row[radius_col]

        # Validate row content
        if angles is None or radii is None:
            continue
        try:
            x = np.asarray(list(angles), dtype=float)
            y = np.asarray(list(radii), dtype=float)
        except Exception:
            continue

        if x.size != y.size or x.size < 2:
            continue

        if use_radians:
            x = np.deg2rad(x)

        # Fit y ~ a*x + b
        X = x.reshape(-1, 1)
        lr.fit(X, y)
        y_pred = lr.predict(X)

        # Errors
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        maxae = float(np.max(np.abs(y - y_pred)))
        errs = np.abs(y - y_pred)

        # Save
        out.at[idx, errs_col] = errs.tolist()
        out.at[idx, rmse_col] = rmse
        out.at[idx, maxae_col] = maxae
        out.at[idx, slope_col] = float(lr.coef_[0])
        out.at[idx, intercept_col] = float(lr.intercept_)

    return out

def combine_unroll_dfs(
    dfs: Union[Sequence[pd.DataFrame], Mapping[Union[str, int], pd.DataFrame]],
    ids: Optional[Sequence[Union[str, int]]] = None,
    id_col: str = "image_id",
    explode_chunks: bool = False,
    preserve_index: bool = False,
) -> pd.DataFrame:
    """
    Combine multiple per-layer DataFrames (one per image) into a single DataFrame,
    adding an image identifier column. Optionally explode chunk arrays to long format.

    Parameters
    ----------
    dfs : sequence of DataFrames or mapping of id -> DataFrame
        Each DataFrame should have one row per layer and include the columns:
        'layer', 'angular_positions', 'radial_positions', 'chunk_thkn'.
        (Extra columns like 'rmse_lr', 'maxae_lr', etc. are preserved.)
    ids : sequence of str|int, optional
        Image IDs aligned with `dfs` if `dfs` is a sequence. If `dfs` is a mapping,
        this is ignored and keys are used as IDs.
    id_col : str, optional
        Name of the column that will store the image ID. Default 'image_id'.
    explode_chunks : bool, optional
        If True, convert the per-layer list columns into per-chunk rows using
        `DataFrame.explode`. Adds a 'chunk_idx' column and explodes
        ['chunk_idx', 'angular_positions', 'radial_positions', 'chunk_thkn'] together.
        Requires that these lists are the same length per row. Default False.
    preserve_index : bool, optional
        If True, keep original row indices on concat. Default False (reset index).

    Returns
    -------
    pandas.DataFrame
        - If `explode_chunks=False`: one row per (image, layer).
        - If `explode_chunks=True`: one row per (image, layer, chunk).

    Raises
    ------
    ValueError
        If `dfs` is a sequence and `ids` length does not match.
        If `explode_chunks=True` and list columns have mismatched lengths per row.
    """
    # Normalize input to parallel lists of (id, df)
    if isinstance(dfs, Mapping):
        pairs = list(dfs.items())
    else:
        if ids is None:
            raise ValueError("When `dfs` is a sequence, you must provide `ids` (same length).")
        if len(dfs) != len(ids):
            raise ValueError(f"Length mismatch: len(dfs)={len(dfs)} vs len(ids)={len(ids)}.")
        pairs = list(zip(ids, dfs))

    tagged = []
    for image_id, df in pairs:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Each item in `dfs` must be a pandas.DataFrame, got {type(df)}.")
        # Attach ID
        tagged.append(df.assign(**{id_col: image_id}))

    combined = pd.concat(tagged, ignore_index=not preserve_index, sort=False)

    required = ["layer", "angular_positions", "radial_positions", "chunk_thkn"]
    missing = [c for c in required if c not in combined.columns]
    if missing:
        raise ValueError(f"Combined DataFrame missing required columns: {missing}")

    # Ensure 'layer' is int-like if present
    try:
        combined["layer"] = combined["layer"].astype(int)
    except Exception:
        pass  # leave as-is if not castable

    if not explode_chunks:
        return combined

    # Validate equal list lengths per row and create chunk_idx
    def _len_or_nan(x) -> Optional[int]:
        try:
            return len(x)
        except Exception:
            return None

    lengths_ok = []
    chunk_idx_col: List[Optional[List[int]]] = []
    for ang, rad, thk in zip(
        combined["angular_positions"], combined["radial_positions"], combined["chunk_thkn"]
    ):
        la, lr, lt = _len_or_nan(ang), _len_or_nan(rad), _len_or_nan(thk)
        ok = la is not None and lr is not None and lt is not None and (la == lr == lt)
        lengths_ok.append(ok)
        chunk_idx_col.append(list(range(la)) if ok else None)

    if not all(lengths_ok):
        bad = np.where(np.asarray(lengths_ok) == False)[0][:10]
        raise ValueError(
            "Cannot explode: 'angular_positions', 'radial_positions', and 'chunk_thkn' "
            "must be lists of equal length per row. Offending row indices (first 10): "
            f"{bad.tolist()}"
        )

    combined = combined.assign(chunk_idx=chunk_idx_col)
    # Explode all chunk-wise columns together to keep alignment
    combined = combined.explode(["chunk_idx", "angular_positions", "radial_positions", "chunk_thkn"],
                                ignore_index=not preserve_index)

    # Optional: enforce numeric dtypes after explode
    for c in ["chunk_idx", "angular_positions", "radial_positions", "chunk_thkn"]:
        combined[c] = pd.to_numeric(combined[c], errors="coerce")

    return combined

def plot_unrolled_layers(
    df: pd.DataFrame,
    image_id: Optional[object] = None,
    id_col: str = "image_id",
    angle_col: str = "angular_positions",
    radius_col: str = "radial_positions",
    metric: Optional[str] = "maxae",  # {"maxae", "rmse", None}
    rmse_col: str = "rmse_lr",
    maxae_col: str = "maxae_lr",
    # --- error coloring options ---
    color_by_error: bool = False,
    error_col: str = "errors",
    cmap: str = "RdYlGn_r",            # low=green, high=red
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = True,
    scatter_size: float = 18.0,
    line_alpha: float = 0.6,           # for grey context line with error coloring
    line_width: float = 1.2,
    # --- axis/figure options ---
    title: Optional[str] = None,
    legend_outside: bool = True,
    xlabel: str = "Angular position (deg)",
    ylabel: str = "Radial position (px)",
    figsize: Sequence[float] = (7.0, 5.0),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot unrolled spiral layers (radius vs. angle).

    Modes
    -----
    - color_by_error=False (default): each layer uses the Matplotlib color cycle (unique color).
    - color_by_error=True: draw a faint grey line for each layer and overlay a scatter
      colored by per-chunk absolute error from `error_col`. A horizontal colorbar is shown
      at the bottom (if `show_colorbar`).

    Notes
    -----
    - No on-the-fly metric computation. If `metric` is 'maxae' or 'rmse', it is shown
      in the legend only if the corresponding column exists and is finite.
    """
    # --- validation ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"`df` must be a pandas.DataFrame, got {type(df)}.")
    required_cols = {"layer", angle_col, radius_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    data = df
    if image_id is not None:
        if id_col not in df.columns:
            raise ValueError(f"`image_id` provided but column '{id_col}' not found.")
        data = df[df[id_col] == image_id]
        if data.empty:
            raise ValueError(f"No rows found for {id_col} == {image_id!r}.")

    created_ax = False
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        created_ax = True

    # which metric column to show (if present)
    metric = None if metric is None else str(metric).lower()
    if metric not in {None, "maxae", "rmse"}:
        raise ValueError("`metric` must be one of {None, 'maxae', 'rmse'}.")
    metric_col = {"maxae": maxae_col, "rmse": rmse_col}.get(metric, None)
    metric_label = {"maxae": "MaxAE", "rmse": "RMSE"}.get(metric, None)

    # If coloring by error, precompute vmin/vmax across selected rows (if not provided)
    use_error_colors = bool(color_by_error) and (error_col in data.columns)
    if use_error_colors and vmin is None and vmax is None:
        all_errs = []
        for _, row in data.iterrows():
            try:
                errs = np.asarray(row[error_col], dtype=float)
                angs = np.asarray(row[angle_col], dtype=float)
                rads = np.asarray(row[radius_col], dtype=float)
                if errs.shape == angs.shape == rads.shape:
                    all_errs.append(errs[np.isfinite(errs)])
            except Exception:
                continue
        if len(all_errs):
            errs_concat = np.concatenate(all_errs) if len(all_errs) > 1 else all_errs[0]
            if np.isfinite(errs_concat).any():
                vmin = float(np.nanmin(errs_concat))
                vmax = float(np.nanmax(errs_concat))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    vmin, vmax = None, None

    last_scatter = None

    # --- plot per layer ---
    for layer_id in sorted(pd.unique(data["layer"].astype(int))):
        row = data[data["layer"] == layer_id]
        if row.empty:
            continue

        try:
            angles = np.asarray(row[angle_col].values[0], dtype=float)
            radii  = np.asarray(row[radius_col].values[0], dtype=float)
        except Exception:
            continue
        if angles.size == 0 or radii.size == 0:
            continue

        # legend label
        label = f"Layer {layer_id}"
        if metric_col is not None and metric_col in row.columns:
            try:
                val = float(row[metric_col].values[0])
                if np.isfinite(val):
                    label = f"{label}: {metric_label} {val:.3f}"
            except Exception:
                pass

        if use_error_colors:
            # faint grey line for context + colored scatter by error
            ax.plot(angles, radii, color="0.7", lw=line_width, alpha=line_alpha, label=label)
            try:
                errs = np.asarray(row[error_col].values[0], dtype=float)
                if errs.shape == angles.shape == radii.shape and np.isfinite(errs).any():
                    last_scatter = ax.scatter(
                        angles, radii,
                        c=errs, cmap=cmap, vmin=vmin, vmax=vmax,
                        s=scatter_size, edgecolors="none"
                    )
            except Exception:
                pass
        else:
            # UNIQUE COLOR PER LAYER: let Matplotlib cycle colors by not specifying 'color'
            ax.plot(angles, radii, lw=line_width, label=label)

    # labels & title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is None:
        title = f"Unrolled layers (image: {image_id})" if image_id is not None else "Unrolled layers"
    ax.set_title(title)

    # legend
    if legend_outside:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    else:
        ax.legend()

    # colorbar ALWAYS at bottom (horizontal), only if using error colors
    if use_error_colors and show_colorbar and last_scatter is not None:
        cbar = plt.colorbar(
            last_scatter,
            ax=ax,
            orientation="horizontal",
            pad=0.12,
            fraction=0.08,
            aspect=40
        )
        cbar.set_label("Absolute error per chunk")

    # layout
    if created_ax:
        plt.tight_layout()

    return ax
    