import tifffile as tiff
import numpy as np
import skimage
from skimage.measure import find_contours, EllipseModel, ransac
from skimage.draw import polygon2mask
from scipy.ndimage import center_of_mass as c_of_m
from scipy.ndimage import distance_transform_edt as dist_trans
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import math
import utils.plotting_utils as plot

def extract_inner_outer_contours(mask, as_mask = False):
    contours = find_contours((mask).astype(float), 0.5)
    outer = max(contours, key=lambda c: 0.5*np.abs(np.sum(c[:,1]*np.roll(c[:,0],-1) - c[:,0]*np.roll(c[:,1],-1))))
    inner = min(contours, key=lambda c: 0.5*np.abs(np.sum(c[:,1]*np.roll(c[:,0],-1) - c[:,0]*np.roll(c[:,1],-1))))
    if as_mask == True:
        outer = polygon2mask(mask.shape, outer).astype(np.uint8)
        inner = polygon2mask(mask.shape, inner).astype(np.uint8)
    return outer, inner

def equivalent_diameter_area(mask):
    area = np.sum(mask)
    return math.sqrt(area/math.pi)*2

def surface_pixels_convolve(img):
    img8 = img.astype(np.uint8)
    s = convolve(img8, np.ones((3,3), np.uint8), mode='constant', cval=0)
    # interior: all 9 in the 3x3 (8-connected)
    interior = (img & (s == 9))
    return img & ~interior

def calculate_can_metrics(
    mask,
    *,
    # RANSAC / ellipse fit controls
    use_ransac: bool = True,
    ransac_min_samples: int = 5,
    ransac_residual_threshold: float = 1.0,
    ransac_max_trials: int = 2000,
    ransac_stop_probability: float | None = None,
    random_state: int | None = None,

    # Dent profile controls
    n_bins: int = 180,
    profile_agg: str = "mean",      # "mean" or "median"
    smooth_window: int | None = None,  # optional circular smoothing (bins)

    # Output controls
    pixel_size: float | None = None,   # e.g. µm/px. If None → keep pixel units
    return_model: bool = False,        # also return the fitted EllipseModel
):
    """
    Compute CAN (cylinder/annulus) metrics from a binary ring mask.

    Parameters
    ----------
    mask : 2D ndarray of bool/int
        Binary image where True/1 are wall pixels of the ring.
    use_ransac : bool
        Fit the ellipse with RANSAC (robust to dents/outliers). If False, plain LS fit.
    ransac_* : tuning parameters for skimage.measure.ransac (ellipse fit).
    n_bins : int
        Angular bins for dent profile.
    profile_agg : {"mean","median"}
        Aggregation within each angular bin.
    smooth_window : int or None
        Optional circular moving average over profile bins.
    pixel_size : float or None
        Physical units per pixel. If given, physical-unit fields are added (and
        max dent/thickness/diameters are also reported in physical units).
    return_model : bool
        If True, return (metrics_dict, ellipse_model). Otherwise, just metrics_dict.

    Returns
    -------
    metrics : dict
        {
          "diameter": {...},
          "thickness": {...},
          "ellipse": {...},
          "dent": {...}
        }
    (optional) model : skimage.measure.EllipseModel
    """
    # -- Ensure boolean mask
    mask = mask.astype(bool)

    # -- (1) Inner/outer filled regions from contours (same as your logic)
    outer_fill, inner_fill = extract_inner_outer_contours(mask, as_mask=True)
    outer_fill = outer_fill.astype(bool)
    # Remove any wall pixels from the inner mask, leaving just the void (hole)
    inner_hole = inner_fill.astype(bool) & (~mask)

    # -- (2) Area-equivalent diameters (pixels)
    OD_px = equivalent_diameter_area(outer_fill)
    ID_px = equivalent_diameter_area(inner_hole)

    # -- (3) Thickness via EDT from inner wall to outer border (pixels)
    outer_border = surface_pixels_convolve(outer_fill).astype(bool)  # one-pixel wide outer boundary
    inner_wall   = surface_pixels_convolve(inner_hole).astype(bool)  # one-pixel wide inner boundary

    # Distance to outer border for all non-border pixels:
    # EDT returns distance-to-zero for non-zero pixels; so set border to 0, everything else to 1.
    edt_to_outer = dist_trans(~outer_border)
    thickness_vals_px = edt_to_outer[inner_wall]
    thick_mean_px = float(thickness_vals_px.mean()) if thickness_vals_px.size else float("nan")
    thick_min_px  = float(thickness_vals_px.min())  if thickness_vals_px.size else float("nan")
    thick_max_px  = float(thickness_vals_px.max())  if thickness_vals_px.size else float("nan")
    thick_std_px  = float(thickness_vals_px.std())  if thickness_vals_px.size else float("nan")

    # -- (4) Fit ellipse to the OUTER contour (as in your code)
    contours = find_contours(mask.astype(float), 0.5)
    if not contours:
        raise ValueError("No contours found for ellipse fitting.")

    # largest-area contour is the outer boundary
    def _poly_area_abs(c):
        x, y = c[:, 1], c[:, 0]
        return 0.5 * abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

    outer_contour_rc = max(contours, key=_poly_area_abs)
    pts_xy = np.column_stack([outer_contour_rc[:, 1], outer_contour_rc[:, 0]]).astype(float)

    model = EllipseModel()
    if use_ransac:
        ransac_kwargs = dict(
            min_samples=ransac_min_samples,
            residual_threshold=ransac_residual_threshold,
            max_trials=ransac_max_trials,
        )
        if ransac_stop_probability is not None:
            ransac_kwargs["stop_probability"] = ransac_stop_probability
        if random_state is not None:
            ransac_kwargs["random_state"] = random_state

        model, inliers = ransac(
            pts_xy, EllipseModel, **ransac_kwargs
        )
    else:
        ok = model.estimate(pts_xy)
        if not ok:
            raise RuntimeError("Ellipse fit failed without RANSAC.")

    # params: xc, yc, a, b, phi  (semi-axes)
    xc, yc, a, b, phi = model.params
    # ensure A >= B for consistency
    A, B = (a, b) if a >= b else (b, a)
    if B == 0:
        eccentricity = float("nan")
    else:
        eccentricity = float(np.sqrt(max(0.0, 1.0 - (B / A) ** 2)))

    # -- (5) Dent profile vs. this ellipse (same flow as your code; mean per bin)
    vx = pts_xy[:, 0] - xc
    vy = pts_xy[:, 1] - yc
    theta = np.arctan2(vy, vx)              # [-pi, pi)
    r_obs = np.hypot(vx, vy)

    # ellipse radius at each theta
    ct = np.cos(theta - phi)
    st = np.sin(theta - phi)
    r_ell = (a * b) / np.sqrt((b * ct) ** 2 + (a * st) ** 2)

    delta = r_ell - r_obs                   # + = inward dent

    # binning
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    idx = np.digitize(theta, bins) - 1
    idx[idx == n_bins] = 0

    prof = np.full(n_bins, np.nan)
    for k in range(n_bins):
        m = idx == k
        if np.any(m):
            if profile_agg.lower() == "median":
                prof[k] = float(np.nanmedian(delta[m]))
            else:
                prof[k] = float(np.nanmean(delta[m]))

    if smooth_window and smooth_window > 1:
        from scipy.ndimage import uniform_filter1d
        prof = uniform_filter1d(prof, size=int(smooth_window), mode="wrap")

    dmax_px = float(np.nanmax(prof)) if np.any(np.isfinite(prof)) else float("nan")

    # -- (6) Package metrics; defer scaling until here
    def _scale(v):
        return None if pixel_size is None or v is None or not np.isfinite(v) else float(v * pixel_size)

    metrics = {
        "diameter": {
            "outer_equiv_px": float(OD_px),
            "inner_equiv_px": float(ID_px),
            **(
                {} if pixel_size is None else
                {"outer_equiv_units": _scale(OD_px), "inner_equiv_units": _scale(ID_px)}
            ),
        },
        "thickness": {
            "mean_px": thick_mean_px,
            "min_px": thick_min_px,
            "max_px": thick_max_px,
            "std_px": thick_std_px,
            **(
                {} if pixel_size is None else
                {
                    "mean_units": _scale(thick_mean_px),
                    "min_units": _scale(thick_min_px),
                    "max_units": _scale(thick_max_px),
                    "std_units": _scale(thick_std_px),
                }
            ),
        },
        "ellipse": {
            "center_xy_px": (float(xc), float(yc)),
            "semi_axes_px": (float(A), float(B)),  # reported sorted (A>=B)
            "angle_rad": float(phi),
            "eccentricity": float(eccentricity),
            **(
                {} if pixel_size is None else
                {
                    "center_xy_units": ( _scale(xc), _scale(yc) ),
                    "semi_axes_units": ( _scale(A), _scale(B) ),
                }
            ),
        },
        "dent": {
            "max_px": dmax_px,
            **({} if pixel_size is None else {"max_units": _scale(dmax_px)}),
            "n_bins": int(n_bins),
            "agg": profile_agg,
            # optional: expose full profile if you want downstream plotting
            "profile": {
                "theta_rad": centers.tolist(),
                "delta_px": [None if not np.isfinite(v) else float(v) for v in prof],
                **(
                    {} if pixel_size is None else
                    {"delta_units": [None if not np.isfinite(v) else _scale(v) for v in prof]}
                ),
            },
        },
    }

    return (metrics, model) if return_model else metrics
