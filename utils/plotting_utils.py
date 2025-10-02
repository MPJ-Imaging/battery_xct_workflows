import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ipywidgets import IntSlider, FloatRangeSlider, FloatSlider, Dropdown, Checkbox, Button, VBox, HBox

def show_overlay(img, msk, ax, title=""):
    # basic checks
    if img.shape != msk.shape:
        raise ValueError(f"Image/mask shapes differ: {img.shape} vs {msk.shape}")

    ax.imshow(img, cmap="gray", interpolation="nearest")
    alpha = (msk > 0).astype(float) * 0.5   # 0 where mask==0, 0.5 where mask>0
    ax.imshow(msk, cmap="Reds", alpha=alpha, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

def view_axis0(volume, *, vmin=None, vmax=None, cmap='gray',
               downsample=1, continuous_update=False):
    """
    Axis-0 viewer for 3D arrays (Z, Y, X) with absolute contrast range.

    Parameters
    ----------
    volume : 3D ndarray
        Base intensity volume shaped (Z, Y, X).
    vmin, vmax : float or None
        Intensity range for display. Defaults to (min, max) of the data.
    cmap : str
        Matplotlib colormap for the base image.
    downsample : int
        Factor for Y/X downsampling (Z unchanged).
    continuous_update : bool
        Whether z-slider updates continuously.
    """
    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError("volume must be a 3D array shaped (Z, Y, X)")

    v = vol[:, ::downsample, ::downsample] if downsample > 1 else vol

    if vmin is None:
        vmin = float(v.min())
    if vmax is None:
        vmax = float(v.max())

    with plt.ioff():
        fig, ax = plt.subplots()
        im = ax.imshow(v[v.shape[0] // 2], cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    z = IntSlider(min=0, max=v.shape[0]-1, value=v.shape[0]//2,
                  description='z', continuous_update=continuous_update)
    cm = Dropdown(options=sorted(plt.colormaps()), value=cmap,
                  description='cmap')
    clim_slider = FloatRangeSlider(
        min=float(v.min()), max=float(v.max()), step=(v.max()-v.min())/500,
        value=[vmin, vmax], description='range', continuous_update=False)

    def update_slice(*_):
        im.set_data(v[z.value]); fig.canvas.draw_idle()

    def update_cmap(*_):
        im.set_cmap(cm.value); fig.canvas.draw_idle()

    def update_clim(*_):
        lo, hi = clim_slider.value
        if lo >= hi:
            lo, hi = float(v.min()), float(v.max())
        im.set_clim(lo, hi); fig.canvas.draw_idle()

    z.observe(update_slice, 'value')
    cm.observe(update_cmap, 'value')
    clim_slider.observe(update_clim, 'value')

    return VBox([HBox([z, cm]), clim_slider, fig.canvas])

def view_axis0_with_labels(volume, labels, *,
                           vmin=None, vmax=None,
                           img_cmap='gray', label_alpha=0.5,
                           downsample=1, continuous_update=False, seed=None):
    """
    Axis-0 viewer for 3D arrays (Z, Y, X) with label overlay
    and absolute intensity range sliders.
    """
    vol = np.asarray(volume)
    lab = np.asarray(labels)

    if vol.ndim != 3 or lab.ndim != 3:
        raise ValueError("volume and labels must be 3D arrays shaped (Z, Y, X)")
    if vol.shape != lab.shape:
        raise ValueError(f"volume and labels must have the same shape. Got {vol.shape} vs {lab.shape}")

    if downsample > 1:
        v = vol[:, ::downsample, ::downsample]
        l = lab[:, ::downsample, ::downsample]
    else:
        v, l = vol, lab

    if vmin is None:
        vmin = float(v.min())
    if vmax is None:
        vmax = float(v.max())

    rng = np.random.default_rng(seed)
    max_label = int(l.max()) if l.size else 0

    def make_label_cmap(alpha):
        if max_label <= 0:
            colors = np.zeros((2, 4))
            colors[0, 3] = 0.0
            colors[1, :3] = 1.0
            colors[1, 3] = alpha
            return ListedColormap(colors)
        colors = np.zeros((max_label + 1, 4))
        colors[0, 3] = 0.0
        colors[1:, :3] = rng.random((max_label, 3))
        colors[1:, 3] = alpha
        return ListedColormap(colors)

    with plt.ioff():
        fig, ax = plt.subplots()
        im = ax.imshow(v[v.shape[0] // 2], cmap=img_cmap,
                       vmin=vmin, vmax=vmax, interpolation='nearest')
        label_cmap = make_label_cmap(label_alpha)
        im_lab = ax.imshow(l[l.shape[0] // 2], cmap=label_cmap,
                           vmin=0, vmax=max(1, max_label), interpolation='nearest')
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    z = IntSlider(min=0, max=v.shape[0]-1, value=v.shape[0]//2,
                  description='z', continuous_update=continuous_update)
    cm = Dropdown(options=sorted(plt.colormaps()), value=img_cmap,
                  description='cmap')
    clim_slider = FloatRangeSlider(
        min=float(v.min()), max=float(v.max()), step=(v.max()-v.min())/500,
        value=[vmin, vmax], description='range', continuous_update=False)
    show_lab = Checkbox(value=True, description='show labels')
    alpha_slider = FloatSlider(min=0.0, max=1.0, step=0.05, value=float(label_alpha),
                               description='label α', continuous_update=continuous_update)
    shuffle_btn = Button(description='reshuffle colors')

    def update_slice(*_):
        im.set_data(v[z.value]); im_lab.set_data(l[z.value]); fig.canvas.draw_idle()

    def update_cmap(*_):
        im.set_cmap(cm.value); fig.canvas.draw_idle()

    def update_clim(*_):
        lo, hi = clim_slider.value
        if lo >= hi:
            lo, hi = float(v.min()), float(v.max())
        im.set_clim(lo, hi); fig.canvas.draw_idle()

    def update_label_alpha(*_):
        im_lab.set_cmap(make_label_cmap(alpha_slider.value)); fig.canvas.draw_idle()

    def toggle_labels(*_):
        im_lab.set_visible(show_lab.value); fig.canvas.draw_idle()

    def reshuffle_colors(_):
        nonlocal rng
        rng = np.random.default_rng()
        im_lab.set_cmap(make_label_cmap(alpha_slider.value)); fig.canvas.draw_idle()

    z.observe(update_slice, 'value')
    cm.observe(update_cmap, 'value')
    clim_slider.observe(update_clim, 'value')
    alpha_slider.observe(update_label_alpha, 'value')
    show_lab.observe(toggle_labels, 'value')
    shuffle_btn.on_click(reshuffle_colors)

    im_lab.set_visible(show_lab.value)

    controls_row1 = HBox([z, cm, show_lab])
    controls_row2 = HBox([clim_slider])
    controls_row3 = HBox([alpha_slider, shuffle_btn])
    return VBox([controls_row1, controls_row2, controls_row3, fig.canvas])

def ellipse_points(xc, yc, a, b, phi, n=400):
    """
    Parametric ellipse sampled at n points.
    (xc, yc): center
    a, b    : semi-axes (a = major, b = minor)
    phi     : rotation (radians), from +x toward +y
    """
    t = np.linspace(0, 2*np.pi, n)
    ct, st = np.cos(t), np.sin(t)
    c, s = np.cos(phi), np.sin(phi)
    x = xc + a*ct*c - b*st*s
    y = yc + a*ct*s + b*st*c
    return x, y

def plot_ellipse_on_mask(mask, model, pts_xy=None, inliers=None, show_axes=True, title="Ellipse fit"):
    """
    mask   : 2D image (shown as background)
    model  : skimage EllipseModel (after .estimate or RANSAC)
    pts_xy : optional Nx2 array of (x,y) points used for fitting
    inliers: optional boolean mask (same length as pts_xy) from RANSAC
    """
    xc, yc, a, b, phi = model.params
    ex, ey = ellipse_points(xc, yc, a, b, phi)

    plt.figure(figsize=(6,6))
    plt.imshow(mask, cmap="gray", interpolation="nearest")  # origin='upper' (image-like)
    
    # plot points if provided
    if pts_xy is not None:
        if inliers is None:
            plt.plot(pts_xy[:,0], pts_xy[:,1], '.', ms=2, alpha=0.4, label="points")
        else:
            plt.plot(pts_xy[~inliers,0], pts_xy[~inliers,1], '.', ms=2, alpha=0.2, label="outliers")
            plt.plot(pts_xy[inliers,0],  pts_xy[inliers,1],  '.', ms=2, alpha=0.6, label="inliers")

    # ellipse + center
    plt.plot(ex, ey, '-', lw=2, label="fitted ellipse")
    plt.scatter([xc], [yc], s=50, marker='+', label="center")

    # draw major/minor axes
    if show_axes:
        major = np.array([a*np.cos(phi), a*np.sin(phi)])
        minor = np.array([-b*np.sin(phi), b*np.cos(phi)])
        c = np.array([xc, yc])
        p1, p2 = c - major, c + major
        p3, p4 = c - minor, c + minor
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', lw=1, label="major axis")
        plt.plot([p3[0], p4[0]], [p3[1], p4[1]], '--', lw=1, label="minor axis")

    plt.axis('equal')
    plt.legend(loc='lower right', fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.show()



