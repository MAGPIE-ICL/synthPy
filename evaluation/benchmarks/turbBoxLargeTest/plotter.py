import h5py
import numpy as np
import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import export_image
import pandas as pd
import os

import sys
import os

### DO NOT REMOVE UNLESS YOU ARE VERY CERTAIN OF CORRECT PACKAGING

# os.getcwd()                                   # - don't want cwd, want the dir of this file
# os.path.dirname(os.path.realpath(__file__))   # cannot be called interactively - ? - seems's fine though
# sys.path[0]                                   # haven't tested...
# os.path.abspath(sys.argv[0])                  # haven't tested...

try:
    current_file = os.path.realpath(__file__)
except NameError:
    # __file__ not defined
    if sys.argv[0]:  # Might still work in some IDEs
        current_file = os.path.realpath(sys.argv[0])
    else:
        # Fallback to current working directory (e.g. Jupyter)
        current_file = os.getcwd()

#top_level_path = resolve_path(str(os.path.dirname(os.path.realpath(__file__))) + "/../")
top_level_path = os.path.abspath(os.path.join(os.path.dirname(current_file), '../../../src'))
print(top_level_path)
sys.path.insert(0, top_level_path)
import processing.diagnostics as diag

import matplotlib.pyplot as plt

def rays_to_dfs(rays):
    """
    Converts ray data into DataFrames for position and angle spaces.
    """

    print(rays)
    rays[0] *= 1e3
    rays[2] *= 1e3
    rays, _ = diag.lens_cutoff(rays)
    print(rays)

    mask = (rays[0] >= -10) & (rays[0] <= 10) & \
           (rays[2] >= -10) & (rays[2] <= 10)

    # Position in mm
    df_position = pd.DataFrame({
        'x': rays[0] * 1e3,
        'y': rays[2] * 1e3
    })

    mask = (rays[1] >= 0) & (rays[1] <= 0.5) & \
           (rays[3] >= -5) & (rays[3] <= 5)

    df_angles = pd.DataFrame({
        'theta': rays[1][mask],
        'phi': rays[3][mask]
    })

    return df_position, df_angles

def render_histogram(df, x_col, y_col, filename, cmap='jet'):
    """
    Render a 2D histogram using datashader with fixed axis limits and colormap.
    """

    nbins = 256  # Number of bins for histograms

    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()

    margin_x = (x_max - x_min) * 0.05
    margin_y = (y_max - y_min) * 0.05

    x_range = (x_min - margin_x, x_max + margin_x)
    y_range = (y_min - margin_y, y_max + margin_y)

    cvs = ds.Canvas(plot_width = nbins, plot_height = nbins, x_range = x_range, y_range = y_range)
    print(df[x_col])
    print(df[y_col])
    agg = cvs.points(df, x_col, y_col, agg = ds.count())

    # Convert matplotlib colormap to datashader format
    from matplotlib.pyplot import get_cmap
    colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in get_cmap(cmap, nbins)(np.linspace(0, 1, nbins))]

    img = tf.shade(agg, cmap = colors, how = 'eq_hist')

    export_image(img, filename = filename, background = "black")

def general_ray_plots(rf, nbins, lwl = 1032e-9, *, l_x = 0, u_x = 0.3, l_y = -5, u_y = 5, extra_info = True):
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # diag.lens_cutoff may not make a difference to the angle plot (after already masked seperately) - but it does to this
    # diag.lens_cutoff(...) passes a tuple of rf and Jf (= None), not just rf
    rf, _ = diag.lens_cutoff(rf)

    _, _, _, im1 = ax1.hist2d(rf[0] * 1e3, rf[2] * 1e3, bins=(nbins, nbins), cmap=plt.cm.jet)
    plt.colorbar(im1, ax = ax1)
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("y (mm)")

    #rf = rf.at[1].set(rf[1] * 1e3)
    #rf = rf.at[3].set(rf[3] * 1e3)

    x_theta = rf[1] * 1e3
    y_theta = rf[3] * 1e3

    mask = (x_theta >= l_x) & (x_theta <= u_x) & (y_theta >= l_y) & (x_theta <= u_y)

    _, _, _, im2 = ax2.hist2d(x_theta[mask], y_theta[mask], bins=(nbins, nbins), cmap=plt.cm.jet);
    plt.colorbar(im2, ax = ax2)
    ax2.set_xlabel(r"$\theta$ (mrad)")
    ax2.set_ylabel(r"$\phi$ (mrad)")

    ax2.set_xlim(l_x, u_x)
    ax2.set_ylim(l_y, u_y)

    fig1.tight_layout()
    plt.show()



    fig2, axis = plt.subplots(1, 2, figsize = (20, 5))

    axis[0].set_xlabel("x (mm)")
    axis[0].set_ylabel("y (mm)")

    for i in range(len(axis)):
        axis[i].grid(False)

    shadowgrapher = diag.Shadowgraphy(lwl, rf)
    shadowgrapher.single_lens_solve()
    shadowgrapher.histogram(bin_scale = 1, clear_mem = False, extra_info = extra_info)

    axis[0].imshow(shadowgrapher.H, cmap = 'hot', interpolation = 'nearest', clim = (0.5, 1))

    refractometer = diag.Refractometry(lwl, rf)
    refractometer.incoherent_solve()
    refractometer.histogram(bin_scale = 1, clear_mem = False, extra_info = extra_info)

    axis[1].imshow(refractometer.H, cmap = 'hot', interpolation = 'nearest', clim = (0.5, 1))
    plt.show()

def main():
    SAVE_DIR = "saves"

    file_list = sorted([
        os.path.join(SAVE_DIR, f)
        for f in os.listdir(SAVE_DIR)
        if f.endswith('.hdf5') or f.endswith('.hdf5.gzip')
    ])

    for i, file_path in enumerate(file_list):
        print(f"[{i+1}/{len(file_list)}] Processing {file_path}")

        with h5py.File(file_path, 'r') as hf:
            data = hf['data'][:]

        df_pos, df_ang = rays_to_dfs(data)

        # Position histogram with fixed axis limits (matching general_ray_plots)
        render_histogram(df_pos, 'x', 'y', filename=f"position_frame_{i:05d}")

        general_ray_plots(data, 256)

        # Angle histogram with fixed axis limits
        #render_histogram(df_ang, 'theta', 'phi', filename=f"angle_frame_{i:05d}")

        # Optional: Plot the shadowgraphy and refractometry images as in general_ray_plots
        # This part depends on your diag module and matplotlib, so keep that as is.

if __name__ == "__main__":
    main()