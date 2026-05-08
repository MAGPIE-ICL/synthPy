import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import processing.diagnostics as diag
from simulator.fresnel_integral import *

def graph_domain(domain, *, save = False, slice = "z", mark = None):
    """
    Graphs a 2D slice of the electron density domain.
    
    :param domain: Domain object containing electron density data
    :type domain: processing.domain.ScalarDomain
    
    :param save: Whether to save the figure to disk
    :type save: bool, default: False
    
    :param slice: Which slice of the domain to graph ('x', 'y', or 'z')
    :type slice: str, default: "z"
    
    :param mark: Optional coordinates (x, y, z) to mark on the plot
    :type mark: tuple, default: None
    
    :return: No return, generates plot
    :rtype: None
    """
    fig, ax = plt.subplots(figsize = (9.5, 9.5))
    fig.subplots_adjust(.15, .15, .95, .95, hspace = 0.5)

    if slice == "x":
        x = domain.y
        y = domain.z

        ne_T = domain.ne[0, :, :].T
    elif slice == "y":
        x = domain.z
        y = domain.x

        ne_T = domain.ne[:, 0, :].T
    elif slice == "z":
        x = domain.x
        y = domain.y

        ne_T = domain.ne[:, :, 0].T

    x *= 1000
    y *= 1000

    norm = domain.ne.max()
    im = ax.imshow(ne_T / norm, cmap = 'jet', origin = 'lower', extent = [x[0], x[-1], y[0], y[-1]], clim = [domain.ne.min() / norm, 1])

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    ax.set_xticks(np.linspace(x[0], x[-1], 5))
    ax.set_yticks(np.linspace(y[0], y[-1], 5))

    axins1 = inset_axes(
        ax,
        width="50%",  # width: 50% of parent_bbox width
        height="5%",  # height: 5%
        loc="lower right",
        bbox_to_anchor = (-0.5, -0.2, 1, 1),
        bbox_transform = ax.transAxes,
        borderpad = 0,
    )

    axins1.xaxis.set_ticks_position("bottom")
    cbar = plt.colorbar(im, cax=axins1, ticks=np.linspace(domain.ne.min(), 1, 5), shrink = 0.4, orientation="horizontal", extend='both')
    mantissa, exponent = f"{norm:e}".split("e+")
    cbar.set_label(r'$ n_e(x, y, z_0)$ ($\times $' + str(mantissa) + "^" + str(exponent) + r'$\ m^{-3}$)', fontsize = 24)
    cbar.ax.tick_params(labelsize = 24)

    if slice == "x":
        ax.set_xlabel('y (mm)', fontsize = 24)
        ax.set_ylabel('z (mm)', fontsize = 24)
    elif slice == "y":
        ax.set_xlabel('z (mm)', fontsize = 24)
        ax.set_ylabel('x (mm)', fontsize = 24)
    elif slice == "z":
        ax.set_xlabel('x (mm)', fontsize = 24)
        ax.set_ylabel('y (mm)', fontsize = 24)

    divider = make_axes_locatable(ax)
    axvert  = divider.append_axes('right', size = '30%', pad = 0.15)
    axhoriz = divider.append_axes('top',   size = '30%', pad = 0.15)

    profile_vert    =   ne_T.sum(axis = 0)
    profile_vert    =   (profile_vert - profile_vert.min() ) / (profile_vert.max() - profile_vert.min())
    #profile_vert /= profile_vert.max()
    axhoriz.plot(x/1000, profile_vert, lw = 3, c = 'k', alpha = 1)
    axhoriz.set_xlim(x[0], x[-1])
    axhoriz.set_ylim(0, 1)
    axhoriz.set_ylabel(r'$n_e$(x)', fontsize = 23)

    profile_hor     =   ne_T.sum(axis = 1)
    profile_hor     =   (profile_hor - profile_hor.min() ) / (profile_hor.max() - profile_hor.min())
    #profile_hor /= profile_hor.max()
    axvert.plot(profile_hor, y/1000, lw = 3, c = 'k', alpha = 1)
    axvert.set_ylim(y[0], y[-1])
    axvert.set_xlabel(r'$n_e$(z)', fontsize = 23)

    ax.tick_params(axis = 'both', labelsize = 24)

    axvert.set_xticks([])
    axhoriz.set_yticks([])
    axvert.set_yticks([])
    axhoriz.set_xticks([])

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(1.5)
        axvert.spines[axis].set_linewidth(1.5)
        axhoriz.spines[axis].set_linewidth(1.5)

    # ax.text(4, 4.2, s = 'a)', c = 'w', fontsize = 30)

    if mark is not None:
        mx, my, mz = mark

        if slice == "x":
            px = my * 1000
            py = mz * 1000
        elif slice == "y":
            px = mz * 1000
            py = mx * 1000
        elif slice == "z":
            px = mx * 1000
            py = my * 1000

        ax.plot(
            px, py,
            marker='+',
            markersize=20,
            markeredgewidth=3,
            color='k'
        )

    # save Figure
    if save:
        from datetime import datetime
        fig.savefig('./analytical 2D electron density distribution - ' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.png', bbox_inches = 'tight', dpi = 600)

from processing.diagnostics import lens_cutoff

def inital_ray_plot(rf, nbins, *, slice = "z"):
    """
    Plots the initial ray positions as a 2D histogram.
    
    :param rf: Array of ray data
    :type rf: jax.Array
    
    :param nbins: Number of bins for the histogram
    :type nbins: int
    
    :param slice: Plane to plot ('x', 'y', or 'z')
    :type slice: str, default: "z"
    
    :return: No return, displays the plot
    :rtype: None
    """
    fig1, ax1 = plt.subplots(1, figsize=(10, 4))

    if rf.shape[0] > 3:
        rf = rf[:3, :]

    if rf.shape[0] < 3:
        print("Not enough information to plot!")

    if slice == "x":
        _, _, _, im1 = ax1.hist2d(rf[1] * 1e3, rf[2] * 1e3, bins=(nbins, nbins), cmap=plt.cm.jet)
        plt.colorbar(im1, ax = ax1)
        ax1.set_xlabel("y (mm)")
        ax1.set_ylabel("z (mm)")

    elif slice == "y":
        _, _, _, im1 = ax1.hist2d(rf[2] * 1e3, rf[0] * 1e3, bins=(nbins, nbins), cmap=plt.cm.jet)
        plt.colorbar(im1, ax = ax1)
        ax1.set_xlabel("z (mm)")
        ax1.set_ylabel("x (mm)")

    elif slice == "z":
        _, _, _, im1 = ax1.hist2d(rf[0] * 1e3, rf[1] * 1e3, bins=(nbins, nbins), cmap=plt.cm.jet)
        plt.colorbar(im1, ax = ax1)
        ax1.set_xlabel("x (mm)")
        ax1.set_ylabel("y (mm)")

def general_ray_plots(rf, nbins, lwl = 1032e-9, *, l_x = 0, u_x = 0.3, l_y = -5, u_y = 5, extra_info = True, ignore_lens = False, initial = False, limit = None):
    """
    Generates a set of general ray plots including spatial and angular histograms, as well as shadowgraphy and refractometry.
    
    :param rf: Array of ray data
    :type rf: jax.Array
    
    :param nbins: Number of bins for the histograms
    :type nbins: int
    
    :param lwl: Wavelength
    :type lwl: float, default: 1032e-9
    
    :param l_x: Lower limit for theta
    :type l_x: float, default: 0
    
    :param u_x: Upper limit for theta
    :type u_x: float, default: 0.3
    
    :param l_y: Lower limit for phi
    :type l_y: float, default: -5
    
    :param u_y: Upper limit for phi
    :type u_y: float, default: 5
    
    :param extra_info: Whether to print extra information
    :type extra_info: bool, default: True
    
    :param ignore_lens: Whether to ignore the lens cutoff
    :type ignore_lens: bool, default: False
    
    :param initial: Indicates if these are initial rays
    :type initial: bool, default: False
    
    :param limit: Spatial range limit for the plot
    :type limit: float, default: None
    
    :return: No return, displays the plots
    :rtype: None
    """
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    if rf.shape[0] > 4 or initial == True:
        rf = rf[:3, :]

    if rf.shape[0] < 4 or initial == True:
        rf = np.vstack((rf, np.full((4 - rf.shape[0], rf.shape[1]), np.nan)))

    # lens_cutoff may not make a difference to the angle plot (after already masked seperately) - but it does to this
    # lens_cutoff(...) passes a tuple of rf and Jf (= None), not just rf
    if ignore_lens == False or initial == False:
        rf, _ = lens_cutoff(rf)

    if limit is None:
        _, _, _, im1 = ax1.hist2d(rf[0] * 1e3, rf[2] * 1e3, bins=(nbins, nbins), cmap=plt.cm.jet)
    else:
        _, _, _, im1 = ax1.hist2d(rf[0] * 1e3, rf[2] * 1e3, bins=(nbins, nbins), cmap=plt.cm.jet, range=[[-limit, limit], [-limit, limit]])
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



    fig2, axis = plt.subplots(1, 2, figsize = (20, 5))

    axis[0].set_xlabel("x (mm)")
    axis[0].set_ylabel("y (mm)")

    for i in range(len(axis)):
        axis[i].grid(False)

    shadowgrapher = diag.Shadowgraphy(lwl, rf, ignore_lens = ignore_lens)
    shadowgrapher.single_lens_solve()
    shadowgrapher.histogram(bin_scale = 1, clear_mem = False, extra_info = extra_info, auto_range = True)

    axis[0].imshow(shadowgrapher.H, cmap = 'hot', interpolation = 'nearest', clim = (0.5, 1))

    refractometer = diag.Refractometry(lwl, rf, ignore_lens = ignore_lens)
    refractometer.incoherent_solve()
    refractometer.histogram(bin_scale = 1, clear_mem = False, extra_info = extra_info, auto_range = True)

    axis[1].imshow(refractometer.H, cmap = 'hot', interpolation = 'nearest', clim = (0.5, 1))

    plt.show()

def stepped_ray_plot(rf, domain, sample_size = 32, *, indexing = "synthPy"):
    """
    Plots the trajectories of a sampled subset of rays through the domain.
    
    :param rf: Array of ray path data
    :type rf: jax.Array or list
    
    :param domain: Domain object containing dimensions
    :type domain: processing.domain.ScalarDomain
    
    :param sample_size: Number of rays to plot
    :type sample_size: int, default: 32
    
    :param indexing: Indexing convention
    :type indexing: str, default: "synthPy"
    
    :return: No return, displays the 3D plot
    :rtype: None
    """
    ##
    ## Matplotlib's plotting means that the axis that we would like to be used as z for display purposes is actually the x
    ## Hence why we assign x, y, z --> z, x, y here when using matplotlib
    ##

    # plotting defaults to assume 'z' probing_direction, some logic has been added to generalise such that it will run
    # however, it has not been checked personally, could look odd potentially and may need customisation for your needs

    probing_index = ['x', 'y', 'z'].index(domain.probing_direction)

    sol_count = len(rf)
    save_points_per_region = rf[0].ys.shape[1]

    if sol_count == 1 and save_points_per_region == 1:
        print("\nNot enough points (1) per ray for plot.")
    else:
        print("\nThere are", save_points_per_region + (sol_count - 1) * (save_points_per_region - 1), "data points available to plot per ray.")

        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')

        for i in np.random.randint(low = rf[0].ys.shape[0], size = sample_size):

            x, y, z = [], [], []

            for j in range(sol_count):
                # save_points_per_region SHOULD be constant between regions
                for k in range(save_points_per_region):
                    x.append(rf[j].ys[i, k, 0])    # is this correct when generalised??
                    y.append(rf[j].ys[i, k, 1])
                    z.append(rf[j].ys[i, k, 2])

            plt.plot(z, x, y, label = 'save_point' + str(k))

        xx, yy = np.meshgrid(domain.x, domain.y)
        z_plane = np.full(len(domain.z), -(domain.z_length / 2))

        ax.plot_wireframe(z_plane, xx, yy, rcount = 5, ccount = 5, color="k")

        margin = 0.0005
        ax.set_xlim(domain.z[0] - margin, domain.z[-1] + margin)
        ax.set_ylim(domain.x[0] - margin, domain.x[-1] + margin)
        ax.set_zlim(domain.y[0] - margin, domain.y[-1] + margin)

        ax.set_xlabel('z (m)')
        ax.set_ylabel('x (m)')
        ax.set_zlabel('y (m)')

        plt.show()

def initial_field(domain, x, y, phases, amplitudes, pix_x, pix_y, title, savefig = False, fname = "hi"):
    """
    Plots the intensity of the initial optical field.
    
    :param domain: Domain object
    :type domain: processing.domain.ScalarDomain
    
    :param x: Ray x positions
    :type x: np.array
    
    :param y: Ray y positions
    :type y: np.array
    
    :param phases: Ray phases
    :type phases: np.array
    
    :param amplitudes: Ray amplitudes
    :type amplitudes: np.array
    
    :param pix_x: Pixels in x direction
    :type pix_x: int
    
    :param pix_y: Pixels in y direction
    :type pix_y: int
    
    :param title: Plot title
    :type title: str
    
    :param savefig: Whether to save the figure
    :type savefig: bool, default: False
    
    :param fname: Filename for the saved plot
    :type fname: str, default: "hi"
    
    :return: No return, shows plot
    :rtype: None
    """
    from scipy.interpolate import LinearNDInterpolator as LND
    phases_interp = LND((x, y), phases, fill_value = 0.0)
    amplitudes_interp = LND((x, y), amplitudes, fill_value = 0.0)

    x = np.linspace(-domain.x_length/2, domain.x_length/2, pix_x)
    y = np.linspace(-domain.y_length/2, domain.y_length/2, pix_y)
    XX, YY = np.meshgrid(x, y)
    phase_grid = phases_interp((XX, YY))
    amplitude_grid = amplitudes_interp((XX, YY))

    initial_field = amplitude_grid * np.exp(-1j * phase_grid)
    fig0, axs0 = plt.subplots()
    im = axs0.imshow(np.absolute(initial_field)**2, extent = (-domain.x_length/2, domain.x_length/2, -domain.y_length/2, domain.y_length/2), origin = "lower")
    axs0.set_xlabel("x position (m)")
    axs0.set_ylabel("y position (m)")
    axs0.set_title(title)
    fig0.colorbar(im, ax = axs0, orientation='vertical', fraction = .1)
    plt.show()
    if savefig is True:
        plt.savefig(f"../../../{fname}.png",dpi=800, bbox_inches='tight', pad_inches=0.1)

def propagated_field(domain, x_pos, y_pos, z, phases, amplitudes, pix_x, pix_y, title, lwl, pad_factor = 2, vmin = None, vmax = None, savefig = False, fname = "hi"):
    """
    Propagates a field and plots its intensity.
    
    :param domain: Domain object
    :type domain: processing.domain.ScalarDomain
    
    :param x_pos: Ray x positions
    :type x_pos: np.array
    
    :param y_pos: Ray y positions
    :type y_pos: np.array
    
    :param z: Propagation distance
    :type z: float
    
    :param phases: Ray phases
    :type phases: np.array
    
    :param amplitudes: Ray amplitudes
    :type amplitudes: np.array
    
    :param pix_x: Resolution in x
    :type pix_x: int
    
    :param pix_y: Resolution in y
    :type pix_y: int
    
    :param title: Plot title
    :type title: str
    
    :param lwl: Laser wavelength
    :type lwl: float
    
    :param pad_factor: Padding factor for FFT
    :type pad_factor: int, default: 2
    
    :param vmin: Minimum intensity for color map
    :type vmin: float, default: None
    
    :param vmax: Maximum intensity for color map
    :type vmax: float, default: None
    
    :param savefig: Whether to save the figure
    :type savefig: bool, default: False
    
    :param fname: Output filename
    :type fname: str, default: "hi"
    
    :return: No return, shows plot
    :rtype: None
    """
    fig, axs = plt.subplots()
    final_field = propagate(lwl, domain, x_pos, y_pos, amplitudes, phases, z, pix_x, pix_y, pad_factor)
    im = axs.imshow(np.absolute(final_field)**2, extent = (-domain.x_length/2, domain.x_length/2, -domain.y_length/2, domain.y_length/2), origin = "lower", cmap = "viridis", vmin = vmin, vmax = vmax)
    axs.set_xlabel("x position (m)")
    axs.set_ylabel("y position (m)")
    axs.set_title(title)
    fig.colorbar(im, ax = axs, fraction = .05, pad = 0.08)
    plt.show()
    if savefig is True:
        plt.savefig(f"../../../{fname}.png",dpi=800, bbox_inches='tight', pad_inches=0.1)

def phase_on_off(params, suptitle, lwl, domain, z, pad_factor = 2, titles = ["Phase off", "Phase on"], savefig = False, fname = "hi", pix_x = 400, pix_y = 400, 
         vmin = None, vmax = None, vmin1 = None, vmax1 = None):
    """
    Plots side-by-side comparison of propagated fields with phase disabled vs phase enabled.
    
    :param params: List of two parameter sets (for phase off and phase on)
    :type params: list
    
    :param suptitle: Plot super title
    :type suptitle: str
    
    :param lwl: Laser wavelength
    :type lwl: float
    
    :param domain: Domain object
    :type domain: processing.domain.ScalarDomain
    
    :param z: Propagation distance
    :type z: float
    
    :param pad_factor: Padding factor
    :type pad_factor: int, default: 2
    
    :param titles: Subplot titles
    :type titles: list, default: ["Phase off", "Phase on"]
    
    :param savefig: Whether to save the figure
    :type savefig: bool, default: False
    
    :param fname: Output filename
    :type fname: str, default: "hi"
    
    :param pix_x: x resolution
    :type pix_x: int, default: 400
    
    :param pix_y: y resolution
    :type pix_y: int, default: 400
    
    :param vmin: min color bound (phase off)
    :type vmin: float, default: None
    
    :param vmax: max color bound (phase off)
    :type vmax: float, default: None
    
    :param vmin1: min color bound (phase on)
    :type vmin1: float, default: None
    
    :param vmax1: max color bound (phase on)
    :type vmax1: float, default: None
    
    :return: No return, shows plot
    :rtype: None
    """
    fig, axs = plt.subplots(1,2)
    fig.suptitle(suptitle, y=0.8)
    fig.tight_layout()
    fig.subplots_adjust(wspace = 0.5, top = 0.9)
    axes = axs.flatten()
    x_pos, y_pos, amplitudes, phases = params[0]
    x_pos1, y_pos1, amplitudes1, phases1 = params[1]
    final_field = propagate(lwl, domain, x_pos, y_pos, amplitudes, phases, z, pix_x, pix_y, pad_factor)
    final_field1 = propagate(lwl, domain, x_pos1, y_pos1, amplitudes1, phases1, z, pix_x, pix_y, pad_factor)
    final_field = np.absolute(final_field)
    final_field1 = np.absolute(final_field1)
    
    im = axes[0].imshow(final_field1**2, extent = (-domain.x_length/2, domain.x_length/2, -domain.y_length/2, domain.y_length/2), cmap = "viridis", vmin = vmin, vmax = vmax, origin = "lower")
    im1 = axes[1].imshow(final_field**2, extent = (-domain.x_length/2, domain.x_length/2, -domain.y_length/2, domain.y_length/2), cmap = "viridis", vmin = vmin1, vmax = vmax1, origin = "lower")
    images = [im, im1] 

    for i, ax in enumerate(axes):
        ax.set_xlabel("x position (mm)")
        ax.set_ylabel("y position (mm)")
        fig.colorbar(images[i], ax = ax, fraction = .05, pad = 0.2)
        ax.set_title(titles[i])
    if savefig is True:
        plt.savefig(f"../../../{fname}.png",dpi=800, bbox_inches='tight', pad_inches=0.1)

def var_distance(domain, x_pos, y_pos, z, phases, amplitudes, pix_x, pix_y, title, lwl, pad_factor = 2, a = 2, b = 3, savefig = False, fname = "hi"):
    """
    Plots propagated fields over multiple distances.
    
    :param domain: Domain object
    :type domain: processing.domain.ScalarDomain
    
    :param x_pos: Initial x positions
    :type x_pos: np.array
    
    :param y_pos: Initial y positions
    :type y_pos: np.array
    
    :param z: Array of distances
    :type z: np.array or list
    
    :param phases: Field phases
    :type phases: np.array
    
    :param amplitudes: Field amplitudes
    :type amplitudes: np.array
    
    :param pix_x: Number of pixels in x
    :type pix_x: int
    
    :param pix_y: Number of pixels in y
    :type pix_y: int
    
    :param title: Suptitle for figure
    :type title: str
    
    :param lwl: Laser wavelength
    :type lwl: float
    
    :param pad_factor: Padding factor
    :type pad_factor: int, default: 2
    
    :param a: Number of subplot rows
    :type a: int, default: 2
    
    :param b: Number of subplot columns
    :type b: int, default: 3
    
    :param savefig: Whether to save
    :type savefig: bool, default: False
    
    :param fname: Output filename
    :type fname: str, default: "hi"
    
    :return: No return, shows plot
    :rtype: None
    """
    if len(z) != a*b:
        raise ValueError("z must have length equal to a*b, i.e. number of plots")

    fig, axs = plt.subplots(a,b)
    fig.suptitle(title)
    fig.subplots_adjust(wspace = 0.3)
    axes = axs.flatten()
    final_fields = []

    for i in range (0,a*b):
        final_field = propagate(lwl, domain, x_pos, y_pos, amplitudes, phases, z[i], pix_x, pix_y, pad_factor)
        final_fields.append(np.absolute(final_field)**2)

    images = []
    counter = 0

    for ax, data in zip(axes, final_fields):
        im = ax.imshow(data, extent = (-domain.x_length/2, domain.x_length/2, -domain.y_length/2, domain.y_length/2), cmap = "viridis", origin = "lower")
        images.append(im)
        if counter == 0:
            ax.set_xlabel("x position (mm)")
            ax.set_ylabel("y position (mm)")
        else:
            ax.set_xticks([])  
            ax.set_yticks([])  
        ax.set_title(f"z = {(z[counter]):.2f} m")
        fig.colorbar(im, ax = ax, fraction = .05, pad = 0.2)
        counter += 1
    if savefig is True:
        plt.savefig(f"../../../{fname}",dpi=800, bbox_inches='tight', pad_inches=0.1)

def interpolated_phase(domain, x_pos, y_pos, phases, pix_x, pix_y, title = "Interpolated Phase", savefig = False, fname = "hi", wrapped = False, vmin = None, vmax = None):
    """
    Plots an interpolated 2D grid of the given phase values.
    
    :param domain: Domain object
    :type domain: processing.domain.ScalarDomain
    
    :param x_pos: Ray x positions
    :type x_pos: np.array
    
    :param y_pos: Ray y positions
    :type y_pos: np.array
    
    :param phases: Ray phases
    :type phases: np.array
    
    :param pix_x: Resolution in x
    :type pix_x: int
    
    :param pix_y: Resolution in y
    :type pix_y: int
    
    :param title: Plot title
    :type title: str, default: "Interpolated Phase"
    
    :param savefig: Whether to save to file
    :type savefig: bool, default: False
    
    :param fname: Output filename
    :type fname: str, default: "hi"
    
    :param wrapped: Whether to wrap phase values Mod 2*pi
    :type wrapped: bool, default: False
    
    :param vmin: Minimum intensity for color map
    :type vmin: float, default: None
    
    :param vmax: Maximum intensity for color map
    :type vmax: float, default: None
    
    :return: No return, shows plot
    :rtype: None
    """
    
    phases_interp = LND((x_pos, y_pos), phases, fill_value = 0.0)
    x = np.linspace(-domain.x_length/2, domain.x_length/2, pix_x)
    y = np.linspace(-domain.y_length/2, domain.y_length/2, pix_y)
    XX, YY = np.meshgrid(x, y)
    phase_grid = phases_interp((XX, YY))
    fig1, ax1 = plt.subplots()
    fig1.suptitle(title)

    if wrapped is True:
        im = ax1.imshow((-phase_grid) % (2*np.pi), extent = (-domain.x_length/2, domain.x_length/2, -domain.y_length/2, domain.y_length/2), origin = "lower", vmin = vmin, vmax = vmax)
    else:
        im = ax1.imshow(phase_grid, extent = (-domain.x_length/2, domain.x_length/2, -domain.y_length/2, domain.y_length/2), origin = "lower", vmin = vmin, vmax = vmax)
   
    ax1.set_xlabel("x position (mm)")
    ax1.set_ylabel("y position (mm)")
    fig1.colorbar(im, ax = ax1, orientation='vertical', fraction = .1)
    if savefig is True:
        plt.savefig(f"../../../{fname}",dpi=800, bbox_inches='tight', pad_inches=0.1)