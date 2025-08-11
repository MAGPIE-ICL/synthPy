import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import processing.diagnostics as diag

def graph_domain(domain, *, save = False):
    fig, ax = plt.subplots(figsize = (9.5, 9.5))
    fig.subplots_adjust(.15, .15, .95, .95, hspace = 0.5)

    im = ax.imshow(domain.ne[:,:,0].T/1e24, cmap = 'jet', origin = 'lower', extent = [-5, 5, -5, 5], clim = [0, 8])
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xticks([-5, -2.5, 0, 2.5, 5])
    ax.set_yticks([-5, -2.5, 0, 2.5, 5])

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
    cbar = plt.colorbar(im, cax=axins1, ticks=[0, 2, 4, 6, 8], shrink = 0.4, orientation="horizontal", extend='both')
    cbar.set_label(r'$ n_e(x, y, z_0)$ ($\times 10^{18}$ $cm^{-3}$)', fontsize = 24)
    cbar.ax.tick_params(labelsize = 24)

    ax.set_xlabel('x (mm)', fontsize = 24)
    ax.set_ylabel('y (mm)', fontsize = 24)

    divider = make_axes_locatable(ax)
    axvert  = divider.append_axes('right', size = '30%', pad = 0.15)
    axhoriz = divider.append_axes('top',   size = '30%', pad = 0.15)

    profile_vert    =   domain.ne[:, :, 0].T.sum(axis = 0)
    profile_vert    =   (profile_vert - profile_vert.min() ) / (profile_vert.max() - profile_vert.min())
    axhoriz.plot(domain.x, profile_vert, lw = 3, c = 'k', alpha = 1)
    axhoriz.set_xlim(-5e-3, 5e-3)
    axhoriz.set_ylim(0, 1)
    axhoriz.set_ylabel(r'$n_e$(x)', fontsize = 23)

    profile_hor     =   domain.ne[:, :, 0].T.sum(axis = 1)
    profile_hor     =   (profile_hor - profile_hor.min() ) / (profile_hor.max() - profile_hor.min())
    profile_hor_theory  =  profile_hor
    axvert.plot(profile_hor, domain.y, lw = 3, c = 'k', alpha = 1)
    axvert.set_ylim(-5e-3, 5e-3)
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

    # save Figure
    if save:
        from datetime import datetime
        fig.savefig('./analytical 2D electron density distribution - ' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.png', bbox_inches = 'tight', dpi = 600)

from processing.diagnostics import lens_cutoff

def general_ray_plots(rf, nbins, lwl = 1032e-9, *, l_x = 0, u_x = 0.3, l_y = -5, u_y = 5, extra_info = True):
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # lens_cutoff may not make a difference to the angle plot (after already masked seperately) - but it does to this
    # lens_cutoff(...) passes a tuple of rf and Jf (= None), not just rf
    rf, _ = lens_cutoff(rf)

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

def stepped_ray_plot(rf, domain, sample_size = 32, *, indexing = "synthPy"):
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

        sample_indices = np.random.randint(low = rf[0].ys.shape[0], size = sample_size)

        x = []
        y = []
        z = []

        for i in sample_indices:
            for j in range(sol_count):
                # save_points_per_region SHOULD be constant between regions
                for k in range(save_points_per_region):
                    values = rf[j].ys[:, k, :].T    # is this correct when generalised??

                    x.append(values[0, i])
                    y.append(values[1, i])
                    z.append(values[2, i])

            plt.plot(z, x, y, label = 'save_point' + str(k))

            x = []
            y = []
            z = []

        xx, yy = np.meshgrid(domain.x, domain.y)
        z_plane = np.full(len(domain.z), -(domain.z_length / 2))

        ax.plot_wireframe(z_plane, xx, yy, rcount = 5, ccount = 5, color="k")

        margin = 0.0005
        ax.set_xlim(-(domain.z_length / 2) - margin, (domain.z_length / 2) + margin)
        ax.set_ylim(-(domain.x_length / 2) - margin, (domain.x_length / 2) + margin)
        ax.set_zlim(-(domain.y_length / 2) - margin, (domain.y_length / 2) + margin)

        ax.set_xlabel('z (m)')
        ax.set_ylabel('x (m)')
        ax.set_zlabel('y (m)')

        plt.show()