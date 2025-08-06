import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import processing.diagnostics as diag

import jax.numpy as jnp
    
def lens_cutoff(rf, *, L = 400, R = 25):
    """
    Masks the Jonesvector resulting array to avoid plotting any values outside of some set limit
        - important as even if you set limits for the histogram to "zoom in", binning is based on raw data
        --> leading to low resolutions if this is not used!

    Args:
        rf (jax.Array): Jonesvector output from solver
        L (int): Length till next lens
        R (int): Radius of lens

    Return:
        rf (jax.Array): Masked Jonesvector
    """

    return jnp.asarray(rf)[:, jnp.pow(jnp.pow(L * jnp.tan(rf[1]) + rf[0], 2) + jnp.pow(L * jnp.tan(rf[3]) + rf[2], 2), 0.5) <= R]

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

def general_ray_plots(rf, lwl, *, l_x, u_x, l_y, u_y):
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    nbins = 201

    _, _, _, im1 = ax1.hist2d(rf[0] * 1e3, rf[2] * 1e3, bins=(nbins, nbins), cmap=plt.cm.jet);
    plt.colorbar(im1, ax = ax1)
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("y (mm)")

    rf = lens_cutoff(rf)

    x_theta = rf[1] * 1e3
    y_theta = rf[3] * 1e3

    #mask = (x_theta >= l_x) & (x_theta <= u_x) & (y_theta >= l_y) & (x_theta <= u_y)

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
    shadowgrapher.histogram(bin_scale = 1, clear_mem = False)

    axis[0].imshow(shadowgrapher.H, cmap = 'hot', interpolation = 'nearest', clim = (0.5, 1))

    refractometer = diag.Refractometry(lwl, rf)
    refractometer.incoherent_solve()
    refractometer.histogram(bin_scale = 1, clear_mem = False)

    axis[1].imshow(refractometer.H, cmap = 'hot', interpolation = 'nearest', clim = (0.5, 1))