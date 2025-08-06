import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def graph_domain(domain, *, save = False):
    fig, ax = plt.subplots(figsize = (9.5, 9.5))
    fig.subplots_adjust(.15,.15,.95,.95, hspace= 0.5)

    im = ax.imshow(domain.ne[:,:,0].T/1e24, cmap = 'jet', origin = 'lower', extent = [-5,5,-5,5], clim = [0, 8])
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_xticks([-5,-2.5, 0, 2.5, 5])
    ax.set_yticks([-5,-2.5, 0, 2.5, 5])

    axins1 = inset_axes(
        ax,
        width="50%",  # width: 50% of parent_bbox width
        height="5%",  # height: 5%
        loc="lower right",
        bbox_to_anchor=(-0.5, -0.2, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    axins1.xaxis.set_ticks_position("bottom")
    cbar = plt.colorbar(im, cax=axins1, ticks=[0, 2, 4, 6, 8], shrink=0.4, orientation="horizontal", extend='both')
    cbar.set_label(r'$ n_e(x, y, z_0)$ ($\times 10^{18}$ $cm^{-3}$)', fontsize = 24)
    cbar.ax.tick_params(labelsize = 24)

    ax.set_xlabel('x (mm)', fontsize = 24)
    ax.set_ylabel('y (mm)', fontsize = 24)

    divider = make_axes_locatable(ax)
    axvert  = divider.append_axes('right', size='30%', pad=0.15)
    axhoriz = divider.append_axes('top',   size='30%', pad=0.15)

    profile_vert    =   domain.ne[:,:,0].T.sum(axis = 0)
    profile_vert    =   (profile_vert - profile_vert.min() ) / (profile_vert.max() - profile_vert.min())
    axhoriz.plot(ne_x, profile_vert, lw = 3, c = 'k', alpha = 1)
    axhoriz.set_xlim(-5e-3,5e-3)
    axhoriz.set_ylim(0,1)
    axhoriz.set_ylabel(r'$n_e$(x)', fontsize = 23)

    profile_hor     =   domain.ne[:,:,0].T.sum(axis = 1)
    profile_hor     =   (profile_hor - profile_hor.min() ) / (profile_hor.max() - profile_hor.min())
    profile_hor_theory  =  profile_hor
    axvert.plot(profile_hor, ne_y, lw = 3, c = 'k', alpha = 1)
    axvert.set_ylim(-5e-3,5e-3)
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
        fig.savefig('./analytical 2D electron density distribution.png', bbox_inches='tight', dpi=600)