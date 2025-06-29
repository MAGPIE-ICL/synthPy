
import sys

sys.path.insert(1, '/rds/general/user/le322/home/synthPy/')
import field_generator.gaussian3D as g3
import field_generator.gaussian2D as g2
import utils.power_spectrum as spectrum
import matplotlib.pyplot as plt
import numpy as np
import solver.minimal_solver as s
import sys

plt.style.use("/rds/general/user/le322/home/synthPy/thesis.mplstyle")





p = int(sys.argv[1])

factors =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.4,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0]




def power_spectrum(k,a):
    return k**-a

def k41(k):
    return power_spectrum(k, p)


for factor in factors:

    print(f'running {factor}')
    l_max = 1
    l_min = 0.01
    extent = 5
    res = 312

    k_min = 2 * np.pi / l_max

    k_max = 2 * np.pi / l_min


    field_3d = g3.gaussian3D(k41)
    ne = field_3d.domain_fft(l_max, l_min, extent, res, factor)


    ne = 1e25 + 9e24* ne

    print(np.mean(ne), np.max(ne), np.min(ne))



#     _, wn, ps_raw = spectrum.radial_3Dspectrum(ne, 2*extent, 2*extent, int(2*extent*factor))
#     ps = ps_raw[~np.isnan(ps_raw)]/np.nanmax(ps_raw)
#     wn = wn[~np.isnan(ps_raw)]


#     ps = ps/np.max(ps)
#     c1 = '#1C6758'   # dark green
#     c2 = '#006e23'   # ciano
#     c3 = '#6e0052'   # violet
#     c4 = '#db4900'   # purple
#     c5 = '#37e67d'   #light green


#     plt.plot(wn, ps, 'o', color = c4, zorder = 4, alpha = 0.5)
#     plt.xscale('log')
#     plt.yscale('log')

#     k_min, k_max = 2*np.pi/l_max, 2*np.pi/l_min

#     min_i = np.where(wn >= k_min)[0][0]
#     max_i = np.where(wn <= k_max)[0][-1]





#     from lmfit.models import ExpressionModel
#     mod = ExpressionModel('c*x + a')
#     mod.make_params()
#     mod.set_param_hint('a', value = 1)
#     mod.set_param_hint('c', value = -5/3)
#     res0     =   mod.fit(np.array(np.log(ps[min_i : max_i])), x = np.log(wn[min_i : max_i]), nan_policy='omit')
#     print(res0.fit_report())

#     plt.plot(wn[min_i: max_i], np.exp(res0.best_fit), label = f"Linear Fit, gradient = {np.round(-res0.params['c'].value, 2)}, ideal={np.round(p, 2)}", linewidth = 3, color = 'blue')





#     plt.xlabel(r'k [mm$^{-1}$]')
#     plt.ylabel('Spectral Power Density')
#     plt.vlines([2*np.pi/l_max, 2*np.pi/l_min], 1e-31, 1, linestyle = 'dashed', color = c4)
#     # plt.text(10**0.5,1e-14, (r'$\delta_{{min}}$', + '= {l_min}', color = 'orange')
#     # plt.text(10**0.5,1e-16, r'$\delta_{{max}}$', + '= {l_max}', color = 'orange')
#     # plt.text(10**0.5,1e-12, f'power law = {np.round(p, 3)}', color = 'orange')
#     # plt.text(10**0.5,1e-10, f'extent = {2*extent*factor}mm', color = 'orange')
# #     plt.annotate(r'$k_{min}$', (2.6, 10**-18), fontsize = 15, color = c4, weight = 'bold')
# #     plt.annotate(r'$k_{max}$', (67, 10**-18), fontsize = 15, color = c4, weight = 'bold')
#     # # plt.ylim(1e9,1e45)
#     # # plt.xlim(1, 1e2)

#     plt.legend(loc = 'lower left')
#     plt.savefig(f'/rds/general/user/le322/home/synthPy/fields/length_scale_full/{str(p)}/turb2_{factor}_spec.png', dpi = 600)
#     plt.show()

#     plt.imshow(ne[:,:,10], cmap = 'plasma', extent = (-5, 5, -5, 5))
#     cbar = plt.colorbar()
#     cbar.set_label(r'$n$ [m$^{-3}$]', fontsize=20, rotation = 270, labelpad = 35)
#     plt.xlabel('x / mm')
#     plt.ylabel('y / mm')
#     plt.title(r'$\delta_{{min}}$' + f'= {l_min}, ' + r'$\delta_{{max}}$' + f'= {l_max} \n' + r'$P_{imposed}$ = ' + f'{np.round(p,2)}', fontsize = 15)
#     plt.grid('off')
#     plt.savefig(f'/rds/general/user/le322/home/synthPy/fields/length_scale_full/{str(p)}/turb2_{factor}.png', dpi = 600)
#     plt.show()

    x, y, z=  np.linspace(-extent, extent, 2*res), np.linspace(-extent, extent, 2*res), np.linspace(-extent*factor, extent*factor, int(2*res*factor))
    domain = s.ScalarDomain(x,y,z)
    domain.external_ne(ne)
    domain.export_scalar_field(fname = f'/rds/general/user/le322/home/synthPy/fields/length_scale_full/{str(p)}/{factor}')
    del x
    del y
    del z
    del ne
