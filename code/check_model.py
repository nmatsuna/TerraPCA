#!/use/bin/env python3
# check_model.py
# Fit PCA-based telluric model to the dataset used for
# constructing the basis spectra. 

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import utils

# ---------------------------------------------
# Configurationi and Argument Parsing
# ---------------------------------------------
TXT_DIR='eigen_txt'
PLOT_DIR='check_model'

if len(sys.argv) != 2:
    print('Usage: python3 {} [order (e.g., m44)]'.format(os.path.basename(__file__)), file=sys.stderr)
    sys.exit(1)

order=sys.argv[1]
if order not in utils.orders:
    print(f'Error: order={order} is not in valid order list.', file=sys.stderr)
    sys.exit(1)

# Ensure output directory exists
os.makedirs(f'{PLOT_DIR}/{order}', exist_ok=True)

# ---------------------------------------------
# Load Spectral Data and PCA Components
# ---------------------------------------------
objs = utils.load_tel_list(list_run=None)

wrange = utils.order_wranges_with_buffer[order]
wmin_str, wmax_str = wrange.split(':')
wmin, wmax = float(wmin_str), float(wmax_str)
waves = np.loadtxt(f'{TXT_DIR}/waves_{order}.txt')
sp_ave = np.loadtxt(f'{TXT_DIR}/ave_{order}.txt')
wmin2, wmax2 = min(sp_ave[:,0]), max(sp_ave[:,0])
print(f'# {order} {wmin_str}:{wmax_str} (with buffer: {wmin2:.0f}:{wmax2:.0f})', file=sys.stderr)

func_ave = CubicSpline(sp_ave[:,0], sp_ave[:,1])
array_base = utils.load_eigen_base(order, add_ones_column=True)

# ---------------------------------------------
# Fit Model to Each Object in Dataset
# ---------------------------------------------
for label, obj in objs.items():
    spfile = utils.get_sp_txt(obj, order)
    if spfile is None:
        print(f'    Skipping {label}: spectrum file not found for order={order}', file=sys.stderr)
        continue

    sp_obs, *_ = utils.load_sp(spfile, frange=[0,1.5])
    wmin_obs, wmax_obs = min(sp_obs[:,0]), max(sp_obs[:,0])

    # Adjust wavelength and flux to match average model
    xadjust, yadjust = utils.calc_adjust(sp_obs, max([wmin,wmin_obs]), min([wmax,wmax_obs]), func_ave)
    sp_obs[:,0] += xadjust
    sp_obs[:,0] += yadjust
    func_obs = CubicSpline(sp_obs[:,0], sp_obs[:,1])

    # Find pixels within overlap
    pixels_use = utils.check_pixels_use(waves, sp_obs[:,0])
    if not np.any(pixels_use):
        print(f' Skipping {label}: no overlapping wavelength range', file=sys.stderr)
        continue

    waves_part = waves[pixels_use]
    array_base_part = array_base[pixels_use]

    # Interpolate and fit
    sp_interp_all = func_obs(waves)
    sp_interp_part = sp_interp_all[pixels_use]
    mean_level = np.mean(sp_interp_part)
    y = sp_interp_part - mean_level

    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(array_base_part, y, rcond=None)
    except np.linalg.LinAlgError as e:
        print(f'   Error fitting {label}: {e}', file=sys.stderr)

    sp_model_part = array_base_part @ coeffs + mean_level
    sigma = np.std(sp_interp_part - sp_model_part)
    print('{} sigma={:.5f} ({:d} pixels, xadjust={:.2f})'.format(label,sigma,np.count_nonzero(pixels_use),xadjust), file=sys.stderr)

    # ---------------------------------------------
    # Plot Observed vs Model Spectrum
    # ---------------------------------------------
    xmin, xmax = wmin-15., wmax+15.
    low, up = min(sp_interp_part), max(sp_interp_part)
    ymin = min([(up+low)/2.-0.6*(up-low),0.95])
    ymax = max([(up+low)/2.+0.6*(up-low),1.03])
    fig = plt.figure(figsize=(15,8))
    axs = {}
    axs['sp'] = fig.add_axes([0.10,0.46,0.80,0.46])
    axs['dev'] = fig.add_axes([0.10,0.10,0.80,0.30])
    axs['sp'].plot(waves_part,sp_interp_part,color='red',alpha=0.7,label='Observed')
    axs['sp'].plot(waves_part,sp_model_part,color='blue',alpha=0.7,label='Model')
    axs['sp'].axhline(1,ls='dotted',lw=0.7,color='k')
    axs['sp'].axvline(wmin,ls='dotted',lw=0.7,color='k')
    axs['sp'].axvline(wmax,ls='dotted',lw=0.7,color='k')
    axs['sp'].legend(bbox_to_anchor=(0.99,0.03),loc='lower right')
    axs['sp'].set_xlim([xmin,xmax])
    axs['sp'].set_ylim([ymin,ymax])
    axs['sp'].set_ylabel('Normalized Flux')
    axs['sp'].set_title(f'{label} ({order}, {wrange})')
    axs['dev'].plot(waves_part,sp_interp_part-sp_model_part,zorder=1,color='gray')
    axs['dev'].axhline(0.,ls='dotted',color='k',zorder=2)
    axs['dev'].axvline(wmin,ls='dotted',lw=0.7,color='k')
    axs['dev'].axvline(wmax,ls='dotted',lw=0.7,color='k')
    axs['dev'].text(0.01*xmin+0.99*xmax,0.11,r'$\sigma ={:.4f}$'.format(sigma),horizontalalignment='right',verticalalignment='top')
    axs['dev'].set_xlim([xmin,xmax])
    axs['dev'].set_ylim([-0.12,0.12])
    axs['dev'].set_xlabel('Wavelength')
    fig.savefig(f'{PLOT_DIR}/{order}/{label}_{order}.png',dpi=200)
    plt.close(fig)
