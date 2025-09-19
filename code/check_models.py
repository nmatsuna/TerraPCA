#!/usr/bin/env python3
# check_models.py
# Fit PCA-based telluric model to the same dataset used for building the PCA basis

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from calc_sp_offset import calc_sp_offset
import utils

# ---------------------------------------------
# Argument Parsing
# ---------------------------------------------
parser = argparse.ArgumentParser(description="Check PCA models on original telluric dataset.")
parser.add_argument("order", type=str, help="Spectral order to test (e.g. m44)")
parser.add_argument("--vac_air", type=str, required=True, choices=["vac","air"],
                    help="Wavelength scale of the telluric standard spectra")
parser.add_argument("-s","--setting", type=str, default=utils.default_setting,
                    help=f"Instrumental setting label (default: {utils.default_setting})")
parser.add_argument("--no_align", action="store_true",
                    help="Avoid aligning each spectrum to running mean before stacking")
parser.add_argument("--no_plot", action="store_true", help="Avoid plotting outputs")
args = parser.parse_args()

setting = args.setting
order = args.order
vac_air = args.vac_air

# ---------------------------------------------
# Load Setting and PCA model files
# ---------------------------------------------
setting_orders = utils.load_setting(setting)
if order not in setting_orders:
    print(f"Error: order={order} not defined in setting_{setting}.txt", file=sys.stderr)
    sys.exit(1)

oconf = setting_orders[order]
wmin, wmax = oconf['wmin'], oconf['wmax']

if not args.no_plot:
    PLOT_DIR = os.path.join(utils.HOME_DIR, "check_models", setting, order)
    os.makedirs(PLOT_DIR, exist_ok=True)

array_base = utils.load_model_base(setting, order, add_ones_column=True)
waves = utils.load_model_waves(setting, order, vac_air=vac_air)
sp_ave = utils.load_model_ave(setting, order, vac_air=vac_air)
func_ave = CubicSpline(sp_ave[:,0], sp_ave[:,1])

print(f"# Checking PCA model for {order} ({wmin:.1f}–{wmax:.1f})", file=sys.stderr)

# ---------------------------------------------
# Loop over telluric standard spectra
# ---------------------------------------------
files = utils.list_telluric_files(setting, order)
for f in files:
    fname = os.path.basename(f)
    sp_obs, *_ = utils.load_sp(f, frange=[0,1.5])
    if sp_obs is None or len(sp_obs)==0:
        print(f"  Skipping {fname}: no data", file=sys.stderr)
        continue

    # Adjust wavelength and flux to match average model
    if not args.no_align:
        xoff, yoff, *_ = calc_sp_offset(sp_obs, sp_ave, list_v=np.arange(-15,15,0.2))
        sp_obs[:,0] *= (1.0 - xoff/utils.c_kms)
        sp_obs[:,1] -= yoff
    func_obs = CubicSpline(sp_obs[:,0], sp_obs[:,1])

    # Mask to overlap pixels
    pixels_use = utils.check_pixels_use(waves, sp_obs[:,0])
    if not np.any(pixels_use):
        print(f"  Skipping {fname}: no overlapping pixels", file=sys.stderr)
        continue

    waves_part = waves[pixels_use]
    array_base_part = array_base[pixels_use]

    # Interpolate onto PCA grid and fit
    sp_interp_all = func_obs(waves)
    sp_interp_part = sp_interp_all[pixels_use]
    mean_level = np.mean(sp_interp_part)
    y = sp_interp_part - mean_level

    try:
        coeffs, *_ = np.linalg.lstsq(array_base_part, y, rcond=None)
    except np.linalg.LinAlgError as e:
        print(f"  Error fitting {fname}: {e}", file=sys.stderr)
        continue

    sp_model_part = array_base_part @ coeffs + mean_level
    sigma = np.std(sp_interp_part - sp_model_part)
    if args.no_align:
        print(f"{fname:30s}  sigma={sigma:.4f} ({pixels_use.sum()} px)")
    else:
        print(f"{fname:30s}  sigma={sigma:.4f} ({pixels_use.sum()} px, xoff={xoff:.1f} km/s)")

    # ---------------------------------------------
    # Plot Observed vs Model Spectrum
    # ---------------------------------------------
    if not args.no_plot:
        xmin, xmax = wmin-15., wmax+15.
        low, up = np.min(sp_interp_part), np.max(sp_interp_part)
        ymin = min([(up+low)/2.-0.6*(up-low),0.95])
        ymax = max([(up+low)/2.+0.6*(up-low),1.03])
    
        fig = plt.figure(figsize=(15,8))
        axs = {}
        axs['sp'] = fig.add_axes([0.10,0.46,0.80,0.46])
        axs['dev'] = fig.add_axes([0.10,0.10,0.80,0.30])
    
        axs['sp'].plot(waves_part, sp_interp_part, color='red', alpha=0.7, label='Observed')
        axs['sp'].plot(waves_part, sp_model_part, color='blue', alpha=0.7, label='Model')
        axs['sp'].axhline(1, ls='dotted', lw=0.7, color='k')
        axs['sp'].axvline(wmin, ls='dotted', lw=0.7, color='k')
        axs['sp'].axvline(wmax, ls='dotted', lw=0.7, color='k')
        axs['sp'].legend(bbox_to_anchor=(0.99,0.03), loc='lower right')
        axs['sp'].set_xlim([xmin, xmax])
        axs['sp'].set_ylim([ymin, ymax])
        axs['sp'].set_ylabel('Normalized Flux')
        axs['sp'].set_title(f'{fname} ({order})')
    
        axs['dev'].plot(waves_part, sp_interp_part - sp_model_part, zorder=1, color='gray')
        axs['dev'].axhline(0., ls='dotted', color='k', zorder=2)
        axs['dev'].axvline(wmin, ls='dotted', lw=0.7, color='k')
        axs['dev'].axvline(wmax, ls='dotted', lw=0.7, color='k')
        axs['dev'].text(0.99*xmax, 0.11, f'σ={sigma:.4f}', ha='right', va='top')
        axs['dev'].set_xlim([xmin, xmax])
        axs['dev'].set_ylim([-0.12, 0.12])
        axs['dev'].set_xlabel(f'{vac_air} wavelength')

        plot_out = os.path.join(PLOT_DIR, f"{os.path.splitext(fname)[0]}_{order}.png")
        fig.savefig(plot_out, dpi=200)
        plt.close(fig)
