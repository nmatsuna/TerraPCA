#!/usr/bin/env python3
#
# Build PCA basis vectors for telluric absorption modelling.

import sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import CubicSpline

from calc_sp_offset import calc_sp_offset
import utils
num_standard_min = utils.num_standard_min
c_kms = utils.c_kms
xoff_alert = 10

# ------------------------------------------
# Parse arguments
# ------------------------------------------
parser = argparse.ArgumentParser(description="Build PCA basis spectra from telluric standards.")
parser.add_argument("--vac_air", type=str, required=True, choices=["vac","air"],
                    help="Wavelength scale of the telluric standard spectra")
parser.add_argument("-s","--setting", type=str, default=utils.default_setting,
                    help=f"Instrumental setting label (default: {utils.default_setting})")
parser.add_argument("-o", "--orders", type=str, default="",
                    help="Comma-separated list of orders to process (default: all orders in the setting)")
parser.add_argument("-v","--verbose", action="store_true", default=False, help="Verbose mode")
parser.add_argument("--no_align", action="store_true",
                    help="Avoid aligning each spectrum to running mean before stacking")
parser.add_argument("--no_plot", action="store_true", help="Avoid plotting outputs")
args = parser.parse_args()
vac_air = args.vac_air
if vac_air == 'vac':
    vac_air2 = 'air'
elif vac_air == 'air':
    vac_air2 = 'vac'
else:
    print(f'Unexpected vac_air = {vac_air} (should be vac or air)')
    exit()

# Decide which orders to analyze
setting_orders = utils.load_setting(args.setting)
if args.orders.strip() == "":
    target_orders = list(setting_orders.keys())
else:
    target_orders = [o.strip() for o in args.orders.split(",")]
    for o in target_orders:
        if o not in setting_orders:
            print(f"[ERROR] order={o} is not defined in setting_{args.setting}.txt", file=sys.stderr)
            sys.exit(1)

# ------------------------------------------
# Load setting
# ------------------------------------------
setting = args.setting
setting_orders = utils.load_setting(setting)
TXT_DIR = os.path.join(utils.HOME_DIR, "models_txt", setting)
os.makedirs(TXT_DIR, exist_ok=True)
print(f"### build_models for {setting} ###")
print(f"# Models to be stored in {TXT_DIR}")
if not args.no_plot:
    PLOT_DIR = os.path.join(utils.HOME_DIR, "models_plot", setting)
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"# Plots to be stored in {PLOT_DIR}")
else:
    print(f"# No plots will be created.")

cmap = cm.get_cmap('Reds').reversed()

def normalize_abs(f, check_continuum_up=True, unity_level='upper', upper_level_frac=97):
    """Normalize spectrum so continuum ~1 and absorption is downward."""
    f = np.copy(f)
    if check_continuum_up and np.mean(f) > np.median(f):
        f *= -1
    if unity_level == 'mean':
        unity = np.mean(f)
    elif unity_level == 'median':
        unity = np.median(f)
    elif unity_level == 'upper':
        unity = np.percentile(f, upper_level_frac)
    else:
        unity = np.max(f)
    f2 = f - unity + 1
    if np.min(f2) <= 0:
        f2 = (f2-1)/(1-np.min(f2))+1.01
    return f2

# ------------------------------------------
# Main Loop
# ------------------------------------------
for order, oconf in setting_orders.items():
    if order not in target_orders:
        continue
    wmin, wmax = oconf['wmin'], oconf['wmax']
    n_pix, n_base = oconf['n_pix'], oconf['n_base']
    print(f"[{order}] range={wmin:.1f}:{wmax:.1f}, n_pix={n_pix}, n_base={n_base}", file=sys.stderr)

    files = utils.list_telluric_files(setting, order)
    if len(files) == 0:
        print(f"[{order}] No files found, skipping", file=sys.stderr)
        continue

    waves = utils.make_order_waves(setting_orders, order)
    np.savetxt(f"{TXT_DIR}/waves_{order}_{vac_air}.txt", waves, fmt='%9.3f')
    waves2 = utils.vac_air_conversion(waves,conversion=f'{vac_air}_to_{vac_air2}')
    np.savetxt(f"{TXT_DIR}/waves_{order}_{vac_air2}.txt", waves2, fmt='%9.3f')

    sp_sum = np.zeros_like(waves)
    sp_mean = None
    n_sp = 0
    list_sp_std = []
    # ------------------------------------------
    # Load and interpolate spectra
    # ------------------------------------------
    for f in files:
        sp, *_ = utils.load_sp(f)
        if sp is None or len(sp) == 0:
            continue

        if (not args.no_align) and (sp_mean is not None):
            xoff, yoff, *_ = calc_sp_offset(sp, np.column_stack((waves,sp_mean)), list_v=np.arange(-15,15,0.2))
            if abs(xoff) > xoff_alert:
                print(f"  {os.path.basename(f)}: xoff={xoff:.1f}km/s - too large!", file=sys.stderr)
                continue
            sp[:,0] *= (1.-xoff/c_kms)
            sp[:,1] += yoff
            if args.verbose:
                print(f"  {os.path.basename(f)}: xoff={xoff:.1f}km/s yoff={yoff:.3f}", file=sys.stderr)

        interp_cur = CubicSpline(sp[:,0], sp[:,1])
        sp_new = interp_cur(waves)
        # Fill outside observed range with continuum=1
        sp_new[(waves < np.min(sp[:,0])) | (waves > np.max(sp[:,0]))] = 1.0
        stdev = sp_new.std()
        if stdev <= 0:
            continue
        n_sp += 1
        list_sp_std.append((sp_new - sp_new.mean()) / stdev)
        sp_sum += sp_new
        sp_mean = sp_sum / n_sp

    if n_sp == 0:
        print(f"[{order}] No usable spectra after loading", file=sys.stderr)
        continue
    elif n_sp < num_standard_min:
        print(f"[{order}] Too few spectra after loading (should be more than {num_standard_min:d})", file=sys.stderr)
        continue

    # ------------------------------------------
    # PCA decomposition
    # ------------------------------------------
    # PCA: build X (n_sp x n_pix), covariance (n_sp x n_sp), eigen-decomp
    # eigh is for symmetric matrices, eigenvalues in ascending order here
    X = np.vstack(list_sp_std).astype(np.float32) # shape (n_sp, n_pix)
    cov = np.cov(X, rowvar=True, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort in descending order
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Construct basis A = X^T @ Ep
    Lp = eigvals[:n_base] # (n_base)
    norm_factor = 1/np.sum(Lp)
    Ep = eigvecs[:, :n_base] # (n_sp, n_base)
    A = X.T @ Ep # (n_pix, n_base) = basis_vectors
    # Align signs of basis vectors so major absorption components are downward
    for i in range(n_base):
        if np.mean(A[:,i]) > np.median(A[:,i]):
            A[:,i] *= -1
    np.savetxt(f"{TXT_DIR}/base_{order}.txt", A, fmt='%10.6f')

    print(f"  -> model created using {n_sp:d} spectra", file=sys.stderr)

    # ------------------------------------------
    # Build average spectrum
    # ------------------------------------------
    try:
        # Fit sp_mean as linear combination of basis spectra A
        coeffs, *_ = np.linalg.lstsq(A, sp_mean - np.mean(sp_mean), rcond=None)
        sp_ave = A @ coeffs + np.mean(sp_mean)
    except np.linalg.LinAlgError as e:
        print(f' Error fitting {label}: {e}', file=sys.stderr)
        #sp_ave = normalize_abs(A @ np.ones(n_base), unity_level='upper')
        sp_ave = normalize_abs(A @ Lp, unity_level='upper')
    np.savetxt(f"{TXT_DIR}/ave_{order}_{vac_air}.txt", np.column_stack([waves, sp_ave]), fmt='%9.3f %10.6f')
    np.savetxt(f"{TXT_DIR}/ave_{order}_{vac_air2}.txt", np.column_stack([waves2, sp_ave]), fmt='%9.3f %10.6f')

    # ------------------------------------------
    # Plots
    # ------------------------------------------
    if not args.no_plot:
        # Eigenvalues
        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(111)
        ax.plot(np.arange(len(eigvals)), eigvals, marker='o')
        ax.axvline(n_base, ls='dotted', color='k')
        ax.set_yscale('log')
        ax.set_title(f"{setting} - eigenvalues for {order}")
        ax.set_xlabel('Component index')
        ax.set_ylabel('Eigenvalue')
        fig.tight_layout()
        fig.savefig(f"{PLOT_DIR}/eigenvalues_{order}.png", dpi=150)
        plt.close(fig)

        # Basis
        fig = plt.figure(figsize=(14,6))
        ax = fig.add_subplot(111)
        for i in range(n_base):
            if i == 0:
                sp_abs = normalize_abs(norm_factor*Lp[i]*A[:,i],unity_level='upper')
            else:
                sp_abs = normalize_abs(norm_factor*Lp[i]*A[:,i],unity_level='mean')
            ax.plot(waves,sp_abs+0.05*(i+1),color=cmap(Normalize(vmin=0, vmax=n_base)(i)),zorder=i)
            #ax.plot(waves,(2**i*(sp_abs-1))+1+0.05*i,color=cmap(Normalize(vmin=0, vmax=n_base)(i)),zorder=i)
        ax.plot(waves, sp_mean, color='deepskyblue', lw=1, label='Average', zorder=n_base+1)
        #ax.plot(waves, sp_ave, color='deepskyblue', lw=0.7, label='PCA-fit', zorder=n_base+2)
        ax.axvline(wmin, ls='dotted', lw=0.5, color='k')
        ax.axvline(wmax, ls='dotted', lw=0.5, color='k')
        ax.set_xlabel(f'{vac_air} wavelength')
        ax.set_ylabel('flux')
        ax.legend(loc='lower right')
        ax.set_title(f"{setting} - basis spectra for order {order}")
        fig.tight_layout()
        fig.savefig(f"{PLOT_DIR}/eigenvectors_{order}.png", dpi=200)
        plt.close(fig)
