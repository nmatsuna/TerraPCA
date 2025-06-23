#!/usr/bin/env python3
#
# Build PCA basis vectors for telluric absorption modelling.

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import CubicSpline

import utils

# ------------------------------------------
# Configurations
# ------------------------------------------
TXT_DIR = utils.HOME_DIR+'/eigen_txt'
PLOT_DIR = utils.HOME_DIR+'/eigen_plot'

p = utils.num_eigen
npix = utils.num_pix_order

cmap = cm.get_cmap('Reds').reversed()
norm = Normalize(vmin=0, vmax=p)

orders = utils.orders
order_wranges = utils.order_wranges_with_buffer

# ------------------------------------------
# Utility Functions
# ------------------------------------------
def data_to_reject(label, order):
    reject = {
            'm44':['HD_143912_230610_v1'],
            'm58':['4_Ari_220915_v2','17_CMa_231102_v1','HD_10538_231031_v1']
            }
    if (order not in reject) or (label not in reject[order]):
        return False
    return True

# Rscale an (already inverted if necessary) absorption spectrum so that
# the continuum simeq 1 and absorption goes downward 
def normalize_abs(f, check_continuum_up=True, unity_level='upper', upper_level_frac=97):
    if check_continuum_up:
        f_median = np.median(f)
        f_mean = np.mean(f)
        if f_mean > f_median:
            f *= -1
    if unity_level == 'mean':
        unity = np.mean(f)
    elif unity_level == 'median':
        unity = np.median(f)
    elif unity_level == 'upper':
        unity = np.percentile(f,upper_level_frac)
    else:
        unity = np.max(f)
    f2 = f - unity + 1
    fmin = np.min(f2)
    if fmin <= 0:
        f2 = (f2-1)/(1-fmin)+1.01 # Rescale so min = 0.01
    return f2

# ------------------------------------------
# Load Target Spectra List
# ------------------------------------------
objs = utils.load_tel_list(list_run=None)
print('{} spectra to be used'.format(len(objs)),file=sys.stderr)

# ------------------------------------------
# Main Loop Over Orders
# ------------------------------------------
for order in orders:
    print(f'Processing order {order} ({order_wranges[order]})',file=sys.stderr)
    # ------------------------------------------
    # Setup wavelength grid
    # ------------------------------------------
    wmin_str, wmax_str = order_wranges[order].split(':')
    wmin, wmax = float(wmin_str), float(wmax_str)
    print(f'Processing order {order} ({wmin_str}:{wmax_str})',file=sys.stderr)
    waves = utils.make_order_waves(order)  # (npix,) 
    np.savetxt(f'{TXT_DIR}/waves_{order}.txt',waves,fmt='%9.3f')
    # ------------------------------------------
    # Load and normalize spectra
    # ------------------------------------------
    list_sp_std = []
    sp_sum = np.array([0.]*len(waves))
    n_sum = 0
    for iobj, (label, obj) in enumerate(objs.items()):
        if data_to_reject(label, order):
            print(f'   Not using the spectrum of {label} for order={order}',file=sys.stderr)
            continue
        spfile = utils.get_sp_txt(obj, order)
        if spfile is None:
            print(f'   Failed to load the spectrum of {label} for order={order}',file=sys.stderr)
            continue
        sp_obj, *_ = utils.load_sp(spfile)
        if sp_obj.size == 0:
            continue
        # Reference spectrum alignment
        if iobj == 0:
            interp_func = CubicSpline(sp_obj[:,0],sp_obj[:,1])
            sp_new = interp_func(waves)
        else:
            xadjust, yadjust = utils.calc_adjust(sp_obj,wmin,wmax,interp_ref)
            sp_obj[:, 0] += xadjust
            sp_obj[:, 1] += yadjust
            print(f'   xadjust={xadjust:.3f} AA, yadjust={yadjust:.3f} for {label}',file=sys.stderr)
            interp_ref = CubicSpline(sp_obj[:,0],sp_obj[:,1])
            sp_new = interp_ref(waves)
        # Pad outside observed range with continuum = 1
        left = min(sp_obj[:,0]); right = max(sp_obj[:,0])
        mask_out = (waves < left) | (waves > right)
        sp_new[mask_out] = 1.0
        n_sum += 1
        sp_sum += sp_new
        sp_mean = sp_sum / n_sum
        interp_ref = CubicSpline(waves, sp_mean)
        # Standardise and store
        sp_std = (sp_new - sp_new.mean()) / sp_new.std()
        list_sp_std.append(sp_std)
    m = len(list_sp_std)
    if m == 0:
        print(f'No usable spectra for {order} - skipping.',file=sys.stderr)
        continue
    print(f'   -> {m:d} spectra being used for {order}',file=sys.stderr)
    sp_mean = sp_sum/m

    # ------------------------------------------
    # PCA Analysis
    # ------------------------------------------
    # PCA: build X (m x npix), covariance (m x m), eigen-decomp
    # eigh is for symmetic matrices, eigenvalues in ascending order here
    X=np.vstack(list_sp_std).astype(np.float32) # shape (m, npix)
    cov_matrix = np.cov(X,rowvar=True,dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # Sort in descending order
    idx_desc = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx_desc]
    eigenvectors = eigenvectors[:, idx_desc]
    # Construct basis A = X^T @ Ep
    Lp = eigenvalues[:p] # (p)
    norm_factor = 1/np.sum(Lp)
    Ep = eigenvectors[:, :p] # (m, p)
    A = X.T @ Ep # (npix, p) = basis vectors
    # Align signs of basis vectors so absorption is downward
    for i in range(p):
       a_median = np.median(A[:,i])
       a_mean = np.mean(A[:,i])
       if a_mean>a_median:
           A[:,i] *= -1
    np.savetxt(f'{TXT_DIR}/base_{order}.txt',A,fmt='%10.6f')
    # ------------------------------------------
    # Compose Average Spectrum
    # ------------------------------------------
    try:
        # Fit sp_mean as linear combination of basis spectra A
        coeffs, residuals, rank, s = np.linalg.lstsq(A, sp_mean - np.mean(sp_mean), rcond=None)
        sp_ave = A @ coeffs + np.mean(sp_mean)
    except np.linalg.LinAlgError as e:
        print(f'   Error fitting {label}: {e}')
        # Composed average spectrum (sum of first p components)
        sp_ave = normalize_abs(A @ Lp, unity_level='upper')
    np.savetxt(f'{TXT_DIR}/ave_{order}.txt',np.column_stack([waves,sp_ave]),fmt='%9.3f %10.6f')

    # ------------------------------------------
    # Plot: Eigenvalue Spectrum
    # ------------------------------------------
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.plot(range(int(m/2)),eigenvalues[0:int(m/2)],marker='o')
    ax.axvline(p,ls='dotted',color='k')
    ax.set_xlabel('Eigenvalue order')
    ax.set_ylabel('Eigenvalue')
    ax.set_yscale('log')
    ax.set_title(order)
    fig.savefig(f'{PLOT_DIR}/eigenvalues_{order}.png')
    plt.close(fig)

    # ------------------------------------------
    # Plot: Basis Vectors and Composition
    # ------------------------------------------
    fig = plt.figure(figsize=(15,6))
    ax = fig.add_subplot(111)
    for i in range(p):
        if i == 0:
            sp_abs = normalize_abs(norm_factor*Lp[i]*A[:,i],unity_level='upper')
        else:
            sp_abs = normalize_abs(norm_factor*Lp[i]*A[:,i],unity_level='mean')
        ax.plot(waves,(2**i*(sp_abs-1))+1+0.05*i,color=cmap(norm(i)),zorder=i)
    ax.plot(waves,sp_mean,lw=1,color='k',label='Average',zorder=p+1)
    ax.plot(waves,sp_ave,color='deepskyblue',lw=0.7,label='Composed',zorder=p+2)
    ax.legend(bbox_to_anchor=(0.99,0.03),loc='lower right')
    ax.set_ylim([0,1.5])
    ax.axvline(wmin,ls='dotted',lw=0.6)
    ax.axvline(wmax,ls='dotted',lw=0.6)
    ax.axhline(0,ls='dotted',lw=0.6)
    ax.axhline(1,ls='dotted',lw=0.6)
    ax.set_xlabel('Wavelength')
    ax.set_title(f'Projection of the first {p:d} eigenvectors ({order}, {wmin_str}:{wmax_str})')
    fig.savefig(f'{PLOT_DIR}/eigenvectors_{order}.png')
    plt.close(fig)
