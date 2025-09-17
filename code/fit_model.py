#!/usr/bin/env python3
#fit_model.py
#Fit PCA-based telluric model to an observed spectrum using a linear
#combination of basis spectra and an offset term.

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import utils
from mask_from_spectrum import mask_from_spectrum
from calc_sp_offset import calc_sp_offset

c_kms = utils.c_kms
tel_min, tel_max = 0.001, 1.2

default_edge_frac = 0.1
default_niter = 5
default_clip_limit = 3
default_clip_margin = 0.3
default_continuum_percentile = 80


# --------------------------------------------
# Utility functions
# --------------------------------------------

def clip_parts(waves,sp_dev,sigma,limit=default_clip_limit,margin=default_clip_margin):
    if limit <= 0:
        return []
    reject = []
    mean = np.mean(sp_dev)
    left, right = None, None
    for w, f in zip(waves,sp_dev):
        if abs(f-mean) > limit*sigma:
            if left is None:
                left = w-margin
            right = w+margin
        elif left is not None:
            reject.append([left,right])
            left, right = None, None
    if left is not None:
        reject.append([left,max(waves)])
    return reject

def calc_sigma(waves, func_obs, func_model, frange=(0.0, 1.5), list_reject=[], tel_abs=[], only_tel_abs=True):
    fmin, fmax = frange
    if only_tel_abs:
        use_ranges = utils.combine_use_and_reject(tel_abs, list_reject)
    else:
        use_ranges = utils.combine_use_and_reject([[np.min(waves),np.max(waves)]], list_reject)
    if len(use_ranges) <= 0:
        return np.nan, 0
    mask = np.ones(len(waves), dtype=bool)
    use_mask = np.zeros(len(waves), dtype=bool)
    for w0, w1 in use_ranges:
        use_mask |= (waves >= w0) & (waves <= w1)
    mask &= use_mask
    sp_obs = func_obs(waves)
    sp_model = func_model(waves)
    valid = (sp_obs>fmin)&(sp_obs<fmax)&(sp_model>fmin)&(sp_model<fmax)&(mask)
    if not np.any(valid):
        return np.nan, 0
    sigma = np.std(sp_obs[valid]-sp_model[valid])
    n_used = np.count_nonzero(valid)
    return sigma, n_used


# --------------------------------------------
# Argument parsing
# --------------------------------------------
parser = argparse.ArgumentParser(description='Fitting a PCA-based telluric absorption model')
parser.add_argument('order',type=str,help='Order (eg. m44)')
parser.add_argument('file_in',type=str,help='Input file (txt or FITS)')
parser.add_argument('file_out',type=str,help='Output file (txt or FITS, same format as input)')
parser.add_argument("-s","--setting", type=str, default=utils.default_setting,
                    help=f"Instrumental setting label (default: {utils.default_setting})")
parser.add_argument('--vac_air', type=str, default='vac', choices=['vac','air'], help='Wavelength scale of input spectrum and model')
parser.add_argument('-x','--xadjust',action='store_true',help='Adjust the wavelength scale')
parser.add_argument('-n','--niter',type=int,default=default_niter,help=f'The number of iteration (default: {default_niter:d})')
parser.add_argument('-c','--continuum_percentile',type=float,default=float(default_continuum_percentile),help='Percentile to determine the continuum level (1-99; default={default_continuum_percentile})')
parser.add_argument('-t','--trim_edge',action='store_true',help='Trim the edge of the input spectrum')
parser.add_argument('-e','--edge_frac',type=float,default=default_edge_frac,help=f'The fraction of the edge part within the observed spectrum to trim (default: {default_edge_frac:.2f})')
parser.add_argument('--reject1',type=str,default='',help=f'Parts of rejection 1')
parser.add_argument('--reject2',type=str,default='',help=f'Parts of rejection 2 (not rejected if overlapped with pre-defined telluric)')
parser.add_argument('--mask_obj',type=str,default='',help=f'Name of the text file of the predicted object spectrum giving the absorption of the target to mask')
parser.add_argument('--obj_rv_add',type=float,default=0.,help=f'RV to add to the object model to calculate the masking wavelength ranges')
parser.add_argument('--obj_thresh',type=float,default=0.95,help=f'Threshold in flux to detect the object absorption to mask (e.g. 0.95)')
parser.add_argument('--clip_margin',type=float,default=default_clip_margin,help=f'Margin around a sigma-clip range (default: {default_clip_margin})')
parser.add_argument('--clip_limit',type=float,default=default_clip_limit,help=f'Limit of sigma-clip in the unit of sigma (default: {default_clip_limit})')
parser.add_argument("--fmin", type=float, default=0.0, help="Minimum flux to include in the analysis")
parser.add_argument("--fmax", type=float, default=1.5, help="Maximum flux to include in the analysis")
parser.add_argument('-p','--plot_out',type=str,default='NONE',help='Plot output file (png, jpg, pdf)')
parser.add_argument('-d','--diag_out',type=str,default='',help='Optional text file to append diagnostics (RMS, etc).')
parser.add_argument('-v','--verbose', action='store_true', default=False, help='Verbose mode')

args = parser.parse_args()
order = args.order
vac_air = args.vac_air
file_in = args.file_in
file_out = args.file_out
verbose = args.verbose
fmin, fmax = args.fmin, args.fmax

setting = args.setting
setting_orders = utils.load_setting(setting)
if order not in setting_orders:
    print(f"[ERROR] order={order} is not defined in setting {setting}", file=sys.stderr)
    sys.exit(1)
wmin_set, wmax_set=setting_orders[order]['wmin'], setting_orders[order]['wmax'] 
print(f'# order = {order} ({wmin_set}:{wmax_set}) for {setting}', file=sys.stderr)

if not os.path.isfile(file_in):
    print(f'{file_in} does not exist!', file=sys.stderr)
    parser.print_help()
basename_obs, extension_obs = os.path.splitext(file_in)
if extension_obs in ['.fits','.FITS']:
    type_in_fits = True
else:
    type_in_fits = False
print(f'# input  = {file_in} (fits={type_in_fits}, {vac_air})', file=sys.stderr)

basename_out, extension_out = os.path.splitext(file_out)
if extension_out in ['.fits','.FITS']:
    type_out_fits = True
else:
    type_out_fits = False
print(f'# output = {file_out} (fits={type_out_fits})', file=sys.stderr)
if (type_out_fits) and (not type_in_fits):
    print(f'Sorry! The output can be a fits only if the input is a fits!', file=sys.stderr)
    parser.print_help()

plot_out = args.plot_out
list_plot_ext = ['.png','.jpg','.pdf']
if plot_out == 'NONE':
    plot_out = None
else:
    base_name, ext = os.path.splitext(plot_out)
    if ext not in list_plot_ext:
        print('plot_out should be in the format of '+'/'.join(list_plot_ext)+'or NONE', file=sys.stderr)
        parser.print_usage()
    print(f'# plot   = {plot_out}', file=sys.stderr)

# --------------------------------------------
# Load Observed Data and Model
# --------------------------------------------
sp_obs, header = utils.load_sp(args.file_in)
wmin_obs, wmax_obs = min(sp_obs[:,0]), max(sp_obs[:,0])
waves = utils.load_model_waves(setting, order, vac_air=vac_air)
base = utils.load_model_base(setting, order, add_ones_column=True)
sp_ave = utils.load_model_ave(setting, order, vac_air=vac_air)
tel_abs = utils.load_tel_abs(setting, order, vac_air=vac_air)
if args.verbose:
    print(f"# Loaded model for {order} ({vac_air}), {len(waves)} pixels", file=sys.stderr)
    print(f"# Found {len(tel_abs)} telluric absorption parts from average model", file=sys.stderr)


# Masking

list_reject1 = utils.parse_ranges_string(args.reject1)
list_reject2 = utils.parse_ranges_string(args.reject2)
if len(list_reject1) > 0:
    print(f'# list_reject1 - {len(list_reject1):d} parts to mask', file=sys.stderr)
if len(list_reject2) > 0:
    print(f'# list_reject2 - {len(list_reject2):d} parts to mask', file=sys.stderr)

if args.mask_obj:
    if os.path.isfile(args.mask_obj):
        mask_obj_ranges = mask_from_spectrum(np.loadtxt(args.mask_obj), rv_to_add=args.obj_rv_add, flux_thresh=args.obj_thresh, buffer_pix=3)
        list_reject1 += mask_obj_ranges
        print(f'# mask_obj={args.mask_obj} - {len(mask_obj_ranges):d} parts to mask', file=sys.stderr)

if args.trim_edge:
    edge_frac = args.edge_frac
    left_edge = wmin_obs+edge_frac*(wmax_obs-wmin_obs)
    right_edge = wmax_obs-edge_frac*(wmax_obs-wmin_obs)
    list_reject1.append([wmin_obs-1,left_edge])
    list_reject1.append([right_edge,wmax_obs+1])
    print(f'# trim_edge - edge_frac={args.edge_frac} (leaving {left_edge}:{right_edge})', file=sys.stderr)

# Optional xadjust
rv_used = 0.0
if args.xadjust:
    xadj, yadj, n_used = calc_sp_offset(sp_obs, sp_ave, use_ranges=[tuple(x) for x in tel_abs], list_v=np.arange(-15,15,0.5), frange=(fmin,fmax))
    sp_obs[:,0] *= (1.0+xadj/c_kms)
    #sp_obs[:,1] += yadj
    rv_used = xadj
    if args.verbose:
        print(f"   xadjust={xadj:.2f} km/s, n_used={n_used}", file=sys.stderr)

func_obs = CubicSpline(sp_obs[:,0], sp_obs[:,1])

# --------------------------------------------
# Iterative fitting loop
# --------------------------------------------
func_model = None
for iiter in range(args.niter):
    pixels_use = utils.check_pixels_use(waves,sp_obs[:,0],list_reject1=list_reject1,list_reject2=list_reject2,tel_abs=tel_abs)
    if not np.any(pixels_use):
        print('No overlapping pixels with model wavelength grid.', file=sys.stderr)
        sys.exit(1)
    waves_part = waves[pixels_use]
    sp_part = func_obs(waves_part)
    base_part = base[pixels_use]
    mean_level = np.mean(sp_part)
    coeffs, *_ = np.linalg.lstsq(base_part, sp_part-mean_level, rcond=None)
    model = base @ coeffs + mean_level
    func_model = CubicSpline(waves, model)
    model_part = func_model(waves_part)
    dev_part = sp_part - model_part
    sigma1, n_used1 = calc_sigma(waves_part, func_obs, func_model, frange=(fmin, fmax), list_reject=list_reject1+list_reject2, tel_abs=tel_abs, only_tel_abs=True)
    sigma2, n_used2 = calc_sigma(waves_part, func_obs, func_model, frange=(fmin, fmax), list_reject=list_reject1+list_reject2, only_tel_abs=False)
    if args.verbose:
        print('   iiter={:d} sigma1={:.5f}({:d}px)/sigma2={:.5f}({:d}px),  N(rej1)={:d}pts, N(rej2)={:d}pts)'.format(iiter,sigma1,n_used1,sigma2,n_used2,len(list_reject1),len(list_reject2)), file=sys.stderr)
    new_rejects = clip_parts(waves_part,dev_part,sigma2,limit=args.clip_limit,margin=args.clip_margin)
    list_reject2 = utils.merge_ranges(list_reject2 + new_rejects)
if func_model is None:
    print('Model fitting failed !!', file=sys.stderr)
    sys.exit(1)

# --------------------------------------------
# Save output
# --------------------------------------------
continuum_level_obs = utils.calc_continuum(sp_obs,continuum_percentile=args.continuum_percentile,reject_ranges=list_reject1+list_reject2+list(tel_abs))
continuum_level_model = utils.calc_continuum(np.column_stack((waves_part, model_part)),continuum_percentile=args.continuum_percentile,reject_ranges=list_reject1+list_reject2+list(tel_abs))

in_band = (sp_obs[:,0] >= waves[0]) & (sp_obs[:,0] <= waves[-1])
model_interp = func_model(sp_obs[:,0])
model_interp = np.clip(model_interp - (continuum_level_model - 1), tel_min, tel_max)
flux_out = np.where(in_band, model_interp, 1.0)
if type_out_fits:
    utils.save_new_fits(file_out,header,flux_out)
else:
    np.savetxt(file_out,np.column_stack([sp_obs[:,0]*(1-rv_used/c_kms),flux_out]),fmt='%.6f')

#pixels_use = utils.check_pixels_use(waves,sp_obs[:,0],list_reject=list_reject,tel_abs=tel_abs)
pixels_use = utils.check_pixels_use(waves,sp_obs[:,0],list_reject1=[],list_reject2=[],tel_abs=[])
waves_part = waves[pixels_use]
sp_part = func_obs(waves_part)-(continuum_level_obs-1)
model_part = func_model(waves_part)-(continuum_level_model-1)
dev_part = sp_part - model_part
sigma1, n_used1 = calc_sigma(waves_part, func_obs, func_model, frange=(fmin, fmax), list_reject=list_reject1+list_reject2, tel_abs=tel_abs, only_tel_abs=True)
sigma2, n_used2 = calc_sigma(waves_part, func_obs, func_model, frange=(fmin, fmax), list_reject=list_reject1+list_reject2, only_tel_abs=False)
print(f'   -> telluric model fitted, sigma1={sigma1:.5f}({n_used1:d}px)/sigma2={sigma2:.5f}({n_used2:d}px)', file=sys.stderr)

if args.diag_out:
    with open(args.diag_out, 'a') as fout:
        fout.write(f"{setting}\t{order}\t{sigma1:.5f}\t{n_used1}\t{sigma2:.5f}\t{n_used2}\t{os.path.basename(file_in)}\n")
    if args.verbose:
        print(f"# Appended diagnostics to {args.diag_out}", file=sys.stderr)

# --------------------------------------------
# Plotting (if enabled)
# --------------------------------------------
if plot_out is not None:
    xmin = min([wmin_obs-0.05*(wmax_obs-wmin_obs), wmin_set-0.05*(wmax_set-wmin_set)])
    xmax = max([wmax_obs+0.05*(wmax_obs-wmin_obs), wmax_set+0.05*(wmax_set-wmin_set)])
    low, up = min(sp_part), max(sp_part)
    ymin = min([(up+low)/2.-0.7*(up-low),0.95])
    ymax = max([(up+low)/2.+0.7*(up-low),1.03])
    ymin = fmin-0.1*(fmax-fmin) if ymin < fmin else ymin
    ymax = fmax+0.1*(fmax-fmin) if ymax > fmax else ymax
    ymin2 = np.min([-0.15, -3.2*sigma1,-3.2*sigma2])
    ymax2 = np.max([0.13, 2.8*sigma1,2.8*sigma2])
    fig = plt.figure(figsize=(12,7))
    axs = {}
    axs['sp'] = fig.add_axes([0.10,0.48,0.80,0.44])
    axs['dev'] = fig.add_axes([0.10,0.10,0.80,0.32])
    axs['sp'].plot(sp_obs[:,0],sp_obs[:,1]-(continuum_level_obs-1),color='red',lw=1.25,alpha=0.7,label='obs')
    axs['sp'].plot(waves_part,model_part,color='blue',lw=1.25,alpha=0.7,label='model')
    if (args.mask_obj != '') and (os.path.isfile(args.mask_obj)):
        mask_obj_sp = np.loadtxt(args.mask_obj)
        mask_obj_sp[:,0] *= 1.0+(args.obj_rv_add/c_kms)
        axs['sp'].plot(mask_obj_sp[:,0],mask_obj_sp[:,1],lw=1.25,ls='dashed',color='limegreen',zorder=1)
        axs['dev'].plot(mask_obj_sp[:,0],mask_obj_sp[:,1]-1,lw=1.25,ls='dashed',color='limegreen',zorder=10)
    for w0, w1 in tel_abs:
        axs['sp'].fill_between([w0,w1],1+0.50*(ymax-1),1+0.57*(ymax-1),color='blue',alpha=0.6)
        axs['dev'].fill_between([w0,w1],0.25*ymin2+0.75*ymax2,0.23*ymin2+0.77*ymax2,color='blue',alpha=0.6)
    for w0, w1 in list_reject2:
        axs['sp'].fill_between([w0,w1],1+0.65*(ymax-1),1+0.72*(ymax-1),color='gray',alpha=0.6)
        axs['dev'].fill_between([w0,w1],0.20*ymin2+0.80*ymax2,0.18*ymin2+0.82*ymax2,color='gray',alpha=0.6)
    for w0, w1 in list_reject1:
        axs['sp'].fill_between([w0,w1],1+0.80*(ymax-1),1+0.87*(ymax-1),color='red',alpha=0.6)
        axs['dev'].fill_between([w0,w1],0.15*ymin2+0.85*ymax2,0.13*ymin2+0.87*ymax2,color='red',alpha=0.6)
    axs['sp'].axhline(1,ls='dotted',lw=0.7,color='k')
    axs['sp'].axvline(wmin_set,ls='dotted',lw=0.7,color='k')
    axs['sp'].axvline(wmax_set,ls='dotted',lw=0.7,color='k')
    axs['sp'].legend(bbox_to_anchor=(0.99,0.03),loc='lower right')
    axs['sp'].set_xlim([xmin,xmax])
    axs['sp'].set_ylim([ymin,ymax])
    axs['sp'].set_ylabel('Normalized Flux')
    axs['sp'].set_title(f'{file_in} ({order}, {wmin_set}-{wmax_set})')
    axs['dev'].plot(waves_part,model_part-1,lw=0.75,color='blue',zorder=2)
    axs['dev'].plot(waves_part,dev_part,zorder=3,lw=1.5,color='gray')
    axs['dev'].axhline(0.,ls='dotted',color='k',zorder=2)
    axs['dev'].axvline(wmin_set,ls='dotted',lw=0.7,color='k')
    axs['dev'].axvline(wmax_set,ls='dotted',lw=0.7,color='k')
    axs['dev'].text(0.01*xmin+0.99*xmax,0.03*ymin2+0.97*ymax2,r'$\sigma_1={:.4f}, \sigma_2={:.4f}$'.format(sigma1,sigma2),horizontalalignment='right',verticalalignment='top')
    axs['dev'].set_xlim([xmin,xmax])
    axs['dev'].set_ylim([ymin2,ymax2])
    axs['dev'].set_xlabel(f'{vac_air} wavelength')
    axs['dev'].set_ylabel('Residual')
    fig.savefig(plot_out,dpi=200)
    plt.close(fig)

