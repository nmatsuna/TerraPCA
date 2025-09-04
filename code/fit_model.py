#!/usr/bin/env python3
#fit_model.py
#
#Fit PCA-based telluric model to an observed spectrum using a linear
#combination of basis spectra and an offset term.

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from mask_from_obj import mask_from_obj
import utils

PLOT_DIR = 'fit_model'
BASE_DIR = '{}/eigen_txt'.format(utils.HOME_DIR)

num_base = utils.num_base
orders = utils.orders
order_wranges = utils.order_wranges_with_buffer

default_edge_frac = '0.1'
default_niter = '5'
default_clip_limit = '3'
default_clip_margin = '0.3'
default_continuum_percentile = '80'

tel_min, tel_max = 0.001, 1.2
flux_upper_limit=1.5

def calc_continuum(waves,fluxs,continuum_percentile=float(default_continuum_percentile),list_reject=[]):
    fluxs_use = []
    for w, f in zip(waves,fluxs):
        reject = False
        for w0, w1 in list_reject:
            if (w-w0)*(w-w1) <= 0:
                reject = True
                break
        if not reject:
            fluxs_use.append(f)
    continuum = np.percentile(fluxs_use,continuum_percentile)
    return continuum

def clip_parts(waves,sp_dev,sigma,limit=float(default_clip_limit),margin=float(default_clip_margin)):
    if limit <= 0:
        return []
    if margin <= 0:
        margin = 0
    reject = []
    mean = np.mean(sp_dev)
    left, right = None, None
    for w, f in zip(waves,sp_dev):
        if abs(f-mean)>limit*sigma:
            if left is None:
                left = w-margin
            right = w+margin
        elif left is not None:
            reject.append([left,right])
            left, right = None, None
    if left is not None:
        right = max(waves)
        reject.append([left,right])
    return reject

# --------------------------------------------
# 1. Parse and Validate Input
# --------------------------------------------
parser = argparse.ArgumentParser(description='Fitting a PCA-based telluric absorption model')
parser.add_argument('order',type=str,help='Order (eg. m44)')
parser.add_argument('file_in',type=str,help='Input file (FITS or text)')
parser.add_argument('file_out',type=str,help='Output file (FITS or text, same format as input)')
parser.add_argument('-p','--plot_out',type=str,default='NONE',help='Plot output file (png, jpg, pdf)')
parser.add_argument('-n','--niter',type=int,default=int(default_niter),help=f'The number of iteration (default: {default_niter})')
parser.add_argument('-a','--air',action='store_true',help='If the wavelength is air (default: vacuum)')
parser.add_argument('-t','--trim_edge',action='store_true',help='Trim the edge of the input spectrum')
parser.add_argument('-e','--edge_frac',type=float,default=float(default_edge_frac),help=f'The fraction of the edge part to trim (default: {default_edge_frac})')
parser.add_argument('-x','--xadjust',action='store_true',help='Adjust the wavelength scale')
parser.add_argument('-c','--continuum_percentile',type=float,default=float(default_continuum_percentile),help='Percentile to determine the continuum level (1-99; default={default_continuum_percentile})')
parser.add_argument('--reject1',type=str,default='',help=f'Parts of rejection 1')
parser.add_argument('--reject2',type=str,default='',help=f'Parts of rejection 2 (not rejected if overlapped with pre-defined telluric)')
parser.add_argument('--mask_obj',type=str,default='',help=f'Name of the text file of the predicted object spectrum giving the absorption of the target to mask')
parser.add_argument('--obj_rv_add',type=float,default=0.,help=f'RV to add to the object model to calculate the masking wavelength ranges')
parser.add_argument('--obj_thresh',type=float,default=0.95,help=f'Threshold in flux to detect the target absorption to mask (e.g. 0.95)')
parser.add_argument('--clip_margin',type=float,default=float(default_clip_margin),help=f'Margin around a sigma-clip range')
parser.add_argument('--clip_limit',type=float,default=float(default_clip_limit),help=f'Limit of sigma-clip in the unit of sigma')

args = parser.parse_args()
niter = args.niter
if niter <= 0:
    print('niter should be positive!', file=sys.stderr)
    parser.print_help()
if args.air:
    vac_air = 'air'
else:
    vac_air = 'vac'
if args.xadjust:
    flag_xadjust = True
else:
    flag_xadjust = False
mask_obj = args.mask_obj
obj_rv_add = args.obj_rv_add
obj_thresh = args.obj_thresh
clip_limit = args.clip_limit
clip_margin = args.clip_margin
continuum_percentile = args.continuum_percentile

order = args.order
if order not in orders:
    print(f'Error: Unknown order {order}', file=sys.stderr)
    parser.print_help()
wrange = order_wranges[order]
wmin_str, wmax_str = wrange.split(':')
wmin, wmax=float(wmin_str), float(wmax_str)
print(f'# order  = {order} ({wmin_str}:{wmax_str})', file=sys.stderr)

list_reject1 = []
if len(args.reject1)>0:
    for part in args.reject1.split(','):
        items = part.split(':')
        w0, w1 = float(items[0]), float(items[1])
        list_reject1.append([min([w0,w1]),max([w0,w1])])
if mask_obj != '':
    if not os.path.isfile(mask_obj):
        print(f'Failed find mask_obj={mask_obj}')
    else:
        masked_ranges = mask_from_obj(mask_obj, rv_to_add=obj_rv_add, depth_thresh=obj_thresh, buffer_pix=3)
        nmask = 0
        for w0, w1 in masked_ranges:
            if (not (wmin <= w0 <= wmax)) and (not (wmin <= w1 <= wmax)):
                continue
            list_reject1.append([min([w0,w1]),max([w0,w1])])
            nmask += 1
        print(f'mask_obj={mask_obj} - {nmask:d} parts to mask', file=sys.stderr)
list_reject2 = []
if len(args.reject2)>0:
    for part in args.reject2.split(','):
        items = part.split(':')
        w0, w1=float(items[0]), float(items[1])
        list_reject2.append([min([w0,w1]),max([w0,w1])])

file_in = str(args.file_in)
if not os.path.isfile(file_in):
    print(f'{file_in} does not exist!', file=sys.stderr)
    parser.print_help()
basename_obs, extension_obs = os.path.splitext(file_in)
if extension_obs in ['.fits','.FITS']:
    type_in_fits = True
else:
    type_in_fits = False
print(f'# input  = {file_in} (fits={type_in_fits}, {vac_air})', file=sys.stderr)

file_out = args.file_out
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
# 2. Load Observed Data and Model
# --------------------------------------------
sp_obs, header = utils.load_sp(file_in)
wmin_obs, wmax_obs = min(sp_obs[:,0]), max(sp_obs[:,0])
func_obs = CubicSpline(np.array(sp_obs[:,0]),np.array(sp_obs[:,1]))
waves = utils.load_eigen_waves(order,vac_air=vac_air)
array_base = utils.load_eigen_base(order,add_ones_column=True)
sp_ave = np.loadtxt(f'{BASE_DIR}/ave_{order}.txt')
if vac_air == 'air':
    sp_ave[:,0] = utils.vac2air(sp_ave[:,0])
wmin2, wmax2 = min(sp_ave[:,0]), max(sp_ave[:,0])
tel_abs = utils.load_tel_abs(order,vac_air=vac_air)

npix, npix_tel = len(waves), 0
for w in waves:
    for w0, w1 in tel_abs:
        if w0 <= w <= w1:
            npix_tel += 1
            break
print('{:d} pixels in the original model, {:d} pixels within tel_abs ({:d} parts)'.format(npix,npix_tel,len(tel_abs)), file=sys.stderr)

# --------------------------------------------
# 3. Optional Wavelength Adjustments
# --------------------------------------------
if flag_xadjust:
    func_ave = CubicSpline(sp_ave[:,0],sp_ave[:,1])
    xadjust, yadjust = utils.calc_adjust(sp_obs,max([wmin,wmin_obs]),min([wmax,wmax_obs]),func_ave)
    print(f'xadjust = {xadjust:.3f}, yadjust = {yadjust:.3f}', file=sys.stderr)
    waves -= xadjust
    sp_ave[:,0] -= xadjust
    sp_ave[:,1] -= yadjust
    tel_abs[:,0] -= xadjust
    tel_abs[:,1] -= xadjust
func_ave = CubicSpline(sp_ave[:,0],sp_ave[:,1])

if args.trim_edge:
    edge_frac = args.edge_frac
    list_reject1.append([wmin_obs-1,wmin_obs+edge_frac*(wmax_obs-wmin_obs)])
    list_reject1.append([wmax_obs-edge_frac*(wmax_obs-wmin_obs),wmax_obs+1])

# --------------------------------------------
# 4. Sigma-Clipping and Model Fitting Loop
# --------------------------------------------
for iiter in range(niter):
    #print(list_reject1, file=sys.stderr)
    #print(list_reject2, file=sys.stderr)
    pixels_use = utils.check_pixels_use(waves,sp_obs[:,0],list_reject1=list_reject1,list_reject2=list_reject2,tel_abs=tel_abs)
    if not np.any(pixels_use):
        print('No overlapping pixels with model wavelength grid.', file=sys.stderr)
        sys.exit(1)
    waves_part = waves[pixels_use]
    sp_interp_all = func_obs(waves) 
    sp_interp_part = sp_interp_all[pixels_use]
    array_base_part = array_base[pixels_use]
    mean_level = np.mean(sp_interp_part)
    x, residuals, rank, s = np.linalg.lstsq(array_base_part,sp_interp_part-mean_level,rcond=None)
    sp_model_part = array_base_part@x+mean_level
    sp_dev_part = sp_interp_part-sp_model_part
    sp_dev_partpart = []
    for f1, f2 in zip(sp_interp_part, sp_model_part):
        if 0 <= f1 <= flux_upper_limit:
            sp_dev_partpart.append(f1-f2)
    sigma = np.std(sp_dev_partpart)
    list_reject2 += clip_parts(waves_part,sp_dev_part,sigma,limit=clip_limit,margin=clip_margin)
    print('iiter={:d} sigma={:.5f} ({:d} pixels used; N(reject1)={:d}, N(reject2)={:d})'.format(iiter,sigma,np.count_nonzero(pixels_use),len(list_reject1),len(list_reject2)), file=sys.stderr)

# --------------------------------------------
# 5. Final Model Evaluation and Output    
# --------------------------------------------
#pixels_use = utils.check_pixels_use(waves,sp_obs[:,0],list_reject=list_reject,tel_abs=tel_abs)
pixels_use = utils.check_pixels_use(waves,sp_obs[:,0],list_reject1=[],list_reject2=[],tel_abs=[])
waves_part = waves[pixels_use]
array_base_part = array_base[pixels_use]
sp_interp_part = func_obs(waves_part)
sp_model_part = array_base_part@x+mean_level
continuum_level = calc_continuum(waves_part,sp_model_part,continuum_percentile=continuum_percentile,list_reject=list_reject1+list_reject2+list(tel_abs))
sp_model_part += (1-continuum_level)
sp_dev_part = sp_interp_part-sp_model_part

func_out = CubicSpline(waves_part,sp_model_part)
w0, w1 = min(waves_part), max(waves_part)
flux_out = []
for w in sp_obs[:,0]:
    if w0 <= w <= w1:
        f = func_out(w)
        if f < tel_min:
            f = tel_min
        if f > tel_max:
            f = tel_max
        flux_out.append(f)
    else:
        flux_out.append(1.)
if type_out_fits:
    utils.save_new_fits(file_out,header,flux_out)
else:
    np.savetxt(file_out,np.column_stack([sp_obs[:,0],flux_out]),fmt='%.6f')
    #np.savetxt(file_out,np.column_stack([waves_part,sp_model_part,sp_interp_part,sp_dev_part]),fmt='%.6f')

# --------------------------------------------
# 6. Plotting (if enabled)
# --------------------------------------------
if plot_out is not None:
    xmin, xmax = wmin_obs-15., wmax_obs+15.
    low, up = min(sp_interp_part), max(sp_interp_part)
    ymin = min([(up+low)/2.-0.7*(up-low),0.95])
    ymax = max([(up+low)/2.+0.7*(up-low),1.03])
    if ymin < 0:
        ymin = 0
    if ymax > flux_upper_limit:
        ymax = flux_upper_limit
    fig = plt.figure(figsize=(12,7))
    axs = {}
    axs['sp'] = fig.add_axes([0.10,0.48,0.80,0.44])
    axs['dev'] = fig.add_axes([0.10,0.10,0.80,0.32])
    axs['sp'].plot(sp_obs[:,0],sp_obs[:,1],color='red',lw=1.25,alpha=0.7,label='obs')
    axs['sp'].plot(waves_part,sp_model_part,color='blue',lw=1.25,alpha=0.7,label='model')
    if (mask_obj != '') and (os.path.isfile(mask_obj)):
        mask_obj_sp = np.loadtxt(mask_obj)
        mask_obj_sp[:,0] *= 1.0+(obj_rv_add/300000.0)
        axs['sp'].plot(mask_obj_sp[:,0],mask_obj_sp[:,1],lw=1.25,ls='dashed',color='limegreen',zorder=1)
        axs['dev'].plot(mask_obj_sp[:,0],mask_obj_sp[:,1]-1,lw=1.25,ls='dashed',color='limegreen',zorder=10)
    for w0, w1 in tel_abs:
        axs['sp'].plot([w0,w1],[0.3*1+0.7*ymax,0.3*1+0.7*ymax],color='blue')
        axs['dev'].plot([w0,w1],[0.085,0.085],color='blue')
    for w0, w1 in list_reject2:
        axs['sp'].plot([w0,w1],[0.6*1+0.4*ymax,0.6*1+0.4*ymax],color='gray')
        axs['dev'].plot([w0,w1],[0.065,0.065],color='gray')
    for w0, w1 in list_reject1:
        axs['sp'].plot([w0,w1],[0.6*1+0.4*ymax,0.6*1+0.4*ymax],color='red')
        axs['dev'].plot([w0,w1],[0.065,0.065],color='red')
    axs['sp'].axhline(1,ls='dotted',lw=0.7,color='k')
    axs['sp'].axvline(wmin,ls='dotted',lw=0.7,color='k')
    axs['sp'].axvline(wmax,ls='dotted',lw=0.7,color='k')
    axs['sp'].legend(bbox_to_anchor=(0.99,0.03),loc='lower right')
    axs['sp'].set_xlim([xmin,xmax])
    axs['sp'].set_ylim([ymin,ymax])
    axs['sp'].set_ylabel('Normalized Flux')
    axs['sp'].set_title(f'{file_in} ({order}, {wrange})')
    axs['dev'].plot(waves_part,sp_model_part-1,lw=0.75,color='blue',zorder=2)
    axs['dev'].plot(waves_part,sp_dev_part,zorder=3,lw=1.5,color='gray')
    axs['dev'].axhline(0.,ls='dotted',color='k',zorder=2)
    axs['dev'].axvline(wmin,ls='dotted',lw=0.7,color='k')
    axs['dev'].axvline(wmax,ls='dotted',lw=0.7,color='k')
    axs['dev'].text(0.01*xmin+0.99*xmax,0.11,r'$\sigma ={:.4f}$'.format(sigma),horizontalalignment='right',verticalalignment='top')
    axs['dev'].set_xlim([xmin,xmax])
    axs['dev'].set_ylim([-0.12,0.12])
    if vac_air == 'air':
        axs['dev'].set_xlabel('Air Wavelength (AA)')
    else:
        axs['dev'].set_xlabel('Vacuum Wavelength (AA)')
    fig.savefig(plot_out,dpi=200)
    plt.close(fig)

