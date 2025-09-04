import os
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from astropy.io import fits
from scipy.interpolate import interp1d

#HOME_DIR='/mnt/SharedDisk/WINERED/telluric'
HOME_DIR = '/home/nmatsuna/WINERED/telluric'
CODE_DIR = f'{HOME_DIR}/code/'
DATA_DIR = f'{HOME_DIR}/data/'

TEL_LIST = DATA_DIR + '/tel_list'

orders_all = ['m42','m43','m44','m45','m46','m47','m48','m49','m50','m51','m52','m53','m54','m55','m56','m57','m58','m59','m60','m61']
#orders = ['m43','m44','m45','m46','m47','m48','m52','m55','m56','m57','m58']
orders = ['m43','m44','m45','m46','m47','m48','m51','m52','m55','m56','m57','m58','m61']
order_wranges_with_buffer = {'m43':'12870:13210', 'm44':'12570:12930', 'm45':'12290:12630', 'm46':'12020:12350', 'm47':'11775:12075', 'm48':'11535:11825', 'm51':'10880:11110','m52':'10660:10910', 'm53':'10460:10700', 'm54':'10260:10500', 'm55':'10080:10300', 'm56':'9900:10120', 'm57':'9740:9940', 'm58':'9560:9780','m61':'9110:9290'}
order_wranges = {'m42':'13190:13510','m43':'12900:13190', 'm44':'12600:12900', 'm45':'12320:12600', 'm46':'12050:12320', 'm47':'11800:12050', 'm48':'11560:11800', 'm49':'11320:11560', 'm50':'11100:11320', 'm51':'10890:11100', 'm52':'10680:10890', 'm53':'10480:10680', 'm54':'10280:10480', 'm55':'10100:10280', 'm56':'9920:10100', 'm57':'9760:9920', 'm58':'9580:9760', 'm59':'9420:9580', 'm60':'9280:9420', 'm61':'9120:9280'}
num_eigen = 6
num_base = num_eigen
num_pix_order = 3072

def make_order_waves(order):
    wrange = order_wranges_with_buffer[order]
    wmin_str, wmax_str = wrange.split(':')
    wmin, wmax = float(wmin_str), float(wmax_str)
    return np.linspace(wmin,wmax,num_pix_order)

def which_order(wave_f):
    for order, wrange in order_wranges.items():
        w0s, w1s = wrange.split(':')
        w0, w1 = float(w0s), float(w1s)
        if (wave_f-w0)*(wave_f-w1) <= 0:
            return order
    return None

def calc_adjust(sp_obj,wmin,wmax,interp_func,list_v=np.arange(-30,30,1),frange=[0,1.5]):
    chi2_min, xadjust, yadjust = np.nan, 0., 0.
    wcen = (wmin+wmax)/2.
    for dw in (wcen*(list_v/300000.)):
        list_dev = []
        for w, fobs in sp_obj:
            if (wmin <= w+dw <= wmax) and (frange[0] <= fobs <= frange[1]):
                list_dev.append(fobs-interp_func(w+dw))
        if len(list_dev) <= 0:
            continue
        chi2=np.std(list_dev)
        if (np.isnan(chi2_min)) or (chi2<chi2_min):
            chi2_min = chi2
            xadjust = dw
            yadjust = -np.median(list_dev)
    return xadjust, yadjust


def load_tel_list(tel_list=TEL_LIST, list_run=None):
    objs = {}
    fh = open(tel_list)
    for line in fh.readlines():
        if line.startswith('#'):
            continue
        run1, label, star, date = line.split()
        if (list_run is None) or (run1 in list_run):
            objs[label] = {'run':run1, 'label':label, 'star':star, 'date':date}
    fh.close()
    return objs

def get_sp_txt(obj, order):
    sp_txt = '/'.join([obj['run'],obj['label'],'txt',order+'.txt'])
    if not os.path.isfile(sp_txt):
        return None
    return sp_txt

def load_sp(file_in, voff=np.nan, yoff=np.nan, wrange=None, frange=None):
    base_name, extension = os.path.splitext(file_in)
    if extension in ['.fits','.FITS']:
        return load_sp_fits(file_in,voff=voff,yoff=yoff)
    sp = load_sp_txt(file_in,voff=voff,yoff=yoff,wrange=wrange,frange=frange)
    return sp, None

def load_sp_txt(file_in, wrange=None, frange=None, voff=np.nan, yoff=np.nan):
    if not os.path.isfile(file_in):
        print(f'{file_in} does not exist!')
        return None
    sp = np.loadtxt(file_in)
    if not np.isnan(voff):
        sp[:,0] *= (1.+voff/300000.0)
    if not np.isnan(yoff):
        sp[:,1] += yoff
    if (wrange is None) and (frange is None):
        return sp[:, 0:2]
    wmin, wmax, fmin, fmax = 0,100000, -99999, 99999
    if wrange is not None:
        wmin, wmax = np.min(wrange), np.max(wrange)
    if frange is not None:
        fmin, fmax = np.min(frange), np.max(frange)
    sp_part = []
    for w, f, *_ in sp:
        if (wmin < w < wmax) and (fmin < f < fmax):
            sp_part.append([w, f])
    return np.array(sp_part)

def load_sp_fits(file_in, voff=np.nan, yoff=np.nan):
    if not os.path.isfile(file_in):
        print(f'{file_in} does not exist!')
        return None
    hdul = fits.open(file_in)
    header = hdul[0].header
    flux = hdul[0].data
    crval1 = header['CRVAL1']
    crpix1 = header['CRPIX1']
    cdelt1 = header['CDELT1']
    num_pixels = len(flux)
    pixel_values = np.arange(1,num_pixels+1)
    wave = crval1+(pixel_values-crpix1)*cdelt1
    sp = np.column_stack([wave,flux])
    if not np.isnan(voff):
        sp[:,0] *= (1.+voff/300000.0)
    if not np.isnan(yoff):
        sp[:,1] += yoff
    hdul.close()
    return sp, header

def save_new_fits(file_out,header,data,overwrite=True):
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdul_new = fits.HDUList([hdu])
    hdul_new.writeto(file_out, overwrite=overwrite)

def vac2air(waves):
    list_w_air = []
    for w_vac in waves:
        s = 1./w_vac
        n = 1.0 + 5.792105e-2 / (238.0185 - s**2) + 1.67917e-3/(57.362 - s**2)
        w_air = w_vac/n
        list_w_air.append(w_air)
    return np.array(list_w_air)

def load_eigen_waves(order, vac_air='vac'):
    waves = np.loadtxt(f'{HOME_DIR}/eigen_txt/waves_{order}.txt')
    if vac_air not in ['vac', 'VAC', 'vacuum', 'VACUUM']:
        return vac2air(waves)
    return waves

def load_eigen_base(order, add_ones_column=False):
    list_base = np.loadtxt(f'{HOME_DIR}/eigen_txt/base_{order}.txt')
    if add_ones_column:
        nline, nrow = list_base.shape
        ones_column = np.ones((nline,1))
        list_base = np.hstack((ones_column,list_base))
    return list_base

def load_tel_abs(order, vac_air='vac'):
    list_tel_abs = np.loadtxt(f'{HOME_DIR}/eigen_txt/tel_abs_{order}.txt')
    if (vac_air == 'air') and (len(list_tel_abs) > 0):
        list_tel_abs[:,0] = vac2air(list_tel_abs[:,0])
        list_tel_abs[:,1] = vac2air(list_tel_abs[:,1])
    return list_tel_abs

def check_pixels_use(w_grid,w_obs,list_reject1=[],list_reject2=[],tel_abs=[]):
    pixels_use = [True]*len(w_grid)
    wmin_obs, wmax_obs = np.min(w_obs), np.max(w_obs)
    wmin_grid, wmax_grid = np.min(w_grid), np.max(w_grid)
    list_out_grid = []
    if wmin_obs > wmin_grid:
        list_out_grid.append([wmin_grid-1,wmin_obs])
    if wmax_obs < wmax_grid:
        list_out_grid.append([wmax_obs,wmax_grid+1])
    for i, w in enumerate(w_grid):
        reject1, reject2, telluric = False, False, False
        for w0, w1 in list_out_grid:
            if w0 <= w <= w1:
                reject1 = True
                break
        for w0, w1 in list_reject1:
            if w0 <= w <= w1:
                reject1 = True
                break
        if not reject1:
            for w0, w1 in list_reject2:
                if w0 <= w <= w1:
                    reject2=True
                    break
            if reject2:
                for w0, w1 in tel_abs:
                    if w0 <= w <= w1:
                        telluric=True
                        break
        if (reject1) or ((reject2) and (not telluric)):
            pixels_use[i]=False
    return np.array(pixels_use)

