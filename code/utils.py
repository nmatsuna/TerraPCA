import os, glob
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from astropy.io import fits
from scipy.interpolate import interp1d
c_kms = 299792.458
num_standard_min = 5

#HOME_DIR='/mnt/SharedDisk/WINERED/telluric'
HOME_DIR = '/home/nmatsuna/WINERED/telluric'
CODE_DIR = f'{HOME_DIR}/code/'
DATA_DIR = f'{HOME_DIR}/data/'
SETTINGS_DIR = f'{HOME_DIR}/settings/'

default_setting = "WINERED_WIDE"

TEL_LIST = DATA_DIR + '/tel_list'

def load_setting(setting_label):
    """
    Load configuration from setting_[label].txt.
    Returns:
      dict with global params, and list of orders with wmin, wmax
    """
    filename = f"{SETTINGS_DIR}/setting_{setting_label}.txt"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Setting file {filename} for {setting_label} not found.")
    
    setting_orders = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            else:
                parts = line.split()
                if len(parts) == 5:
                    order, wmin, wmax, n_pix, n_base = parts[0], float(parts[1]), float(parts[2]), int(parts[3]), int(parts[4])
                    setting_orders[order] = {'wmin':wmin, 'wmax':wmax, 'n_pix':n_pix, 'n_base':n_base}
    return setting_orders

def make_order_waves(setting_orders, order):
    if (order not in setting_orders):
        return None
    o = setting_orders[order]
    return np.linspace(o['wmin'], o['wmax'], o['n_pix']) 

def parse_ranges_string(s):
    """Parse string like '10300.0:10310.0,10325.0:10330.0' to list of (start, end) tuples."""
    ranges = []
    if not s:
        return ranges
    for part in s.split(','):
        w0, w1 = map(float, part.split(':'))
        ranges.append([w0, w1])
    return ranges

def merge_ranges(regions):
    """
    Merge overlapping or adjacent (start, end) wavelength ranges.
    Parameters:
        regions (list of tuple): List of (start, end) tuples.
    Returns:
        List of merged (start, end) tuples.
    """
    if not regions:
        return []
    # Ensure sorted by start wavelength
    regions = sorted(regions, key=lambda x: x[0])
    merged = []
    for current_start, current_end in regions:
        if not merged:
            merged.append([current_start, current_end])
        else:
            last_start, last_end = merged[-1]
            if current_start <= last_end:
                # Overlap or adjacent -> merge
                merged[-1] = [last_start, max(last_end, current_end)]
            else:
                merged.append([current_start, current_end])
    return merged

def combine_use_and_reject(list_use, list_reject):
    """Return regions in list_use that are not in list_reject."""
    if len(list_use)<=0:
        return []
    final = []
    for u_start, u_end in list_use:
        cur_start = u_start
        for r_start, r_end in sorted(list(list_reject)):
            if r_end <= cur_start:
                continue
            if r_start >= u_end:
                break
            if r_start > cur_start:
                final.append([cur_start, min(r_start, u_end)])
            cur_start = max(cur_start, r_end)
        if cur_start < u_end:
            final.append([cur_start, u_end])
    return final

def apply_use_and_reject_ranges(sp, rv=0., use_ranges=[], reject_ranges=[]):
    """
    Applies both inclusion (use) and exclusion (reject) wavelength masks to a spectrum.

    Parameters:
        sp (np.ndarray): Input spectrum as an Nx2 array [(wavelength, flux), ...]
        rv (float): RV in km/s. The function checks if the wavelength after subtracting the redshift is within the ranges.
        use_ranges (list of tuples): List of (wmin, wmax) to keep. If None or empty, no cut.
        reject_ranges (list of tuples): List of (wmin, wmax) to exclude. If None or empty, no mask.

    Returns:
        np.ndarray: Masked spectrum
    """
    if isinstance(use_ranges, np.ndarray):
        use_ranges = use_ranges.tolist()
    if isinstance(reject_ranges, np.ndarray):
        reject_ranges = reject_ranges.tolist()
    wave = sp[:, 0] * (1.0 - rv/c_kms)
    mask = np.ones(len(wave), dtype=bool)
    if len(use_ranges) > 0:
        use_mask = np.zeros(len(wave), dtype=bool)
        for w0, w1 in use_ranges:
            use_mask |= (wave >= w0) & (wave <= w1)
        mask &= use_mask
    if len(reject_ranges) > 0:
        for w0, w1 in reject_ranges:
            mask &= ~((wave >= w0) & (wave <= w1))
    return np.column_stack((sp[:,0][mask], sp[:,1][mask]))

def list_telluric_files(setting_label, order):
    """
    List all telluric standard star spectra files for a given setting and order.

    The files are expected under:
        data/{setting_label}/{order}/

    Returns:
        list of file paths (both .txt and .fits allowed)
    """
    base_dir = os.path.join(HOME_DIR, "data", setting_label, order)
    if not os.path.isdir(base_dir):
        return []

    # Match both .txt and .fits
    files = sorted(glob.glob(os.path.join(base_dir, "*.txt")) +
                   glob.glob(os.path.join(base_dir, "*.fits")) +
                   glob.glob(os.path.join(base_dir, "*.FITS")))
    return files

# if voff or yoff is given, it will be subtracted.
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
        sp[:,0] *= (1.-voff/c_kms)
    if not np.isnan(yoff):
        sp[:,1] -= yoff
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
        sp[:,0] *= (1.-voff/c_kms)
    if not np.isnan(yoff):
        sp[:,1] -= yoff
    hdul.close()
    return sp, header

def save_new_fits(file_out,header,data,overwrite=True):
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdul_new = fits.HDUList([hdu])
    hdul_new.writeto(file_out, overwrite=overwrite)

#def vac2air(waves):
#    list_w_air = []
#    for w_vac in waves:
#        s = 1./w_vac
#        n = 1.0 + 5.792105e-2 / (238.0185 - s**2) + 1.67917e-3/(57.362 - s**2)
#        w_air = w_vac/n
#        list_w_air.append(w_air)
#    return np.array(list_w_air)

# IAU conversion in Morton (1991)
# w in Angstrom
def vac_to_air(w_vac):
    s2 = (1e4 / w_vac)**2
    n = 1.0 + 0.0000834254 + 0.02406147/(130 - s2) + 0.00015998/(38.9 - s2)
    return w_vac / n

def air_to_vac(w_air):
    s2 = (1e4 / w_air)**2
    n = 1.0 + 0.00008336624212083 + 0.02408926869968/(130.1065924522 - s2) \
        + 0.0001599740894897/(38.92568793293 - s2)
    return w_air * n

def vac_air_conversion(w, conversion='vac_to_air'):
    if conversion == 'vac_to_air':
        return vac_to_air(w)
    if conversion == 'air_to_vac':
        return air_to_vac(w)
    print(f'!! Unknown conversion={conversion} !!')
    return w

def load_model_waves(setting, order, vac_air='vac'):
    waves = np.loadtxt(f'{HOME_DIR}/models_txt/{setting}/waves_{order}_{vac_air}.txt')
    return waves

def load_model_base(setting, order, add_ones_column=False):
    list_base = np.loadtxt(f'{HOME_DIR}/models_txt/{setting}/base_{order}.txt')
    if add_ones_column:
        nline, nrow = list_base.shape
        ones_column = np.ones((nline,1))
        list_base = np.hstack((ones_column,list_base))
    return list_base

def load_model_ave(setting, order, vac_air='vac'):
    sp = np.loadtxt(f'{HOME_DIR}/models_txt/{setting}/ave_{order}_{vac_air}.txt')
    return sp

def load_tel_abs(setting, order, vac_air='vac'):
    return np.loadtxt(f'{HOME_DIR}/models_txt/{setting}/tel_abs_{order}_{vac_air}.txt')

def calc_continuum(sp, continuum_percentile=80, rv=0., use_ranges=None, reject_ranges=None):
    """
    Calculate continuum level as a percentile of the flux
    using only pixels in use_ranges minus reject_ranges.
    """
    if use_ranges is None:
        use_ranges = [(np.min(sp[:,0]), np.max(sp[:,0]))]
    if reject_ranges is None:
        reject_ranges = []
    sp_sel = apply_use_and_reject_ranges(sp, rv=rv, use_ranges=use_ranges, reject_ranges=reject_ranges)
    if len(sp_sel) == 0:
        return np.nan
    return np.percentile(sp_sel[:,1], continuum_percentile)

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

