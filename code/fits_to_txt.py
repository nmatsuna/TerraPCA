import os
import numpy as np
import astropy.io.fits as fits

orders=['m42','m43','m44','m45','m46','m47','m48','m52','m53','m54','m55','m56','m57','m58','m60','m61']
#orders=['m42','m43','m44','m45','m46','m47','m48','m49','m50','m51','m52','m53','m54','m55','m56','m57','m58','m59','m60','m61']

fh=open('tel_list')
for line in fh.readlines():
    if line.startswith('#'):
        continue
    run, label, obj, date=line.split()
    out_dir=f'{run}/{label}/txt'
    if run in ['NTT17a']:
        fits_dir=f'{run}/{label}/telluric/FITS/cut5'
    elif run in ['NTT17b']:
        fits_dir=f'{run}/{label}/telluric/FITS/cut1'
    elif run in ['LCO23a']:
        fits_dir=f'{run}/{label}/telluric/FITS/nocut'
    else:
        fits_dir=f'{run}/{label}/telluric/FITS/fsr1.30'
    if not os.path.isdir(fits_dir):
        print(f'{fits_dir} does not exist!')
        continue
    for order in orders:
        fitsname=f'{fits_dir}/telluric_{order}.fits'
        if not os.path.isfile(fitsname):
            print(f'Failed find the fits (order={order}) for {run}/{label}')
            continue
        print(f'{run} {label:13s} {order}')
        hdulist=fits.open(fitsname)
        flux=hdulist[0].data
        header=hdulist[0].header
        crval1=header['CRVAL1']
        crpix1=header['CRPIX1']
        cdelt1=header['CDELT1']
        num_pixels=len(flux)
        pixel_values=np.arange(1,num_pixels+1)
        wave=crval1+(pixel_values-crpix1)*cdelt1
        output=f'{out_dir}/{order}.txt'
        fout=open(output,'w')
        for w, f in zip(wave,flux):
            print(f'{w:<12.6f} {f:9.6f}',file=fout)
        fout.close()
fh.close()
