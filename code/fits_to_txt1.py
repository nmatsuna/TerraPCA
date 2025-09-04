import os
import sys
import numpy as np
import astropy.io.fits as fits


orders=['m42','m43','m44','m45','m46','m47','m48','m51','m52','m53','m54','m55','m56','m57','m58','m60','m61']
wranges={'m42':'13200:13500','m43':'12900:13200', 'm44':'12600:12900', 'm45':'12320:12600', 'm46':'12050:12320', 'm47':'11800:12050', 'm48':'11560:11800', 'm49':'11320:11560', 'm50':'11100:11320', 'm51':'10890:11100', 'm52':'10680:10890', 'm53':'10480:10680', 'm54':'10280:10480', 'm55':'10100:10280', 'm56':'9920:10100', 'm57':'9760:9920', 'm58':'9580:9760', 'm59':'9420:9580', 'm60':'9280:9420', 'm61':'9120:9280'}

if len(sys.argv)!=3:
    print('python3 {} input.fits output.txt'.format(os.path.basename(__file__)))
    exit()
input_fits=sys.argv[1]
output_txt=sys.argv[2]
if not os.path.isfile(input_fits):
    print(f'{input_fits} does not exist!')
    exit()

hdulist=fits.open(input_fits)
flux=hdulist[0].data
header=hdulist[0].header
crval1=header['CRVAL1']
crpix1=header['CRPIX1']
cdelt1=header['CDELT1']
num_pixels=len(flux)
pixel_values=np.arange(1,num_pixels+1)
wave=crval1+(pixel_values-crpix1)*cdelt1
fout=open(output_txt,'w')
for w, f in zip(wave,flux):
    print(f'{w:<12.6f} {f:9.6f}',file=fout)
fout.close()

