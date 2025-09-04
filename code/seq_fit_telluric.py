import os
import sys
import glob
import copy
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u

orders=['m43', 'm44', 'm45', 'm46', 'm47', 'm48', 'm51', 'm52', 'm55', 'm56', 'm57', 'm58','m61']

runs=['LCO24a','LCO24b','LCO25a']

def which_order_txt(txtfile):
    for order in orders:
        if order+'.txt' in txtfile:
            return order
    return None

for run in runs:
    targets=glob.glob(f'{run}/*')
    for target in targets:
        if not os.path.isdir(target):
            continue
        txtfiles=glob.glob(f'{target}/txt/*.txt')
        for txtfile in txtfiles:
            order=which_order_txt(txtfile)
            if order is None:
                continue
            file_out=target+'/model/'+(txtfile.split('/'))[-1]
            plot_out=target+'/fitted/'+(txtfile.split('/'))[-1].replace('txt','png')
            print(f'python3 /mnt/SharedDisk/WINERED/telluric/fit_model.py {order} {txtfile} {file_out} -p {plot_out} --clip_margin 1 -t -x')

