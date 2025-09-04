import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import utils

png_dir='comp_tel'

npix=2048
buffer=20.

orders=['m43','m44','m45','m46','m47','m48','m52','m53','m54','m55','m56','m57','m60','m61']
#orders=['m42','m43','m44','m45','m46','m47','m48','m49','m50','m51','m52','m53','m54','m55','m56','m57','m58','m59','m60','m61']
order_wranges={'m42':'13200:13500','m43':'12900:13200', 'm44':'12600:12900', 'm45':'12320:12600', 'm46':'12050:12320', 'm47':'11800:12050', 'm48':'11560:11800', 'm49':'11320:11560', 'm50':'11100:11320', 'm51':'10890:11100', 'm52':'10680:10890', 'm53':'10480:10680', 'm54':'10280:10480', 'm55':'10100:10280', 'm56':'9920:10100', 'm57':'9760:9920', 'm58':'9580:9760', 'm59':'9420:9580', 'm60':'9280:9420', 'm61':'9120:9280'}

objs=utils.load_tel_list('tel_list')
for order in orders:
    plot_out=f'{png_dir}/{order}.png'
    wmin_str, wmax_str=order_wranges[order].split(':')
    wmin=float(wmin_str); wmax=float(wmax_str)
    waves=np.linspace(wmin-0.3*buffer,wmax+0.3*buffer,npix)
    fig=plt.figure(figsize=(15,12))
    axs={}
    axs['flux']=fig.add_axes([0.1,0.46,0.83,0.46])
    axs['std']=fig.add_axes([0.1,0.10,0.83,0.30])
    list_sp_obj=[]; list_label=[]
    for iobj, (label, obj) in enumerate(objs.items()):
        spfile=utils.get_sp_txt(obj,order)
        if spfile is None:
            continue
        sp_obj=utils.load_sp(spfile)
        axs['flux'].plot(sp_obj[:,0],sp_obj[:,1],label='{} ({})'.format(obj['star'],obj['date']))
        if iobj==0:
            interp_func_ref=CubicSpline(np.array(sp_obj[:,0]),np.array(sp_obj[:,1]))
            sp_new=np.vstack((waves,interp_func_ref(waves))).T
        else:
            xadjust=utils.calc_xadjust(sp_obj,wmin,wmax,interp_func_ref)
            sp_obj[:,0]+=xadjust
            yadjust=np.median(interp_func_ref(np.array(sp_obj[:,0]))-np.array(sp_obj[:,1]))
            sp_obj[:,1]+=yadjust
            interp_func=CubicSpline(np.array(sp_obj[:,0]),np.array(sp_obj[:,1]))
            sp_new=np.vstack((waves,interp_func(waves))).T
        list_sp_obj.append(sp_new)
        list_label.append(label)
    sp_median=np.median(list_sp_obj,axis=0)
    interp_func_median=CubicSpline(np.array(sp_median[:,0]),np.array(sp_median[:,1]))
    axs['flux'].plot(sp_median[:,0],sp_median[:,1],label='median',color='k',alpha=0.5,lw=3)
    axs['flux'].set_ylabel('Normalized Flux')
    axs['flux'].set_title(f'order={order}, [{wmin_str}, {wmax_str}]')
    axs['flux'].set_xlim([wmin-buffer,wmax+buffer])
    axs['flux'].set_ylim([0.45,1.05])
    axs['flux'].axvline
    
    sp_std_y=np.std([sp[:,1] for sp in list_sp_obj],axis=0)
    axs['std'].plot(waves,sp_std_y)
    axs['std'].set_xlabel('Air Wavelenth (AA)')
    axs['std'].set_ylabel('STDDEV')
    axs['std'].set_xlim([wmin-buffer,wmax+buffer])
    axs['std'].set_ylim([0,0.25])

    for panel, ax in axs.items():
        for w in [wmin,wmax]:
            ax.axvline(w,ls='dotted',color='k')
    fig.savefig(plot_out,dpi=240)
    plt.close(fig)


