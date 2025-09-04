import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import utils

txt_dir='eigen_txt'

orders=utils.orders
order_wranges=utils.order_wranges_with_buffer

objs=utils.load_tel_list('tel_list')

for order in orders:
    wrange=order_wranges[order]
    wmin_str, wmax_str=wrange.split(':')
    wmin=float(wmin_str); wmax=float(wmax_str)
    waves=np.loadtxt(f'{txt_dir}/waves_{order}.txt')
    sp_ave=np.loadtxt(f'{txt_dir}/ave_{order}.txt')
    wmin2=min(sp_ave[:,0]); wmax2=min(sp_ave[:,0])
    print(f'# {order} {wmin_str}:{wmax_str}')
    func_ave=CubicSpline(np.array(sp_ave[:,0]),np.array(sp_ave[:,1]))
    eigens=utils.load_eigen(order)
    for label, obj in objs.items():
        spfile=utils.get_sp_txt(obj,order)
        if spfile is None:
            continue
        sp_obs=utils.load_sp(spfile,frange=[0,1.5])
        xadjust=utils.calc_xadjust(sp_obs,wmin,wmax,func_ave)
        sp_obs[:,0]+=xadjust
        wmin_obs=min(sp_obs[:,0]); wmax_obs=max(sp_obs[:,0])
        func_obs=CubicSpline(np.array(sp_obs[:,0]),np.array(sp_obs[:,1]))
        pixels_use=utils.check_pixels_use(waves,sp_obs[:,0])
        waves_part=waves[pixels_use]
        eigens_part=eigens[pixels_use]
        sp_interp=func_obs(waves_part)
        x, residuals, rank, s=np.linalg.lstsq(eigens_part,sp_interp,rcond=None)
        sp_model=eigens_part@x
        sp_model+=np.mean(sp_interp-sp_model)
        sigma=np.std(sp_interp-sp_model)
        print(f'{label} {sigma:.5f}')
        xmin=wmin-15.; xmax=wmax+15.
        low, up=min(sp_obs[:,1]), max(sp_obs[:,1])
        ymin=min([(up+low)/2.-0.6*(up-low),0.95])
        ymax=max([(up+low)/2.+0.6*(up-low),1.03])
        fig=plt.figure(figsize=(15,8))
        axs={}
        axs['sp']=fig.add_axes([0.10,0.46,0.80,0.46])
        axs['dev']=fig.add_axes([0.10,0.10,0.80,0.30])
        axs['sp'].plot(sp_obs[:,0],sp_obs[:,1],color='red',alpha=0.7,label='obs')
        axs['sp'].plot(waves_part,sp_model,color='blue',alpha=0.7,label='model')
        axs['sp'].axhline(1,ls='dotted',lw=0.7,color='k')
        axs['sp'].axvline(wmin,ls='dotted',lw=0.7,color='k')
        axs['sp'].axvline(wmax,ls='dotted',lw=0.7,color='k')
        axs['sp'].legend(bbox_to_anchor=(0.99,0.03),loc='lower right')
        axs['sp'].set_xlim([xmin,xmax])
        axs['sp'].set_ylim([ymin,ymax])
        axs['sp'].set_ylabel('Normalized Flux')
        axs['sp'].set_title(f'{label} ({order}, {wrange})')
        axs['dev'].plot(waves_part,sp_interp-sp_model,zorder=1,color='gray')
        axs['dev'].axhline(0.,ls='dotted',color='k',zorder=2)
        axs['dev'].axvline(wmin,ls='dotted',lw=0.7,color='k')
        axs['dev'].axvline(wmax,ls='dotted',lw=0.7,color='k')
        axs['dev'].text(0.01*xmin+0.99*xmax,0.11,r'$\sigma ={:.4f}$'.format(sigma),horizontalalignment='right',verticalalignment='top')
        axs['dev'].set_xlim([xmin,xmax])
        axs['dev'].set_ylim([-0.12,0.12])
        axs['dev'].set_xlabel('Wavelength')
        fig.savefig(f'model_tel/{label}_{order}.png',dpi=240)
        plt.close(fig)
