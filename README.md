# TerraPCA - Telluric Empirical Reduction & Reconstruction Analysis via PCA
Python toolkit for empirical modelling and correction of telluric absorption  
in **WINERED** spectra (WIDE, HIRES-Y, etc.) using Principal-Component Analysis (PCA).

---

## Why PCA?
Ground-based NIR spectra are riddled with time-variable telluric lines.
By observing many hot, feature-poor standards over a range of airmass and
humidity conditions, we can describe those lines with only a handful of
principal components.  A small number (typically 5–8) of components can
reproduce the WINERED telluric profile to < 1 % RMS per pixel across multiple orders.

---

## Repository structure
    data/{setting_label}/{order}/        telluric standard spectra (*.fits / *.txt)
    settings/setting_{label}.txt          configuration files (orders, ranges, n_pix, n_base)
    models_txt/{setting_label}/           PCA model outputs (waves_*, base_*, ave_*, tel_abs_*)
    models_plot/{setting_label}/           diagnostic plots from model building

---

## Quick-start

### 0. Install
    git clone https://github.com/your-user/winered-telluric-pca.git
    cd winered-telluric-pca
    pip install -r requirements.txt     # or use conda/pipx

### 1. Prepare spectra (once)
Put 2-column text or FITS spectra of telluric standards in:
    data/{setting_label}/{order}/
To check available spectra to use:
    python3 list_standard_data.py

### 2. Build PCA basis (once)
    python3 build_models.py --vac_air vac -s WINERED_WIDE
This creates:
    models_txt/WINERED_WIDE/waves_{order}_vac.txt
    models_txt/WINERED_WIDE/ave_{order}_vac.txt
    models_txt/WINERED_WIDE/base_{order}.txt
    models_plot/WINERED_WIDE/*.png

### 3. Mark strong telluric pixels (once for each order)
    bash make_tel_abs.sh
This uses list_tel_abs.py and saves:
    models_txt/WINERED_WIDE/tel_abs_{order}_vac.txt
    models_txt/WINERED_WIDE/tel_abs_{order}_air.txt

### 4. (Optional) Validate PCA basis on training set
    python3 check_models.py m44 --vac_air vac -s WINERED_WIDE
Plots are saved to:
    check_models/WINERED_WIDE/m44/

### 5. Fit any science spectrum
    python3 fit_model.py m44 path/to/sci.txt sci_model.txt \
           --vac_air vac -s WINERED_WIDE \
           --plot_out sci_fit.png --xadjust -t -e 0.1
The PNG shows:
   observed (red) vs. model (blue),
   blue bars = tel_abs regions,
   grey = sigma-clipped reject2,
   red = user reject1,
   residual panel with RMS.

---

## Script overview

### Basic utilities
    fits_to_txt.py               Convert WINERED FITS spectra to plain-text (λ flux)
    mask_from_spectrum.py        Detect and return wavelength ranges to mask based on flux threshold
    calc_sp_offset.py            Estimate RV/flux offset by residual minimization between spectra
    crosscorr_sp_offset.py       Estimate RV offset using normalized cross-correlation
    trace_continuum.py           Trace the smooth continuum of a 1D spectrum

### Model construction and validation
    build_models.py              Build order-by-order PCA basis; saves base_*, ave_*, waves_*
    list_tel_abs.py               Identify telluric absorption pixels from ave_*; saves tel_abs_*
    check_models.py               Fit each standard with the PCA basis and report RMS + PNGs

### Model application
    fit_model.py                  Fit a science spectrum using the PCA basis and output telluric model
    measure_telluric_offsets.py   Measure velocity offsets of multi-order object spectra vs PCA models

---

## Adapting to other instruments
a. Create `settings/setting_{LABEL}.txt` for the data set:
       m43 12870 13210 3072 6
       m44 12570 12930 3072 6
       ...
   (order wmin wmax n_pix n_base)

b. Place telluric standard spectra in:
       data/{LABEL}/{order}/

c. LABEL should be consistent throughout the analysis, and create the following directories.
       models_txt/{LABEL}/
       models_plot/{LABEL}/
       check_models/{LABEL}/{order}/

d. Run steps 1-4, listed above, to build PCA basis and tel_abs ranges.

---

## Citation
If this toolkit helps your research, please cite this GitHub repository.

## Acknowledgements
I'm grateful to Hiroaki Sameshima for his suggestion to use the PCA approach for
the WINERED telluric correction and following discussions. I also thank Daisuke
Taniguchi for discussions on the algorithm and toolkit. The WINERED HIRES-Y band
data of telluric standard stars are kindly provided by Emily Pass (MIT).
