# winered-telluric-pca
Python toolkit for empirical modelling and correction of telluric absorption  
in **WINERED** WIDE-mode spectra using Principal-Component Analysis (PCA).

---

## Why PCA?
Ground-based NIR spectra are riddled with time-variable telluric lines.
By observing many hot, feature-poor standards over a range of airmass and
humidity conditions, we can describe those lines with only a handful of
principal components.  Six PCs reproduce the WINERED telluric profile to
< 1 % RMS per pixel across 11 orders (m43–m48, m52, m55–m58).

---

## Repository contents
data/ list of standard star data, zipped reference spectra 
code/ Python scripts, shell scripts

---

## Quick-start

### 0. Install
```bash
git clone https://github.com/your-user/winered-telluric-pca.git
cd winered-telluric-pca
pip install -r requirements.txt     # or use conda/pipx

1. Prepare spectra (once)
Convert all FITS files listed in data/tel_list into 2-column text:
python3 fits_to_txt.py
2. Build PCA basis (once)
python3 get_eigens.py
3. Mark strong telluric pixels (once for all orders)
bash make_tel_abs.sh   
4. (Option) Validate basis on training set
bash check_model.sh           # saves PNGs in check_model/<order>/
5. Fit any science spectrum
python fit_model.py m44 path/to/sci.txt sci_model.txt \
       --plot sci_fit.png --xadjust -t -e 0.1
The PNG shows:
       observed (red) vs. model (blue),
       blue bars = tel_abs regions,
       grey = sigma-clipped reject2,
       red = user reject1,
       residual panel with RMS.

Script overview
script purpose
fits_to_txt.py	Convert WINERED FITS spectra to plain-text (λ flux).
get_eigens.py	Build order-by-order PCA basis, save base_*, ave_*, diagnostic plots.
make_tel_abs.py	Find pixels with flux < threshold in ave_*, add ±3-pixel buffer, write tel_abs_*.
check_model.py	Fit each standard star with the PCA basis, output per-star PNG + statistics.
fit_model.py	Fit a new spectrum; supports vacuum/air wavelengths, edge trimming, σ-clipping, user masks.

Wrapper shells (*.sh) automate running some steps for all orders.

Adapting to other instruments
Change:
    utils.order_wranges_with_buffer – order limits & buffers
    wavelength grid in utils.make_order_waves
    prepare a new tel_list + FITS paths, then rerun steps 0–3.

Citation
If this toolkit helps your research, please cite this GitHub repository.

