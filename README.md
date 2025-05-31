# TAP-BD: Turbulence-Aware Poisson Blind Deconvolution

TAP-BD is a framework for robust image reconstruction under strong atmospheric turbulence and extremely low-photon conditions.  
This repository provides two implementations: one for simulation and one for experimental data.
---

## Main Scripts

### `Main_TAPBD_SimHighSignal.m` (Simulation)
- Designed for high-photon, simulation-based testing
- Uses FFT-based forward propagation
- Recovers both the target image and turbulence phase
- **Poisson denoiser not used**
- Outputs reconstructions across different numbers of SLM patterns

### `Main_TAPBD_ExpLowSignal.m` (Experiment)
- Designed for real, low-photon experimental data
- Applies Anscombe + wavelet-based **Poisson denoising**
- Uses angular spectrum propagation model
- Incorporates optical system parameters (focal length, wavelength, etc.)
- Recovers both object and turbulence phase

---

## Dataset

Required `.mat` files are already available in the `/DataSet/` folder and will be loaded automatically by each script.
