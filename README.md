# TAP-BD: Turbulence-Aware Poisson Blind Deconvolution

**TAP-BD** is a framework for robust image reconstruction under strong atmospheric turbulence and extremely low-photon conditions.  
This repository provides three Matlab-based implementations: two for simulation and one for experimental data.

---

## Main Scripts

### `Main_TAPBD_SimSmallMeas.m` (Simulation)
- Simulation study for small number of measurements challenge
- Uses FFT-based forward propagation  
- Recovers both the target image and turbulence phase  
- High photon cases; **Poisson denoiser not used**
  
### `Main_TAPBD_SimExtremeTurb.m` (Simulation)
- Simulation study for extreme turbulence challenge
- Uses FFT-based forward propagation  
- Recovers both the target image and turbulence phase  
- High photon cases; **Poisson denoiser not used**
- 
### `Main_TAPBD_ExpLowSignal.m` (Experiment)
- Low-photon experimental study
- Uses angular spectrum propagation model  
- Recovers both object and turbulence phase
- Low photon cases; Applies Anscombe + wavelet-based **Poisson denoising**

**Note**: All scripts are designed to be run section-by-section (not as a single run).  
> Each script is structured into modular sections, allowing the user to run and inspect results step-by-step.
---

## Dataset

Required `.mat` files are already available in the `/DataSet/` folder and are loaded automatically by each script:
