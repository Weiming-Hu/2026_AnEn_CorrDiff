# Comparing Analog Ensemble and Corrective Diffusion for Western US Precipitation Downscaling from GraphCast

This repository contains the research code for the associated manuscript [... DOI to be included here ...]. The work compares two approaches: Analog Ensemble method and corrective diffusion models for downscaling GraphCast global weather forecasts to high resolution PRISM observations.

## Overview

Precipitation downscaling transforms coarse resolution numerical weather predictions into fine scale estimates suitable for regional applications. This codebase implements and evaluates multiple approaches for downscaling GraphCast forecast data (approximately 25 km resolution) to PRISM observational grid spacing (4 km resolution) over the Western United States.

### Key Methods

**Analog Ensemble**: An efficient ensemble forecast generation method. For more information and how to run AnEn, please refer to [the official website](https://weiming-hu.github.io/AnalogsEnsemble/).

**Corrective Diffusion**: A two-stage approach where a denoising diffusion probabilistic model (DDPM) learns to correct residuals from either raw GraphCast predictions or the direct U-Net baseline. The diffusion model employs a noise prediction objective with reconstruction regularization and generates ensemble predictions through multiple stochastic sampling passes.

## Project Structure

```
.
├── datasets/               # Data loading and preprocessing
│   ├── run_graphcast.py    # GraphCast forecasts generation script
│   ├── GCnPRISM.py        # GraphCast and PRISM data pairing
│   ├── Residual.py        # Residual computation for corrective approach
│   └── PickleFolder.py    # Utilities for pickled data management
├── models/                 # Model implementations
│   ├── Direct.py          # Direct U-Net downscaling model
│   └── Diffusion.py       # DDPM corrective diffusion model
├── networks/               # Neural network architectures
│   ├── AttResUnet.py      # Attention-based residual U-Net
│   ├── SR3Unet.py         # Super-resolution diffusion U-Net
│   └── SimpleUnet.py      # Basic U-Net implementation
├── utils/                  # Verification and utility functions
│   ├── crps.py            # CRPS score decomposition (Hersbach 2000)
│   ├── rel.py             # Reliability diagram calculations
│   ├── download.py        # Data download utilities
│   └── presets.py         # Visualization presets
├── Project_EarthData_main.py      # Main training and evaluation script
├── Project_EarthData_Training.py  # Training utilities and callbacks
├── PRISM_Download.ipynb           # PRISM data acquisition
└── PRISM_Restructure.ipynb        # PRISM data preprocessing
```

## Data Sources

**GraphCast Forecasts**: Global atmospheric forecasts generated using the GraphCast model via the ECMWF ai-models framework (https://github.com/ecmwf-lab/ai-models). GraphCast provides deterministic weather forecasts at 0.25 degree resolution with multiple atmospheric variables.

**PRISM Observations**: High resolution gridded precipitation observations from the Parameter-elevation Relationships on Independent Slopes Model ([PRISM](https://www.prism.oregonstate.edu/)), providing daily precipitation at 4 km resolution over the conterminous United States.

## Usage

### Training and Evaluation

The main script does the training and evaluation tasks:

```bash
python Project_EarthData_main.py \
    --lead_time 1 \
    --unet_type AttRes \
    --val_period 2022-10-01_2023-09-30 \
    --test_period 2023-10-01_2024-09-30 \
    --gen_members 15
```

**Parameters**:
* `lead_time`: Forecast lead time in days (1 through 7)
* `unet_type`: Architecture choice (AttRes or SR3)
* `val_period`: Validation period in YYYY-MM-DD_YYYY-MM-DD format
* `test_period`: Test period in YYYY-MM-DD_YYYY-MM-DD format
* `gen_members`: Number of ensemble members for diffusion sampling

### Configuration

Model and training parameters are specified in the `args` dictionary within `Project_EarthData_main.py`:

```python
args = {
    'io': {
        'forecast_path': 'path/to/GraphCast/daily.nc',
        'observation_path': 'path/to/PRISM/PRISM_daily.nc',
    },
    'model': {
        'base_channels': 128,
    },
    'training': {
        'batch_size': 8,
        'unet_lr': 1e-3,
        'unet_max_epochs': 200,
        'diffusion_lr': 1e-4,
        'diffusion_max_epochs': 2000,
        'corrective_baseline': 'Unet',  # 'Unet' or 'GC'
    },
}
```

## Verification Metrics

The codebase implements comprehensive forecast verification:

* **RMSE**: Root mean squared error for deterministic accuracy
* **CRPS**: Continuous Ranked Probability Score with Hersbach decomposition into reliability, resolution, and uncertainty components
* **Reliability Diagrams**: Calibration assessment for probabilistic forecasts

## HPC Integration

Batch submission scripts are provided for high performance computing environments:

* `Project_EarthData_main.sh`: Shell wrapper for environment setup
* `Project_EarthData_main.job`: SLURM job configuration
* `Project_EarthData_main.submit`: Job submission with multiple configurations

## Output

Results are saved in the `lightning_logs` directory with separate folders for each model type and test period. The final predictions file contains:

* `obs`: Ground truth PRISM observations (mm)
* `gc`: Raw GraphCast precipitation (mm) 
* `unet`: Direct U-Net predictions (mm)
* `cd`: Corrective diffusion predictions (mm)
* `cd_ens`: Full ensemble of diffusion predictions (members × samples × height × width)
* Temporal metadata (initialization times, valid times)
* Model checkpoint references

## Citation [TODO after manuscript acceptance]

If you use this code for your research, please cite the associated paper:

```
Comparing Analog Ensemble and Corrective Diffusion for Western US Precipitation 
Downscaling from GraphCast
```

## License

See LICENSE file for details.
