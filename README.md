# Learning low-dimensional representations of ensemble forecast fields using autoencoder-based methods

This repository provides python codes for the dimensionality reduction approaches to compressing ensemble forecast fields accompanying the paper

> Chen, J., Janke, T., Steinke, F. and Lerch, S., 2025. Learning low-dimensional representations of ensemble forecast fields using autoencoder-based methods. *arXiv preprint arXiv:2502.04409*. https://arxiv.org/abs/2502.04409

## Data

The data needed for reproducing the results is publicly available:

> Chen, Jieyu; Lerch, Sebastian (2025). Gridded dataset of daily 2-m temperature forecast from the ECMWF 50-member ensemble forecast. figshare. Dataset. https://doi.org/10.6084/m9.figshare.28151213.v2

> Chen, Jieyu; Lerch, Sebastian (2025). Gridded dataset of daily U component of 10-m wind speed forecast from the ECMWF 50-member ensemble forecast. figshare. Dataset. https://doi.org/10.6084/m9.figshare.28151372.v1

> Chen, Jieyu; Lerch, Sebastian (2025). Gridded dataset of daily V component of 10-m wind speed forecast from the ECMWF 50-member ensemble forecast. figshare. Dataset. https://doi.org/10.6084/m9.figshare.28151411.v1

> Chen, Jieyu; Lerch, Sebastian (2025). Gridded dataset of daily geopotential height at 500 hPa forecast from the ECMWF 50-member ensemble forecast. figshare. Dataset. https://doi.org/10.6084/m9.figshare.28151444.v1


**ECMWF forecasts from TIGGE dataset**

https://software.ecmwf.int/wiki/display/TIGGE/Models

- Forecast data: 2-days ahead 50-member ensemble forecasts
- Forecast time range: Daily forecasts from 2007-01-03 to 2017-01-02
- Meteorological variables

|Variable| Description|
|-------------|---------------|
|t2m| 2-m temperature|
|z500| Geopotential height at 500 hPa|
|u10| 10-m U component of wind|
|v10| 10-m V component of wind|

## Explanation of the code files

- For reproducing the reconstructed forecast fields presented in the main paper (*please download the datasets above from figshare first*):

|File name| Explanation |
|-------------|---------------|
|**`drf_pca_eachens.py`**| Python script to implement the PCA-based approach. |
|**`drf_ae_simple_eachens.py`**| Python script to implement the AE-based approach. |
|**`drf_ivae_simple_eachens.py`**| Python script to implement the invariant VAE approach. |

- Others:

|File name| Explanation |
|-------------|---------------|
|`hptune_ae_simple.py`| Code for hyperparameter tuning of the AE neural network model. |
