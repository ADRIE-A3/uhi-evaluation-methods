# uhi-evaluation-methods

Code for the paper

> Adriaensen, L., Vandelanotte, K., Dewettinck, W., Top, S., Caluwaerts, S.
> *Same model, different skill: how Urban Heat Island definitions shape climate model evaluation.*
> Submitted to PNAS.

We apply seven urban heat island (UHI) evaluation methods to one convection-permitting regional
climate model simulation (ALARO1-SFX, 2.5 km, 2018–2022) for nine European cities (Amsterdam,
Barcelona, Berlin, Birmingham, Ghent, Lisbon, Rennes, Turku, Zurich). We then show how the
choice of method changes the model's apparent skill for the summer nighttime UHI (UHIN). The
methods differ in three characteristics:

| Characteristic | Options |
|---|---|
| Urban/rural grid-cell selection | point-to-point (`p2p`), area-to-area (`a2a`, URCLIMASK masks), point-to-area (`p2a`) |
| Observational network | Urban Climate Network (`UCN`), Meteorological Service (`MS`) |
| Station classification | urban centre (`uc`, GHS-UCDB), Local Climate Zone (`lcz`), distance to the centre (`dist`) |

### Method names in the code

The file and variable names predate the paper's naming. They translate as follows:

| Paper name | Code: variables / columns (`UHIN_<name>_{sim,obs,bias}`) | Computed in |
|---|---|---|
| p2p_UCN_uc | `p2p_un_ucdb` | `method_3_urclimask/` |
| p2p_UCN_lcz | `p2p_un_lcz` | `method_3_urclimask/` |
| p2p_MS_uc | `p2p_ms_ucdb` | `method_3_urclimask/` |
| a2a_UCN_uc | `urclim_un_ucdb` | `method_3_urclimask/` |
| a2a_UCN_lcz | `urclim_un_lcz` | `method_3_urclimask/` |
| a2a_MS_uc | `urclim_ms_ucdb` | `method_3_urclimask/` |
| p2a_MS_dist | `blend_ms_ucdb` ("method 2", blended approach) | `method_2_blended_approach/` |

Other code abbreviations: `un` = urban (climate) network (UCN), `ms` = meteorological service,
`ucdb` = urban-centre classification (uc), `city_network` = UCN, `meteo_stations` = MS,
`combined_network` = UCN and MS together. Some figure folders are called `method_2_urclimask_*`
for historical reasons; they hold p2p/a2a output.

* Data (Zenodo): https://doi.org/10.5281/zenodo.23591294
* Code archive (Zenodo): every release of this repository is archived on Zenodo; the DOI badge is added after the first release.

## Repository layout

```
scripts/
  preprocessing/
    metobs_cleaning_scripts/<City>_metobs_datacleaning.ipynb   QC + gap filling per city (MetObs-toolkit)
    clean_obs_to_datasets.ipynb                                 cleaned series -> hourly/daily/monthly netCDF
  analysis/
    method_2_blended_approach/UHI_blended_approach_<City>.ipynb p2a method (p2a_MS_dist)
      src/                                                      helper modules
    method_3_urclimask/Eurocities_urclimmask_<City>_.ipynb     p2p and a2a methods, station classification,
                                                                method comparison per city
      Eurocities_urclimmask_Rennes_illustration_methods.ipynb   Fig. 1 (Rennes worked example)
      urclimask/                                                modified copy of URCLIMASK (see NOTICE)
    comparing_methods/compare_methods.ipynb                     cross-city tables and SI dot plots
  plotting/
    fig2_bias_spread_rennes.ipynb                               Fig. 2
    domain_visualisation.ipynb                                  SI simulation-domain map
results/
  data/<City>/        per-city UHI annual cycles and biases (csv; netCDF on Zenodo)
  tables/             station land-cover fractions and the LaTeX tables of the paper
  figures/            figures used in the paper and SI
data/                 not in git: download from Zenodo (see below)
```

## Installation

```bash
conda env create -f environment.yml
conda activate UHI_eval_sens
```

The notebooks import the local `urclimask` package via a relative path, so run them from the
folder they are in.

## Data

Download the data record from Zenodo (DOI above) and unpack it into `data/` at the repository
root:

```
data/
  metadata/masks/                         urban/rural simulation masks
  metadata/simulations/                   static model fields (urban fraction, orography, land-sea mask)
  metadata/UCDB_shapefiles/               GHS-UCDB 2015 (download from JRC, see below)
  processed/simulations/monthly_means/    simulated TN, ALARO1-SFX
  processed/reanalysis/CERRA/             CERRA TN, regridded to the model grid, monthly means
  processed/reanalysis/ANASTASIA/         ANASTASIA TN, regridded to the model grid, monthly means
  processed/observations/<City>/<network>/  quality-controlled station data and station metadata
```

The regridded CERRA (Copernicus Climate Change Service) and ANASTASIA (Météo-France, Licence
Ouverte 2.0) fields are redistributed in the Zenodo record with attribution. The other
third-party data are not redistributed. Get them from the original providers:

| Dataset | Use | Source |
|---|---|---|
| GHCNh | MS station data (raw) | NOAA NCEI |
| FAIRUrbTemp | UCN station data (raw) | Amini et al. (2026) |
| Lisboa Aberta, XEMA | UCN data for Lisbon and Barcelona (raw) | Câmara Municipal de Lisboa; Meteocat |
| GHS-UCDB 2015 | urban centres | European Commission JRC |
| ERA5-Land | gap filling of station data | Copernicus Climate Data Store |
| CERRA, ANASTASIA (original grids) | observed rural reference (p2a) | Copernicus Climate Data Store; Météo-France |

## Reproducing the paper

The full chain, from raw observations to figures:

1. **Quality control** of the raw station data: `scripts/preprocessing/metobs_cleaning_scripts/<City>_metobs_datacleaning.ipynb`, then `clean_obs_to_datasets.ipynb`. This step needs the raw third-party data. You can skip it, because the Zenodo record contains its output.
2. **Methods per city**: first `method_2_blended_approach/UHI_blended_approach_<City>.ipynb` (p2a), then `method_3_urclimask/Eurocities_urclimmask_<City>_.ipynb` (p2p, a2a), which adds the p2a result. Together they write `results/data/<City>/`.
3. **Cross-city comparison**: `comparing_methods/compare_methods.ipynb`.

To regenerate only the figures and tables, step 3 and the plotting notebooks are enough. They
read the per-city results in `results/data/`.

| Paper item | Notebook | Output |
|---|---|---|
| Fig. 1 | `analysis/method_3_urclimask/Eurocities_urclimmask_Rennes_illustration_methods.ipynb` | `results/figures/Rennes/method_2_urclimask_city_network/illustration_different_methodsRennes_tasmin_v2.pdf` (assembled into `results/figures/Rennes/methods_illustration_rennes.png`) |
| Fig. 2 | `plotting/fig2_bias_spread_rennes.ipynb` | `results/figures/Rennes/bias_spread_rennes.png` |
| Table 1 (bias, observed and simulated UHIN) | `analysis/comparing_methods/compare_methods.ipynb` | `results/tables/comparing_methods/uhin_merged_table.tex` (separate blocks in the other `*.tex` files there) |
| SI simulation domain | `plotting/domain_visualisation.ipynb` | `results/figures/Supplementary_Information/simulation_domain_with_cities.png` |
| SI dot plots | `analysis/comparing_methods/compare_methods.ipynb` | `results/figures/Supplementary_Information/dotplot_UHIN_{bias,obs}_JJA_per_method.pdf` |
| SI network tables | `analysis/method_3_urclimask/Eurocities_urclimmask_<City>_.ipynb` | `results/data/<City>/obs_attributes_*.tex` |

## License and attribution

The code is released under the [Apache License 2.0](LICENSE).
`scripts/analysis/method_3_urclimask/urclimask/` contains a modified copy of
[URCLIMASK](https://github.com/FPS-URB-RCC/URCLIMASK) (Copyright 2023, European Union, Apache-2.0).
See [NOTICE](NOTICE) for the changes, and cite Diez-Sierra et al. (2025),
*Scientific Data* 12, 1961, https://doi.org/10.1038/s41597-025-06257-1.

Observations were quality-controlled with the
[MetObs-toolkit](https://github.com/vergauwenthomas/MetObs_toolkit) (Vergauwen et al., 2024).

## Citation

See [CITATION.cff](CITATION.cff). Please cite both the paper and the Zenodo archive of this code.

## Contact

Leon Adriaensen, Ghent University (e-mail: leon.adriaensen@ugent.be)
