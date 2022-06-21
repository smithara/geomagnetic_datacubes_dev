# Geomagnetic Datacubes Development

This repository holds resources used for prototyping the geomagnetic datacube concept.

## Setup

Assumes using Linux and conda environments

```
git clone https://github.com/smithara/geomagnetic_datacubes_dev.git
cd geomagnetic_datacubes_dev
```

Install conda/mamba. I recommend using [mambaforge](https://github.com/conda-forge/miniforge#mambaforge) if you have not already got conda installed. To install it:
```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
```
(but the following steps will work with conda too - just replace `mamba` with `conda`)


### Option A - recreate exact environment

1. Install [conda-lock](https://conda-incubator.github.io/conda-lock/)  
    You might do this with:  
    ```
    mamba install -n base pipx
    pipx install conda-lock[pip_support]
    ```
    (this installs the tool independently of your conda setup, as an application)
2. Create the `geomagcubes` environment using the `conda-lock.yml`:
    ```
    conda-lock install --name geomagcubes conda-lock.yml
    ```
3. Install the pip packages not covered in the conda-lock file:
    ```
    mamba activate geomagcubes
    pip install viresclient==0.10.1 hapiclient==0.2.3 chaosmagpy==0.8
    pip install -e .
    ```
    (the last one installs the code within this repository (under `src`) as an editable package)


### Option B - create similar environment (suitable if you need to customise it)
   

Using the `environment.yml` to create the environment:
```
mamba env create --file environment.yml --name geomagcubes
```
(the file already specifies the pip packages so there is no need to install them separately)

### Getting started

Activate the environment and launch JupyterLab from within it
```
mamba activate geomagcubes
jupyter lab
```

Browse to `./notebooks/00_datacube_quicklooks.ipynb` to download the datacube and view it.

**Important**: That notebook will download and store the prototype datacube at `./data/interim/SwA_20140501-20190501_proc1.nc` (~5GB) and a copy (for working on) at `./notebooks/datacube_test.zarr`. You may need to delete these manually and re-download if you previously used an older version of the code.

## Building the datacube from scratch

Code for building the datacube is within `src/data/`:
- Some configuration settings are within `src/data/proc_env.py`
- The processing steps can be run individually by running the scripts, `src/data/proc*.py`, or with just `src/data/make_dataset.py` which runs them all in series
  1. Fetching raw data over the internet (populating `data/raw/`):
    - `proc0a_fetch_raw_vires.py`: Fetches the data from VirES in chunks (takes many hours)
      (Swarm magnetic measurements and model predictions. Some auxiliary data such as Kp, MLT, QDLat etc.)
    - `proc0b_fetch_raw_hapi.py`: Fetches the OMNI solar wind data via HAPI (CDAWEB)
  2. Additional computation, merging, and translation to analysis-ready data format, (populating `data/interim/`):
    - `proc1_build_interim.py`: Merges data into one NetCDF file, computes smoothed solar wind data, assigns icosphere grid points.
  3. Create "processed" datasets in `data/processed/`:
    - `proc2_filter_bin.py`:

<!-- ## To-do

- Check memory usage and how to adjust the file chunk sizes and dask settings appropriately
- Remake data pipeline with `snakemake`?
- Automate updating of RC index? (http://www.spacecenter.dk/files/magnetic-models/RC/) -->
