# Geomagnetic Datacubes Development

This repository holds resources used for prototyping the geomagnetic datacube concept.

## Setup

Create a conda environment (it will be called `geomagcubes`):
```
conda env create --file environment.yml --name geomagcubes
```
Activate that environment and install the repository package itself (making it editable):
```
conda activate geomagcubes
pip install -e .
```
Now code within the `src` directory is available as a package called `src`

## Building the datacube

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

## To-do

- Check memory usage and how to adjust the file chunk sizes and dask settings appropriately
- Remake data pipeline with `snakemake`?
- Automate updating of RC index? (http://www.spacecenter.dk/files/magnetic-models/RC/)
