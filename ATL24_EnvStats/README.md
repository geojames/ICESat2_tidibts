# ATL24 Environmental Stats  
Calculating environmental parameters from labeled ATL03 Datasets

Prerequsites:
Python 3.11
- glob, h5py, numpy, pandas, requests (2.31), rasterio (1.2.10)

Must have labeled data (from ICEVIS) and the full ATL03 granule H5 files

Run Options:

options:
  -h, --help            show this help message and exit
  -g GRAN, --gran GRAN  Folder path to ICEsat-2 *.h5 files
  -l LABEL, --label LABEL
                        Top level folder path to labeled files
  -d DEM, --dem DEM     Path to Reference Elevation DEM - Full path
  -o OUTPATH, --outpath OUTPATH
                        Path to output files
  -v, --verbose         increase output verbosity
