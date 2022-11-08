# tart2ms

[![PyPI package](https://img.shields.io/badge/pip%20install-tart2ms-brightgreen)](https://pypi.org/project/tart2ms) [![version number](https://img.shields.io/pypi/v/tart2ms?color=green&label=version)](https://github.com/tart-telescope/tart2ms/releases) [![License](https://img.shields.io/github/license/tart-telescope/tart2ms)](https://github.com/tart-telescope/tart2ms/blob/master/LICENSE.txt)


Convert data from a [TART radio telescope](https://tart.elec.ac.nz) to measurement set format. This module relies on the excellent dask-ms module as a helper to create the measurement sets. This packate requires python-casacore to be installed on your system

## Install

    sudo aptitude install python3-casacore
    sudo pip3 install tart2ms

## Examples

Download data from the TART in real time via the RESTful API (defaults to using the API at https://tart.elec.ac.nz/signal):

    tart2ms --ms data.ms

To convert a previously downloads JSON file to a measurement set (MS):

    tart2ms --json data.json --ms data.ms

JSON-based datasets currently may only contain a single timestamp. This limits their usefulness
when it comes to more general imaging. It is possible to make a concatenated database from multiple
single-dump json databases, e.g.
    
    tart2ms --json ../tart_data/NZ_2022_10_19_json/*.json

HDF5 format archive files may contain multiple timestamps and may also be concatenated into
longer observations as is the case for JSON archive files.

    tart2ms --hdf ../tart_data/NZ_2022_10_19/*.hdf

JSON databases may be exported from HDF5 archives using
   
    tart_vis2json --vis ../NZ_2022_10_19/*.hdf 

Currently each such exported JSON database will contain a single timestamp (thus multiple JSON databases
may result from a single HDF5 archive).

Your telescope name may not be in the JPL list of recognized observatories which at present
raises an error in casacore and hence some casa tasks like listobs or plotants, even though the
antenna table contains valid ITRF coordinates for the antennae. We recommend that if problems are
encountered the telescope name is changed to an existing observatory like kat-7 or MeerKAT.

    tart2ms --json ../tart_data/NZ_2022_10_19_json/*.json -c --telescope_name 'kat-7'

Standard CASA tasks may be executed with the CASA memo 229-compliant (MSv2.0) databases written by tart2ms.
These may include (tested):
  - listobs
  - plotms
  - fixvis
  - plotants
  - clean

To synthesize (using wsclean) the image from the measurement set:

    wsclean -name test -size 1280 1280 -scale 0.0275 -niter 0 data.ms
 
This will create an image called test-image.fits. You will need to install wsclean on your system.

## Usage

    usage: tart2ms [-h] [--json JSON] [--ms MS] [--api API] [--catalog CATALOG]
                [--vis VIS] [--pol2]

    Generate measurement set from a JSON file from the TART radio telescope.

    optional arguments:
    -h, --help         show this help message and exit
    --json JSON        Snapshot observation JSON file (visiblities, positions
                        and more). (default: None)
    --hdf HDF          Visibility hdf5 file (One minutes worth of visibility data). (default: None)
    --ms MS            Output MS table name. (default: tart.ms)
    --api API          Telescope API server URL. (default:
                        https://tart.elec.ac.nz/signal)
    --catalog CATALOG  Catalog API URL. (default:
                        https://tart.elec.ac.nz/catalog)
    --vis VIS          Use a local JSON file containing the visibilities for
                        visibility data (default: None)
    --pol2             Fake a second polarization. Some pipelines choke if there
                        is only one. (default: False)

## Credits

Thanks to Simon Perkins and Oleg Smirnov for help in interpreting the measurement set documentation.


## TODO

- 

## Changelog

- 0.4.0b1 Add a tart2ms.read_ms function (from disko)
          Add utilities for resolution calculations.
- 0.3.0b3 New features for keeping the phase center constant during a long measurement set (fringe phasing)
          Fix progress bar.
- 0.3.0b1 Fix many CASA compatability issues. Antenna positions are done. Many thanks to Ben Hugo of SARAO.
- 0.2.0b5 Fix bug in the timensions of the TIME_CENTROID column in the MAIN table. Issue 8.
- 0.2.0b3 Fix bug in the timensions of the TIME column in the MAIN table. Issue 7.
- 0.2.0b3 Move to the tart-telescope organization on github..
- 0.2.0b2 Place all visibilities from HDF5 files into a single measurement set..
- 0.2.0b1 Add importing of HDF5 files saved from the web app.
- 0.1.4b4 clean up some bitrot in dask-ms (dealing with chunking objects)
- 0.1.4b3 Add SIGMA, FLAG, FLAG_CATEGORY to main table (:/)
- 0.1.4b1 Add RESOLUTION and EFFECTIVE_BW to the SPECTRAL_WINDOW
- 0.1.3b1 Sort out the timestamps correctly, added a handy function for converting to epoch time.
- 0.1.2 Correct pointing direction of the array (in J2000).
- 0.1.1 Added -pol2 switch to generate a second polarization.
- 0.1.0 first functioning release.
