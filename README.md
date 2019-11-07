# tart2ms

Convert data from a [TART radio telescope](https://tart.elec.ac.nz) to measurement set format. This module relies on the excellent dask-ms module as a helper to create the measurement sets. 

## Examples

Download data from the TART in real time via the RESTful API (defaults to using the API at https://tart.elec.ac.nz/signal):

    tart2ms --ms data.ms

To convert a previously downloads JSON file to a measurement set (MS):

    tart2ms --json data.json --ms data.ms

To synthesize (using wsclean) the image from the measurement set:

    wsclean -name test -size 1280 1280 -scale 0.0275 -niter 0 data.ms
 
This will create an image called test-image.fits

## Usage

    usage: tart2ms [-h] [--json JSON] [--ms MS] [--api API] [--catalog CATALOG]
                [--vis VIS]

    Generate measurement set from a JSON file from the TART radio telescope.

    optional arguments:
    -h, --help         show this help message and exit
    --json JSON        Snapshot observation JSON file (visiblities, positions
                        and more). (default: None)
    --ms MS            Output MS table name. (default: tart.ms)
    --api API          Telescope API server URL. (default:
                        https://tart.elec.ac.nz/signal)
    --catalog CATALOG  Catalog API URL. (default:
                        https://tart.elec.ac.nz/catalog)
    --vis VIS          Use a local JSON file containing the visibilities for
                        visibility data (default: None)

## Credits

Thanks to Simon Perkins and Oleg Smirnov for help in interpreting the measurement set documentation.


## TODO

- Add the correct pointing direction of the array (in J2000).
- Sort out the correct polarization settings for LHCP.

## Changelog

-0.1.1 Added switch to generate a second polarization.
- 0.1.0 first functioning release.
